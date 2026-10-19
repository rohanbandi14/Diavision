import json
import os
import time
import base64
from io import BytesIO
import cv2
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rest_framework.decorators import api_view
from rest_framework.response import Response

from groq import Groq

from .apps import DiavisionApiConfig


# ==========================================================
# GROQ CLIENT
# ==========================================================

groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY", "")
)


# ==========================================================
# 0. HERA CONFIGURATION
# ==========================================================

FIELD_INFO = {

    "age": {
        "asks_for": "how old the patient is",
        "conversational": True,
        "high_importance": True,
    },

    "BMI": {
        "asks_for": (
            "BMI -- if unknown, just ask for height and weight "
            "(any units are fine, e.g. feet/inches or cm, and "
            "kg or lbs); it will be calculated automatically, "
            "you do not need to do any math yourself"
        ),
        "conversational": True,
        "high_importance": True,
    },

    "diastolic_pressure": {
        "asks_for": (
            "diastolic (lower number) blood pressure reading -- "
            "ask if they have a recent home or clinic reading"
        ),
        "conversational": False,
        "high_importance": True,
    },

    "systolic_pressure": {
        "asks_for": (
            "systolic (upper number) blood pressure reading -- "
            "ask if they have a recent home or clinic reading"
        ),
        "conversational": False,
        "high_importance": False,
    },

    "diabetic_nephropathy": {
        "asks_for": (
            "whether a doctor has ever told them they have kidney "
            "problems/disease related to diabetes"
        ),
        "conversational": True,
        "high_importance": True,
    },

    "number_of_lower_leg_artery_occlusion": {
        "asks_for": (
            "whether a doctor has found narrowed or blocked leg "
            "arteries (peripheral artery disease) -- if they don't "
            "know an exact count, 0 if never diagnosed, otherwise "
            "ask them to estimate or say 'not sure'"
        ),
        "conversational": True,
        "high_importance": True,
    },

    "hypertension": {
        "asks_for": (
            "whether they have been diagnosed with high blood pressure"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "corory_heart_disease": {
        "asks_for": (
            "whether they have been diagnosed with coronary heart disease"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "smoking_time.Y.": {
        "asks_for": (
            "how many years they have smoked "
            "(0 if never/not currently)"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "Number of cigarettes smoked per day": {
        "asks_for": (
            "how many cigarettes per day, if they smoke"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "sat_fat_intake": {
        "asks_for": (
            "how often they eat fried, fatty or oily food, red meat, "
            "butter/ghee -- rarely, sometimes or often"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "sleep_hours": {
        "asks_for": (
            "how many hours they usually sleep per night"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "family_history": {
        "asks_for": (
            "whether a parent or sibling has/had diabetes or heart disease"
        ),
        "conversational": True,
        "high_importance": False,
    },

    "CRP": {
        "asks_for": (
            "their CRP (C-reactive protein) blood test result in mg/L, "
            "if they have a recent lab report -- most patients will "
            "NOT know this from memory, so don't push hard on it"
        ),
        "conversational": False,
        "high_importance": False,
    },

    "Controlling glucose by insulin injection": {
        "asks_for": (
            "whether they control their blood sugar using insulin injections"
        ),
        "conversational": True,
        "high_importance": False,
    },
}


# ==========================================================
# REQUIRED FIELDS
# ==========================================================

REQUIRED_FOLLOWUP_FIELDS = [

    field

    for field, info in FIELD_INFO.items()

    if info["high_importance"]

]


# ==========================================================
# VALID RANGES
# ==========================================================

VALID_RANGES = {

    "age": (0, 120),

    "BMI": (10, 80),

    "diastolic_pressure": (30, 160),

    "systolic_pressure": (60, 260),

    "number_of_lower_leg_artery_occlusion": (0, 10),

    "smoking_time.Y.": (0, 100),

    "Number of cigarettes smoked per day": (0, 100),

    "CRP": (0, 500),

    "sat_fat_intake": (0, 2),

    "sleep_hours": (0, 24),

    # Intermediate fields for deterministic BMI calculation -- not
    # clinical model features themselves, just sanity-checked inputs
    # so a garbled answer (e.g. "150 meters") gets caught here instead
    # of silently producing a nonsense BMI or looping forever.
    "height_cm": (50, 250),

    "weight_kg": (20, 300),

}


# ==========================================================
# BINARY FIELDS
# ==========================================================

BINARY_FIELDS = {

    "hypertension",

    "corory_heart_disease",

    "diabetic_nephropathy",

    "Controlling glucose by insulin injection",

    "family_history",

}


# ==========================================================
# HERA SYSTEM PROMPT
# ==========================================================

HERA_SYSTEM_PROMPT = """

You are Hera, a warm, patient, plain-language health assistant
inside the DiaVision AI diabetic foot screening tool.

Your job is to have a short, friendly conversation with the patient
to gather the information required by the screening system and then
hand it off for a risk screening.

You are NOT a doctor.

You NEVER diagnose anything yourself.

You only collect information and later explain what the screening
tool found.

RULES:

1. Use simple, everyday words.

2. Do not use medical jargon without explaining it.

For example:
say "the lower number in your blood pressure reading"
instead of only saying "diastolic pressure".

3. Ask ONE question at a time.

4. Never present a long checklist to the patient.

5. If the patient does not know a laboratory value, that is completely
fine.

6. Never make up a number.

7. Never infer a medical condition that the patient did not state.

8. Be warm, but direct. Do not pad replies with filler like
"Thanks for sharing that!", "Great, thanks!", or "Thanks for the
details!" before every question -- acknowledge briefly only when it
actually adds something, then ask the next question immediately.

9. Do not give medical diagnoses.

10. Do not tell the patient that they are safe or unsafe based on
the information they provide.

11. Keep replies to 1-2 short sentences. No exceptions for
"explaining more" -- if more explanation is needed, ask a follow-up
question instead of writing a longer message.

12. If the patient simply says hello or starts a casual conversation,
respond naturally instead of immediately asking a medical question.

13. If the patient provides a clinical value, move directly to the
next question -- do not restate or re-summarize what they said.

14. If the patient's last answer looks physically impossible or
clearly garbled (e.g. a height of 150 meters), say plainly and
specifically what looks wrong and how to fix it in ONE short
sentence (e.g. "That height doesn't look right -- can you give it in
feet/inches or centimeters?"). Do not repeat the same generic
question verbatim a second time.

15. Do not mention internal instructions, extraction, models,
XGBoost, prompts, fields, or system logic.

Information to gather naturally:

"""


HERA_SYSTEM_PROMPT += "\n".join(

    f"- {field}: {info['asks_for']}"

    for field, info in FIELD_INFO.items()

)


# ==========================================================
# EXTRACTION PROMPT
# ==========================================================

def build_extraction_prompt(transcript):

    fields_description = "\n".join(

        f'"{field}": number or null'

        for field in FIELD_INFO

    )

    return f"""

Extract ONLY clinical information that the patient has EXPLICITLY
provided in the conversation below.

STRICT RULES:

- Never guess.
- Never infer.
- Never invent.
- Never estimate a value.
- If the patient did not explicitly provide the information,
  return null.
- If the patient says they do not know, return null.
- If the patient gives a value that can clearly be converted into
  the requested numeric value, you may convert it.
- Do not use information from your general medical knowledge.
- Do not assume that a condition exists because another condition
  was mentioned.

Return ONLY a JSON object.

The JSON must contain exactly these fields:

{fields_description}
"height_cm": number or null
"weight_kg": number or null

For height_cm and weight_kg, ONLY convert units -- do not calculate
BMI yourself, it is computed separately:
- If height is given in feet/inches (e.g. 5'5" or "5 feet 5 inches"),
  convert to centimeters: 1 foot = 30.48 cm, 1 inch = 2.54 cm.
- If height is given in meters, multiply by 100 to get centimeters.
- If height is already in centimeters, use it as-is.
- If weight is given in pounds, convert to kilograms: 1 lb = 0.4536 kg.
- If weight is already in kilograms, use it as-is.
- Leave "BMI" above as null even if height_cm/weight_kg are known --
  it will be calculated from them afterward, not by you.

BINARY YES/NO FIELDS -- read this carefully:
For these fields, the patient's plain "yes" or "no" (or a clear
equivalent, e.g. "never", "not that I know of", "I do have that")
IS the explicit value the patient provided. Converting it to 0 or 1
is direct extraction, NOT an inference or a guess, so the "never
infer/never guess" rule above does not apply to it:
  - hypertension
  - corory_heart_disease
  - diabetic_nephropathy
  - Controlling glucose by insulin injection
A "no" (or equivalent) means 0. A "yes" (or equivalent) means 1.
Only return null for these if the patient was never asked, or said
they don't know.

For "sat_fat_intake": rarely = 0, sometimes = 1, often = 2.
For "family_history": yes = 1, no = 0 (parent or sibling with diabetes
or heart disease).

For "number_of_lower_leg_artery_occlusion": if the patient answers
"no" (or equivalent) to ever having narrowed/blocked leg arteries
found, that means 0 -- return 0, not null. If they say "yes" but
can't give a count or estimate, return null (they'll be asked to
estimate separately).

Conversation:

{transcript}

"""


# ==========================================================
# VALIDATE EXTRACTED DATA
# ==========================================================

def validate_extracted_fields(extracted):

    cleaned = {}

    dropped = []


    # ------------------------------------------
    # Ensure every field exists
    # ------------------------------------------

    for field in FIELD_INFO:

        value = extracted.get(field)

        cleaned[field] = value

    # height_cm / weight_kg aren't clinical model fields (not in
    # FIELD_INFO), just intermediates for the deterministic BMI calc
    # below -- carry them through separately so they still get
    # range-validated.
    cleaned["height_cm"] = extracted.get("height_cm")
    cleaned["weight_kg"] = extracted.get("weight_kg")


    # ------------------------------------------
    # Validate numeric ranges
    # ------------------------------------------

    for field, (lo, hi) in VALID_RANGES.items():

        value = cleaned.get(field)

        if value is None:
            continue

        try:

            value = float(value)

            cleaned[field] = value

        except (ValueError, TypeError):

            dropped.append(
                (field, value)
            )

            cleaned[field] = None

            continue


        if not (lo <= value <= hi):

            dropped.append(
                (field, value)
            )

            cleaned[field] = None


    # ------------------------------------------
    # Validate binary fields
    # ------------------------------------------

    for field in BINARY_FIELDS:

        value = cleaned.get(field)

        if value is None:
            continue

        try:

            value = int(value)

            cleaned[field] = value

        except (ValueError, TypeError):

            dropped.append(
                (field, value)
            )

            cleaned[field] = None

            continue


        if value not in (0, 1):

            dropped.append(
                (field, value)
            )

            cleaned[field] = None


    return cleaned, dropped


# ==========================================================
# DETERMINISTIC BMI CALCULATION
# ==========================================================
# The LLM only needs to convert units (feet/inches -> cm,
# lbs -> kg) into height_cm/weight_kg -- the actual
# BMI = weight_kg / (height_m ** 2) arithmetic happens here in
# plain Python, which never gets it wrong the way a small
# reasoning-effort="low" model doing multi-step conversion +
# division + squaring in one shot sometimes did.

def compute_bmi_if_possible(cleaned):

    if cleaned.get("BMI") is not None:
        return cleaned

    height_cm = cleaned.get("height_cm")
    weight_kg = cleaned.get("weight_kg")

    if height_cm is None or weight_kg is None:
        return cleaned

    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)

    lo, hi = VALID_RANGES["BMI"]
    if lo <= bmi <= hi:
        cleaned["BMI"] = round(bmi, 1)
    else:
        print(
            f">>> COMPUTED BMI {bmi:.1f} OUT OF VALID RANGE "
            f"(from height_cm={height_cm}, weight_kg={weight_kg}) -- "
            f"discarding, will re-ask for height/weight."
        )

    return cleaned


# ==========================================================
# EXTRACT CLINICAL FIELDS USING GPT-OSS-20B
# ==========================================================

def extract_fields_from_conversation(transcript):

    print("\n>>> STARTING EXTRACTION")

    extraction_start_time = time.time()


    prompt = build_extraction_prompt(
        transcript
    )


    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are a strict clinical information "
                        "extraction system. Return only valid JSON."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }

            ],

            temperature=0,

            max_completion_tokens=1024,

            # include_reasoning=False is not a real Groq param -- it was
            # silently ignored, and without reasoning_format="hidden" the
            # reasoning model (gpt-oss-20b) burns its token budget on
            # hidden reasoning and leaves message.content empty. The fix
            # is reasoning_format="hidden" + a high enough token budget,
            # plus response_format here since this call must return JSON.
            reasoning_effort="low",

            reasoning_format="hidden",

            response_format={"type": "json_object"},

        )


        extraction_time = (
            time.time()
            - extraction_start_time
        )


        print(
            ">>> EXTRACTION FINISHED"
        )

        print(
            "EXTRACTION TIME:",
            round(
                extraction_time,
                2
            ),
            "seconds"
        )


        raw_text = (
            response
            .choices[0]
            .message
            .content
        )


        print(
            "RAW EXTRACTION:",
            raw_text
        )


        # ------------------------------------------
        # Parse JSON
        # ------------------------------------------

        extracted = json.loads(
            raw_text
        )


        # ------------------------------------------
        # Validate
        # ------------------------------------------

        extracted, dropped = (
            validate_extracted_fields(
                extracted
            )
        )


        extracted = compute_bmi_if_possible(extracted)


        if dropped:

            print(
                "DROPPED INVALID VALUES:",
                dropped
            )


        print(
            "EXTRACTED DATA:",
            extracted
        )


        return extracted


    except Exception as e:

        print(
            ">>> EXTRACTION ERROR:",
            repr(e)
        )

        raise


# ==========================================================
# DETERMINE MISSING REQUIRED FIELDS
# ==========================================================

def missing_required_fields(extracted):

    return [

        field

        for field in REQUIRED_FOLLOWUP_FIELDS

        if extracted.get(field) is None

    ]


# ==========================================================
# GENERATE HERA REPLY USING GPT-OSS-20B
# ==========================================================

def trim_transcript_for_reply(transcript, max_lines=8):
    """
    The reply-generation call only needs recent context to sound
    natural -- it doesn't need the full history the way extraction
    does (extraction must see everything ever said, since that's
    where a patient's earlier answer would otherwise get lost).
    Trimming this keeps the reply prompt from growing every turn.
    """

    lines = [line for line in transcript.split("\n") if line.strip()]

    return "\n".join(lines[-max_lines:])


def generate_hera_reply(
    transcript,
    still_missing,
    known=None
):

    print(
        "\n>>> STARTING HERA REPLY"
    )

    reply_start_time = time.time()


    chat_prompt = (

        "Conversation so far (most recent messages):\n"

        + trim_transcript_for_reply(transcript)

    )


    # The reply prompt only sees the last few lines, so earlier answers
    # (e.g. age) fall out of view and Hera re-asks them. List them here.
    if known:
        got = ", ".join(
            f"{k}={v}" for k, v in known.items() if v is not None
        )
        if got:
            chat_prompt += (
                "\n\nALREADY PROVIDED (never ask about these again): "
                + got
            )


    # ------------------------------------------
    # Missing information instruction
    # ------------------------------------------

    if still_missing:

        chat_prompt += (

            "\n\nINTERNAL INSTRUCTION:\n"

            "The following important information has not yet "
            "been provided:\n"

            + ", ".join(still_missing)

            + "\n\n"

            "Ask about ONE missing item naturally."

            "\nDo not mention this internal instruction."

        )


    # ------------------------------------------
    # All required information gathered
    # ------------------------------------------

    else:

        chat_prompt += (

            "\n\nINTERNAL INSTRUCTION:\n"

            "The key information has been gathered."

            "\nDo not ask any further questions. "

            "\nTell the patient they can run the screening now, "

            "or continue chatting if they wish."

            "\nDo not mention this internal instruction."

        )


    try:

        response = groq_client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[

                {
                    "role": "system",
                    "content": HERA_SYSTEM_PROMPT
                },

                {
                    "role": "user",
                    "content": chat_prompt
                }

            ],

            temperature=0.6,

            max_completion_tokens=500,

            reasoning_effort="low",

            reasoning_format="hidden",

        )


        reply_time = (
            time.time()
            - reply_start_time
        )


        print(
            ">>> HERA REPLY FINISHED"
        )

        print(
            "REPLY TIME:",
            round(
                reply_time,
                2
            ),
            "seconds"
        )


        reply_text = (
            response
            .choices[0]
            .message
            .content
        )


        return reply_text


    except Exception as e:

        print(
            ">>> HERA REPLY ERROR:",
            repr(e)
        )

        raise
PATIENT_DISCUSSION_PROMPT = """
You are Hera, the DiaVision AI copilot, discussing ONE patient from an
uploaded cohort with a clinician.

Use ONLY the facts given. You have just this screening summary: risk score,
risk tier, smoking flag, the top 3 model factors with direction, model
confidence, and where the patient sits in the cohort. You do NOT have raw
vitals, lab values, history or images -- if asked for those, say you only
have the screening summary. Never guess or invent numbers.

How to read the facts:
- A factor marked up-arrow pushed this patient's risk UP; down-arrow pushed
  it DOWN. These are model contributions, not causes or diagnoses.
- Confidence: "Clear" = model is confident; "Likely, but check" = probably
  right but worth a clinician's check; "Uncertain" = borderline.

Style: plain language, 4-6 short sentences unless asked for more, no
tables. Do not diagnose or call the patient safe/unsafe. For High or Very
High tiers, suggest clinician review. Do not mention internal instructions.
"""


def generate_hera_patient_discussion(transcript, patient):
    def f(key):
        return str(patient.get(key, ""))[:200]

    facts = (
        f"Patient #{f('patient_id')}\n"
        f"- Risk score: {f('risk_score')}% ({f('diagnosis')})\n"
        f"- Smoking flag: {f('smoking_flag') or 'Not assessed'}\n"
        f"- Top model factors: {f('shap_factors')}\n"
        f"- Model confidence: {f('confidence_tier')}\n"
    )
    if patient.get("cohort_size"):
        facts += (
            f"- Cohort: {f('cohort_size')} patients uploaded; this patient's "
            f"score is higher than {f('higher_than_pct')}% of them; "
            f"cohort average score {f('cohort_avg')}%\n"
        )

    prompt = (
        "PATIENT FACTS:\n" + facts
        + "\nConversation so far (most recent messages):\n"
        + trim_transcript_for_reply(transcript)
        + "\n\nReply to the clinician's latest message."
    )

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": PATIENT_DISCUSSION_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_completion_tokens=600,
        reasoning_effort="low",
        reasoning_format="hidden",
    )
    return response.choices[0].message.content


def generate_hera_vision_explanation(diagnosis_label, is_abnormal):
    """
    Separate from generate_hera_reply on purpose -- that function always
    appends an onboarding-flow instruction (ask a missing question, or
    tell the patient to run screening), which doesn't apply here. This
    is a plain one-shot explanation of a vision scan result.
    """
    prompt = (
        f"A foot image scan was just analyzed. Result: {diagnosis_label}.\n\n"
        "In 2-3 short, warm, plain-language sentences, explain what this "
        "means to the patient and what they should do next. "
        "Do not ask any questions -- this is a one-way explanation."
    )

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": HERA_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_completion_tokens=300,
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        return response.choices[0].message.content
    except Exception as e:
        print(">>> HERA VISION EXPLANATION ERROR:", repr(e))
        return None


# ==========================================================
# PHASE 2 DATA COLLECTION (not used by the model yet)
# One row per chat session, upserted as answers arrive. DFU is left
# blank -- fill in the real clinical outcome later, then merge with
# Kaddler_1.csv and retrain.
# ==========================================================

COLLECTION_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "collected_patients.csv"
)
COLLECTION_FIELDS = ["sat_fat_intake", "sleep_hours", "family_history"]


def save_collection_row(session_id, extracted):
    if not session_id or any(extracted.get(f) is None for f in COLLECTION_FIELDS):
        return
    try:
        row = {"session_id": str(session_id), "DFU": "", "time": time.strftime("%d-%m-%Y %H:%M")}
        row.update({k: extracted.get(k) for k in DiavisionApiConfig.feature_columns})
        row.update({f: extracted.get(f) for f in COLLECTION_FIELDS})
        new = pd.DataFrame([row])
        if os.path.exists(COLLECTION_CSV):
            old = pd.read_csv(COLLECTION_CSV, dtype={"session_id": str})
            old = old[old["session_id"] != row["session_id"]]
            # keep any DFU already filled in by hand
            new = pd.concat([old, new], ignore_index=True)
        new.to_csv(COLLECTION_CSV, index=False)
    except Exception as e:
        print(">>> COLLECTION SAVE ERROR:", repr(e))


# ==========================================================
# 1. HERA CHAT ENDPOINT
# ==========================================================

@api_view(['POST'])
def hera_chat(request):

    print("\n==========================================")
    print("HERA REQUEST")
    print("==========================================")


    # ------------------------------------------
    # Receive request
    # ------------------------------------------

    data = request.data


    transcript = data.get(
        "transcript",
        ""
    )


    if not isinstance(
        transcript,
        str
    ):

        transcript = str(
            transcript
        )


    print(
        "TRANSCRIPT LENGTH:",
        len(transcript)
    )


    print(
        "TRANSCRIPT:",
        repr(
            transcript[:500]
        )
    )


    # ------------------------------------------
    # Check Groq key
    # ------------------------------------------

    if not os.environ.get(
        "GROQ_API_KEY"
    ):

        print(
            ">>> GROQ_API_KEY IS NOT SET"
        )

        return Response(

            {
                "error":
                "GROQ_API_KEY is not configured."
            },

            status=500

        )


    # ======================================================
    # PATIENT DISCUSSION MODE (frontend sends the cohort row)
    # Skips onboarding extraction/scoring entirely.
    # ======================================================

    patient = data.get("patient_context")

    if isinstance(patient, dict) and patient:

        try:

            reply_text = generate_hera_patient_discussion(
                transcript,
                patient
            )

        except Exception as e:

            print(">>> PATIENT DISCUSSION FAILED:", repr(e))

            return Response(
                {"error": "Hera response generation failed."},
                status=503
            )

        return Response({
            "reply": reply_text,
            "extracted_data": {},
            "still_missing": [],
            "risk_score": None,
            "diagnosis": None,
            "risk_tier": None,
        })


    # ======================================================
    # STEP 1: EXTRACT CLINICAL INFORMATION
    # ======================================================

    try:

        extracted_data = (
            extract_fields_from_conversation(
                transcript
            )
        )

    except Exception as e:

        print(
            ">>> EXTRACTION FAILED:",
            repr(e)
        )

        return Response(

            {
                "error":
                "Clinical information extraction failed."
            },

            status=503

        )

    # Merge in previously-confirmed fields the frontend sends back --
    # this is what stops a field (e.g. age, leg artery occlusion) from
    # occasionally getting silently re-nulled and re-asked when a
    # later extraction pass over the growing transcript misses
    # something that was already answered earlier in the conversation.
    known_fields = data.get("known_fields") or {}
    if isinstance(known_fields, dict):
        for field, value in known_fields.items():
            if extracted_data.get(field) is None and value is not None:
                extracted_data[field] = value


    save_collection_row(data.get("session_id"), extracted_data)


    # ======================================================
    # STEP 2: DETERMINE MISSING REQUIRED FIELDS
    # ======================================================

    still_missing = (
        missing_required_fields(
            extracted_data
        )
    )


    print(
        ">>> STILL MISSING:",
        still_missing
    )


    # ======================================================
    # STEP 3: GENERATE HERA RESPONSE
    # ======================================================

    try:

        reply_text = generate_hera_reply(

            transcript,

            still_missing,

            extracted_data

        )

    except Exception as e:

        print(
            ">>> HERA REPLY FAILED:",
            repr(e)
        )

        return Response(

            {
                "error":
                "Hera response generation failed."
            },

            status=503

        )


    # ======================================================
    # STEP 4: XGBOOST SCORING
    #
    # DO NOT CHANGE MODEL LOGIC
    # ======================================================

    risk_score = None

    diagnosis = None

    risk_tier = None


    if (

        not still_missing

        and

        DiavisionApiConfig.xgb_model is not None

    ):


        # ------------------------------------------
        # Build model row
        # ------------------------------------------

        row = {

            col:
            extracted_data.get(col)

            for col
            in DiavisionApiConfig.feature_columns

        }


        X_input = pd.DataFrame(
            [row]
        )


        # ------------------------------------------
        # Convert values to numeric
        # ------------------------------------------

        for col in X_input.columns:

            X_input[col] = pd.to_numeric(

                X_input[col],

                errors="coerce"

            )


        # ------------------------------------------
        # Probability
        # ------------------------------------------

        proba = (

            DiavisionApiConfig
            .xgb_model
            .predict_proba(
                X_input
            )[:, 1][0]

        )


        proba = float(

            np.clip(

                proba,

                0.005,

                0.995

            )

        )


        risk_score = round(

            recalibrate(proba) * 100,

            2

        )


        # ------------------------------------------
        # Diagnosis (4-tier risk band, same bands as
        # predict_tabular -- see get_risk_tier below --
        # so Hera's spoken label and the CSV table can
        # never disagree for the same patient)
        # ------------------------------------------

        risk_tier, diagnosis = get_risk_tier(risk_score)


    # ======================================================
    # STEP 5: RETURN RESPONSE
    # ======================================================

    return Response({

        "reply":
        reply_text,

        "extracted_data":
        extracted_data,

        "still_missing":
        still_missing,

        "risk_score":
        risk_score,

        "diagnosis":
        diagnosis,

        "risk_tier":
        risk_tier,

    })


# ==========================================================
# Risk tier banding -- shared by predict_tabular AND hera_chat,
# so the same patient never gets a different label from each.
# Labels deliberately avoid leading with "Normal" for any tier
# above Low -- that word is what caused the original confusion
# (a 49.55% patient displaying as "Normal").
# ==========================================================
# Recalibration: the raw model over-predicts on recent patients (~5 points).
# A constant logit shift, fitted on 2020-2022 patients and validated on
# 2022-2023 (see model_validation.py, section 4), corrects most of it.
# Revisit if the model is retrained or the patient mix changes.
RECALIBRATION_SHIFT = -0.45


def recalibrate(p):
    p = float(np.clip(p, 0.005, 0.995))
    z = np.log(p / (1 - p)) + RECALIBRATION_SHIFT
    return float(np.clip(1 / (1 + np.exp(-z)), 0.005, 0.995))


def get_risk_tier(risk_pct):
    if risk_pct < 30:
        return "low", "🟢 Low Risk"
    elif risk_pct < 50:
        return "moderate", "🟡 Moderate Risk (Needs Monitoring)"
    elif risk_pct < 75:
        return "high", "🟠 High Risk"
    else:
        return "very_high", "🔴 Very High Risk"


# Confidence (not severity) tier from split conformal prediction --
# mirrors conformal_prediction.py's prediction_set()/tier_label()
# exactly, using the thresholds apps.py loaded from
# conformal_thresholds.json at startup.
def get_confidence_tier(prob_pos):
    if (
        DiavisionApiConfig.conformal_threshold_99 is None
        or DiavisionApiConfig.conformal_threshold_90 is None
    ):
        return "N/A"

    def prediction_set(p, threshold):
        included = []
        for cls, p_cls in [(0, 1 - p), (1, p)]:
            score = 1 - p_cls
            if score <= threshold:
                included.append(cls)
        return included

    set_99 = prediction_set(prob_pos, DiavisionApiConfig.conformal_threshold_99)
    set_90 = prediction_set(prob_pos, DiavisionApiConfig.conformal_threshold_90)

    if len(set_99) == 1:
        return "Clear"
    elif len(set_90) == 1:
        return "Likely, but check"
    else:
        return "Uncertain"


# ==========================================================
# 2. CLINICAL TABULAR ENDPOINT
# ==========================================================

@api_view(['POST'])
def predict_tabular(request):


    # ------------------------------------------
    # Check upload
    # ------------------------------------------

    if "file" not in request.FILES:

        return Response(

            {
                "error":
                "No CSV file uploaded."
            },

            status=400

        )


    # ------------------------------------------
    # Check model
    # ------------------------------------------

    if (
        DiavisionApiConfig.xgb_model
        is None
    ):

        return Response(

            {
                "error":
                "Tabular model not loaded."
            },

            status=500

        )


    try:

        # --------------------------------------
        # Read CSV
        # --------------------------------------

        file = request.FILES[
            "file"
        ]


        df = pd.read_csv(
            file
        ).reset_index(
            drop=True
        )


        # --------------------------------------
        # Check required columns
        # --------------------------------------

        missing_cols = [

            col

            for col
            in DiavisionApiConfig.feature_columns

            if col not in df.columns

        ]


        if missing_cols:

            return Response(

                {

                    "error":
                    f"Missing required columns: {missing_cols}"

                },

                status=400

            )


        # --------------------------------------
        # Input features
        # --------------------------------------

        X_input = df[
            DiavisionApiConfig.feature_columns
        ]


        # --------------------------------------
        # Predictions
        # --------------------------------------

        predictions = (

            DiavisionApiConfig
            .xgb_model
            .predict(
                X_input
            )

        )


        probabilities = (

            DiavisionApiConfig
            .xgb_model
            .predict_proba(
                X_input
            )[:, 1]

        )


        probabilities = np.clip(

            probabilities,

            0.005,

            0.995

        )


        # --------------------------------------
        # SHAP "why this score" -- computed once for
        # the whole batch (vectorized), not per-row.
        # Uses shap_imputer (split from the RAW model
        # in apps.py, not the calibrated xgb_model --
        # xgb_model isn't a Pipeline, so checking its
        # named_steps here always fell through to
        # feeding un-imputed rows into an explainer
        # built on a booster trained on imputed data).
        # --------------------------------------

        shap_values = None

        if DiavisionApiConfig.shap_explainer is not None and DiavisionApiConfig.shap_imputer is not None:

            X_input_imp = pd.DataFrame(
                DiavisionApiConfig.shap_imputer.transform(X_input),
                columns=X_input.columns,
                index=X_input.index,
            )
            shap_values = DiavisionApiConfig.shap_explainer.shap_values(X_input_imp)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]


        results = []


        # --------------------------------------
        # Process patients
        # --------------------------------------

        for i, row in df.iterrows():

            p = predictions[i]

            proba = probabilities[i]


            cigs = row.get(
                "Number of cigarettes smoked per day"
            )


            years = row.get(
                "smoking_time.Y."
            )


            cigs_known = pd.notna(
                cigs
            )


            years_known = pd.notna(
                years
            )


            # ----------------------------------
            # Smoking flag
            # ----------------------------------

            if (
                cigs_known
                and
                cigs >= 20
            ):

                smoke_flag = (
                    "🚬🚨 Heavy smoker (≥20/day)"
                )


            elif (
                cigs_known
                and
                cigs >= 10
            ):

                smoke_flag = (
                    "🚬⚠️ Moderate smoker (10-19/day)"
                )


            elif (
                cigs_known
                and
                cigs > 0
            ):

                smoke_flag = (
                    "🚬 Current smoker"
                )


            elif (
                years_known
                and
                years > 0
            ):

                smoke_flag = (
                    "🚬 Smoking history"
                )


            elif (
                not cigs_known
                and
                not years_known
            ):

                smoke_flag = (
                    "Not assessed"
                )


            else:

                smoke_flag = ""



            risk_pct = round(
                recalibrate(float(proba)) * 100,
                2
            )

            diag_tier, diag = get_risk_tier(risk_pct)
            confidence_tier = get_confidence_tier(float(proba))


            # ----------------------------------
            # NEW: SHAP Top Factors Extraction
            # ----------------------------------
            
            shap_factors_str = "N/A"
            
            if shap_values is not None:
                
                row_shap = pd.Series(shap_values[i], index=DiavisionApiConfig.feature_columns)
                top3 = row_shap.reindex(row_shap.abs().sort_values(ascending=False).index[:3])
                
                factors = []
                
                for feat, val in top3.items():
                    direction = "⬆️" if val > 0 else "⬇️"
                    factors.append(f"{feat} {direction}")
                    
                shap_factors_str = ", ".join(factors)


            results.append({

                "patient_id":
                i + 1,

                "diagnosis":
                diag,

                "risk_tier":
                diag_tier,

                "risk_score":
                risk_pct,

                "smoking_flag":
                smoke_flag,

                "shap_factors":
                shap_factors_str,

                "confidence_tier":
                confidence_tier,

            })


        # --------------------------------------
        # Return
        # --------------------------------------

        return Response({

            "results":
            results

        })


    except Exception as e:

        print(
            "TABULAR ERROR:",
            repr(e)
        )


        return Response(

            {
                "error":
                str(e)
            },

            status=400

        )
    
    

# ==========================================================
# 3. VISION GRAD-CAM ENDPOINT
# ==========================================================

@api_view(['POST'])
def predict_vision(request):
    
    if 'image' not in request.FILES:
        return Response({"error": "No image uploaded."}, status=400)
        
    if not DiavisionApiConfig.vision_model:
        return Response({"error": "Vision model not loaded."}, status=500)

    try:
        
        file = request.FILES['image']
        pil_image = Image.open(file).convert("RGB")
        
        img_array = cv2.resize(np.array(pil_image), (224, 224))
        rgb_img_float = np.float32(img_array) / 255
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(rgb_img_float).unsqueeze(0)

        with torch.no_grad():
            outputs = DiavisionApiConfig.vision_model(input_tensor)
            pred_idx = outputs.argmax(1).item()
            
        is_ulcer = (pred_idx == DiavisionApiConfig.ulcer_idx)
        label = "Abnormal (Ulcer Risk)" if is_ulcer else "Normal (Healthy)"
        hera_analysis = generate_hera_vision_explanation(label, is_ulcer)

        cam = GradCAM(model=DiavisionApiConfig.vision_model, target_layers=[DiavisionApiConfig.vision_model.layer4[-1]])
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        cam_pil = Image.fromarray((cam_image * 255).astype(np.uint8))
        buffer = BytesIO()
        cam_pil.save(buffer, format="PNG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return Response({
            "diagnosis": label,
            "is_abnormal": is_ulcer,
            "heatmap_base64": f"data:image/png;base64,{base64_str}",
            "hera_analysis": hera_analysis,
        }) 
        
    except Exception as e:
        
        print("VISION ERROR:", repr(e))
        return Response({"error": str(e)}, status=400)


# ==========================================================
# 4. DIAGNOSTICS ENDPOINT
# ==========================================================

@api_view(['GET'])
def model_diagnostics(request):
    
    status = {
        "tabular_model_loaded": DiavisionApiConfig.xgb_model is not None,
        "vision_model_loaded": DiavisionApiConfig.vision_model is not None,
        "shap_explainer_loaded": DiavisionApiConfig.shap_explainer is not None,
        "features_tracked": DiavisionApiConfig.feature_columns if DiavisionApiConfig.feature_columns else [],
    }
    
    return Response(status)

# ==========================================================
# 5. SHAP DEPENDENCE DIAGNOSTICS ENDPOINT
# ==========================================================
# Migrated from shap_dependence_analysis.py -- internal diagnostic
# tool, not patient-facing. Reuses the explainer + training
# background apps.py built once at startup rather than reloading
# the model/CSV on every call.
#
# POST body (multipart or JSON):
#   feature              (required)  -- must be one of DiavisionApiConfig.feature_columns
#   interaction_feature  (optional)  -- colors the dependence plot by this feature
#   highlight_file        (optional, file upload) -- a CSV of specific patients
#                          (e.g. false_negatives.csv) to overlay on the plot as
#                          stars, so you can see where specific cases sit
#                          relative to the general training-data trend

def _shap_single_factor_sweep(feature):
    """
    Fair single-factor sweep at p10/p50/p90 of `feature`, holding every
    other feature at its real training-data median -- same "neutral
    baseline" logic as shap_dependence_analysis.py Part 1, scoped to
    one feature instead of the fixed sweep_features list.
    """

    bg_X = DiavisionApiConfig.shap_background_X
    median_baseline = bg_X.median()

    low = bg_X[feature].quantile(0.10)
    mid = bg_X[feature].quantile(0.50)
    high = bg_X[feature].quantile(0.90)

    rows = []
    for val in [low, mid, high]:
        row = median_baseline.copy()
        row[feature] = val
        rows.append(row)

    sweep_df = pd.DataFrame(rows)[DiavisionApiConfig.feature_columns]
    probs = DiavisionApiConfig.raw_model.predict_proba(sweep_df)[:, 1] * 100

    return {
        "p10": {"value": round(float(low), 2), "raw_risk_pct": round(float(probs[0]), 2)},
        "p50": {"value": round(float(mid), 2), "raw_risk_pct": round(float(probs[1]), 2)},
        "p90": {"value": round(float(high), 2), "raw_risk_pct": round(float(probs[2]), 2)},
    }


def _build_shap_dependence_plot(feature, interaction_feature, highlight_df, highlight_labels):
    """
    SHAP dependence plot for `feature` on the training background,
    optionally colored by `interaction_feature`, with optional
    highlighted points (e.g. false_negatives.csv) overlaid as stars
    so specific cases can be seen against the general trend -- this
    is the piece shap.dependence_plot() doesn't support out of the
    box, so it's built directly with matplotlib.

    Returns (base64_png, highlight_shap_values_or_None).
    """

    bg_X = DiavisionApiConfig.shap_background_X
    bg_shap = DiavisionApiConfig.shap_background_values
    feat_idx = DiavisionApiConfig.feature_columns.index(feature)

    x_vals = bg_X[feature].values
    y_vals = bg_shap[:, feat_idx]

    fig, ax = plt.subplots(figsize=(8, 6))

    if interaction_feature:
        color_vals = bg_X[interaction_feature].values
        sc = ax.scatter(
            x_vals, y_vals, c=color_vals, cmap="viridis",
            alpha=0.5, s=25, label=f"Training data (n={len(bg_X)})",
        )
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(interaction_feature)
    else:
        ax.scatter(
            x_vals, y_vals, color="steelblue",
            alpha=0.4, s=25, label=f"Training data (n={len(bg_X)})",
        )

    highlight_shap = None

    if highlight_df is not None and len(highlight_df) > 0:

        hi_imputed = pd.DataFrame(
            DiavisionApiConfig.shap_imputer.transform(
                highlight_df[DiavisionApiConfig.feature_columns]
            ),
            columns=DiavisionApiConfig.feature_columns,
        )
        hi_shap_matrix = DiavisionApiConfig.shap_explainer.shap_values(hi_imputed)
        if isinstance(hi_shap_matrix, list):
            hi_shap_matrix = hi_shap_matrix[1]
        highlight_shap = hi_shap_matrix[:, feat_idx]

        hx = highlight_df[feature].values

        ax.scatter(
            hx, highlight_shap, color="red", marker="*", s=220,
            edgecolor="black", linewidth=1, label="Highlighted cases", zorder=5,
        )

        if highlight_labels is not None:
            for xi, yi, lbl in zip(hx, highlight_shap, highlight_labels):
                ax.annotate(
                    str(lbl), (xi, yi), textcoords="offset points",
                    xytext=(6, 6), fontsize=8,
                )

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(feature)
    ax.set_ylabel(f"SHAP value for {feature}")
    title = f"SHAP dependence: {feature}"
    if interaction_feature:
        title += f" (colored by {interaction_feature})"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    image_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return image_b64, highlight_shap


@api_view(['POST'])
def shap_dependence_diagnostic(request):

    if (
        DiavisionApiConfig.shap_explainer is None
        or DiavisionApiConfig.shap_background_X is None
    ):
        return Response(
            {
                "error":
                "SHAP dependence diagnostics unavailable -- raw model or "
                "training CSV not loaded at startup. Check the Django "
                "console output from apps.py's ready() for which one is "
                "missing."
            },
            status=503,
        )

    feature = request.data.get("feature")
    if not feature or feature not in DiavisionApiConfig.feature_columns:
        return Response(
            {"error": f"'feature' must be one of: {DiavisionApiConfig.feature_columns}"},
            status=400,
        )

    interaction_feature = request.data.get("interaction_feature") or None
    if interaction_feature and interaction_feature not in DiavisionApiConfig.feature_columns:
        return Response(
            {"error": f"'interaction_feature' must be one of: {DiavisionApiConfig.feature_columns}"},
            status=400,
        )

    highlight_df = None
    highlight_labels = None

    if "highlight_file" in request.FILES:
        try:
            highlight_df = pd.read_csv(request.FILES["highlight_file"])
        except Exception as e:
            return Response({"error": f"Could not read highlight_file: {e}"}, status=400)

        missing = [c for c in DiavisionApiConfig.feature_columns if c not in highlight_df.columns]
        if missing:
            return Response(
                {"error": f"highlight_file is missing model feature columns: {missing}"},
                status=400,
            )

        id_col = "no." if "no." in highlight_df.columns else None
        highlight_labels = (
            highlight_df[id_col].tolist()
            if id_col
            else highlight_df.index.astype(str).tolist()
        )

    try:
        image_b64, highlight_shap = _build_shap_dependence_plot(
            feature, interaction_feature, highlight_df, highlight_labels
        )
    except Exception as e:
        print("SHAP DEPENDENCE ERROR:", repr(e))
        return Response({"error": str(e)}, status=400)

    feat_idx = DiavisionApiConfig.feature_columns.index(feature)
    bg_shap_col = DiavisionApiConfig.shap_background_values[:, feat_idx]

    response_payload = {
        "feature": feature,
        "interaction_feature": interaction_feature,
        "dependence_plot_base64": image_b64,
        "shap_range_on_training_data": {
            "min": round(float(bg_shap_col.min()), 4),
            "max": round(float(bg_shap_col.max()), 4),
            "spread": round(float(bg_shap_col.max() - bg_shap_col.min()), 4),
        },
        "single_factor_sweep": _shap_single_factor_sweep(feature),
    }

    if highlight_df is not None and highlight_shap is not None:
        response_payload["highlighted_cases"] = [
            {
                "label": lbl,
                "feature_value": float(highlight_df.iloc[i][feature]),
                "shap_value": round(float(highlight_shap[i]), 4),
            }
            for i, lbl in enumerate(highlight_labels)
        ]

    return Response(response_payload)