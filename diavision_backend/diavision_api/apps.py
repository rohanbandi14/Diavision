import os
import json
import joblib
import torch
import torch.nn as nn
import pandas as pd
from torchvision import models
from django.apps import AppConfig
import shap  # <-- NEW: Import SHAP

class DiavisionApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'diavision_api'

    # Global placeholders for your AI models
    xgb_model = None
    feature_columns = None
    vision_model = None
    ulcer_idx = 1
    shap_explainer = None  # <-- NEW: Global explainer
    raw_model = None              # raw (pre-calibration) pipeline, has .predict_proba
    shap_imputer = None           # imputer split out of raw_model's pipeline
    shap_background_X = None      # imputed background sample (DataFrame)
    shap_background_values = None # SHAP values for shap_background_X
    conformal_threshold_99 = None # split conformal thresholds (see conformal_prediction.py)
    conformal_threshold_90 = None

    def ready(self):
        base_dir = r"C:\Users\rohan\Downloads"
        
        clinical_model_path = os.path.join(base_dir, "diavision_clinical_model.pkl")
        features_path = os.path.join(base_dir, "feature_columns.json")
        vision_model_path = os.path.join(base_dir, "foot_image_model.pt")
        class_names_path = os.path.join(base_dir, "class_names.json")

        # ==========================================
        # 1. Load Clinical Tabular Model & SHAP
        # ==========================================
        try:
            if os.path.exists(clinical_model_path) and os.path.exists(features_path):
                DiavisionApiConfig.xgb_model = joblib.load(clinical_model_path)
                
                with open(features_path, 'r') as f:
                    DiavisionApiConfig.feature_columns = json.load(f)
                    
                print("✅ Clinical Model loaded successfully!")
            else:
                print(f"⚠️ Missing Clinical Model files in: {base_dir}")
        except Exception as e:
            print(f"❌ Error loading Clinical Model: {e}")

        # ==========================================
        # 1b. Load RAW (pre-calibration) model + build SHAP
        # ==========================================
        # xgb_model above is the CALIBRATED deployed model and isn't a
        # Pipeline, so its named_steps check always fell through and fed
        # un-imputed rows into an explainer built on a booster trained on
        # imputed data. TreeExplainer needs the raw pipeline instead.
        try:
            raw_model_path = os.path.join(base_dir, "diavision_clinical_model_raw.pkl")
            training_csv_path = os.path.join(base_dir, "Kaddler 1.csv")

            if os.path.exists(raw_model_path) and DiavisionApiConfig.feature_columns:
                raw_pipeline = joblib.load(raw_model_path)
                DiavisionApiConfig.raw_model = raw_pipeline

                if hasattr(raw_pipeline, 'named_steps'):
                    DiavisionApiConfig.shap_imputer = raw_pipeline.named_steps.get('imputer')
                    booster = raw_pipeline.named_steps['model']
                else:
                    booster = raw_pipeline

                DiavisionApiConfig.shap_explainer = shap.TreeExplainer(booster)

                if os.path.exists(training_csv_path):
                    df = pd.read_csv(training_csv_path)
                    bg_sample = df[DiavisionApiConfig.feature_columns].sample(
                        n=min(500, len(df)), random_state=42
                    )
                    if DiavisionApiConfig.shap_imputer is not None:
                        bg_X = pd.DataFrame(
                            DiavisionApiConfig.shap_imputer.transform(bg_sample),
                            columns=DiavisionApiConfig.feature_columns,
                        )
                    else:
                        bg_X = bg_sample

                    DiavisionApiConfig.shap_background_X = bg_X
                    shap_vals = DiavisionApiConfig.shap_explainer.shap_values(bg_X)
                    DiavisionApiConfig.shap_background_values = (
                        shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                    )
                    print("✅ SHAP Explainer built and background sample loaded into memory!")
                else:
                    print(f"⚠️ Missing training CSV at {training_csv_path}, SHAP dependence diagnostics disabled")
            else:
                print(f"⚠️ Missing raw model at {raw_model_path}, SHAP disabled")
        except Exception as e:
            print(f"❌ Error building SHAP explainer/background: {e}")

        # ==========================================
        # 2. Load Vision Model (ResNet18)
        # ==========================================
        try:
            if os.path.exists(vision_model_path) and os.path.exists(class_names_path):
                with open(class_names_path, 'r') as f:
                    class_names = json.load(f)
                
                for i, name in enumerate(class_names):
                    if "bnormal" in name or "lcer" in name:
                        DiavisionApiConfig.ulcer_idx = i
                        break

                model = models.resnet18(weights=None)
                model.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.fc.in_features, len(class_names))
                )
                
                model.load_state_dict(torch.load(vision_model_path, map_location="cpu", weights_only=True))
                model.eval()
                
                DiavisionApiConfig.vision_model = model
                print("✅ Vision Model loaded successfully!")
            else:
                print(f"⚠️ Missing Vision Model files in: {base_dir}")
        except Exception as e:
            print(f"❌ Error loading Vision Model: {e}")

        # ==========================================
        # 3. Load Conformal Prediction Thresholds
        # ==========================================
        try:
            thresholds_path = os.path.join(base_dir, "conformal_thresholds.json")
            if os.path.exists(thresholds_path):
                with open(thresholds_path, 'r') as f:
                    thresholds = json.load(f)
                DiavisionApiConfig.conformal_threshold_99 = thresholds["threshold_99"]
                DiavisionApiConfig.conformal_threshold_90 = thresholds["threshold_90"]
                print("✅ Conformal thresholds loaded successfully!")
            else:
                print(f"⚠️ Missing conformal_thresholds.json in: {base_dir}")
        except Exception as e:
            print(f"❌ Error loading conformal thresholds: {e}")