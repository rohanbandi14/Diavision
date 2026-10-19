import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import './table-modal.css';
import './briefing-panel.css';

// ==========================================================
// ICONS (inline, single-stroke, no icon library dependency)
// ==========================================================
const Icon = {
  Dossier: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="4" y="3" width="16" height="18" rx="2" /><path d="M8 8h8M8 12h8M8 16h5" /></svg>
  ),
  Record: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="1.6" /><path d="M4 18l5-5 4 3 3-3 4 4" /></svg>
  ),
  Triage: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M4 20V10M12 20V4M20 20v-7" /></svg>
  ),
  Hera: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" /><circle cx="12" cy="12" r="3" /></svg>
  ),
  Send: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 12L20 4l-6 16-3-7-7-1z" /></svg>
  ),
  ChevronLeft: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 18l-6-6 6-6" /></svg>
  ),
  ChevronRight: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 18l6-6-6-6" /></svg>
  ),
  Reset: () => (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 4v5h5" /><path d="M4.5 13a8 8 0 1 0 2.3-6.3L4 9" /></svg>
  ),
};

const DiaVisionLogo = ({ size = 40, className = '' }) => (
  <svg className={className} width={size} height={size} viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <filter id="eye-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" result="blur" />
        <feComposite in="SourceGraphic" in2="blur" operator="over" />
      </filter>
    </defs>
    <path d="M 15 50 C 35 25 65 25 85 50 C 65 75 35 75 15 50 Z" fill="none" stroke="var(--red)" strokeWidth="5" strokeLinejoin="round" />
    <circle cx="50" cy="50" r="14" fill="none" stroke="#34d399" strokeWidth="4.5" filter="url(#eye-glow)" />
    <circle cx="50" cy="50" r="4" fill="#ffffff" />
    <circle cx="33" cy="50" r="3" fill="#34d399" />
    <circle cx="67" cy="50" r="3" fill="#34d399" />
  </svg>
);

// ==========================================================
// SHAP -> RADAR MAPPING
// Maps the model's raw feature names (fixed on the backend) onto
// six systemic-risk axes for the biometric radar. Only the top-3
// SHAP factors are available per patient, so unlisted axes render
// at the population-average baseline (50).
// ==========================================================
const AXES = ['BMI', 'Blood Pressure', 'Smoking', 'Kidney Function', 'Vascular', 'Inflammation'];

const FEATURE_TO_AXIS = {
  'BMI': 'BMI',
  'systolic_pressure': 'Blood Pressure',
  'diastolic_pressure': 'Blood Pressure',
  'hypertension': 'Blood Pressure',
  'smoking_time.Y.': 'Smoking',
  'diabetic_nephropathy': 'Kidney Function',
  'number_of_lower_leg_artery_occlusion': 'Vascular',
  'corory_heart_disease': 'Vascular',
  'CRP': 'Inflammation',
};

const SHAP_LABELS = {
  'BMI': 'Body Mass Index', 'systolic_pressure': 'Systolic Pressure', 'diastolic_pressure': 'Diastolic Pressure',
  'hypertension': 'Hypertension', 'smoking_time.Y.': 'Smoking Duration', 'diabetic_nephropathy': 'Diabetic Nephropathy',
  'number_of_lower_leg_artery_occlusion': 'Leg Artery Occlusion', 'corory_heart_disease': 'Coronary Heart Disease', 'CRP': 'C-Reactive Protein',
};
const shapLabel = (n) => SHAP_LABELS[n] || n.replace(/[_.]+/g, ' ').trim();

function parseShapFactors(shapStr) {
  if (!shapStr || shapStr === 'N/A') return [];
  return shapStr.split(',').map((chunk) => {
    const trimmed = chunk.trim();
    const up = trimmed.includes('⬆️');
    const name = trimmed.replace('⬆️', '').replace('⬇️', '').trim();
    return { name, up };
  });
}

const FEATURE_TO_ZONE = {
  systolic_pressure: 'heart', diastolic_pressure: 'heart', hypertension: 'heart', corory_heart_disease: 'heart',
  'smoking_time.Y.': 'lungs', diabetic_nephropathy: 'kidney',
  number_of_lower_leg_artery_occlusion: 'foot', BMI: 'metabolic', CRP: 'metabolic',
};
const ZONES = {
  lungs: { side: 'L', base: 130, anchors: [[438, 241]] },
  heart: { side: 'R', base: 200, anchors: [[510, 257]] },
  kidney: { side: 'A', base: 380, anchors: [[460, 354], [540, 350]] },
  metabolic: { side: 'R', base: 580, anchors: [[498, 458]] },
  foot: { anchors: [[446, 540]] },
  brain: { anchors: [[502, 48]] },
};
const TIER_RANK = ['low', 'moderate', 'high', 'very_high'];
const zoneOf = (p) => {
  if (p.risk_tier === 'low') return 'brain';
  const f = parseShapFactors(p.shap_factors).find((x) => FEATURE_TO_ZONE[x.name]);
  return f ? FEATURE_TO_ZONE[f.name] : 'metabolic';
};
const featOf = (p) => {
  const fs = parseShapFactors(p.shap_factors);
  const f = fs.find((x) => FEATURE_TO_ZONE[x.name]) || fs[0];
  return f ? shapLabel(f.name) : 'Risk factor';
};

function BodyMap({ cohort, zone, onZone, sel, onPick }) {
  const col = (t) => TIER_TO_COLOR[t] || 'var(--green)';
  const tag = (p) => `#${String(p.patient_id).padStart(3, '0')}`;
  const byZone = {};
  cohort.forEach((p, i) => { const z = zoneOf(p); (byZone[z] = byZone[z] || []).push({ p, i }); });
  const worst = (list) => TIER_RANK[list.reduce((m, { p }) => Math.max(m, TIER_RANK.indexOf(p.risk_tier)), 0)];
  const cur = { L: 0, R: 0 };
  const els = [];
  const dots = [];
  const chip = (key, x, y, p, i) => (
    <g key={key} className="zone-chip" onClick={() => onPick(i)} transform={`translate(${x},${y})`}>
      <rect width="60" height="24" rx="12" fill={col(p.risk_tier)} fillOpacity={i === sel ? 0.45 : 0.18} stroke={col(p.risk_tier)} strokeWidth={i === sel ? 2.5 : 1.2} />
      <text x="30" y="16.5" textAnchor="middle" fontSize="13" fontWeight="600" fill="var(--text-hi)">{tag(p)}</text>
    </g>
  );
  const pill = (key, x, y, w, label, color, k) => (
    <g key={key} className="zone-badge" onClick={() => k && onZone(zone === k ? null : k)}>
      <rect x={x} y={y} width={w} height="30" rx="15" fill={color} fillOpacity="0.16" stroke={color} strokeOpacity="0.8" strokeWidth={zone === k ? 2.5 : 1.2} />
      <text x={x + w / 2} y={y + 20} textAnchor="middle" fontSize="14" fontWeight="600" fill={color}>{label}</text>
    </g>
  );
  Object.entries(ZONES).forEach(([k, z]) => {
    const list = byZone[k];
    if (!list) return;
    list.sort((a, b) => Number(b.p.risk_score) - Number(a.p.risk_score));
    const shownDots = list.slice(0, 14);
    const selItem = list.find((it) => it.i === sel);
    if (selItem && !shownDots.includes(selItem)) shownDots.push(selItem);
    shownDots.forEach(({ p, i }, n) => {
      const a = z.anchors[n % z.anchors.length], q = Math.floor(n / z.anchors.length);
      const r = Math.min(13 * Math.sqrt(q), k === 'brain' ? 30 : 40);
      dots.push(<circle key={`d${i}`} className={p.risk_tier === 'very_high' ? 'zone-dot pulse' : 'zone-dot'} cx={a[0] + r * Math.cos(q * 2.4)} cy={a[1] + r * Math.sin(q * 2.4)} r="8" fill={col(p.risk_tier)} stroke="#fff" strokeOpacity={i === sel ? 1 : 0.4} strokeWidth={i === sel ? 2.5 : 1.2} onClick={() => onPick(i)} />);
    });
    if (k === 'brain') {
      list.slice(0, 5).forEach(({ p, i }, n) => els.push(chip(`b${i}`, 566, 30 + n * 34, p, i)));
      if (list.length > 5) els.push(<text key="bm" x="566" y={30 + 5 * 34 + 14} fontSize="12" fill="var(--text-mid)">+{list.length - 5} more</text>);
      return;
    }
    if (k === 'foot') {
      list.slice(0, 6).forEach(({ p, i }, n) => {
        const y = 620 + n * 44, label = featOf(p), w = label.length * 8.4 + 30, c = col(p.risk_tier);
        els.push(<line key={`fl${i}`} x1="240" y1={y} x2="412" y2={y} stroke={c} strokeOpacity="0.6" />);
        els.push(pill(`fp${i}`, 240 - w, y - 15, w, label, c, 'foot'));
        els.push(chip(`fc${i}`, 412, y - 12, p, i));
      });
      if (list.length > 6) els.push(<text key="fm" x="442" y={620 + 6 * 44} textAnchor="middle" fontSize="12" fill="var(--text-mid)">+{list.length - 6} more</text>);
      return;
    }
    const groups = {};
    list.forEach((it) => { const f = featOf(it.p); (groups[f] = groups[f] || []).push(it); });
    Object.entries(groups).forEach(([label, g], gi) => {
      const side = z.side === 'A' ? (gi % 2 ? 'R' : 'L') : z.side;
      const y = Math.max(z.base, cur[side]), w = label.length * 8.4 + 30, c = col(worst(g));
      const ex = side === 'L' ? 250 : 690, an = z.anchors[gi % z.anchors.length];
      els.push(<line key={`l${k}${gi}`} x1={ex} y1={y + 15} x2={an[0]} y2={an[1]} stroke={c} strokeOpacity="0.6" />);
      els.push(pill(`p${k}${gi}`, side === 'L' ? ex - w : ex, y, w, label, c, k));
      const shown = g.slice(0, 6), rows = Math.ceil(shown.length / 2);
      shown.forEach(({ p, i }, n) => els.push(chip(`c${i}`, (side === 'L' ? 126 : 690) + (n % 2) * 66, y + 38 + Math.floor(n / 2) * 30, p, i)));
      if (g.length > 6) els.push(<text key={`m${k}${gi}`} x={side === 'L' ? 250 : 690} y={y + 38 + rows * 30 + 12} textAnchor={side === 'L' ? 'end' : 'start'} fontSize="12" fill="var(--text-mid)">+{g.length - 6} more</text>);
      cur[side] = y + 38 + rows * 30 + (g.length > 6 ? 22 : 8) + 14;
    });
  });
  return (
    <div className="body-map">
      <svg viewBox="0 0 1024 993" width="100%">
        <image href={`${process.env.PUBLIC_URL || ''}/body-map.png`} x="0" y="0" width="1024" height="993" />
        {els}
        {dots}
      </svg>
    </div>
  );
}
const TIER_TO_COLOR = { very_high: 'var(--red)', high: 'var(--orange)', moderate: 'var(--amber)', low: 'var(--green)' };

function buildRadarValues(shapStr) {
  const factors = parseShapFactors(shapStr);
  const values = {};
  AXES.forEach((a) => { values[a] = 50; });
  const offsets = [34, 24, 15];
  factors.forEach((f, i) => {
    const axis = FEATURE_TO_AXIS[f.name];
    if (!axis) return;
    const delta = offsets[i] ?? 10;
    values[axis] = Math.max(6, Math.min(96, 50 + (f.up ? delta : -delta)));
  });
  return values;
}

// ==========================================================
// RADAR / SPIDER CHART (pure SVG, no dependency)
// ==========================================================
const RadarChart = ({ values }) => {
  const size = 240;
  const center = size / 2;
  const maxR = 88;
  const n = AXES.length;
  const angleFor = (i) => (Math.PI * 2 * i) / n - Math.PI / 2;

  const pointAt = (i, val) => {
    const r = (val / 100) * maxR;
    const a = angleFor(i);
    return [center + r * Math.cos(a), center + r * Math.sin(a)];
  };

  const dataPoints = AXES.map((axis, i) => pointAt(i, values[axis] ?? 50));
  const baselinePoints = AXES.map((_, i) => pointAt(i, 50));
  const toPath = (pts) => pts.map((p) => p.join(',')).join(' ');

  const rings = [25, 50, 75, 100];

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {rings.map((r) => (
        <polygon
          key={r}
          points={toPath(AXES.map((_, i) => pointAt(i, r)))}
          fill="none"
          stroke="var(--hairline)"
          strokeWidth="1"
        />
      ))}
      {AXES.map((_, i) => {
        const [x, y] = pointAt(i, 100);
        return <line key={i} x1={center} y1={center} x2={x} y2={y} stroke="var(--hairline)" strokeWidth="1" />;
      })}
      <polygon points={toPath(baselinePoints)} fill="none" stroke="var(--text-dim)" strokeWidth="1.2" strokeDasharray="3,3" />
      <polygon points={toPath(dataPoints)} fill="rgba(34,211,238,0.16)" stroke="var(--cyan)" strokeWidth="2" />
      {dataPoints.map(([x, y], i) => (
        <circle key={i} cx={x} cy={y} r="3" fill="var(--cyan)" />
      ))}
      {AXES.map((axis, i) => {
        const a = angleFor(i);
        const lx = center + (maxR + 26) * Math.cos(a);
        const ly = center + (maxR + 26) * Math.sin(a);
        return (
          <text
            key={axis}
            x={lx}
            y={ly}
            textAnchor="middle"
            dominantBaseline="middle"
            fill="var(--text-mid)"
            fontSize="9.5"
            fontFamily="var(--font-mono)"
          >
            {axis}
          </text>
        );
      })}
    </svg>
  );
};

// ==========================================================
// SMALL HELPERS
// ==========================================================
const TIER_CLASS = { low: 'pill-normal', moderate: 'pill-moderate', high: 'pill-low', very_high: 'pill-high' };
const TIER_ORDER = ['very_high', 'high', 'moderate', 'low'];
const TIER_LABEL = { very_high: 'Very High', high: 'High', moderate: 'Moderate', low: 'Low' };
const TIER_BAR_COLOR = { very_high: 'var(--red)', high: 'var(--orange)', moderate: 'var(--amber)', low: 'var(--green)' };

const confidenceClass = (tier) => {
  if (tier === 'Clear') return 'pill-conf-clear';
  if (tier === 'Likely, but check') return 'pill-conf-check';
  if (tier === 'Uncertain') return 'pill-conf-uncertain';
  return 'pill-unknown';
};

const smokingClass = (text) => {
  if (!text) return 'pill-normal';
  if (text.includes('Heavy') || text.includes('≥20')) return 'pill-high';
  if (text.includes('Moderate') || text.includes('10-19')) return 'pill-moderate';
  if (text === '🚬 Smoking history' || text.includes('Current')) return 'pill-low';
  if (text.includes('Not assessed')) return 'pill-unknown';
  return 'pill-normal';
};

const classifyLevel = (value) => {
  if (value >= 75) return { label: 'High', color: 'var(--red)', cls: 'pill-level-high' };
  if (value >= 50) return { label: 'Elevated', color: 'var(--orange)', cls: 'pill-level-elevated' };
  if (value >= 30) return { label: 'Moderate', color: 'var(--amber)', cls: 'pill-level-moderate' };
  return { label: 'Normal', color: 'var(--green)', cls: 'pill-level-normal' };
};

const getRiskRecommendation = (tier) => {
  switch (tier) {
    case 'very_high': return 'Urgent podiatry referral recommended — schedule an in-person foot exam within 1–2 weeks.';
    case 'high': return 'Podiatry referral recommended, with closer monitoring of glycemic and vascular risk factors.';
    case 'moderate': return 'Reinforce foot self-care education and re-screen at the next routine visit.';
    default: return 'No elevated screening signal. Continue routine annual foot screening.';
  }
};

const getVisionRecommendation = (result) => {
  if (!result) return '';
  return result.is_abnormal
    ? 'Visual findings suggest a possible abnormality. Recommend an in-person podiatry examination and wound assessment.'
    : 'No abnormal tissue pattern detected in this capture. Continue routine foot checks and re-scan if symptoms develop.';
};

// ==========================================================
// CIRCULAR RISK GAUGE (pure SVG)
// ==========================================================
const CircularGauge = ({ value = 0, size = 108, stroke = 9, label = 'RISK' }) => {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const dash = (pct / 100) * c;
  const ringColor = pct >= 75 ? 'var(--red)' : pct >= 50 ? 'var(--orange)' : pct >= 30 ? 'var(--amber)' : 'var(--green)';
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="gauge-svg">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--hairline)" strokeWidth={stroke} />
      <circle
        cx={size / 2} cy={size / 2} r={r} fill="none"
        stroke={ringColor} strokeWidth={stroke} strokeLinecap="round"
        strokeDasharray={`${dash} ${c - dash}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
        style={{ filter: `drop-shadow(0 0 5px ${ringColor})`, transition: 'stroke-dasharray 0.6s ease' }}
      />
      <text x="50%" y="46%" textAnchor="middle" dominantBaseline="middle" fill="var(--text-hi)" fontSize={size * 0.22} fontWeight="700" fontFamily="var(--font-display)">
        {Math.round(pct)}%
      </text>
      <text x="50%" y="65%" textAnchor="middle" dominantBaseline="middle" fill="var(--text-dim)" fontSize={size * 0.078} fontFamily="var(--font-mono)" letterSpacing="1">
        {label}
      </text>
    </svg>
  );
};

// ==========================================================
// PATIENT HEADER (shared by Dossier + Record views)
// ==========================================================
const PatientHeader = ({ patient, contextLabel, actions }) => {
  if (!patient) return null;
  return (
    <div className="panel patient-header">
      <div className="patient-id-block">
        <div className="patient-avatar"><Icon.Record /></div>
        <div>
          <div className="patient-id-row">
            <span className="patient-id">Patient #{String(patient.patient_id).padStart(3, '0')}</span>
            <span className={`pill ${TIER_CLASS[patient.risk_tier] || 'pill-normal'}`}>{patient.diagnosis}</span>
          </div>
          {contextLabel && <div className="patient-meta">{contextLabel}</div>}
          <div className="patient-chip-row">
            <span className={`pill ${confidenceClass(patient.confidence_tier)}`}>{patient.confidence_tier}</span>
            <span className={`pill ${smokingClass(patient.smoking_flag)}`}>{patient.smoking_flag || '✅ Non-smoker'}</span>
          </div>
        </div>
      </div>
      <div className="patient-gauge-block">
        <CircularGauge value={Number(patient.risk_score)} />
        <span className="gauge-caption">AI ESTIMATED RISK</span>
      </div>
      {actions && <div className="patient-actions">{actions}</div>}
    </div>
  );
};

// ==========================================================
// APP
// ==========================================================
const INITIAL_HERA_MESSAGE = {
  sender: 'hera',
  text: 'Hera online. I\'m watching the workspace — ask me about a patient, a protocol, or a risk factor anytime.',
};

// ==========================================================
// COHORT TABLE (compact preview + full-window modal)
// ==========================================================
const PREVIEW_ROWS = 5;

function CohortTable({ rows, selectedPatientIdx, onSelect, expandedIdx, onToggle }) {
  return (
    <div className="data-table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            <th>Patient</th>
            <th>Diagnosis</th>
            <th>Score</th>
            <th>Confidence</th>
            <th>Smoking</th>
            <th>SHAP Factors</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ p, idx }) => {
            const fs = parseShapFactors(p.shap_factors);
            const ups = fs.filter((f) => f.up).length;
            const risky = ups > 0 && ups >= fs.length - ups;
            const open = expandedIdx === idx;
            return (
              <React.Fragment key={idx}>
                <tr className={idx === selectedPatientIdx ? 'selected' : ''} onClick={() => onSelect(idx)}>
                  <td>#{String(p.patient_id).padStart(3, '0')}</td>
                  <td><span className={`pill ${TIER_CLASS[p.risk_tier] || 'pill-normal'}`}>{p.diagnosis}</span></td>
                  <td>{p.risk_score}%</td>
                  <td><span className={`pill ${confidenceClass(p.confidence_tier)}`}>{p.confidence_tier}</span></td>
                  <td><span className={`pill ${smokingClass(p.smoking_flag)}`}>{p.smoking_flag || '✅ Non-smoker'}</span></td>
                  <td>
                    {fs.length ? (
                      <button className={`shap-pill ${risky ? 'risk' : 'safe'} ${open ? 'open' : ''}`} aria-expanded={open}
                        onClick={(e) => { e.stopPropagation(); onToggle(idx); }}>
                        {ups ? `${ups} Active Risk Factor${ups > 1 ? 's' : ''}` : `${fs.length} Protective`}
                        <span className="shap-chev">▼</span>
                      </button>
                    ) : <span style={{ color: 'var(--text-dim)' }}>—</span>}
                  </td>
                </tr>
                <tr className="shap-drawer-row">
                  <td colSpan={6}>
                    <div className={`shap-drawer ${open ? 'open' : ''}`}>
                      <div className="shap-drawer-inner">
                        <div className="shap-grid">
                          {fs.map((f, k) => (
                            <React.Fragment key={k}>
                              <span className="shap-key">{shapLabel(f.name)}</span>
                              <span className={`shap-val ${f.up ? 'risk' : 'safe'}`}>{f.up ? '▲ Risk' : '▼ Normal'}<b>rank #{k + 1}</b></span>
                            </React.Fragment>
                          ))}
                        </div>
                      </div>
                    </div>
                  </td>
                </tr>
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CohortModal({ rows, total, onClose, onExport, ...tableProps }) {
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev; };
  }, [onClose]);

  return (
    <div className="cohort-modal-backdrop" onClick={onClose}>
      <div className="cohort-modal" role="dialog" aria-modal="true" aria-label="Full cohort table" onClick={(e) => e.stopPropagation()}>
        <div className="cohort-modal-head">
          <div>
            <h2 className="cohort-modal-title">Cohort Results</h2>
            <div className="cohort-modal-sub">{rows.length}{rows.length !== total ? ` of ${total}` : ''} patients</div>
          </div>
          <div className="cohort-modal-actions">
            <button className="btn btn-ghost" onClick={onExport}>Export Cohort (CSV)</button>
            <button className="btn btn-ghost" onClick={onClose} aria-label="Close">✕ Close</button>
          </div>
        </div>
        <div className="cohort-modal-body">
          <CohortTable rows={rows} {...tableProps} />
        </div>
      </div>
    </div>
  );
}

// ==========================================================
// PATIENT BRIEFING PANEL (right of the body map)
// ==========================================================
function BriefingPanel({ cohort, zoneFilter, selectedIdx, onSelect, dismissed, escalated, onDismiss, onEscalate, onExport, onClose }) {
  const items = cohort
    .map((p, idx) => ({ p, idx }))
    .filter(({ p, idx }) => !dismissed.has(idx) && (!zoneFilter || zoneOf(p) === zoneFilter))
    .sort((a, b) => Number(b.p.risk_score) - Number(a.p.risk_score));
  const gaugeColor = (t) => TIER_TO_COLOR[t] || 'var(--green)';

  return (
    <aside className="briefing-panel">
      <div className="briefing-head">
        <span>Patient Briefing Card</span>
        <button className="briefing-close" onClick={onClose} aria-label="Close briefing panel">✕</button>
      </div>
      <div className="briefing-list">
        {items.length === 0 && <div className="empty-state">No patients to brief{zoneFilter ? ' in this zone' : ''}.</div>}
        {items.map(({ p, idx }) => {
          const isEsc = escalated.has(idx);
          const score = Number(p.risk_score) || 0;
          const c = gaugeColor(p.risk_tier);
          return (
            <div key={idx} className={`briefing-card ${idx === selectedIdx ? 'selected' : ''} ${isEsc ? 'escalated' : ''}`} onClick={() => onSelect(idx)}>
              <div className="briefing-card-top">
                <span className="briefing-id">#{String(p.patient_id).padStart(3, '0')}</span>
                <span className={`pill ${TIER_CLASS[p.risk_tier] || 'pill-normal'}`}>{p.diagnosis}</span>
              </div>
              <div className="briefing-body">
                <div className="briefing-ring" style={{ '--c': c, '--pct': `${Math.min(score, 100)}%` }}>
                  <span>{Math.round(score)}</span>
                </div>
                <dl className="briefing-facts">
                  <dt>Risk Score</dt><dd>{p.risk_score}%</dd>
                  <dt>Smoking Flag</dt><dd><span className={`pill ${smokingClass(p.smoking_flag)}`}>{p.smoking_flag || '✅ Non-smoker'}</span></dd>
                  <dt>Confidence</dt><dd><span className={`pill ${confidenceClass(p.confidence_tier)}`}>{p.confidence_tier}</span></dd>
                </dl>
              </div>
              <div className="briefing-actions">
                <button className="b-export" onClick={(e) => { e.stopPropagation(); onExport(p); }}>Export</button>
                <button className={`b-escalate ${isEsc ? 'on' : ''}`} onClick={(e) => { e.stopPropagation(); onEscalate(idx); }}>{isEsc ? 'Escalated ✓' : 'Escalate'}</button>
                <button className="b-dismiss" onClick={(e) => { e.stopPropagation(); onDismiss(idx); }}>Dismiss</button>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [activeView, setActiveView] = useState('dossier');
  const [isLightMode, setIsLightMode] = useState(false);
  const [copilotOpen, setCopilotOpen] = useState(true);

  // Hera chat state
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([INITIAL_HERA_MESSAGE]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const threadRef = useRef(null);
  const sessionIdRef = useRef(Date.now().toString(36));
  const discussedRef = useRef(null);
  const handleResetChat = () => {
    sessionIdRef.current = Date.now().toString(36);
    discussedRef.current = null;
    setChatHistory([INITIAL_HERA_MESSAGE]);
    setChatInput('');
  };

  // Clinical dossier state
  const [csvFile, setCsvFile] = useState(null);
  const [tabularResult, setTabularResult] = useState(null);
  const [clinicalError, setClinicalError] = useState('');
  const [isClinicalLoading, setIsClinicalLoading] = useState(false);
  const [selectedPatientIdx, setSelectedPatientIdx] = useState(0);
  const [expandedIdx, setExpandedIdx] = useState(null);
  const [tableModalOpen, setTableModalOpen] = useState(false);
  const [briefingOpen, setBriefingOpen] = useState(true);
  const [dismissedIdx, setDismissedIdx] = useState(() => new Set());
  const [escalatedIdx, setEscalatedIdx] = useState(() => new Set());
  const [zoneFilter, setZoneFilter] = useState(null);

  // Vision / unified record state
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [visionResult, setVisionResult] = useState(null);
  const [visionError, setVisionError] = useState('');
  const [isVisionLoading, setIsVisionLoading] = useState(false);

  // Recent activity feed (client-side, real timestamps of real actions)
  const [activityLog, setActivityLog] = useState([]);
  const pushActivity = (text) => {
    setActivityLog((log) => [
      { time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }), text },
      ...log,
    ].slice(0, 8));
  };

  useEffect(() => {
    const timer = setTimeout(() => setShowSplash(false), 3000);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight;
  }, [chatHistory, isChatLoading]);

  // ---- HERA CHAT ----
  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    const newHistory = [...chatHistory, { sender: 'user', text: chatInput }];
    setChatHistory(newHistory);
    setChatInput('');

    // "patient 3" / "patient #03" -> Hera discusses that cohort row
    if (/^\s*(exit|done|back to screening)\s*$/i.test(chatInput)) discussedRef.current = null;
    const pm = chatInput.match(/(?:patient|pt)\s*#?\s*0*(\d+)/i);
    if (pm) {
      const found = tabularResult && tabularResult.find((p) => Number(p.patient_id) === Number(pm[1]));
      if (!found) {
        setChatHistory([...newHistory, { sender: 'hera', text: tabularResult ? `I can't find patient #${pm[1]} in the loaded cohort (1–${tabularResult.length}).` : 'Upload a cohort CSV in the Dossier view first, then ask me about a patient.' }]);
        return;
      }
      const scores = tabularResult.map((p) => Number(p.risk_score));
      discussedRef.current = {
        ...found,
        cohort_size: scores.length,
        higher_than_pct: Math.round((100 * scores.filter((v) => v < Number(found.risk_score)).length) / scores.length),
        cohort_avg: Math.round(scores.reduce((a, b) => a + b, 0) / scores.length),
      };
    }

    setIsChatLoading(true);

    const fullTranscript = newHistory.map((m) => `${m.sender === 'user' ? 'Patient' : 'Hera'}: ${m.text}`).join('\n');

    try {
      const response = await fetch('http://127.0.0.1:8000/api/hera/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ transcript: fullTranscript, session_id: sessionIdRef.current, patient_context: discussedRef.current || undefined }),
      });
      const data = await response.json();
      if (response.ok) {
        let updated = [...newHistory, { sender: 'hera', text: data.reply }];
        if (data.risk_score !== null && data.risk_score !== undefined) {
          updated.push({ sender: 'hera', isRiskCard: true, diagnosis: data.diagnosis, score: data.risk_score, context: data.extracted_data });
        }
        setChatHistory(updated);
      } else {
        setChatHistory([...newHistory, { sender: 'hera', text: 'Sorry, I encountered an error connecting to my servers.' }]);
      }
    } catch (err) {
      setChatHistory([...newHistory, { sender: 'hera', text: 'Connection error. Is the Django server running?' }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  // ---- CLINICAL DOSSIER ----
  const handleDownloadCSV = () => {
    if (!tabularResult || tabularResult.length === 0) return;
    const headers = 'Patient ID,AI Diagnosis,Risk Score (%),Smoking Risk Flag,Top Risk Factors (SHAP),Confidence Tier';
    const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const rows = tabularResult
      .map((p) => [p.patient_id, p.diagnosis, p.risk_score, p.smoking_flag || '✅ Non-smoker', p.shap_factors, p.confidence_tier].map(esc).join(','))
      .join('\n');
    const blob = new Blob([`\uFEFF${headers}\n${rows}`], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', 'diavision_risk_results.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleExportPatient = (p) => {
    const headers = 'Patient ID,AI Diagnosis,Risk Score (%),Smoking Risk Flag,Top Risk Factors (SHAP),Confidence Tier';
    const esc = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const row = [p.patient_id, p.diagnosis, p.risk_score, p.smoking_flag || '✅ Non-smoker', p.shap_factors, p.confidence_tier].map(esc).join(',');
    const blob = new Blob([`\uFEFF${headers}\n${row}`], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `diavision_patient_${String(p.patient_id).padStart(3, '0')}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const toggleEscalate = (idx) => {
    const pid = tabularResult && tabularResult[idx] ? String(tabularResult[idx].patient_id).padStart(3, '0') : idx;
    setEscalatedIdx((prev) => {
      const next = new Set(prev);
      if (next.has(idx)) { next.delete(idx); } else { next.add(idx); pushActivity(`Patient #${pid} escalated for clinical review`); }
      return next;
    });
  };

  const dismissPatient = (idx) => setDismissedIdx((prev) => new Set(prev).add(idx));

  const handleCsvUpload = async (e) => {
    e.preventDefault();
    if (!csvFile) return;
    setClinicalError('');
    setTabularResult(null);
    setIsClinicalLoading(true);
    const formData = new FormData();
    formData.append('file', csvFile);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict/tabular/', { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setTabularResult(data.results);
        setSelectedPatientIdx(0);
        setDismissedIdx(new Set());
        setEscalatedIdx(new Set());
        pushActivity(`Cohort analyzed — ${data.results.length} patient${data.results.length === 1 ? '' : 's'} scored`);
      } else {
        setClinicalError(data.error || 'Error running tabular prediction.');
      }
    } catch (err) {
      setClinicalError('Error connecting to the server. Make sure Django is running.');
    } finally {
      setIsClinicalLoading(false);
    }
  };

  // ---- VISION / UNIFIED RECORD ----
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImageFile(file);
      setImagePreview(URL.createObjectURL(file));
      setVisionResult(null);
      setVisionError('');
      pushActivity('New foot scan image selected');
    }
  };

  const handleVisionUpload = async (e) => {
    e.preventDefault();
    if (!imageFile) return;
    setVisionError('');
    setIsVisionLoading(true);
    const formData = new FormData();
    formData.append('image', imageFile);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/predict/vision/', { method: 'POST', body: formData });
      const data = await response.json();
      if (response.ok) {
        setVisionResult(data);
        pushActivity(`Grad-CAM analysis completed — ${data.diagnosis || (data.is_abnormal ? 'Abnormal' : 'Normal')}`);
      } else {
        setVisionError(data.error || 'Error analyzing image.');
      }
    } catch (err) {
      setVisionError('Error connecting to the server. Make sure Django is running.');
    } finally {
      setIsVisionLoading(false);
    }
  };

  const selectedPatient = tabularResult && tabularResult[selectedPatientIdx];
  const radarValues = selectedPatient ? buildRadarValues(selectedPatient.shap_factors) : {};
  const selectedFactors = selectedPatient ? parseShapFactors(selectedPatient.shap_factors) : [];
  const cohortAvg = tabularResult && tabularResult.length
    ? Math.round(tabularResult.reduce((sum, p) => sum + Number(p.risk_score || 0), 0) / tabularResult.length)
    : null;

  const triageGroups = TIER_ORDER.reduce((acc, tier) => {
    acc[tier] = (tabularResult || []).filter((p) => p.risk_tier === tier);
    return acc;
  }, {});

  const overviewPatient = tabularResult && tabularResult[0];
  const overviewValues = overviewPatient ? buildRadarValues(overviewPatient.shap_factors) : null;

  // ---- NAV CONFIG ----
  const NAV_ITEMS = [
    { id: 'dossier', label: 'Dossier', Icon: Icon.Dossier },
    { id: 'record', label: 'Record', Icon: Icon.Record },
    { id: 'triage', label: 'Triage', Icon: Icon.Triage },
    { id: 'hera', label: 'Hera', Icon: Icon.Hera },
  ];

  const renderCopilotThread = () => (
    <div className="copilot-thread scrollbar-styled" ref={threadRef}>
      {chatHistory.map((msg, i) =>
        msg.isRiskCard ? (
          <div key={i} className="risk-card-msg" style={{ borderLeftColor: msg.diagnosis?.includes('High') ? 'var(--red)' : 'var(--green)' }}>
            <h4>Clinical Risk Assessment</h4>
            <p><span className={msg.diagnosis?.includes('High') ? 'pill pill-high' : 'pill pill-normal'}>{msg.diagnosis}</span></p>
            <p>Model score: {msg.score}%</p>
            {msg.context && ['sat_fat_intake', 'sleep_hours', 'family_history'].some((k) => msg.context[k] != null) && (
              <p style={{ opacity: 0.7, fontSize: '0.8em' }}>
                Intake only, not used in score: fatty food {['rarely', 'sometimes', 'often'][msg.context.sat_fat_intake] ?? '—'} · sleep {msg.context.sleep_hours != null ? `${msg.context.sleep_hours} h` : '—'} · family history {msg.context.family_history == null ? '—' : msg.context.family_history ? 'yes' : 'no'}
              </p>
            )}
          </div>
        ) : msg.sender === 'user' ? (
          <div key={i} className="msg msg-user">{msg.text}</div>
        ) : (
          <div key={i} className="msg msg-hera">
            <div className="msg-hera-label">HERA</div>
            {msg.text}
          </div>
        )
      )}
      {isChatLoading && <div className="msg msg-loading">Hera is analyzing…</div>}
    </div>
  );

  const renderCopilotForm = () => (
    <form onSubmit={handleChatSubmit} className="copilot-form">
      <input
        type="text"
        value={chatInput}
        onChange={(e) => setChatInput(e.target.value)}
        placeholder="Ask Hera anything…"
        disabled={isChatLoading}
      />
      <button type="submit" className="copilot-send" disabled={isChatLoading || !chatInput.trim()} aria-label="Send">
        <Icon.Send />
      </button>
    </form>
  );

  return (
    <div className={`App ${isLightMode ? 'light-theme' : ''}`}>
      {showSplash && (
        <div className="splash-screen">
          <DiaVisionLogo size={140} className="splash-blink" />
          <h1 className="splash-title">DIAVISION</h1>
          <p className="splash-subtitle">AI DIAGNOSTIC COMMAND CENTER</p>
        </div>
      )}

      <div className={`shell ${activeView === 'hera' ? 'no-sidebar' : (copilotOpen ? '' : 'sidebar-collapsed')}`}>
        {/* ---------------- LEFT NAV ---------------- */}
        <nav className="nav-rail">
          <div className="nav-brand">
            <DiaVisionLogo size={30} />
            <span className="nav-brand-label">DIAVISION AI</span>
          </div>

          {NAV_ITEMS.map(({ id, label, Icon: ItemIcon }) => (
            <button
              key={id}
              className={`nav-item ${activeView === id ? 'active' : ''}`}
              onClick={() => setActiveView(id)}
              title={label}
            >
              <ItemIcon />
              <span className="nav-item-label">{label}</span>
            </button>
          ))}

          <div className="nav-bottom">
            <button className="theme-toggle-btn" onClick={() => setIsLightMode(!isLightMode)} title="Toggle theme">
              {isLightMode ? '🌙' : '☀️'}
            </button>
          </div>
        </nav>

        {/* ---------------- CENTER STAGE ---------------- */}
        <main className="stage scrollbar-styled">
          {/* ===== DOSSIER ===== */}
          {activeView === 'dossier' && (
            <>
              <div className="stage-header">
                <div>
                  <h1 className="stage-title">Clinical Risk Dossier</h1>
                  <div className="stage-sub">systemic risk model // tabular inference</div>
                </div>
                <span className="status-chip"><span className="status-dot" />MODEL READY</span>
              </div>

              <div className="panel">
                <p className="panel-label">UPLOAD PATIENT COHORT (.CSV)</p>
                <form onSubmit={handleCsvUpload} className="upload-row">
                  <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files[0])} />
                  <button type="submit" className="btn" disabled={isClinicalLoading || !csvFile}>
                    {isClinicalLoading ? 'Analyzing…' : 'Run Risk Prediction'}
                  </button>
                </form>
                {clinicalError && <div className="error-box">{clinicalError}</div>}
              </div>

              {tabularResult && tabularResult.length > 0 && selectedPatient && (
                <PatientHeader
                  patient={selectedPatient}
                  contextLabel="Clinical Risk Dossier — tabular cohort inference"
                />
              )}

              {tabularResult && tabularResult.length > 0 && selectedPatient && (
                <div className="dossier-top-grid">
                  <div className="panel">
                    <p className="panel-label">KEY CLINICAL SIGNALS</p>
                    {selectedFactors.length > 0 ? (
                      <ul className="signal-list">
                        {selectedFactors.map((f, i) => (
                          <li className="signal-item" key={i}>
                            <span className={`signal-dir ${f.up ? 'signal-up' : 'signal-down'}`}>{f.up ? '▲' : '▼'}</span>
                            <span className="signal-name">{f.name}</span>
                            <span className={`signal-tag ${f.up ? 'signal-tag-up' : 'signal-tag-down'}`}>{f.up ? 'Increases risk' : 'Lowers risk'}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div className="empty-state">No dominant factors reported for this patient.</div>
                    )}
                    <div className="cohort-compare">
                      <div className="cohort-compare-item">
                        <span className="cohort-compare-label">This patient</span>
                        <span className="cohort-compare-val" style={{ color: TIER_BAR_COLOR[selectedPatient.risk_tier] }}>{selectedPatient.risk_score}%</span>
                      </div>
                      <div className="cohort-compare-item">
                        <span className="cohort-compare-label">Cohort average</span>
                        <span className="cohort-compare-val">{cohortAvg}%</span>
                      </div>
                    </div>
                  </div>

                  <div className="panel radar-wrap">
                    <p className="panel-label" style={{ alignSelf: 'flex-start' }}>
                      RISK FACTOR PROFILE — HIGHER RISK →
                    </p>
                    <RadarChart values={radarValues} />
                    <div className="radar-legend">
                      <span><span className="legend-swatch" style={{ background: 'var(--cyan)' }} />This patient</span>
                      <span><span className="legend-swatch" style={{ background: 'var(--text-dim)' }} />Population avg</span>
                    </div>
                  </div>

                  <div className="panel">
                    <p className="panel-label">MODEL EXPLANATION (RANKED SHAP)</p>
                    {selectedFactors.length > 0 ? (
                      <div className="shap-bars">
                        {selectedFactors.map((f, i) => {
                          const width = [88, 62, 40][i] ?? 30;
                          return (
                            <div className="shap-bar-row" key={i}>
                              <span className="shap-bar-label">{f.name}</span>
                              <div className="bar-track">
                                <div className="bar-fill" style={{ width: `${width}%`, background: f.up ? 'var(--red)' : 'var(--green)' }} />
                              </div>
                            </div>
                          );
                        })}
                        <p className="radar-caption" style={{ textAlign: 'left' }}>
                          Bar length reflects contribution rank (1st–3rd), not exact SHAP magnitude.
                        </p>
                      </div>
                    ) : (
                      <div className="empty-state">No SHAP contributors returned for this patient.</div>
                    )}
                    <div className="recommend-box">
                      <p className="recommend-box-title">Clinical Recommendation</p>
                      <p>{getRiskRecommendation(selectedPatient.risk_tier)}</p>
                    </div>
                  </div>
                </div>
              )}

              {tabularResult && tabularResult.length > 0 && (
                <div className="panel">
                  <div className="legend-row">
                    <span><span className="pill pill-normal">Low</span> 0–29%</span>
                    <span><span className="pill pill-moderate">Moderate</span> 30–49%</span>
                    <span><span className="pill pill-low">High</span> 50–74%</span>
                    <span><span className="pill pill-high">Very High</span> 75–100%</span>
                    {!briefingOpen && (
                      <button className="btn btn-ghost" style={{ marginLeft: 'auto' }} onClick={() => setBriefingOpen(true)}>
                        Show Briefing Cards
                      </button>
                    )}
                    <button className="btn btn-ghost" style={briefingOpen ? { marginLeft: 'auto' } : undefined} onClick={handleDownloadCSV}>
                      Export Cohort (CSV)
                    </button>
                  </div>

                  <div className={`map-briefing ${briefingOpen ? '' : 'briefing-closed'}`}>
                    <BodyMap cohort={tabularResult} zone={zoneFilter} onZone={setZoneFilter} sel={selectedPatientIdx} onPick={setSelectedPatientIdx} />
                    {briefingOpen && (
                      <BriefingPanel
                        cohort={tabularResult}
                        zoneFilter={zoneFilter}
                        selectedIdx={selectedPatientIdx}
                        onSelect={setSelectedPatientIdx}
                        dismissed={dismissedIdx}
                        escalated={escalatedIdx}
                        onDismiss={dismissPatient}
                        onEscalate={toggleEscalate}
                        onExport={handleExportPatient}
                        onClose={() => setBriefingOpen(false)}
                      />
                    )}
                  </div>

                  {(() => {
                    const allRows = tabularResult.map((p, idx) => ({ p, idx })).filter(({ p }) => !zoneFilter || zoneOf(p) === zoneFilter);
                    const preview = allRows.slice(0, PREVIEW_ROWS);
                    const openFull = (idx) => { setExpandedIdx(idx); setSelectedPatientIdx(idx); setTableModalOpen(true); };
                    const tableProps = {
                      selectedPatientIdx,
                      onSelect: setSelectedPatientIdx,
                      expandedIdx,
                    };
                    return (
                      <>
                        <CohortTable rows={preview} {...tableProps} onToggle={openFull} />
                        <div className="table-footer">
                          <span>Showing {preview.length} of {allRows.length} patients</span>
                          <button className="btn btn-ghost" onClick={() => { setExpandedIdx(null); setTableModalOpen(true); }}>
                            ⤢ Expand full table
                          </button>
                        </div>
                        {tableModalOpen && (
                          <CohortModal
                            rows={allRows}
                            total={tabularResult.length}
                            onClose={() => setTableModalOpen(false)}
                            onExport={handleDownloadCSV}
                            {...tableProps}
                            onToggle={(idx) => setExpandedIdx(expandedIdx === idx ? null : idx)}
                          />
                        )}
                      </>
                    );
                  })()}
                </div>
              )}
            </>
          )}

          {/* ===== UNIFIED RECORD ===== */}
          {activeView === 'record' && (
            <>
              <div className="stage-header">
                <div>
                  <h1 className="stage-title">Unified Patient Record</h1>
                  <div className="stage-sub">grad-cam vision inference // thermal overlay</div>
                </div>
                <span className="status-chip"><span className="status-dot" />VISION ONLINE</span>
              </div>

              {overviewPatient && (
                <PatientHeader
                  patient={overviewPatient}
                  contextLabel="Unified Patient Record — synced from cohort row #1"
                />
              )}

              <div className="panel">
                <p className="panel-label">UPLOAD FOOT SCAN</p>
                <form onSubmit={handleVisionUpload} className="upload-row">
                  <input type="file" accept="image/*" onChange={handleImageChange} />
                  <button type="submit" className="btn" disabled={!imageFile || isVisionLoading}>
                    {isVisionLoading ? 'Scanning…' : 'Generate Grad-CAM Analysis'}
                  </button>
                </form>
                {visionError && <div className="error-box">{visionError}</div>}
              </div>

              <div className="record-split">
                <div className="panel">
                  <p className="panel-label">SYSTEMIC RISK OVERVIEW</p>
                  {overviewValues ? (
                    <>
                      {AXES.map((axis) => {
                        const lvl = classifyLevel(overviewValues[axis]);
                        return (
                          <div className="bar-metric" key={axis}>
                            <div className="bar-metric-label">
                              <span>{axis}</span>
                              <span className="bar-metric-right">
                                {Math.round(overviewValues[axis])}%
                                <span className={`pill ${lvl.cls}`}>{lvl.label}</span>
                              </span>
                            </div>
                            <div className="bar-track">
                              <div className="bar-fill" style={{ width: `${overviewValues[axis]}%`, background: lvl.color }} />
                            </div>
                          </div>
                        );
                      })}

                      <div className="overall-risk-box">
                        <div className="overall-risk-label">
                          <span>Overall Clinical Risk</span>
                          <span className={`pill ${TIER_CLASS[overviewPatient.risk_tier] || 'pill-normal'}`}>{overviewPatient.diagnosis}</span>
                        </div>
                        <div className="bar-track" style={{ height: 8 }}>
                          <div className="bar-fill" style={{ width: `${overviewPatient.risk_score}%`, background: TIER_BAR_COLOR[overviewPatient.risk_tier] }} />
                        </div>
                        <span className="overall-risk-score">{overviewPatient.risk_score}%</span>
                      </div>

                      <p className="radar-caption" style={{ textAlign: 'left', marginTop: 12 }}>
                        From cohort record #{String(overviewPatient.patient_id).padStart(3, '0')} — upload a CSV in the Dossier view to sync a specific patient.
                      </p>
                    </>
                  ) : (
                    <div className="empty-state">No clinical dossier loaded yet.<br />Upload a patient CSV from the Dossier view to populate systemic risk bars here.</div>
                  )}
                </div>

                <div className="panel">
                  <p className="panel-label">FOOT SCAN VIEWPORT</p>
                  {visionResult ? (
                    <>
                      <div className="scan-grid">
                        <div className="scan-cell">
                          <img src={imagePreview} alt="Original upload" />
                          <p className="scan-cell-label">ORIGINAL CAPTURE</p>
                        </div>
                        <div className="scan-cell">
                          <img src={visionResult.heatmap_base64} alt="Grad-CAM thermal overlay" />
                          <p className="scan-cell-label">
                            GRAD-CAM // <span className={visionResult.is_abnormal ? 'pill pill-high' : 'pill pill-normal'}>{visionResult.diagnosis}</span>
                          </p>
                        </div>
                      </div>

                      <div className="heat-legend">
                        <span className="heat-legend-label">LOW ATTENTION</span>
                        <div className="heat-legend-bar" />
                        <span className="heat-legend-label">HIGH ATTENTION</span>
                      </div>

                      <div className="analysis-recommend-row">
                        <div className="analysis-box">
                          <p className="recommend-box-title">AI Image Analysis</p>
                          <span className={visionResult.is_abnormal ? 'pill pill-high' : 'pill pill-normal'}>
                            {visionResult.is_abnormal ? 'Abnormality Detected' : 'No Abnormality Detected'}
                          </span>
                        </div>
                        <div className="recommend-box">
                          <p className="recommend-box-title">Clinical Recommendation</p>
                          <p>{getVisionRecommendation(visionResult)}</p>
                        </div>
                      </div>

                      <div className="hera-interp">
                        <div className="hera-interp-head"><span className="pulse-dot" />HERA CLINICAL INTERPRETATION</div>
                        {(visionResult.hera_analysis || 'Analyzing scan results…').split('\n').filter(Boolean).map((para, i) => (
                          <p key={i}>{para}</p>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="empty-state">Awaiting scan.<br />Upload a foot image above to generate the thermal Grad-CAM overlay and Hera's interpretation.</div>
                  )}
                </div>
              </div>

              {activityLog.length > 0 && (
                <div className="panel">
                  <p className="panel-label">RECENT ACTIVITY</p>
                  <ul className="activity-list">
                    {activityLog.map((a, i) => (
                      <li className="activity-item" key={i}>
                        <span className="activity-time">{a.time}</span>
                        <span className="activity-text">{a.text}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}

          {/* ===== TRIAGE BOARD ===== */}
          {activeView === 'triage' && (
            <>
              <div className="stage-header">
                <div>
                  <h1 className="stage-title">Triage Board</h1>
                  <div className="stage-sub">cohort sorted by risk tier</div>
                </div>
                <span className="status-chip"><span className="status-dot" />{(tabularResult || []).length} PATIENTS</span>
              </div>

              {tabularResult && tabularResult.length > 0 ? (
                <div className="triage-board">
                  {TIER_ORDER.map((tier) => (
                    <div className="triage-col" key={tier}>
                      <div className="triage-col-head">
                        <span style={{ color: TIER_BAR_COLOR[tier] }}>{TIER_LABEL[tier].toUpperCase()}</span>
                        <span className="triage-count">{triageGroups[tier].length}</span>
                      </div>
                      {triageGroups[tier].map((p, i) => (
                        <div className="triage-card" key={i} style={{ '--border-color': TIER_BAR_COLOR[tier] }}>
                          <div className="triage-card-id">#{String(p.patient_id).padStart(3, '0')}</div>
                          <div className="triage-card-score">{p.risk_score}% · <span className={`pill ${confidenceClass(p.confidence_tier)}`} style={{ fontSize: '0.62rem', padding: '1px 6px' }}>{p.confidence_tier}</span></div>
                          <div className="triage-card-factors">{p.shap_factors}</div>
                        </div>
                      ))}
                      {triageGroups[tier].length === 0 && <div className="empty-state" style={{ padding: '20px 4px' }}>None</div>}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="panel"><div className="empty-state">No cohort loaded.<br />Upload a patient CSV from the Dossier view to populate the triage board.</div></div>
              )}
            </>
          )}

          {/* ===== HERA COMMAND CENTER (full page) ===== */}
          {activeView === 'hera' && (
            <div className="hera-command">
              <div className="stage-header" style={{ marginBottom: 8 }}>
                <div>
                  <h1 className="stage-title">Hera AI Command Center</h1>
                  <div className="stage-sub">consult on protocols, extractions, and risk factors at scale</div>
                </div>
                <div className="header-actions">
                  <button className="btn btn-ghost" onClick={handleResetChat} title="Reset chat">
                    <Icon.Reset /> Reset
                  </button>
                  <span className="status-chip"><span className="status-dot" />ONLINE</span>
                </div>
              </div>
              {renderCopilotThread()}
              {renderCopilotForm()}
            </div>
          )}
        </main>

        {/* ---------------- RIGHT HERA COPILOT DOCK ---------------- */}
        {activeView !== 'hera' && (
          copilotOpen ? (
            <aside className="copilot">
              <div className="copilot-head">
                <DiaVisionLogo size={26} />
                <div className="copilot-head-text">
                  <span className="copilot-title">Hera Copilot</span>
                  <span className="copilot-status"><span className="pulse-dot" />watching workflow</span>
                </div>
                <div className="copilot-head-btns">
                  <button className="copilot-icon-btn" onClick={handleResetChat} title="Reset chat">
                    <Icon.Reset />
                  </button>
                  <button className="copilot-icon-btn" onClick={() => setCopilotOpen(false)} title="Hide Hera Copilot">
                    <Icon.ChevronRight />
                  </button>
                </div>
              </div>
              {renderCopilotThread()}
              {renderCopilotForm()}
            </aside>
          ) : (
            <aside className="copilot copilot-collapsed">
              <button className="copilot-icon-btn" onClick={() => setCopilotOpen(true)} title="Show Hera Copilot">
                <Icon.ChevronLeft />
              </button>
              <DiaVisionLogo size={26} />
              <span className="copilot-collapsed-label">HERA</span>
              <span className="pulse-dot" />
            </aside>
          )
        )}
      </div>
    </div>
  );
}

export default App;