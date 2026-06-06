import streamlit.components.v1 as components

import mmcv
import numpy as np
import streamlit as st
from PIL import Image

from vision_rag_claims.vision.detector import DamageDetector
from vision_rag_claims.vision.visualizer import draw_detections

st.set_page_config(
    page_title="ClaimVision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS (targets native Streamlit components) ──────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background-color: #0B1120 !important;
    background-image: radial-gradient(circle at top right, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                      radial-gradient(circle at bottom left, rgba(139, 92, 246, 0.08) 0%, transparent 40%) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1240px;
}

.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
    transform: translateY(-2px) !important;
}

[data-testid="stMetric"] {
    background: rgba(30,41,59,0.5) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    padding: 20px 24px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stMetricLabel"] p {
    color: #94A3B8 !important; font-size: 12px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.1em !important; margin-bottom: 8px !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important; font-size: 28px !important; font-weight: 700 !important;
    background: linear-gradient(to right, #FFFFFF, #CBD5E1) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

[data-testid="stFileUploader"] section {
    background: rgba(30,41,59,0.3) !important;
    border: 2px dashed rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"] section:hover { 
    border-color: #3B82F6 !important; 
    background: rgba(59,130,246,0.05) !important;
}

[data-testid="stExpander"] {
    background: rgba(30,41,59,0.4) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    backdrop-filter: blur(8px) !important;
}
[data-testid="stExpander"] summary {
    background: transparent !important; color: #E2E8F0 !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 16px 20px !important;
}
[data-testid="stExpander"] summary:hover { color: #FFFFFF !important; background: rgba(255,255,255,0.02) !important; }

[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
}

hr { border-color: rgba(255,255,255,0.05) !important; margin: 2rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Shared iframe CSS (injected into every components.html call) ───────────────

_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Inter', system-ui, sans-serif; background: transparent; }
</style>
"""


def _iframe(body: str, height: int) -> None:
    components.html(f"<!DOCTYPE html><html><head>{_BASE_CSS}</head><body>{body}</body></html>",
                    height=height, scrolling=False)


# ── Severity config ───────────────────────────────────────────────────────────

_SEV = {
    "minor":    ("#FFFFFF", "linear-gradient(135deg, #10B981, #059669)"),
    "moderate": ("#FFFFFF", "linear-gradient(135deg, #F59E0B, #D97706)"),
    "severe":   ("#FFFFFF", "linear-gradient(135deg, #EF4444, #B91C1C)"),
}

_PIPELINE_STEPS = [
    ("detect_damage",   "Damage Detection"),
    ("assess_severity", "Severity Assessment"),
    ("retrieve_policy", "Policy Retrieval"),
    ("check_coverage",  "Coverage Analysis"),
    ("generate_report", "Report Generation"),
]

# ── HTML component builders ───────────────────────────────────────────────────

def _render_header() -> None:
    _iframe("""
    <div style="background: linear-gradient(145deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.7) 100%);
                border: 1px solid rgba(255,255,255,0.05);
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                border-radius: 16px;
                padding: 24px 32px; display: flex; align-items: center; justify-content: space-between;
                backdrop-filter: blur(12px);">
        <div style="display:flex;align-items:center;gap:20px;">
            <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
                        display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 12px rgba(59,130,246,0.3);">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
            </div>
            <div>
                <div style="color:#FFFFFF;font-size:24px;font-weight:700;letter-spacing:-0.03em;
                            background: linear-gradient(to right, #FFFFFF, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ClaimVision</div>
                <div style="color:#94A3B8;font-size:13px;margin-top:4px;font-weight:500;">
                    Intelligent Damage Assessment</div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="color:#64748B;font-size:10px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.1em; margin-bottom: 8px;">Enterprise Stack</div>
            <div style="display:flex; gap: 8px;">
                <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:4px 10px; border-radius:20px; color:#E2E8F0; font-size:11px; font-weight:600;">Mask R-CNN</span>
                <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:4px 10px; border-radius:20px; color:#E2E8F0; font-size:11px; font-weight:600;">ChromaDB</span>
                <span style="background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); padding:4px 10px; border-radius:20px; color:#E2E8F0; font-size:11px; font-weight:600;">GPT-4o</span>
            </div>
        </div>
    </div>
    """, height=100)


def _render_detection_table(detections: list) -> None:
    rows = []
    for i, det in enumerate(detections, start=1):
        sep = "border-top:1px solid rgba(255,255,255,0.05);" if i > 1 else ""
        conf_color = "#10B981" if det.confidence >= 0.85 else "#F59E0B" if det.confidence >= 0.70 else "#EF4444"
        x1, y1, x2, y2 = det.bbox
        rows.append(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    padding:16px 24px;{sep} transition: background 0.2s ease;">
            <div style="display:flex;align-items:center;gap:16px;">
                <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:8px;
                            width:32px;height:32px;display:flex;align-items:center;
                            justify-content:center;font-weight:600;color:#94A3B8;
                            font-size:13px;flex-shrink:0; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);">
                    {i}
                </div>
                <div>
                    <div style="font-weight:600;color:#F8FAFC;text-transform:capitalize;
                                font-size:15px; letter-spacing:-0.01em;">{det.damage_type}</div>
                    <div style="color:#64748B;font-size:12px;margin-top:4px;
                                font-family:'JetBrains Mono', 'Courier New', monospace;">
                        <span style="opacity:0.7">bbox</span> [{x1}, {y1}] &rarr; [{x2}, {y2}]
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:32px;flex-shrink:0;">
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px;">Confidence</div>
                    <div style="font-weight:700;color:{conf_color};font-size:18px;">
                        {det.confidence:.0%}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px;">Mask Px</div>
                    <div style="font-weight:600;color:#E2E8F0;font-size:18px;">
                        {det.mask_area_pixels:,}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:4px;">Area</div>
                    <div style="font-weight:600;color:#E2E8F0;font-size:18px;">
                        {det.bbox_area_pixels:,}</div>
                </div>
            </div>
        </div>""")

    _iframe(
        '<div style="background:linear-gradient(180deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.5) 100%);'
        'border-radius:16px;border:1px solid rgba(255,255,255,0.05); box-shadow: 0 4px 24px rgba(0,0,0,0.2);'
        'backdrop-filter:blur(10px); overflow:hidden;">'
        + "".join(rows) + "</div>",
        height=len(detections) * 80 + 20,
    )


def _pipeline_html(completed: set[str], active: str | None) -> str:
    """Uses block-level layout (no flex) so st.markdown renders it reliably."""
    rows = []
    for i, (key, label) in enumerate(_PIPELINE_STEPS, start=1):
        if key in completed:
            border_color, num_fg, num_bg, text_style = "#10B981", "#FFFFFF", "linear-gradient(135deg, #10B981, #059669)", "color:#94A3B8;"
        elif key == active:
            border_color, num_fg, num_bg, text_style = "#3B82F6", "#FFFFFF", "linear-gradient(135deg, #3B82F6, #2563EB)", "color:#F8FAFC;font-weight:600; text-shadow: 0 0 10px rgba(59,130,246,0.5);"
        else:
            border_color, num_fg, num_bg, text_style = "rgba(255,255,255,0.05)", "#64748B", "rgba(255,255,255,0.05)", "color:#475569;"

        rows.append(
            f'<div style="border-left:2px solid {border_color};padding:8px 0 8px 16px;'
            f'margin-left:8px; transition: all 0.3s ease;">'
            f'<span style="background:{num_bg};color:{num_fg};border-radius:50%;'
            f'display:inline-block;width:24px;height:24px;text-align:center;line-height:24px;'
            f'font-size:12px;font-weight:700;vertical-align:middle;margin-right:12px;'
            f'box-shadow: 0 2px 8px rgba(0,0,0,0.2);">{"✓" if key in completed else i}</span>'
            f'<span style="font-size:14px;vertical-align:middle;{text_style} transition: color 0.3s ease;">{label}</span>'
            f'</div>'
        )
    header = ('<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
              'letter-spacing:0.1em;margin-bottom:16px;">Processing Pipeline</div>')
    return (
        '<div style="background:rgba(30,41,59,0.4);border-radius:16px;padding:24px;'
        f'border:1px solid rgba(255,255,255,0.05);margin-bottom:16px; backdrop-filter: blur(10px);'
        f'box-shadow: 0 4px 20px rgba(0,0,0,0.1);">{header}{"".join(rows)}</div>'
    )


def _label(text: str, fg: str, bg: str) -> str:
    return (f'<span style="background:{bg};color:{fg};padding:2px 9px;'
            f'border-radius:4px;font-size:11px;font-weight:700;letter-spacing:0.05em;">'
            f'{text}</span>')


def _coverage_html(decision) -> str:
    if decision.is_covered:
        accent, tag = "#10B981", _label("COVERED", "#FFFFFF", "linear-gradient(135deg, #10B981, #059669)")
    else:
        accent, tag = "#EF4444", _label("NOT COVERED", "#FFFFFF", "linear-gradient(135deg, #EF4444, #B91C1C)")
    ded_color = "#94A3B8" if decision.deductible.upper() in ("N/A", "NA", "NOT APPLICABLE") else "#FFFFFF"

    citations = ""
    if decision.citations:
        items = "".join(
            f'<div style="padding:12px 16px;background:rgba(15,23,42,0.6);border-radius:8px;'
            f'font-size:14px;color:#CBD5E1;line-height:1.6;margin-top:8px; border:1px solid rgba(255,255,255,0.02);">{c}</div>'
            for c in decision.citations
        )
        citations = (
            '<div style="margin-top:16px;">'
            '<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
            'letter-spacing:0.1em;margin-bottom:6px;">Policy Citations</div>'
            f'{items}</div>'
        )

    return (
        f'<div style="background:linear-gradient(145deg, rgba(30,41,59,0.4) 0%, rgba(15,23,42,0.4) 100%);'
        f'border-radius:16px;padding:24px 28px;'
        f'border:1px solid rgba(255,255,255,0.05);border-left:4px solid {accent};margin-bottom:16px;'
        f'box-shadow: 0 4px 20px rgba(0,0,0,0.1); backdrop-filter: blur(10px);">'
        f'<div style="overflow:hidden;margin-bottom:16px;">'
        f'<div style="float:right;text-align:right;">'
        f'<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
        f'letter-spacing:0.1em;">Deductible</div>'
        f'<div style="font-weight:700;color:{ded_color};font-size:22px;margin-top:2px;">'
        f'{decision.deductible}</div></div>'
        f'<div style="overflow:hidden; display: flex; align-items: center; gap: 12px;">'
        f'<span style="font-weight:700;color:#F8FAFC;font-size:20px;text-transform:capitalize;">'
        f'{decision.damage_type}</span>{tag}</div></div>'
        f'<p style="color:#E2E8F0;font-size:15px;line-height:1.7;margin:0; font-weight:400;">'
        f'{decision.reasoning}</p>'
        f'{citations}</div>'
    )


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def load_detector() -> DamageDetector:
    return DamageDetector()


@st.cache_resource
def load_graph():
    from vision_rag_claims.agent.graph import build_graph
    return build_graph()


# ── App ───────────────────────────────────────────────────────────────────────

_render_header()

with st.spinner("Loading model..."):
    detector = load_detector()

uploaded_file = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown(
        '<div style="background:#1E293B;border-radius:10px;padding:32px;'
        'border:1px solid #334155;text-align:center;margin-top:8px;">'
        '<div style="color:#334155;font-size:14px;font-weight:500;">'
        'Upload a vehicle photo to begin assessment</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ── Detection ─────────────────────────────────────────────────────────────────

pil_image = Image.open(uploaded_file).convert("RGB")
image_rgb = np.array(pil_image)
image_bgr = mmcv.image.rgb2bgr(image_rgb)

with st.spinner("Running damage detection..."):
    result = detector.detect(image_bgr)

palette = detector.model.dataset_meta["palette"]
class_names = list(detector.classes)

annotated_bgr = draw_detections(image=image_bgr, result=result, palette=palette, class_names=class_names)
annotated_rgb = mmcv.image.bgr2rgb(annotated_bgr)

col1, col2 = st.columns(2, gap="medium")
with col1:
    st.markdown('<div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:6px;">Original</div>', unsafe_allow_html=True)
    st.image(image_rgb, use_container_width=True)
with col2:
    st.markdown('<div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:6px;">Annotated</div>', unsafe_allow_html=True)
    st.image(annotated_rgb, use_container_width=True)

n = len(result.detections)
st.markdown(
    f'<div style="font-size:16px;font-weight:700;color:#E2E8F0;margin:20px 0 10px;">'
    f'{n} Damage{"s" if n != 1 else ""} Detected</div>',
    unsafe_allow_html=True,
)

if not result.detections:
    st.warning("No damages detected above the confidence threshold.")
    st.stop()

_render_detection_table(result.detections)

# ── Analysis trigger ──────────────────────────────────────────────────────────

st.divider()

left, right = st.columns([2, 5], gap="medium")
with left:
    run_analysis = st.button("Run Claim Analysis", type="primary", use_container_width=True)
with right:
    st.markdown(
        '<div style="padding:9px 0;color:#475569;font-size:13px;">'
        'Retrieves matching policy clauses via semantic search and evaluates '
        'coverage for each detected damage.</div>',
        unsafe_allow_html=True,
    )

if not run_analysis:
    st.stop()

# ── Agent pipeline ────────────────────────────────────────────────────────────

graph = load_graph()
progress_placeholder = st.empty()
completed_nodes: set[str] = set()
final_state: dict = {}
_node_order = [k for k, _ in _PIPELINE_STEPS]

with st.spinner(""):
    for event in graph.stream({"image": image_bgr, "detection_result": result}, stream_mode="updates"):
        print(f"DEBUG event: {event}")
        for node_name, state_update in event.items():
            completed_nodes.add(node_name)
            idx = _node_order.index(node_name) if node_name in _node_order else -1
            next_node = _node_order[idx + 1] if 0 <= idx < len(_node_order) - 1 else None
            progress_placeholder.markdown(
                _pipeline_html(completed_nodes, next_node),
                unsafe_allow_html=True,
            )
            if state_update:
                final_state.update(state_update)

progress_placeholder.empty()

report = final_state.get("report")
if report is None:
    st.error("The agent did not produce a report. Check your API key and retry.")
    st.stop()

# ── Summary ───────────────────────────────────────────────────────────────────

severity = final_state.get("severity")
sev_val = severity.value if severity else "minor"
sev_fg, sev_bg = _SEV.get(sev_val, ("#94A3B8", "#1E293B"))
sev_badge = _label(sev_val.upper(), sev_fg, sev_bg)

# Float-based layout — no flex
st.markdown(
    f'<div style="background:linear-gradient(145deg, rgba(30,41,59,0.5) 0%, rgba(15,23,42,0.5) 100%);'
    f'border-radius:16px;padding:24px 28px;'
    f'border:1px solid rgba(255,255,255,0.05);margin-bottom:16px;overflow:hidden;'
    f'box-shadow: 0 4px 20px rgba(0,0,0,0.1); backdrop-filter: blur(10px);">'
    f'<div style="float:right;text-align:right;padding-left:28px;border-left:1px solid rgba(255,255,255,0.1);margin-left:28px;">'
    f'<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
    f'letter-spacing:0.1em;margin-bottom:6px;">Estimated Total</div>'
    f'<div style="font-size:32px;font-weight:700;color:#FFFFFF;letter-spacing:-0.02em;">'
    f'{report.estimated_total}</div></div>'
    f'<div style="overflow:hidden;">'
    f'<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
    f'letter-spacing:0.1em;margin-bottom:12px;">Claim Summary</div>'
    f'<div style="margin-bottom:16px;">'
    f'<span style="font-size:22px;font-weight:700;color:#F8FAFC;margin-right:12px;">'
    f'{len(report.damages)} damage(s) assessed</span>{sev_badge}</div>'
    f'<p style="color:#E2E8F0;font-size:16px;line-height:1.7;margin:0; font-weight:400;">{report.summary}</p>'
    f'</div></div>',
    unsafe_allow_html=True,
)

col_action, col_steps = st.columns([2, 3], gap="medium")
with col_action:
    st.markdown(
        f'<div style="background:rgba(30,41,59,0.4);border-radius:16px;padding:24px;'
        f'border:1px solid rgba(255,255,255,0.05);margin-bottom:16px; height: 100%;'
        f'box-shadow: 0 4px 20px rgba(0,0,0,0.1); backdrop-filter: blur(10px);">'
        f'<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:12px;">Recommended Action</div>'
        f'<p style="color:#FFFFFF;font-size:17px;line-height:1.6;margin:0; font-weight:500;">'
        f'{report.recommended_action}</p></div>',
        unsafe_allow_html=True,
    )
with col_steps:
    items = "".join(
        f'<div style="padding:6px 0; display: flex; align-items: flex-start; gap: 12px;">'
        f'<span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);border-radius:6px;'
        f'color:#60A5FA;font-size:12px;font-weight:700;padding:2px 8px; flex-shrink: 0; margin-top:2px;">{j}</span>'
        f'<span style="color:#CBD5E1;font-size:15px;line-height:1.6;">{s}</span>'
        f'</div>'
        for j, s in enumerate(report.next_steps, start=1)
    )
    st.markdown(
        f'<div style="background:rgba(30,41,59,0.4);border-radius:16px;padding:24px;'
        f'border:1px solid rgba(255,255,255,0.05);margin-bottom:16px; height: 100%;'
        f'box-shadow: 0 4px 20px rgba(0,0,0,0.1); backdrop-filter: blur(10px);">'
        f'<div style="font-size:11px;font-weight:700;color:#64748B;text-transform:uppercase;'
        f'letter-spacing:0.1em;margin-bottom:12px;">Next Steps</div>'
        f'{items}</div>',
        unsafe_allow_html=True,
    )

# ── Coverage breakdown ────────────────────────────────────────────────────────

st.markdown(
    '<div style="font-size:16px;font-weight:700;color:#E2E8F0;margin:20px 0 12px;">'
    'Coverage Breakdown</div>',
    unsafe_allow_html=True,
)

retrieved_chunks = final_state.get("retrieved_chunks", {})

for decision in report.damages:
    st.markdown(_coverage_html(decision), unsafe_allow_html=True)
    chunks = retrieved_chunks.get(decision.damage_type.lower(), [])
    if chunks:
        with st.expander("Policy excerpts"):
            for chunk in chunks:
                st.markdown(
                    f'<div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;'
                    f'letter-spacing:0.06em;margin-bottom:4px;">'
                    f'{chunk.source_document} &nbsp;&middot;&nbsp; {chunk.section}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(chunk.content)
                st.divider()
