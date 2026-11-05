"""
CYBERSHIELD AI - INTELLIGENT NETWORK THREAT DETECTION SYSTEM
Industrial Cyberpunk Minimalism dashboard (Streamlit).

Offline, educational classification of CIC-IDS2017 network-flow CSV files
using the saved Random Forest model:  model/cybershield_model.pkl
(no live packet capture, no network scanning, no attack execution).

Run from the project folder (Windows CMD):
    python -m streamlit run app.py
"""

import html
import io
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# PDF report support (install with: python -m pip install reportlab)
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                    PageBreak, Image as PDFImage, KeepTogether)
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CYBERSHIELD AI",
    page_icon=">",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "cybershield_model.pkl"
DATASET_PATH = BASE_DIR / "dataset" / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

MODEL_ACCURACY = "99.97%"  # saved evaluation result from train_model.py (not live)
GREEN = "#00FF00"
RED = "#FF3333"

KEY_FEATURES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Packet Length Mean",
    "Packet Length Std", "SYN Flag Count", "ACK Flag Count", "Average Packet Size",
    "Active Mean", "Idle Mean",
]

FEATURE_INFO = {
    "Destination Port": "The port number the traffic is sent to (for example 80 = web, 443 = secure web).",
    "Flow Duration": "How long the network conversation (flow) lasted, in microseconds.",
    "Total Fwd Packets": "Number of packets sent from the source to the destination.",
    "Total Backward Packets": "Number of packets sent back from the destination to the source.",
    "Total Length of Fwd Packets": "Total number of bytes in all forward packets.",
    "Total Length of Bwd Packets": "Total number of bytes in all backward packets.",
    "Flow Bytes/s": "Data transfer speed of the flow in bytes per second.",
    "Flow Packets/s": "How many packets per second the flow carries.",
    "Flow IAT Mean": "Average waiting time between two packets in the flow.",
    "Flow IAT Std": "How much the waiting time between packets varies (steady vs bursty).",
    "Flow IAT Max": "Longest gap between two packets in the flow.",
    "Flow IAT Min": "Shortest gap between two packets in the flow.",
    "Packet Length Mean": "Average size of the packets in the flow.",
    "Packet Length Std": "How much packet sizes vary within the flow.",
    "Packet Length Variance": "Variance (squared spread) of packet sizes in the flow.",
    "Max Packet Length": "Size of the largest packet in the flow.",
    "Min Packet Length": "Size of the smallest packet in the flow.",
    "Average Packet Size": "Average size of a packet, including headers.",
    "SYN Flag Count": "How many packets carry the SYN flag (used to start a TCP connection).",
    "ACK Flag Count": "How many packets carry the ACK flag (acknowledging received data).",
    "FIN Flag Count": "How many packets carry the FIN flag (politely closing a connection).",
    "RST Flag Count": "How many packets carry the RST flag (abruptly resetting a connection).",
    "PSH Flag Count": "How many packets carry the PSH flag (push data to the application now).",
    "URG Flag Count": "How many packets carry the URG flag (urgent data).",
    "CWE Flag Count": "How many packets carry the CWE flag (TCP congestion control).",
    "ECE Flag Count": "How many packets carry the ECE flag (TCP congestion notification).",
    "Down/Up Ratio": "Ratio of download (backward) traffic to upload (forward) traffic.",
    "Active Mean": "Average time the flow was active before going idle.",
    "Active Std": "How much the active time varies.",
    "Active Max": "Longest active period of the flow.",
    "Active Min": "Shortest active period of the flow.",
    "Idle Mean": "Average time the flow stayed idle between active periods.",
    "Idle Std": "How much the idle time varies.",
    "Idle Max": "Longest idle period of the flow.",
    "Idle Min": "Shortest idle period of the flow.",
    "Fwd Packets/s": "Forward packets sent per second.",
    "Bwd Packets/s": "Backward packets sent per second.",
    "Init_Win_bytes_forward": "TCP window size (receive buffer) announced in the forward direction at the start.",
    "Init_Win_bytes_backward": "TCP window size (receive buffer) announced in the backward direction at the start.",
    "act_data_pkt_fwd": "Forward packets that actually carry data (not just headers).",
    "min_seg_size_forward": "Smallest TCP segment size seen in the forward direction.",
    "Subflow Fwd Packets": "Average forward packets per sub-flow.",
    "Subflow Fwd Bytes": "Average forward bytes per sub-flow.",
    "Subflow Bwd Packets": "Average backward packets per sub-flow.",
    "Subflow Bwd Bytes": "Average backward bytes per sub-flow.",
}

# ----------------------------------------------------------------------------
# CSS  (Industrial Cyberpunk Minimalism)
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Space+Grotesk:wght@500;700;800&display=swap');

html, body { background:#000 !important; }
.stApp {
    background-color:#000 !important;
    background-image:linear-gradient(#111111 1px, transparent 1px),
                     linear-gradient(90deg, #111111 1px, transparent 1px);
    background-size:40px 40px;
    color:#FFFFFF;
    font-family:'JetBrains Mono', monospace;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background:transparent !important; overflow-x:hidden; }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stSidebar"], [data-testid="collapsedControl"], footer, #MainMenu { display:none !important; }
[data-testid="stMainBlockContainer"], .block-container {
    max-width:100% !important; padding:96px 3rem 0 3rem !important;
}
* { border-radius:0 !important; }
::selection { background:#00FF00; color:#000; }

/* ---------- scanline ---------- */
.scan { position:fixed; left:0; width:100%; height:8px; background:rgba(255,255,255,0.05);
        pointer-events:none; z-index:9999; animation:scan 10s linear infinite; }
@keyframes scan { from { top:-10px; } to { top:100vh; } }
@keyframes pulse { 50% { opacity:0.25; } }
@keyframes reveal { from { opacity:0; transform:translateY(20px); } to { opacity:1; transform:translateY(0); } }
.reveal { animation:reveal 0.9s cubic-bezier(0.16, 1, 0.3, 1) both; }

/* ---------- navigation ---------- */
.nav { position:fixed; top:0; left:0; right:0; height:80px; background:#000; border-bottom:1px solid #FFF;
       display:flex; align-items:center; justify-content:space-between; padding:0 2rem; z-index:1000; }
.logo { font-weight:800; font-size:1.2rem; letter-spacing:0.02em; }
.logo b { color:#00FF00; }
.nav-links { display:flex; gap:1.5rem; }
.nav-links a { color:#FFF; text-decoration:none; font-size:12px; font-weight:500; padding:4px 6px;
               transition:all 0.1s steps(2); }
.nav-links a:hover { background:#FFF; color:#000; box-shadow:4px 4px 0 0 #00FF00; }
.btn-b { display:inline-block; background:#FFF; color:#000 !important; font-weight:700; text-transform:uppercase;
         padding:0.75rem 1.5rem; text-decoration:none; font-size:13px; transition:all 0.1s steps(2); }
.btn-b:hover { background:#00FF00; box-shadow:6px 6px 0 0 #FFF; transform:translate(-3px,-3px); }
@media (max-width:1150px) { .nav-links { display:none; } }
@media (max-width:600px)  { .nav .btn-b { display:none; } }

/* ---------- hero ---------- */
.hero { min-height:calc(100vh - 100px); display:grid; grid-template-columns:1.7fr 1fr; gap:2rem;
        align-items:center; padding:2rem 0 3rem 0; }
@media (max-width:1000px) { .hero { grid-template-columns:1fr; } }
.status { color:#00FF00; font-weight:700; font-size:13px; margin-bottom:1.5rem; }
.status i { font-style:normal; animation:pulse 1s steps(2) infinite; }
.hero-title { font-family:'Space Grotesk', sans-serif; font-weight:800; font-size:clamp(2.8rem, 8.5vw, 8rem);
              line-height:0.85; letter-spacing:-0.05em; text-transform:uppercase; color:#FFF; }
.stroke { color:transparent; -webkit-text-stroke:1.5px #FFF; }
.hero-sub { color:#a1a1aa; font-size:clamp(1rem, 2vw, 1.5rem); margin-top:1.5rem; max-width:720px; }
.hero-tag { color:#00FF00; font-weight:800; font-size:1.1rem; margin-top:1.2rem; letter-spacing:0.08em; }

/* ---------- terminal ---------- */
.term { border:1px solid #333; padding:1rem; background:transparent; color:#666; font-size:10px;
        line-height:1.55; font-family:'JetBrains Mono', monospace; overflow-x:auto; }
.term.big { font-size:13px; color:#b5b5b5; padding:1.5rem; }
.term .g { color:#00FF00; }

/* ---------- data ribbon ---------- */
.fb { width:100vw; margin-left:calc(50% - 50vw); }
.ribbon { display:grid; grid-template-columns:repeat(4, 1fr); border-top:1px solid #FFF; border-bottom:1px solid #FFF; background:#000; }
.rcell { padding:2rem; border-right:1px solid #FFF; transition:all 0.1s steps(2); }
.rcell:last-child { border-right:none; }
.rcell:hover { background:#00FF00; color:#000; }
.rlabel { font-size:12px; letter-spacing:0.1em; font-weight:500; }
.rval { font-size:2.5rem; font-weight:800; margin:0.4rem 0; font-family:'JetBrains Mono', monospace; }
.rnote { font-size:10px; opacity:0.65; }
@media (max-width:900px) { .ribbon { grid-template-columns:repeat(2, 1fr); } .rcell:nth-child(2) { border-right:none; } .rcell { border-bottom:1px solid #FFF; } }
@media (max-width:520px) { .ribbon { grid-template-columns:1fr; } .rcell { border-right:none; } }

/* ---------- sections ---------- */
.sec { border-top:1px solid #FFF; margin-top:4.5rem; padding-top:1.2rem; }
.sec-title { font-family:'Space Grotesk', sans-serif; font-weight:800; font-size:clamp(1.6rem, 3.6vw, 3rem);
             letter-spacing:-0.05em; text-transform:uppercase; line-height:1; margin-bottom:1.2rem; }
.sec-title span { color:#00FF00; }
.note { color:#a1a1aa; font-size:13px; line-height:1.6; margin:0.6rem 0 1.2rem 0; }

.igrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); border-top:1px solid #FFF; border-left:1px solid #FFF; margin-bottom:1rem; }
.icell { padding:1rem 1.2rem; border-right:1px solid #FFF; border-bottom:1px solid #FFF; transition:all 0.1s steps(2); word-break:break-word; }
.icell:hover { background:#FFF; color:#000; box-shadow:6px 6px 0 0 #00FF00; }
.icell .l { font-size:11px; letter-spacing:0.08em; opacity:0.7; }
.icell .v { font-size:1.15rem; font-weight:800; margin-top:0.3rem; }
.icell.good .v { color:#00FF00; } .icell.good { border-color:#00FF00; }
.icell.bad .v { color:#FF3333; }  .icell.bad  { border-color:#FF3333; }
.icell.good:hover .v, .icell.bad:hover .v { color:#000; }

.msg { border:1px solid #FFF; padding:0.9rem 1.1rem; margin:0.5rem 0; font-size:13px; line-height:1.55; }
.msg.good { border-color:#00FF00; color:#00FF00; }
.msg.bad  { border-color:#FF3333; color:#FF3333; }
.msg.warn { border-color:#FFF; color:#FFF; }

/* ---------- tables ---------- */
.twrap { overflow:auto; max-height:540px; border:1px solid #FFF; margin-bottom:1rem; }
table.t { border-collapse:collapse; width:100%; font-size:12px; font-family:'JetBrains Mono', monospace; }
table.t th { position:sticky; top:0; background:#000; color:#00FF00; text-align:left; padding:0.55rem 0.8rem;
             border-bottom:1px solid #FFF; white-space:nowrap; z-index:2; }
table.t td { padding:0.45rem 0.8rem; border-bottom:1px solid #222; white-space:nowrap; vertical-align:top; }
table.t td.wrap { white-space:normal; min-width:260px; }
table.t tr:hover td { background:#FFF; color:#000; }
.pill-ok  { background:#00FF00; color:#000; padding:1px 8px; font-weight:700; }
.pill-bad { background:#FF3333; color:#000; padding:1px 8px; font-weight:700; }

/* ---------- cards / flow / cta ---------- */
.sgrid { display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); border-top:1px solid #FFF; border-left:1px solid #FFF; margin:1.2rem 0; }
.scard { position:relative; padding:2.5rem 2rem 4rem 2rem; border-right:1px solid #FFF; border-bottom:1px solid #FFF; transition:all 0.1s steps(2); }
.scard:hover { background:#FFF; color:#000; }
.scard svg { stroke:currentColor; fill:none; stroke-width:1.5; width:34px; height:34px; margin-bottom:1rem; }
.scard h3 { font-family:'Space Grotesk', sans-serif; font-weight:800; text-transform:uppercase; font-size:1.3rem; margin:0 0 0.8rem 0; padding:0; letter-spacing:-0.03em; }
.scard p { font-size:13px; line-height:1.65; margin:0; }
.stag { position:absolute; left:2rem; bottom:1.4rem; border:1px solid currentColor; padding:2px 8px; font-size:10px; }
.flow { display:flex; flex-wrap:wrap; align-items:stretch; margin:1rem 0; }
.fbox { border:1px solid #FFF; padding:1.1rem 0.8rem; flex:1; min-width:140px; text-align:center; font-weight:700; font-size:12px; transition:all 0.1s steps(2); }
.fbox:hover { background:#00FF00; color:#000; }
.farrow { display:flex; align-items:center; padding:0 0.5rem; color:#00FF00; font-weight:800; }
.cta { background:#00FF00; color:#000; padding:5rem 3rem; display:grid; grid-template-columns:1.6fr 1fr; gap:2rem; align-items:center; margin-top:5rem; }
@media (max-width:900px) { .cta { grid-template-columns:1fr; } }
.cta-title { font-family:'Space Grotesk', sans-serif; font-weight:800; font-size:clamp(2.4rem, 7vw, 6.5rem); line-height:0.9; letter-spacing:-0.05em; text-transform:uppercase; }
.cta-sub { font-weight:700; margin-top:1.4rem; font-size:14px; }
.cta-btn { display:inline-block; background:#000; color:#FFF !important; font-weight:800; font-size:1.5rem; padding:1.5rem 2rem; text-decoration:none; transition:all 0.1s steps(2); }
.cta-btn:hover { background:#FFF; color:#000 !important; box-shadow:6px 6px 0 0 #000; transform:translate(-3px,-3px); }
.foot { text-align:center; color:#666; font-size:11px; padding:1.5rem 0 2rem 0; }

/* ---------- Streamlit widgets ---------- */
.stApp label, .stApp label p, [data-testid="stWidgetLabel"] p { color:#FFF !important; font-family:'JetBrains Mono', monospace !important; font-size:12px !important; letter-spacing:0.06em; text-transform:uppercase; }
.stApp p, .stApp span, .stApp li { font-family:'JetBrains Mono', monospace; }
.stButton > button, .stDownloadButton > button {
    background:#FFF !important; color:#000 !important; border:1px solid #FFF !important; padding:0.75rem 1.5rem !important;
    font-family:'JetBrains Mono', monospace !important; font-weight:700 !important; text-transform:uppercase;
    transition:all 0.1s steps(2); min-height:3rem; }
.stButton > button p, .stDownloadButton > button p { color:#000 !important; font-weight:700 !important; font-family:'JetBrains Mono', monospace !important; }
.stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled) {
    background:#00FF00 !important; box-shadow:6px 6px 0 0 #FFF; transform:translate(-3px,-3px); border-color:#FFF !important; }
.stButton > button:disabled { background:#222 !important; border-color:#444 !important; opacity:0.7; }
.stButton > button:disabled p { color:#666 !important; }
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div, [data-baseweb="textarea"] {
    background:#000 !important; border:1px solid #FFF !important; }
[data-baseweb="input"] input, [data-baseweb="select"] input, [data-baseweb="select"] div { color:#FFF !important; font-family:'JetBrains Mono', monospace !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div { border:2px solid #00FF00 !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background:#000 !important; border:1px solid #FFF; }
[data-baseweb="popover"] li:hover { background:#00FF00 !important; color:#000 !important; }
[data-baseweb="tag"] { background:#00FF00 !important; color:#000 !important; }
[data-baseweb="tag"] span { color:#000 !important; }
[data-testid="stFileUploaderDropzone"] { background:#000 !important; border:1px dashed #FFF !important; }
[data-testid="stFileUploaderDropzone"] * { color:#FFF !important; font-family:'JetBrains Mono', monospace !important; }
[data-testid="stFileUploaderDropzone"] button { background:#FFF !important; color:#000 !important; border:none !important; }
[data-testid="stFileUploaderDropzone"] button * { color:#000 !important; }
[data-testid="stFileUploaderDropzone"]:hover { border:2px solid #00FF00 !important; }
[data-testid="stFileUploaderFile"] * { color:#FFF !important; }
[data-testid="stSpinner"] * { color:#00FF00 !important; }
.stApp img { border:1px solid #333; }
</style>
"""


# ----------------------------------------------------------------------------
# GENERIC HELPERS
# ----------------------------------------------------------------------------
def h(text: str) -> str:
    """Strip indentation / blank lines so Streamlit markdown never turns HTML into a code block."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def esc(value) -> str:
    return html.escape(str(value))


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.2f} {unit}"
        size /= 1024
    return f"{n} B"


def fmt_value(v) -> str:
    if v is None:
        return "NaN"
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:,.4f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    return str(v)


def normalize_label(value) -> str:
    s = str(value).strip()
    if s.upper() == "BENIGN" or s == "0":
        return "BENIGN"
    if s.upper() == "DDOS" or s == "1":
        return "DDoS"
    return s


def section_open(anchor: str, number: str, title: str) -> str:
    return h(f"""
    <div id="{anchor}" class="sec reveal">
    <div class="sec-title"><span>{number} /</span> {title}</div>
    </div>
    """)


def msg_box(text: str, kind: str = "warn") -> str:
    return f'<div class="msg {kind}">{text}</div>'


def info_grid(items) -> str:
    """items: list of (label, value, css_class)"""
    cells = "".join(
        f'<div class="icell {cls}"><div class="l">{esc(label)}</div><div class="v">{esc(value)}</div></div>'
        for label, value, cls in items
    )
    return f'<div class="igrid">{cells}</div>'


def terminal_html(lines, big=False) -> str:
    rendered = []
    for line in lines:
        safe = esc(line)
        stripped = safe.lstrip(" ")
        safe = "&nbsp;" * (len(safe) - len(stripped)) + stripped
        if safe.startswith("&gt;"):
            safe = '<span class="g">&gt;</span>' + safe[4:]
        rendered.append(safe if safe else "&nbsp;")
    cls = "term big" if big else "term"
    return f'<div class="{cls}">' + "<br>".join(rendered) + "</div>"


ASCII_LOGO = [
    r"  ___ ___ ",
    r" / __/ __|",
    r"| (__\__ \ ",
    r" \___|___/ ",
    r"  _  _ ___ ",
    r" | || |_ _|",
    r" |_||_|___|",
    r"CYBERSHIELD_AI",
]


def table_html(df: pd.DataFrame, wrap_cols=()) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for row in df.itertuples(index=False):
        tds = []
        for col, val in zip(df.columns, row):
            if col == "Threat Status":
                cls = "pill-ok" if val == "NORMAL" else "pill-bad"
                tds.append(f'<td><span class="{cls}">{esc(val)}</span></td>')
            elif col == "AI Prediction":
                color = GREEN if val == "BENIGN" else RED
                tds.append(f'<td style="color:{color};font-weight:700">{esc(val)}</td>')
            elif col in wrap_cols:
                tds.append(f'<td class="wrap">{esc(val)}</td>')
            else:
                tds.append(f"<td>{esc(fmt_value(val))}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="twrap"><table class="t"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


# ----------------------------------------------------------------------------
# FEATURE DESCRIPTIONS / GROUPS
# ----------------------------------------------------------------------------
def describe_feature(name: str) -> str:
    if name in FEATURE_INFO:
        return FEATURE_INFO[name]
    n = name.lower()
    if "fwd" in n or "forward" in n:
        direction = " in the forward direction (source to destination)"
    elif "bwd" in n or "backward" in n:
        direction = " in the backward direction (destination to source)"
    else:
        direction = ""
    stat = ""
    for key, text in (("std", "Spread (standard deviation)"), ("variance", "Variance"), ("mean", "Average"),
                      ("avg", "Average"), ("max", "Maximum"), ("min", "Minimum"), ("total", "Total")):
        if key in n:
            stat = text
            break
    subject = ""
    for key, text in (("iat", "the time gaps between packets"), ("packet length", "packet length"),
                      ("segment size", "TCP segment size"), ("bulk", "bulk data transfer"),
                      ("flag", "TCP flag usage"), ("header", "packet header size"),
                      ("win", "TCP window size"), ("active", "the active time of the flow"),
                      ("idle", "the idle time of the flow"), ("packets/s", "packet rate"),
                      ("bytes/s", "byte rate"), ("packets", "number of packets"), ("bytes", "number of bytes")):
        if key in n:
            subject = text
            break
    if subject:
        return f"{stat or 'Measurement'} of {subject}{direction}."
    return f"Numeric measurement describing the traffic flow ({name})."


def feature_group(name: str) -> str:
    n = name.lower()
    if "flag" in n:
        return "TCP FLAGS"
    if "iat" in n:
        return "TIMING (IAT)"
    if "active" in n or "idle" in n:
        return "ACTIVE / IDLE"
    if "packet length" in n or "packet size" in n or "segment" in n:
        return "PACKET SIZE"
    if "bulk" in n:
        return "BULK TRANSFER"
    if "win" in n or "header" in n or "act_data" in n or "min_seg" in n:
        return "HEADER / WINDOW"
    if "fwd" in n or "bwd" in n or "backward" in n or "forward" in n or "total" in n or "subflow" in n:
        return "DIRECTIONAL TRAFFIC"
    if "flow" in n:
        return "FLOW RATES"
    return "GENERAL"


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str, mtime: float):
    bundle = joblib.load(path_str)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("The model file must be a dictionary with the keys 'model' and 'features'.")
    feats = [str(f) for f in list(bundle["features"])]
    return {"model": bundle["model"], "features": feats, "clean": [f.strip() for f in feats]}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()].copy()


def _read_csv(source) -> pd.DataFrame:
    return pd.read_csv(source, low_memory=False)


@st.cache_data(show_spinner=False)
def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = _read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), low_memory=False, encoding="latin-1")
    df = clean_columns(df)
    if df.empty:
        raise ValueError("The CSV file contains no records.")
    return df


@st.cache_data(show_spinner=False)
def read_demo_csv(path_str: str, mtime: float, limit) -> pd.DataFrame:
    try:
        df = _read_csv(path_str)
    except UnicodeDecodeError:
        df = pd.read_csv(path_str, low_memory=False, encoding="latin-1")
    df = clean_columns(df)
    if limit and limit < len(df):
        df = df.sample(n=int(limit), random_state=42).reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
def run_prediction(df: pd.DataFrame, bundle: dict, source_label: str, file_name: str) -> dict:
    started = time.time()
    model = bundle["model"]
    feats = bundle["features"]
    clean = bundle["clean"]

    X = df[clean].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    replaced = int(X.isna().sum().sum())
    X = X.fillna(X.median(numeric_only=True)).fillna(0)  # median per column, 0 if a column is all-empty
    X = X.clip(lower=-3.0e38, upper=3.0e38)               # keep values inside float32 range
    X.columns = feats                                      # exact training names and order

    expected = getattr(model, "n_features_in_", len(feats))
    if expected != len(feats):
        raise ValueError(f"Model expects {expected} features but the saved list has {len(feats)}.")

    model_input = X if hasattr(model, "feature_names_in_") else X.values
    raw_pred = model.predict(model_input)
    labels = pd.Series(raw_pred).map(normalize_label).to_numpy()

    proba = None
    classes = [normalize_label(c) for c in getattr(model, "classes_", [])]
    if hasattr(model, "predict_proba") and "DDoS" in classes:
        proba = model.predict_proba(model_input)[:, classes.index("DDoS")]

    status_map = {"BENIGN": "NORMAL", "DDoS": "SUSPICIOUS"}
    out = df.copy()
    out["AI Prediction"] = labels
    out["Threat Status"] = pd.Series(labels).map(lambda v: status_map.get(v, "UNKNOWN")).to_numpy()

    n = len(out)
    benign = int((labels == "BENIGN").sum())
    ddos = int((labels == "DDoS").sum())

    label_match = None
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is not None:
        label_match = float((out[label_col].astype(str).str.strip().str.upper() == pd.Series(labels).str.upper().to_numpy()).mean() * 100)

    return {
        "out": out, "n": n, "benign": benign, "ddos": ddos, "proba": proba,
        "replaced": replaced, "csv": out.to_csv(index=False).encode("utf-8"),
        "label_match": label_match, "label_col": label_col,
        "source": source_label, "name": file_name, "seconds": time.time() - started,
    }


# ----------------------------------------------------------------------------
# CHARTS (matplotlib, black / white / green)
# ----------------------------------------------------------------------------
plt.rcParams["font.family"] = "monospace"


def new_fig(w=6.2, h_=3.8):
    fig, ax = plt.subplots(figsize=(w, h_))
    fig.patch.set_facecolor("#000000")
    ax.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_color("#FFFFFF")
    ax.tick_params(colors="#FFFFFF", labelsize=8)
    ax.xaxis.label.set_color("#FFFFFF")
    ax.yaxis.label.set_color("#FFFFFF")
    fig.tight_layout(pad=2.0)
    return fig, ax


def finish(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, color="#FFFFFF", fontsize=11, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, color="#FFFFFF", fontsize=9)
    ax.set_ylabel(ylabel, color="#FFFFFF", fontsize=9)


def chart_counts(benign, ddos):
    fig, ax = new_fig()
    vals = [benign, ddos]
    bars = ax.bar(["BENIGN", "DDoS"], vals, color=[GREEN, RED], edgecolor="#FFFFFF", linewidth=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", color="#FFFFFF", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(max(vals) * 1.18, 1))
    ax.grid(axis="y", color="#222222")
    ax.set_axisbelow(True)
    finish(ax, "TRAFFIC CLASSIFICATION DISTRIBUTION", "CLASS", "RECORD COUNT")
    fig.tight_layout(pad=2.0)
    return fig


def chart_percent(benign, ddos, n):
    fig, ax = new_fig()
    pct = [benign / n * 100, ddos / n * 100]
    bars = ax.barh(["DDoS", "BENIGN"], [pct[1], pct[0]], color=[RED, GREEN], edgecolor="#FFFFFF", linewidth=1)
    for b, v in zip(bars, [pct[1], pct[0]]):
        ax.text(min(v + 1.5, 82), b.get_y() + b.get_height() / 2, f"{v:.2f}%", va="center", color="#FFFFFF", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color="#222222")
    ax.set_axisbelow(True)
    finish(ax, "TRAFFIC CLASSIFICATION PERCENTAGE", "PERCENT OF ANALYZED RECORDS (%)", "")
    fig.tight_layout(pad=2.0)
    return fig


def chart_features(out, features, stat):
    fig, ax = new_fig()
    width = 0.38
    xs = np.arange(len(features))
    for i, (cls, color) in enumerate((("BENIGN", GREEN), ("DDoS", RED))):
        vals = []
        for f in features:
            s = pd.to_numeric(out.loc[out["AI Prediction"] == cls, f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            vals.append(float(s.median() if stat == "MEDIAN" else s.mean()) if len(s) else 0.0)
        bars = ax.bar(xs + (i - 0.5) * width, vals, width, color=color, edgecolor="#FFFFFF", linewidth=1, label=cls)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", color="#FFFFFF", fontsize=7, rotation=0)
    ax.set_yscale("symlog")
    ax.set_xticks(xs)
    ax.set_xticklabels([f.replace(" ", "\n", 1) if len(f) > 14 else f for f in features], fontsize=7)
    ax.grid(axis="y", color="#222222")
    ax.set_axisbelow(True)
    leg = ax.legend(facecolor="#000000", edgecolor="#FFFFFF", fontsize=8)
    for t in leg.get_texts():
        t.set_color("#FFFFFF")
    finish(ax, f"NETWORK FEATURE ANALYSIS ({stat} BY PREDICTED CLASS)", "", "VALUE (SYMLOG SCALE)")
    fig.tight_layout(pad=2.0)
    return fig


def chart_probability(proba):
    fig, ax = new_fig()
    bins = np.linspace(0, 1, 21)
    _, edges, patches = ax.hist(proba, bins=bins, edgecolor="#FFFFFF", linewidth=1)
    for edge_left, patch in zip(edges[:-1], patches):
        patch.set_facecolor(GREEN if edge_left < 0.5 else RED)
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="#222222")
    ax.set_axisbelow(True)
    finish(ax, "PREDICTION DISTRIBUTION", "MODEL PROBABILITY OF DDoS (0 = BENIGN, 1 = DDoS)", "RECORDS (LOG SCALE)")
    fig.tight_layout(pad=2.0)
    return fig




# ----------------------------------------------------------------------------
# PDF REPORT GENERATORS
# ----------------------------------------------------------------------------
def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="CyberTitle", parent=styles["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#123047"),
        spaceAfter=10
    ))
    styles.add(ParagraphStyle(
        name="CyberH1", parent=styles["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=colors.HexColor("#087F5B"),
        spaceBefore=8, spaceAfter=7
    ))
    styles.add(ParagraphStyle(
        name="CyberH2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=11, leading=14, textColor=colors.HexColor("#123047"),
        spaceBefore=5, spaceAfter=4
    ))
    styles.add(ParagraphStyle(
        name="CyberBody", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=9, leading=13, spaceAfter=6
    ))
    styles.add(ParagraphStyle(
        name="CyberSmall", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=8, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=4
    ))
    return styles


def _pdf_table(rows, widths=None):
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123047")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C5CC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F7")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _pdf_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#667780"))
    canvas.drawString(18 * mm, 12 * mm, "CYBERSHIELD AI | Educational offline dataset analysis")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_detection_pdf(res):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF library missing. Run: python -m pip install reportlab")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm,
                            topMargin=17*mm, bottomMargin=20*mm)
    styles = _pdf_styles()
    story = [
        Paragraph("CYBERSHIELD AI", styles["CyberTitle"]),
        Paragraph("Network Traffic Detection Report", styles["Heading2"]),
        Paragraph(f"Generated: {time.strftime('%d %B %Y, %I:%M:%S %p')}", styles["CyberSmall"]),
        Spacer(1, 8),
        Paragraph("1. Analysis Overview", styles["CyberH1"]),
    ]
    n = int(res["n"])
    benign = int(res["benign"])
    ddos = int(res["ddos"])
    benign_pct = (benign / n * 100) if n else 0
    ddos_pct = (ddos / n * 100) if n else 0
    rows = [
        ["Field", "Value"],
        ["Dataset file", str(res.get("name", "Not provided"))],
        ["Dataset source", str(res.get("source", "Not provided"))],
        ["Records analyzed", f"{n:,}"],
        ["BENIGN predictions", f"{benign:,} ({benign_pct:.2f}%)"],
        ["DDoS predictions", f"{ddos:,} ({ddos_pct:.2f}%)"],
        ["Analysis duration", f"{float(res.get('seconds', 0)):.2f} seconds"],
        ["Missing/infinite values replaced", f"{int(res.get('replaced', 0)):,}"],
        ["Model", "Random Forest Classifier"],
        ["Dataset family", "CIC-IDS2017"],
        ["Saved evaluation accuracy", MODEL_ACCURACY],
    ]
    story.append(_pdf_table(rows, [62*mm, 105*mm]))
    story += [
        Spacer(1, 12),
        Paragraph("2. Classification Summary", styles["CyberH1"]),
        Paragraph(
            f"The model classified {n:,} flow records. {benign:,} records were predicted as BENIGN "
            f"({benign_pct:.2f}%), while {ddos:,} records were predicted as DDoS "
            f"({ddos_pct:.2f}%). These are model predictions for the supplied CSV file.",
            styles["CyberBody"]
        ),
        Paragraph("3. Model and Method", styles["CyberH1"]),
        Paragraph(
            "The application loads a previously trained Random Forest model. The uploaded CSV is "
            "validated for the model's required features. Numeric conversion is applied, infinite "
            "values are treated as missing, and missing values are replaced before classification. "
            "The output adds AI Prediction and Threat Status columns.",
            styles["CyberBody"]
        ),
        Paragraph("4. Important Limitations", styles["CyberH1"]),
        Paragraph(
            "This is an educational, offline dataset-classification tool. It does not capture live "
            "network packets, scan networks, or execute attacks. The displayed 99.97% is the saved "
            "evaluation result from the project's test split; it is not a guarantee of performance "
            "on new networks. Dataset similarity, capture conditions, class balance, and data leakage "
            "can affect measured performance. Predictions should not be treated as a production "
            "security verdict.",
            styles["CyberBody"]
        ),
        Spacer(1, 8),
        Paragraph("End of report.", styles["CyberSmall"]),
    ]
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    buffer.seek(0)
    return buffer.getvalue()


def build_project_pdf():
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("PDF library missing. Run: python -m pip install reportlab")
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=19*mm, leftMargin=19*mm,
                            topMargin=17*mm, bottomMargin=20*mm)
    styles = _pdf_styles()
    P = lambda text: Paragraph(text, styles["CyberBody"])
    H = lambda text: Paragraph(text, styles["CyberH1"])
    story = []

    # 12 deliberately structured pages for a college project report.
    story += [
        Spacer(1, 35*mm),
        Paragraph("CYBERSHIELD AI", styles["CyberTitle"]),
        Paragraph("INTELLIGENT NETWORK THREAT DETECTION SYSTEM", styles["Heading2"]),
        Spacer(1, 12*mm),
        P("<b>Project Report</b><br/>An educational machine-learning application for "
          "offline classification of network-flow records."),
        Spacer(1, 12*mm),
        P("<b>Submitted by:</b> ______________________________"),
        P("<b>Course / Branch:</b> ______________________________"),
        P("<b>Institute:</b> ______________________________"),
        P("<b>Academic Session:</b> ______________________________"),
        Spacer(1, 25*mm),
        P("Note: Fill in the student, institute, guide, and session details before submission."),
        PageBreak(),

        H("PAGE 2 — ABSTRACT"),
        P("CyberShield AI is an offline, educational network traffic classification system. "
          "It uses a trained Random Forest machine-learning model to classify network-flow CSV "
          "records into two categories: BENIGN and DDoS. The application is implemented using "
          "Python and Streamlit and provides a dashboard for loading a dataset, validating required "
          "features, running predictions, viewing summary statistics, exploring charts, and exporting "
          "the prediction results."),
        P("The project uses selected network-flow features from the CIC-IDS2017 dataset. Before "
          "prediction, feature values are converted to numeric form, infinite values are handled as "
          "missing, and missing values are replaced. The output contains the original data along with "
          "AI Prediction and Threat Status columns."),
        P("The system is designed for academic demonstration and offline analysis. It does not monitor "
          "live traffic or perform network operations. Its reported evaluation score is specific to "
          "the saved model evaluation and must not be interpreted as guaranteed real-world accuracy."),
        H("Keywords"),
        P("Cybersecurity, Machine Learning, Random Forest, CIC-IDS2017, DDoS classification, "
          "Network Flow, Python, Streamlit."),
        PageBreak(),

        H("PAGE 3 — INTRODUCTION"),
        P("Computer networks generate large volumes of traffic records. Reviewing every record "
          "manually can be time-consuming. Machine-learning methods can help identify patterns in "
          "structured traffic data and assign records to learned categories."),
        P("A Distributed Denial of Service (DDoS) event is a network availability threat in which "
          "a service may receive an unusually large volume of requests or traffic. In this project, "
          "the DDoS class is studied only through an existing labelled dataset. No attack is created "
          "or executed."),
        P("CyberShield AI provides a graphical interface to make the classification workflow easier "
          "to demonstrate. Users can upload a compatible CSV file or load the configured demonstration "
          "dataset, validate its columns, run the saved model, inspect results, and download outputs."),
        H("Problem Statement"),
        P("To build an easy-to-use offline application that applies a trained machine-learning model "
          "to network-flow CSV records and presents BENIGN and DDoS predictions in a readable form."),
        PageBreak(),

        H("PAGE 4 — OBJECTIVES AND SCOPE"),
        H("Objectives"),
        P("1. Use a labelled network-flow dataset for academic analysis.<br/>"
          "2. Load a previously trained Random Forest model.<br/>"
          "3. Validate the required input features before prediction.<br/>"
          "4. Classify records as BENIGN or DDoS.<br/>"
          "5. Present counts, percentages, charts, and record-level details.<br/>"
          "6. Export predictions and generate PDF reports."),
        H("Project Scope"),
        P("The application focuses on offline CSV-based classification. It is suitable for "
          "demonstrating dataset preprocessing, model loading, inference, visualization, and report "
          "generation. It is not a live intrusion detection appliance, incident-response platform, "
          "or substitute for professional security monitoring."),
        H("Out of Scope"),
        P("Live packet capture, network scanning, active attack simulation, automatic blocking, "
          "and production deployment are outside the scope of this educational implementation."),
        PageBreak(),

        H("PAGE 5 — EXISTING AND PROPOSED SYSTEM"),
        H("Existing Approach"),
        P("A manual approach requires a person to inspect traffic records and compare values across "
          "many columns. This can be slow and may make consistent classification difficult when "
          "the dataset contains many records."),
        H("Proposed System"),
        P("The proposed system loads a saved machine-learning model and applies it to validated "
          "CSV records. It automates numeric preprocessing and classification, then presents "
          "aggregate and record-level results through a Streamlit dashboard."),
        H("Advantages"),
        P("• Faster batch analysis than manual row-by-row inspection.<br/>"
          "• Consistent application of the saved model.<br/>"
          "• Feature validation before prediction.<br/>"
          "• Interactive visual summaries and downloadable outputs.<br/>"
          "• Offline operation on compatible CSV files."),
        PageBreak(),

        H("PAGE 6 — SYSTEM ARCHITECTURE"),
        P("The workflow is organised into six stages:"),
        _pdf_table([
            ["Stage", "Description"],
            ["1. CSV Input", "User uploads a compatible network-flow CSV or loads the demo dataset."],
            ["2. Validation", "The application checks that required model features are present."],
            ["3. Preprocessing", "Values are converted to numeric form; missing and infinite values are handled."],
            ["4. Model Loading", "The saved Random Forest bundle is loaded from the model folder."],
            ["5. Prediction", "The model classifies each row as BENIGN or DDoS."],
            ["6. Output", "Dashboard summaries, charts, detailed rows, CSV and PDF reports are provided."],
        ], [35*mm, 130*mm]),
        Spacer(1, 8),
        P("<b>Architecture flow:</b><br/>CSV Input → Feature Validation → Data Preprocessing → "
          "Saved Random Forest → Prediction → Dashboard and Reports"),
        PageBreak(),

        H("PAGE 7 — DATASET DESCRIPTION"),
        P("The project is based on CIC-IDS2017, a public cybersecurity dataset containing labelled "
          "network traffic records. The configured demonstration file is "
          "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv. The records describe network flows "
          "using numerical attributes such as duration, packet counts, byte rates, packet lengths, "
          "and TCP flag counts."),
        H("Dataset Classes"),
        P("BENIGN represents normal labelled traffic. DDoS represents records labelled as "
          "Distributed Denial of Service traffic in the dataset."),
        H("Data Handling"),
        P("The application expects the columns required by the saved model. If a required feature "
          "is missing, prediction is blocked and the user is informed. Numeric conversion and "
          "missing-value handling are applied before the model receives the data."),
        PageBreak(),

        H("PAGE 8 — RANDOM FOREST ALGORITHM"),
        P("Random Forest is an ensemble-learning algorithm that combines multiple decision trees. "
          "Each tree produces a class prediction, and the forest combines the tree outputs to produce "
          "the final classification. Using multiple trees can reduce dependence on a single decision "
          "tree and can model non-linear relationships between input features."),
        H("Working Principle"),
        P("1. A collection of decision trees is trained.<br/>"
          "2. Each tree learns patterns from training samples and selected features.<br/>"
          "3. During inference, each tree predicts a class for the input record.<br/>"
          "4. The forest combines the votes to select the final class."),
        H("Why Random Forest"),
        P("It is suitable for structured tabular data, can model complex feature relationships, "
          "and provides a practical baseline for binary classification tasks."),
        PageBreak(),

        H("PAGE 9 — IMPLEMENTATION AND TECHNOLOGIES"),
        _pdf_table([
            ["Technology", "Purpose"],
            ["Python", "Main programming language."],
            ["Pandas", "CSV loading and tabular data processing."],
            ["NumPy", "Numerical operations."],
            ["Scikit-learn", "Machine-learning model and preprocessing pipeline."],
            ["Joblib", "Loading the saved model bundle."],
            ["Streamlit", "Interactive web dashboard."],
            ["Matplotlib", "Charts and data visualizations."],
            ["ReportLab", "PDF report generation."],
        ], [42*mm, 123*mm]),
        Spacer(1, 8),
        P("The application is organised around a Streamlit interface. It loads the model from "
          "model/cybershield_model.pkl and the configured dataset from the dataset directory. "
          "The model bundle stores the trained estimator and feature information."),
        PageBreak(),

        H("PAGE 10 — RESULTS AND OUTPUT"),
        P("The saved training workflow reports 120,000 selected records, an 80/20 train-test split, "
          "and a saved evaluation accuracy of 99.97%. The recorded confusion matrix is "
          "[[10,387, 2], [5, 13,606]], with 23,993 correct predictions out of 24,000 test records."),
        P("These values describe the saved evaluation run and are not recalculated by the dashboard "
          "for every uploaded CSV. Dashboard results are the counts of predictions on the currently "
          "analysed file. If a source Label column is present, the application may also show a "
          "comparison against that column; this is not a held-out test evaluation."),
        H("Application Outputs"),
        P("The dashboard presents total records, BENIGN predictions, DDoS predictions, percentages, "
          "traffic charts, selected feature comparisons, prediction details, and a downloadable "
          "CSV containing the original columns plus prediction labels."),
        PageBreak(),

        H("PAGE 11 — LIMITATIONS AND FUTURE SCOPE"),
        H("Limitations"),
        P("The model is evaluated on a particular dataset split and may not generalize to different "
          "networks, devices, traffic distributions, or future threats. Dataset bias, repeated "
          "capture patterns, class distribution, and possible data leakage can affect reported "
          "metrics. Predictions should be interpreted as educational model outputs, not as a "
          "production security decision."),
        H("Future Scope"),
        P("Possible future work includes testing on independent datasets, improving validation "
          "methodology, adding explainability for model predictions, improving accessibility, "
          "and conducting controlled usability testing. Any production-oriented extension would "
          "require security review, privacy safeguards, and extensive independent evaluation."),
        PageBreak(),

        H("PAGE 12 — CONCLUSION AND REFERENCES"),
        P("CyberShield AI demonstrates an end-to-end offline machine-learning workflow for "
          "classifying network-flow CSV records. It combines feature validation, preprocessing, "
          "Random Forest inference, visual summaries, record-level inspection, and downloadable "
          "reports in a single interface. The project provides a practical academic example of "
          "applying machine learning to a cybersecurity dataset while clearly stating its limits."),
        H("References"),
        P("1. CIC-IDS2017 dataset, Canadian Institute for Cybersecurity, University of New Brunswick.<br/>"
          "2. Scikit-learn documentation: Random Forest and preprocessing modules.<br/>"
          "3. Streamlit documentation: application interface and download components.<br/>"
          "4. Pandas and NumPy documentation: data processing and numerical operations."),
        Spacer(1, 8),
        P("Before submission, add your actual project screenshots, student details, guide name, "
          "institute format, and any references required by your department. This generated report "
          "does not invent screenshots or personal details."),
    ]
    doc.build(story, onFirstPage=_pdf_footer, onLaterPages=_pdf_footer)
    buffer.seek(0)
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# PAGE START
# ----------------------------------------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="scan"></div>', unsafe_allow_html=True)

bundle, model_error = None, None
if not MODEL_PATH.exists():
    model_error = f"Model file not found: {MODEL_PATH}"
else:
    try:
        bundle = load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    except Exception as exc:  # noqa: BLE001
        model_error = f"Could not load the model: {exc}"

model_ok = bundle is not None
features = bundle["clean"] if model_ok else []
model_obj = bundle["model"] if model_ok else None
n_features = len(features)

# ---------------- navigation ----------------
st.markdown(h("""
<div class="nav">
<div class="logo"><b>&gt;</b> CYBERSHIELD_AI</div>
<div class="nav-links">
<a href="#analysis">01_ANALYSIS</a>
<a href="#features">02_FEATURES</a>
<a href="#results">03_RESULTS</a>
<a href="#graphs">04_GRAPHS</a>
<a href="#details">05_DETAILS</a>
<a href="#model">06_MODEL</a>
</div>
<a class="btn-b" href="#analysis">RUN_THREAT_DETECTION.EXE</a>
</div>
"""), unsafe_allow_html=True)

# ---------------- hero ----------------
hero_lines = ASCII_LOGO + [
    "",
    "> CYBERSHIELD AI INITIALIZED",
    "> LOADING TRAINED MODEL",
    f"> MODEL STATUS: {'CONNECTED' if model_ok else 'NOT FOUND'}",
    f"> INPUT FEATURES: {n_features}",
    "> CLASSIFICATION: BENIGN / DDOS",
    "> MODE: OFFLINE DATASET ANALYSIS",
]
status_text = "ONLINE" if model_ok else "MODEL ERROR"
model_text = "CONNECTED" if model_ok else "NOT FOUND"
st.markdown(h(f"""
<div class="hero reveal">
<div>
<div class="status"><i>&#9679;</i> STATUS: {status_text} &nbsp;|&nbsp; MODEL: {model_text}</div>
<div class="hero-title">INTELLIGENT <span class="stroke">NETWORK</span> THREAT <span class="stroke">DETECTION</span></div>
<div class="hero-sub">MACHINE LEARNING POWERED NETWORK TRAFFIC CLASSIFICATION</div>
<div class="hero-tag">DETECT. ANALYZE. CLASSIFY.</div>
</div>
<div>{terminal_html(hero_lines)}</div>
</div>
"""), unsafe_allow_html=True)

# ---------------- data ribbon (filled after analysis runs) ----------------
ribbon_ph = st.empty()


def fill_ribbon(res):
    if res:
        vals = (f"{res['n']:,}", f"{res['benign']:,}", f"{res['ddos']:,}")
        notes = ("FROM LATEST ANALYSIS", "BENIGN PREDICTIONS", "DDOS PREDICTIONS")
    else:
        vals = ("--", "--", "--")
        notes = ("AWAITING ANALYSIS",) * 3
    cells = [
        ("TOTAL RECORDS", vals[0], notes[0]),
        ("NORMAL TRAFFIC", vals[1], notes[1]),
        ("DDOS DETECTED", vals[2], notes[2]),
        ("MODEL ACCURACY", MODEL_ACCURACY, "SAVED EVALUATION RESULT (NOT A LIVE SCORE)"),
    ]
    inner = "".join(
        f'<div class="rcell"><div class="rlabel">{l}</div><div class="rval">{v}</div><div class="rnote">{n}</div></div>'
        for l, v, n in cells
    )
    ribbon_ph.markdown(f'<div class="fb ribbon">{inner}</div>', unsafe_allow_html=True)


st.session_state.setdefault("result", None)

if not model_ok:
    fill_ribbon(None)
    st.markdown(msg_box(f"<b>MODEL ERROR:</b> {esc(model_error)}<br>Expected location: <b>model/cybershield_model.pkl</b> "
                        "next to app.py. Run train_model.py first if the file does not exist.", "bad"), unsafe_allow_html=True)
    st.stop()

# ----------------------------------------------------------------------------
# 01 / NETWORK_TRAFFIC_ANALYSIS
# ----------------------------------------------------------------------------
st.markdown(section_open("analysis", "01", "NETWORK_TRAFFIC_ANALYSIS"), unsafe_allow_html=True)
st.markdown('<div class="note">Upload a CIC-IDS2017 style network-flow CSV, or leave the uploader empty to demonstrate '
            'with the project dataset. Analysis is offline classification of recorded data only.</div>', unsafe_allow_html=True)

col_l, col_r = st.columns([1, 1.25], gap="large")
with col_l:
    uploaded = st.file_uploader("UPLOAD NETWORK CSV", type=["csv"])
    demo_choice = st.selectbox("DEMO DATASET ROWS (USED WHEN NO FILE IS UPLOADED)", [5000, 10000, 25000, 50000, "ALL"], index=1)

df, source_label, file_name, file_size, load_error = None, "", "--", "--", None
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    try:
        df = read_uploaded_csv(raw_bytes)
        source_label, file_name, file_size = "UPLOADED FILE", uploaded.name, fmt_bytes(len(raw_bytes))
    except Exception as exc:  # noqa: BLE001
        load_error = f"Could not read this file as a CSV: {exc}"
elif DATASET_PATH.exists():
    try:
        limit = None if demo_choice == "ALL" else int(demo_choice)
        with st.spinner("LOADING DEMO DATASET..."):
            df = read_demo_csv(str(DATASET_PATH), DATASET_PATH.stat().st_mtime, limit)
        source_label = "PROJECT DEMO DATASET"
        file_name, file_size = DATASET_PATH.name, fmt_bytes(DATASET_PATH.stat().st_size)
    except Exception as exc:  # noqa: BLE001
        load_error = f"Could not read the demo dataset: {exc}"
else:
    load_error = "No file uploaded and the demo dataset was not found at dataset/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv."

missing = [f for f in features if df is None or f not in df.columns]
present_count = n_features - len(missing) if df is not None else 0

with col_r:
    st.markdown(info_grid([
        ("FILE NAME", file_name, ""),
        ("FILE SIZE", file_size, ""),
        ("RECORDS LOADED", f"{len(df):,}" if df is not None else "--", ""),
        ("AVAILABLE FEATURES", f"{len(df.columns)} COLUMNS" if df is not None else "--", ""),
        ("REQUIRED FEATURES", f"{n_features} ({present_count} PRESENT)" if df is not None else str(n_features),
         "good" if df is not None and not missing else ("bad" if df is not None else "")),
        ("DATASET SOURCE", source_label or "--", ""),
    ]), unsafe_allow_html=True)
    if load_error:
        st.markdown(msg_box(f"<b>FILE ERROR:</b> {esc(load_error)}", "bad"), unsafe_allow_html=True)
    elif missing:
        shown = ", ".join(esc(m) for m in missing[:25]) + (f" ... (+{len(missing) - 25} more)" if len(missing) > 25 else "")
        st.markdown(msg_box(f"<b>VALIDATION FAILED:</b> {len(missing)} required model feature(s) are missing from this CSV, "
                            f"so prediction is blocked.<br>MISSING: {shown}", "bad"), unsafe_allow_html=True)
    elif df is not None:
        st.markdown(msg_box(f"<b>VALIDATION PASSED:</b> all {n_features} required model features were found.", "good"), unsafe_allow_html=True)

run_clicked = st.button("RUN_THREAT_DETECTION.EXE", disabled=(df is None or bool(missing)), key="run_btn")
if run_clicked and df is not None and not missing:
    try:
        with st.spinner("RUNNING RANDOM FOREST CLASSIFICATION..."):
            st.session_state["result"] = run_prediction(df, bundle, source_label, file_name)
    except Exception as exc:  # noqa: BLE001
        st.session_state["result"] = None
        st.markdown(msg_box(f"<b>PREDICTION ERROR:</b> {esc(exc)}", "bad"), unsafe_allow_html=True)

result = st.session_state["result"]
fill_ribbon(result)
if result:
    extra = f" Missing/infinite values replaced with column medians: {result['replaced']:,}." if result["replaced"] else ""
    st.markdown(msg_box(f"ANALYSIS COMPLETE: {result['n']:,} records from <b>{esc(result['name'])}</b> "
                        f"classified in {result['seconds']:.1f}s.{extra}", "good"), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# PDF REPORTS
# ----------------------------------------------------------------------------
st.markdown(section_open("reports", "PDF", "REPORT_GENERATION"), unsafe_allow_html=True)
if not REPORTLAB_AVAILABLE:
    st.warning("PDF support needs ReportLab. In CMD run: python -m pip install reportlab")
else:
    report_col1, report_col2 = st.columns(2, gap="large")
    with report_col1:
        st.markdown("**DETECTION REPORT**")
        st.caption("A PDF summary of the latest completed CSV analysis.")
        if result:
            try:
                detection_pdf = build_detection_pdf(result)
                st.download_button(
                    "DOWNLOAD_DETECTION_REPORT.PDF",
                    data=detection_pdf,
                    file_name="cybershield_detection_report.pdf",
                    mime="application/pdf",
                    key="download_detection_pdf",
                )
            except Exception as pdf_exc:
                st.error(f"Could not generate detection PDF: {pdf_exc}")
        else:
            st.info("Run an analysis first to enable the detection report.")
    with report_col2:
        st.markdown("**COMPLETE PROJECT REPORT**")
        st.caption("A structured 12-page academic report. Add your details and screenshots before submission.")
        try:
            project_pdf = build_project_pdf()
            st.download_button(
                "DOWNLOAD_PROJECT_REPORT.PDF",
                data=project_pdf,
                file_name="cybershield_ai_project_report.pdf",
                mime="application/pdf",
                key="download_project_pdf",
            )
        except Exception as pdf_exc:
            st.error(f"Could not generate project PDF: {pdf_exc}")


# ----------------------------------------------------------------------------
# 02 / NETWORK_FEATURE_EXPLORER
# ----------------------------------------------------------------------------
st.markdown(section_open("features", "02", "NETWORK_FEATURE_EXPLORER"), unsafe_allow_html=True)
st.markdown('<div class="note">These features describe network traffic behaviour. The Machine Learning model studies their '
            'numerical patterns to classify traffic as BENIGN or DDoS.</div>', unsafe_allow_html=True)
st.markdown(msg_box(f"The saved model uses <b>{n_features}</b> input features. Names below come directly from the model's saved feature list.", "warn"),
            unsafe_allow_html=True)

if df is None:
    st.markdown(msg_box("Load a dataset above to see real feature values.", "warn"), unsafe_allow_html=True)
else:
    rec_idx = st.number_input("RECORD TO INSPECT (ROW INDEX, STARTING AT 0)", min_value=0, max_value=len(df) - 1, value=0, step=1)

    key_present = [f for f in KEY_FEATURES if f in features and f in df.columns]
    if key_present:
        cells = [(f, fmt_value(df[f].iloc[int(rec_idx)]), "") for f in key_present]
        st.markdown('<div class="note">KEY_SIGNALS: commonly used traffic indicators found in the model, with the value from the selected record.</div>',
                    unsafe_allow_html=True)
        st.markdown(info_grid(cells), unsafe_allow_html=True)

    groups = sorted({feature_group(f) for f in features})
    c1, c2, c3 = st.columns([2, 1.3, 1])
    with c1:
        query = st.text_input("SEARCH FEATURE NAME", "")
    with c2:
        group_sel = st.selectbox("FEATURE GROUP", ["ALL"] + groups)
    filtered = [f for f in features if (query.strip().lower() in f.lower()) and (group_sel == "ALL" or feature_group(f) == group_sel)]
    page_size = 20
    pages = max(1, -(-len(filtered) // page_size))
    with c3:
        page = st.number_input(f"PAGE (1-{pages})", min_value=1, max_value=pages, value=1, step=1)

    subset = filtered[(int(page) - 1) * page_size: int(page) * page_size]
    rows = []
    for f in subset:
        if f in df.columns:
            col_num = pd.to_numeric(df[f], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean_val = col_num.mean()
            rows.append({"#": features.index(f) + 1, "FEATURE NAME": f, "GROUP": feature_group(f),
                         "VALUE (SELECTED RECORD)": fmt_value(df[f].iloc[int(rec_idx)]),
                         "DATASET MEAN": fmt_value(mean_val),
                         "WHAT IT MEANS": describe_feature(f)})
        else:
            rows.append({"#": features.index(f) + 1, "FEATURE NAME": f, "GROUP": feature_group(f),
                         "VALUE (SELECTED RECORD)": "MISSING", "DATASET MEAN": "MISSING",
                         "WHAT IT MEANS": describe_feature(f)})
    st.markdown(f'<div class="note">SHOWING {len(subset)} OF {len(filtered)} MATCHING FEATURES ({n_features} TOTAL)</div>', unsafe_allow_html=True)
    if rows:
        st.markdown(table_html(pd.DataFrame(rows), wrap_cols=("WHAT IT MEANS",)), unsafe_allow_html=True)
    else:
        st.markdown(msg_box("No features match this search.", "warn"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 03 / AI_DETECTION_RESULTS
# ----------------------------------------------------------------------------
st.markdown(section_open("results", "03", "AI_DETECTION_RESULTS"), unsafe_allow_html=True)
if not result:
    st.markdown(msg_box("NO ANALYSIS YET. Click RUN_THREAT_DETECTION.EXE in section 01 to generate results.", "warn"), unsafe_allow_html=True)
else:
    n, benign, ddos = result["n"], result["benign"], result["ddos"]
    pb, pd_ = benign / n * 100, ddos / n * 100
    st.markdown(info_grid([
        ("TOTAL RECORDS ANALYZED", f"{n:,}", ""),
        ("NORMAL TRAFFIC COUNT", f"{benign:,}", "good"),
        ("DDOS COUNT", f"{ddos:,}", "bad" if ddos else ""),
        ("NORMAL TRAFFIC PERCENTAGE", f"{pb:.2f}%", "good"),
        ("DDOS PERCENTAGE", f"{pd_:.2f}%", "bad" if ddos else ""),
    ]), unsafe_allow_html=True)

    st.markdown(f'<div class="note">RESULTS FOR: {esc(result["name"])} ({esc(result["source"])})</div>', unsafe_allow_html=True)
    if ddos == 0:
        st.markdown(msg_box(f"No DDoS predictions found. All {n:,} records were classified as normal (BENIGN) traffic.", "good"), unsafe_allow_html=True)
    elif benign == 0:
        st.markdown(msg_box(f"Suspicious traffic detected. All {n:,} records were classified as DDoS.", "bad"), unsafe_allow_html=True)
    else:
        st.markdown(msg_box(f"Suspicious traffic detected: {ddos:,} records ({pd_:.2f}%) were classified as DDoS.", "bad"), unsafe_allow_html=True)
        st.markdown(msg_box(f"Normal traffic detected: {benign:,} records ({pb:.2f}%) were classified as BENIGN.", "good"), unsafe_allow_html=True)
    if result["label_match"] is not None:
        st.markdown(msg_box(f"LABEL COLUMN CHECK: {result['label_match']:.2f}% of predictions match the '{esc(result['label_col'])}' column in this file. "
                            "This compares against the data the model was trained on, so it is not a held-out test. "
                            f"The official evaluation accuracy is {MODEL_ACCURACY}.", "warn"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 04 / TRAFFIC_VISUALIZATION
# ----------------------------------------------------------------------------
st.markdown(section_open("graphs", "04", "TRAFFIC_VISUALIZATION"), unsafe_allow_html=True)
if not result:
    st.markdown(msg_box("Charts appear after the analysis has been run.", "warn"), unsafe_allow_html=True)
else:
    n, benign, ddos, out = result["n"], result["benign"], result["ddos"], result["out"]
    g1, g2 = st.columns(2, gap="large")
    with g1:
        fig = chart_counts(benign, ddos)
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        fig = chart_percent(benign, ddos, n)
        st.pyplot(fig)
        plt.close(fig)

    g3, g4 = st.columns(2, gap="large")
    with g3:
        numeric_ok = [f for f in features if f in out.columns]
        defaults = [f for f in ("Flow Duration", "Flow Bytes/s", "Flow Packets/s", "Packet Length Mean") if f in numeric_ok]
        chosen = st.multiselect("FEATURES TO COMPARE", numeric_ok, default=defaults, max_selections=8)
        stat = st.radio("STATISTIC", ["MEDIAN", "MEAN"], horizontal=True)
        if chosen:
            fig = chart_features(out, chosen, stat)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(msg_box("Select at least one feature.", "warn"), unsafe_allow_html=True)
    with g4:
        if result["proba"] is not None:
            st.markdown('<div class="note">Distribution of the model\'s DDoS probability across all analyzed records '
                        '(green = classified BENIGN, red = classified DDoS).</div>', unsafe_allow_html=True)
            fig = chart_probability(result["proba"])
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown(msg_box("This model does not provide class probabilities, so the prediction distribution chart is unavailable.", "warn"), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# 05 / PREDICTION_DETAILS
# ----------------------------------------------------------------------------
st.markdown(section_open("details", "05", "PREDICTION_DETAILS"), unsafe_allow_html=True)
if not result:
    st.markdown(msg_box("Prediction details appear after the analysis has been run.", "warn"), unsafe_allow_html=True)
else:
    out = result["out"]
    pred_cols = ["AI Prediction", "Threat Status"]
    original_cols = [c for c in out.columns if c not in pred_cols]
    preferred = [c for c in ("Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
                             "Flow Bytes/s", "Flow Packets/s", "Label") if c in original_cols]
    default_cols = preferred if preferred else original_cols[:6]

    d1, d2, d3 = st.columns([1, 1, 1])
    with d1:
        status_filter = st.selectbox("FILTER BY THREAT STATUS", ["ALL", "NORMAL", "SUSPICIOUS"])
    with d2:
        rows_per_page = st.selectbox("ROWS PER PAGE", [25, 50, 100, 250], index=1)
    view = out if status_filter == "ALL" else out[out["Threat Status"] == status_filter]
    total_pages = max(1, -(-len(view) // rows_per_page))
    with d3:
        page_no = st.number_input(f"PAGE (1-{total_pages})", min_value=1, max_value=total_pages, value=1, step=1, key="pred_page")
    shown_cols = st.multiselect("ORIGINAL CSV COLUMNS TO DISPLAY", original_cols, default=default_cols)

    start = (int(page_no) - 1) * rows_per_page
    chunk = view.iloc[start:start + rows_per_page]
    if len(chunk):
        display = chunk[shown_cols + pred_cols].copy()
        display.insert(0, "RECORD #", chunk.index + 1)
        display = display[["RECORD #"] + pred_cols + shown_cols]
        st.markdown(f'<div class="note">SHOWING RECORDS {start + 1:,} - {start + len(chunk):,} OF {len(view):,} MATCHING RECORDS</div>', unsafe_allow_html=True)
        st.markdown(table_html(display), unsafe_allow_html=True)
    else:
        st.markdown(msg_box("No records match this filter.", "warn"), unsafe_allow_html=True)

    st.markdown('<div class="note">The download contains every original CSV column plus AI Prediction and Threat Status '
                '(BENIGN = NORMAL, DDoS = SUSPICIOUS) for all analyzed records.</div>', unsafe_allow_html=True)
    st.download_button("DOWNLOAD_PREDICTIONS.CSV", data=result["csv"], file_name="cybershield_predictions.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# 06 / MODEL_INTELLIGENCE
# ----------------------------------------------------------------------------
st.markdown(section_open("model", "06", "MODEL_INTELLIGENCE"), unsafe_allow_html=True)
st.markdown(info_grid([
    ("MODEL ALGORITHM", "Random Forest Classifier", ""),
    ("DATASET", "CIC-IDS2017", ""),
    ("CLASSIFICATION TYPES", "BENIGN / DDoS", ""),
    ("INPUT FEATURES", str(n_features), ""),
    ("TRAINING RECORDS", "120,000", ""),
    ("TRAIN TEST SPLIT", "80/20", ""),
    ("MODEL ACCURACY", f"{MODEL_ACCURACY} (EVALUATION RESULT)", "good"),
]), unsafe_allow_html=True)

ICON_TREE = '<svg viewBox="0 0 24 24"><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>'
ICON_DB = '<svg viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/><path d="M3 12a9 3 0 0 0 18 0"/></svg>'
ICON_ACT = '<svg viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>'
ICON_SHIELD = '<svg viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>'

st.markdown(h(f"""
<div class="sgrid">
<div class="scard">{ICON_TREE}<h3>What is Random Forest?</h3>
<p>A Random Forest is a team of many decision trees. Each tree looks at the traffic numbers and votes BENIGN or DDoS. The class with the most votes becomes the final prediction, which makes the result stable and accurate.</p>
<span class="stag">STACK</span></div>
<div class="scard">{ICON_DB}<h3>What is CIC-IDS2017?</h3>
<p>A public cybersecurity dataset from the Canadian Institute for Cybersecurity. It contains recorded network traffic, both normal and attack traffic, described as flow records with labelled classes.</p>
<span class="stack stag">STACK</span></div>
<div class="scard">{ICON_ACT}<h3>What are traffic features?</h3>
<p>Numbers that summarise a network conversation, such as duration, packet counts, bytes per second, packet sizes and TCP flag counts. They turn raw traffic into measurable behaviour the model can learn from.</p>
<span class="stag">PROTO</span></div>
<div class="scard">{ICON_SHIELD}<h3>How are BENIGN and DDoS told apart?</h3>
<p>During training the forest learned the numerical patterns of normal and DDoS flows. For new records it compares the {n_features} feature values against those learned patterns. DDoS floods typically show unusual packet rates, sizes and timing.</p>
<span class="stag">PROTO</span></div>
</div>
"""), unsafe_allow_html=True)

steps = ["NETWORK CSV", "DATA PREPROCESSING", "FEATURE EXTRACTION", "RANDOM FOREST MODEL", "PREDICTION", "RESULTS"]
flow = '<div class="farrow">&gt;&gt;</div>'.join(f'<div class="fbox">{s}</div>' for s in steps)
st.markdown(f'<div class="flow">{flow}</div>', unsafe_allow_html=True)
st.markdown('<div class="note">Preprocessing: column names are trimmed, features are converted to numbers, infinity values are '
            'treated as missing, and missing values are filled with column medians before the model predicts.</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SYSTEM STATUS TERMINAL
# ----------------------------------------------------------------------------
st.markdown(section_open("status", "07", "SYSTEM_STATUS"), unsafe_allow_html=True)
estimators = getattr(model_obj, "n_estimators", None)
status_lines = [
    "> CYBERSHIELD AI INITIALIZED",
    "> LOADING TRAINED MODEL",
    "> MODEL STATUS: CONNECTED",
    f"> ESTIMATOR: {type(model_obj).__name__}" + (f" ({estimators} TREES)" if estimators else ""),
    "> ALGORITHM: RANDOM FOREST",
    f"> INPUT FEATURES: {n_features}",
    "> CLASSIFICATION: BENIGN / DDOS",
    "> DATASET: CIC-IDS2017",
    f"> LOADED RECORDS: {len(df):,}" if df is not None else "> LOADED RECORDS: NONE",
    f"> LAST ANALYSIS: {result['n']:,} RECORDS" if result else "> LAST ANALYSIS: NOT RUN YET",
    "> MODE: OFFLINE CSV CLASSIFICATION (NO LIVE PACKET CAPTURE)",
    "> SYSTEM STATUS: READY",
]
st.markdown(terminal_html(status_lines, big=True), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# FINAL CTA
# ----------------------------------------------------------------------------
st.markdown(h("""
<div class="fb cta">
<div>
<div class="cta-title">DETECT. ANALYZE. CLASSIFY.</div>
<div class="cta-sub">INTELLIGENT NETWORK THREAT DETECTION POWERED BY MACHINE LEARNING.</div>
</div>
<div><a class="cta-btn" href="#analysis">RUN_THREAT_DETECTION.EXE</a></div>
</div>
<div class="foot">&gt; CYBERSHIELD_AI // EDUCATIONAL OFFLINE DATASET CLASSIFICATION PROJECT</div>
"""), unsafe_allow_html=True)