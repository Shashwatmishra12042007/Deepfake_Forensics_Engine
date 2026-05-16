from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SMFE | Cyber Forensics Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import hashlib
import sys
from pathlib import Path
from typing import Any

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DEEPFAKE_MODEL_PATH,
    DEEPFAKE_MODEL_REPO,
    FOOTER_AUTHOR_NAME,
    FOOTER_UNIVERSITY,
    SENSOR_PROFILES,
    SIDEBAR_LOGO_CANDIDATES,
)
from src.analysis.forensic_report import AnalysisContext, build_detection_factors
from src.analysis.forensic_score import compute_evidence_breakdown
from src.models.detector import (
    analyze_media,
    finalize_authenticity_with_vouching,
    get_last_forensic_details,
    set_denoise_enabled,
)
from src.models.gradcam import generate_gradcam
from src.models.loader import load_detector
from src.processors.audio_processor import analyze_audio, load_audio
from src.processors.metadata_extractor import MetadataReport, extract_file_metadata
from src.processors.video_processor import extract_frames_one_per_second
from src.utils.constants import CLASS_FAKE
from src.utils.errors import ForensicsError, MediaLoadError, ProcessingError
from src.utils.sensor_profile import (
    get_authenticity_warn_pct,
    get_profile_display_name,
    get_sensor_profile_config,
    set_active_sensor_profile,
)

# ── Session state (before any widgets) ─────────────────────────────────────
if "sensor_profile" not in st.session_state:
    st.session_state["sensor_profile"] = "Standard Camera"
if "sensitivity" not in st.session_state:
    st.session_state["sensitivity"] = 85
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "metadata_report" not in st.session_state:
    st.session_state["metadata_report"] = None
if "denoising_enabled" not in st.session_state:
    st.session_state["denoising_enabled"] = False
if "last_upload_fp" not in st.session_state:
    st.session_state["last_upload_fp"] = None
if "last_analysis_cfg" not in st.session_state:
    st.session_state["last_analysis_cfg"] = None
if "persisted_file_path" not in st.session_state:
    st.session_state["persisted_file_path"] = None

UPLOAD_TYPES = ["mp4", "wav", "mp3", "jpg", "jpeg", "png"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
UPLOAD_CACHE_DIR = ROOT / ".streamlit_cache" / "uploads"

PROFILE_LABELS = ["Standard Camera", "Pro/Raw Image"]
PROFILE_LABEL_TO_KEY = {
    "Standard Camera": "standard_camera",
    "Pro/Raw Image": "pro_raw",
}


def _profile_key_from_label(label: str) -> str:
    return PROFILE_LABEL_TO_KEY.get(label, "standard_camera")


def _normalize_sensor_profile_state() -> None:
    """Keep session label valid for the selectbox."""
    if st.session_state["sensor_profile"] not in PROFILE_LABELS:
        if st.session_state["sensor_profile"] in SENSOR_PROFILES:
            key = st.session_state["sensor_profile"]
            for label, profile_key in PROFILE_LABEL_TO_KEY.items():
                if profile_key == key:
                    st.session_state["sensor_profile"] = label
                    return
        st.session_state["sensor_profile"] = "Standard Camera"


def inject_command_center_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

        .stApp,
        [data-testid="stAppViewContainer"],
        .main .block-container {
            background-color: #0E1117 !important;
            padding-bottom: 3.5rem;
        }

        header[data-testid="stHeader"] {
            background-color: rgba(14, 17, 23, 0.92) !important;
        }

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        div[data-testid="stTabs"] button {
            font-family: 'JetBrains Mono', ui-monospace, monospace !important;
            letter-spacing: 0.04em;
        }

        .command-header {
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-left: 4px solid #00e5a0;
            background: rgba(20, 26, 36, 0.85);
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            border-radius: 8px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.45);
        }

        .command-header h1 {
            font-size: 1.55rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            color: #00e5a0;
            margin: 0 0 0.35rem 0;
            text-shadow: 0 0 18px rgba(0, 229, 160, 0.45);
        }

        .command-header p {
            font-size: 0.9rem;
            color: #8ba3c7;
            margin: 0;
            letter-spacing: 0.06em;
        }

        .label-authentic {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #00e5a0;
            text-shadow: 0 0 12px rgba(0, 229, 160, 0.65);
        }

        .label-suspicious {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #ff4d6a;
            text-shadow: 0 0 12px rgba(255, 77, 106, 0.65);
        }

        .verdict-banner {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.88rem;
            color: #a8b8d0;
            margin: 0.5rem 0 1rem 0;
        }

        [data-testid="stVerticalBlockBorderWrapper"],
        [data-testid="stMetric"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"],
        .forensic-rail {
            border: 1px solid rgba(255, 255, 255, 0.09) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
        }

        [data-testid="stMetric"] {
            background: rgba(18, 22, 30, 0.75);
            padding: 0.65rem 0.85rem;
        }

        .forensic-rail {
            background: rgba(18, 22, 30, 0.85);
            border-top: 3px solid #3d7cff;
            padding: 1rem 1.1rem;
            min-height: 320px;
        }

        .forensic-rail h3 {
            font-size: 0.85rem;
            letter-spacing: 0.15em;
            color: #5b9cff;
            margin-top: 0;
            text-transform: uppercase;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.gauge-panel-label) {
            background: rgba(18, 22, 30, 0.9);
            border-color: rgba(0, 229, 160, 0.35) !important;
            padding: 0.75rem 1rem 0.25rem 1rem;
            margin: 0.75rem 0 1.25rem 0;
        }

        .gauge-panel-label {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.2em;
            color: #00e5a0;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        div[data-testid="stSidebar"] {
            background-color: #0E1117 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .digital-shield-wrap {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 0.5rem 0 1rem 0;
        }

        .digital-shield-wrap svg {
            width: 88px;
            filter: drop-shadow(0 0 12px rgba(0, 229, 160, 0.45));
        }

        .digital-shield-brand {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.28em;
            color: #00e5a0;
            margin-top: 0.55rem;
        }

        .digital-shield-tagline {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            color: #6b8299;
            margin-top: 0.2rem;
            text-transform: uppercase;
        }

        .smfe-sticky-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-wrap: wrap;
            gap: 0.35rem 0.65rem;
            padding: 0.55rem 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #8ba3c7;
            background: linear-gradient(180deg, rgba(14, 17, 23, 0.92) 0%, #0E1117 100%);
            border-top: 1px solid rgba(0, 229, 160, 0.22);
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.45);
        }

        .smfe-footer-sep { color: rgba(107, 130, 153, 0.55); }
        .smfe-footer-uni { color: #5b9cff; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sticky_footer() -> None:
    st.markdown(
        f"""
        <footer class="smfe-sticky-footer">
            <span>Developed for Digital Forensic Analysis</span>
            <span class="smfe-footer-sep">|</span>
            <span>{FOOTER_AUTHOR_NAME}</span>
            <span class="smfe-footer-sep">|</span>
            <span class="smfe-footer-uni">{FOOTER_UNIVERSITY}</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_command_header() -> None:
    st.markdown(
        """
        <div class="command-header">
            <h1>SYNTHETIC MEDIA FORENSICS ENGINE</h1>
            <p>Digital Trust &amp; Verification Protocol</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_sidebar_logo_path() -> Path | None:
    return next((path for path in SIDEBAR_LOGO_CANDIDATES if path.is_file()), None)


def render_sidebar_logo() -> None:
    logo_path = _resolve_sidebar_logo_path()
    if logo_path is not None:
        st.sidebar.image(str(logo_path), use_container_width=True)
        return

    st.sidebar.markdown(
        """
        <div class="digital-shield-wrap">
            <svg viewBox="0 0 120 132" xmlns="http://www.w3.org/2000/svg" aria-label="Digital Shield">
                <defs>
                    <linearGradient id="shieldFill" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#0f1a28"/>
                        <stop offset="100%" stop-color="#0a1018"/>
                    </linearGradient>
                    <linearGradient id="shieldEdge" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stop-color="#00e5a0"/>
                        <stop offset="100%" stop-color="#3d7cff"/>
                    </linearGradient>
                </defs>
                <path d="M60 8 L104 28 L104 72 C104 98 84 118 60 126
                         C36 118 16 98 16 72 L16 28 Z"
                      fill="url(#shieldFill)" stroke="url(#shieldEdge)" stroke-width="2.2"/>
                <circle cx="60" cy="62" r="14" fill="none" stroke="#00e5a0" stroke-width="1.5"/>
                <path d="M60 54 L60 70 M54 62 L66 62" stroke="#00e5a0" stroke-width="2"/>
            </svg>
            <div class="digital-shield-brand">SMFE</div>
            <div class="digital-shield-tagline">Digital Shield</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_controls() -> str:
    """Sensor profile selectbox + sensitivity slider; returns backend profile key."""
    _normalize_sensor_profile_state()

    st.header("◈ Control Panel")
    st.selectbox(
        "Sensor Profile",
        options=PROFILE_LABELS,
        key="sensor_profile",
        help="Standard Camera lowers false positives on smartphone photos.",
    )
    st.slider(
        "Detection Sensitivity",
        min_value=0,
        max_value=100,
        key="sensitivity",
        help="Higher values increase sensitivity to synthetic artifacts.",
    )
    st.toggle(
        "Denoising",
        key="denoising_enabled",
        help="fastNlMeans denoising for images/video before CNN inference.",
    )

    profile_key = _profile_key_from_label(st.session_state["sensor_profile"])
    set_active_sensor_profile(profile_key)
    profile_cfg = get_sensor_profile_config(profile_key)
    st.caption(
        f"Alert threshold: authenticity < **{profile_cfg['gauge_warn_pct']:.0f}%** · "
        f"Sensitivity: **{st.session_state['sensitivity']}%**"
    )
    st.markdown("---")
    weights_ready = DEEPFAKE_MODEL_PATH.is_file()
    st.caption(
        f"CNN: `{DEEPFAKE_MODEL_REPO}`\n\n"
        f"Weights cached: **{'yes' if weights_ready else 'no'}**"
    )
    return profile_key


def _upload_fingerprint(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    """Stable id for the current upload buffer."""
    blob = uploaded.getvalue()
    digest = hashlib.md5(blob).hexdigest()
    return f"{uploaded.name}:{uploaded.size}:{digest}"


def _analysis_config_key(profile_key: str) -> str:
    return (
        f"{profile_key}:"
        f"{st.session_state['denoising_enabled']}:"
        f"{st.session_state['sensitivity']}"
    )


def _clear_analysis_state() -> None:
    st.session_state["analysis_results"] = None
    st.session_state["metadata_report"] = None
    st.session_state["last_upload_fp"] = None
    st.session_state["last_analysis_cfg"] = None
    st.session_state["persisted_file_path"] = None


def persist_uploaded_file(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Save upload to a local path so backend modules can read it across tabs."""
    try:
        UPLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = Path(uploaded.name).name.replace("..", "_")
        digest = hashlib.md5(uploaded.getvalue()).hexdigest()[:10]
        dest = UPLOAD_CACHE_DIR / f"{digest}_{safe_name}"
        dest.write_bytes(uploaded.getvalue())
        st.session_state["persisted_file_path"] = str(dest)
        return dest
    except OSError as exc:
        raise ProcessingError("Could not save uploaded file", str(exc)) from exc


def _try_load_spectrum_audio(file_path: Path) -> tuple[np.ndarray | None, int | None]:
    """Load waveform for mel-spectrogram (audio files and video soundtracks)."""
    try:
        y, sr = load_audio(file_path)
        return y, sr
    except (MediaLoadError, ProcessingError):
        return None, None


def should_run_analysis(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile | None,
    profile_key: str,
    *,
    force: bool = False,
) -> bool:
    if uploaded is None:
        return False
    if force:
        return True
    fp = _upload_fingerprint(uploaded)
    cfg = _analysis_config_key(profile_key)
    if st.session_state.get("last_upload_fp") != fp:
        return True
    if st.session_state.get("last_analysis_cfg") != cfg:
        return True
    return st.session_state.get("analysis_results") is None


def execute_forensic_analysis(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile,
    profile_key: str,
    progress: st.progress,
) -> tuple[dict[str, Any], MetadataReport]:
    """Run full backend pipeline and cache results in session state."""
    results, metadata_report = run_analysis_pipeline(
        uploaded,
        progress,
        profile_key,
        st.session_state["denoising_enabled"],
    )
    st.session_state["analysis_results"] = results
    st.session_state["metadata_report"] = metadata_report
    st.session_state["last_upload_fp"] = _upload_fingerprint(uploaded)
    st.session_state["last_analysis_cfg"] = _analysis_config_key(profile_key)
    return results, metadata_report


def _bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ensure_deepfake_model(progress: st.progress, base: float = 0.05) -> None:
    try:
        progress.progress(base, text="Loading FaceForensics++ deepfake weights…")
        load_detector()
    except ForensicsError as exc:
        st.warning(f"Deepfake CNN unavailable; using forensic pre-processing only. ({exc})")


def _apply_metadata_vouching(
    ai_authenticity_pct: float,
    metadata_report: MetadataReport,
) -> tuple[float, dict]:
    try:
        return finalize_authenticity_with_vouching(ai_authenticity_pct, metadata_report)
    except ProcessingError as exc:
        st.warning(f"Metadata vouching skipped: {exc}")
        return ai_authenticity_pct, {}


def _build_report_ctx(metadata_report: MetadataReport, **kwargs: Any) -> AnalysisContext:
    ctx = AnalysisContext(metadata_report=metadata_report, **kwargs)
    ctx.evidence = compute_evidence_breakdown(ctx)
    return ctx


def _visual_analysis_details(media_type: str, authenticity_pct: float, **extra) -> dict:
    details = {
        "media_type": media_type,
        "authenticity_confidence_pct": authenticity_pct,
        "backbone": DEEPFAKE_MODEL_REPO,
        "preprocessing": ["denoise", "gaussian", "moire_fft", "spectral_residual"],
    }
    forensic = get_last_forensic_details()
    if forensic:
        details["forensics"] = forensic
    details.update(extra)
    return details


def compute_gradcam_meta(
    bgr: np.ndarray,
    authenticity_pct: float,
    warn_pct: float,
) -> dict | None:
    try:
        target_class = CLASS_FAKE if authenticity_pct < warn_pct else None
        return generate_gradcam(bgr, target_class=target_class)
    except ForensicsError as exc:
        st.warning(f"Grad-CAM unavailable: {exc}")
        return None


def run_analysis_pipeline(
    uploaded: st.runtime.uploaded_file_manager.UploadedFile,
    progress: st.progress,
    sensor_profile_key: str,
    denoising_enabled: bool = False,
) -> tuple[dict[str, Any], MetadataReport]:
    set_active_sensor_profile(sensor_profile_key)
    set_denoise_enabled(denoising_enabled)
    file_path = persist_uploaded_file(uploaded)
    suffix = file_path.suffix.lower()
    warn_pct = get_authenticity_warn_pct(sensor_profile_key)

    progress.progress(0.02, text="Extracting file metadata…")
    metadata_report = extract_file_metadata(file_path)

    results: dict[str, Any] = {
        "uploaded_name": uploaded.name,
        "file_path": str(file_path),
        "suffix": suffix,
        "media_type": "unknown",
        "authenticity_pct": None,
        "gradcam_meta": None,
        "detection_factors": [],
        "analysis_details": {},
        "image_bytes": uploaded.getvalue() if suffix in IMAGE_SUFFIXES else None,
        "audio_y": None,
        "audio_sr": None,
        "sensor_profile": sensor_profile_key,
        "sensor_profile_label": st.session_state["sensor_profile"],
        "sensitivity": st.session_state["sensitivity"],
        "warn_pct": warn_pct,
        "denoising_enabled": denoising_enabled,
    }

    if suffix in IMAGE_SUFFIXES:
        results["media_type"] = "image"
        _ensure_deepfake_model(progress, 0.1)
        progress.progress(0.2, text="Running analyze_media() on image…")
        ai_authenticity_pct = analyze_media(
            file_path, sensor_profile=sensor_profile_key, denoise=denoising_enabled
        )
        bgr = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise MediaLoadError("Could not read image for Grad-CAM", str(file_path))
        authenticity_pct, vouching = _apply_metadata_vouching(
            ai_authenticity_pct, metadata_report
        )
        progress.progress(0.85, text="Generating Grad-CAM heatmap…")
        gradcam_meta = compute_gradcam_meta(bgr, authenticity_pct, warn_pct)
        report_ctx = _build_report_ctx(
            metadata_report,
            media_type="image",
            authenticity_pct=authenticity_pct,
            ai_authenticity_pct=ai_authenticity_pct,
            metadata_vouching=vouching,
            file_path=file_path,
            gradcam_meta=gradcam_meta,
        )
        results.update(
            {
                "authenticity_pct": authenticity_pct,
                "ai_authenticity_pct": ai_authenticity_pct,
                "metadata_vouching": vouching,
                "gradcam_meta": gradcam_meta,
                "report_ctx": report_ctx,
            }
        )
        details = _visual_analysis_details("image", authenticity_pct)
        details["ai_authenticity_pct"] = ai_authenticity_pct
        details["metadata_vouching"] = vouching
        if gradcam_meta:
            details["gradcam"] = {
                "target_label": gradcam_meta["target_label"],
                "prob_real": gradcam_meta["prob_real"],
                "prob_fake": gradcam_meta["prob_fake"],
                "focus_regions": gradcam_meta["focus_regions"],
            }
        results["analysis_details"] = details

    elif suffix == ".mp4":
        results["media_type"] = "video"
        _ensure_deepfake_model(progress, 0.02)
        progress.progress(0.05, text="OpenCV: extracting 1 frame/sec…")
        frames = extract_frames_one_per_second(file_path)
        total = len(frames)
        if total == 0:
            raise ProcessingError("No frames extracted from video")

        progress.progress(0.12, text="analyze_media() on sampled frames…")
        scores: list[float] = []
        worst_score = 101.0
        worst_frame = frames[0]
        worst_index = 0

        for i, frame in enumerate(frames):
            score = analyze_media(
                frame,
                sensor_profile=sensor_profile_key,
                denoise=denoising_enabled,
            )
            scores.append(score)
            if score < worst_score:
                worst_score = score
                worst_frame = frame
                worst_index = i
            pct = 0.12 + 0.73 * ((i + 1) / total)
            progress.progress(
                pct,
                text=f"Frame {i + 1}/{total}: CNN + forensic scoring…",
            )

        ai_authenticity_pct = round(float(np.mean(scores)), 2)
        authenticity_pct, vouching = _apply_metadata_vouching(
            ai_authenticity_pct, metadata_report
        )
        progress.progress(0.9, text="Generating Grad-CAM for key frame…")
        gradcam_meta = compute_gradcam_meta(worst_frame, authenticity_pct, warn_pct)
        y_audio, sr_audio = _try_load_spectrum_audio(file_path)
        report_ctx = _build_report_ctx(
            metadata_report,
            media_type="video",
            authenticity_pct=authenticity_pct,
            ai_authenticity_pct=ai_authenticity_pct,
            metadata_vouching=vouching,
            file_path=file_path,
            gradcam_meta=gradcam_meta,
            frame_scores=scores,
            audio_waveform=y_audio,
            audio_sample_rate=sr_audio,
        )
        details = _visual_analysis_details(
            "video",
            authenticity_pct,
            sampling="1_frame_per_second",
            frame_count=total,
            gradcam_frame_second=worst_index,
            frame_score_std=round(float(np.std(scores)), 2) if scores else None,
        )
        if gradcam_meta:
            details["gradcam"] = {
                "target_label": gradcam_meta["target_label"],
                "prob_real": gradcam_meta["prob_real"],
                "prob_fake": gradcam_meta["prob_fake"],
                "focus_regions": gradcam_meta["focus_regions"],
            }
        details["ai_authenticity_pct"] = ai_authenticity_pct
        details["metadata_vouching"] = vouching
        results.update(
            {
                "authenticity_pct": authenticity_pct,
                "ai_authenticity_pct": ai_authenticity_pct,
                "metadata_vouching": vouching,
                "gradcam_meta": gradcam_meta,
                "report_ctx": report_ctx,
                "key_frame_index": worst_index,
                "frame_scores": scores,
                "audio_y": y_audio,
                "audio_sr": sr_audio,
                "analysis_details": details,
            }
        )

    elif suffix in {".wav", ".mp3"}:
        results["media_type"] = "audio"
        progress.progress(0.15, text="Librosa: loading audio…")
        y, sr = load_audio(file_path)
        progress.progress(0.45, text="Librosa: forensic feature extraction…")
        features = analyze_audio(y, sr)
        synthetic = features["heuristic_synthetic_score"]
        ai_authenticity_pct = round((1.0 - synthetic) * 100.0, 2)
        authenticity_pct, vouching = _apply_metadata_vouching(
            ai_authenticity_pct, metadata_report
        )
        report_ctx = _build_report_ctx(
            metadata_report,
            media_type="audio",
            authenticity_pct=authenticity_pct,
            ai_authenticity_pct=ai_authenticity_pct,
            metadata_vouching=vouching,
            file_path=file_path,
            audio_waveform=y,
            audio_sample_rate=sr,
        )
        results.update(
            {
                "authenticity_pct": authenticity_pct,
                "ai_authenticity_pct": ai_authenticity_pct,
                "metadata_vouching": vouching,
                "audio_y": y,
                "audio_sr": sr,
                "report_ctx": report_ctx,
                "analysis_details": {
                    "media_type": "audio",
                    "authenticity_confidence_pct": authenticity_pct,
                    "ai_authenticity_pct": ai_authenticity_pct,
                    "metadata_vouching": vouching,
                    "engine": "librosa_heuristics",
                    "librosa_features": features,
                },
            }
        )
    else:
        raise ProcessingError(f"Unsupported file type: {suffix}")

    progress.progress(1.0, text="Analysis complete.")
    return results, metadata_report


def render_authenticity_gauge(authenticity_pct: float, warn_pct: float) -> None:
    try:
        color = (
            "#00e5a0"
            if authenticity_pct >= warn_pct
            else "#ffb020"
            if authenticity_pct >= warn_pct - 10
            else "#ff4d6a"
        )
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=authenticity_pct,
                title={
                    "text": "AUTHENTICITY CONFIDENCE",
                    "font": {"size": 14, "color": "#8ba3c7", "family": "JetBrains Mono"},
                },
                number={
                    "suffix": "%",
                    "font": {"size": 44, "color": "#e8f0ff", "family": "JetBrains Mono"},
                },
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#4a6080"},
                    "bar": {"color": color, "thickness": 0.3},
                    "bgcolor": "#0E1117",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, warn_pct], "color": "rgba(255, 77, 106, 0.25)"},
                        {"range": [warn_pct, warn_pct + 10], "color": "rgba(255, 176, 32, 0.2)"},
                        {"range": [warn_pct + 10, 100], "color": "rgba(0, 229, 160, 0.2)"},
                    ],
                    "threshold": {
                        "line": {"color": "#5b9cff", "width": 2},
                        "thickness": 0.75,
                        "value": warn_pct,
                    },
                },
            )
        )
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=50, b=10),
            font={"family": "JetBrains Mono, monospace", "color": "#c5d4e8"},
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except (ValueError, TypeError) as exc:
        st.error(f"Could not render gauge chart: {exc}")


def render_gauge_panel(authenticity_pct: float, warn_pct: float) -> None:
    with st.container(border=True):
        st.markdown(
            '<div class="gauge-panel-label">◈ Threat Assessment Index</div>',
            unsafe_allow_html=True,
        )
        render_authenticity_gauge(authenticity_pct, warn_pct)


def render_mel_spectrogram(y: np.ndarray, sr: int) -> None:
    try:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0a0e14")
        ax.set_facecolor("#0a0e14")
        librosa.display.specshow(
            mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
        )
        ax.set_title("Mel-spectrogram", color="#8ba3c7", fontsize=11, family="monospace")
        ax.tick_params(colors="#6b8299")
        fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Could not render mel-spectrogram: {exc}")


def display_gradcam_panel(cam: dict, *, frame_label: str | None = None) -> None:
    title = "◈ Grad-CAM — Neural Attention Map"
    if frame_label:
        title += f" ({frame_label})"
    st.markdown(f"**{title}**")
    st.caption(
        f"Target: **{cam['target_label']}** · P(real)={cam['prob_real']:.1%} · "
        f"P(fake)={cam['prob_fake']:.1%} · Focus: {', '.join(cam['focus_regions'])}"
    )
    col_orig, col_cam = st.columns(2)
    with col_orig:
        original = cam.get("original_bgr") if isinstance(cam, dict) else getattr(cam, "original_bgr", None)
        if original is not None:
            try:
                import numpy as np
                safe_orig = original
                if hasattr(safe_orig, 'detach'):
                    safe_orig = safe_orig.detach().cpu().numpy()
                if isinstance(safe_orig, np.ndarray):
                    safe_orig = np.squeeze(safe_orig)
                    if safe_orig.dtype in [np.float32, np.float64]:
                        safe_orig = (np.clip(safe_orig, 0, 1) * 255).astype(np.uint8)
                st.image(safe_orig, channels="BGR", caption="Source frame", use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM source error: {e}")
        else:
            st.info("No source frame available for heatmap overlay.")
            
    with col_cam:
        heatmap_overlay = cam.get("overlay") if isinstance(cam, dict) else getattr(cam, "overlay", None)
        if heatmap_overlay is not None:
            try:
                import numpy as np
                safe_heat = heatmap_overlay
                if hasattr(safe_heat, 'detach'):
                    safe_heat = safe_heat.detach().cpu().numpy()
                if isinstance(safe_heat, np.ndarray):
                    safe_heat = np.squeeze(safe_heat)
                    if safe_heat.dtype in [np.float32, np.float64]:
                        safe_heat = (np.clip(safe_heat, 0, 1) * 255).astype(np.uint8)
                st.image(safe_heat, channels="BGR", caption="Grad-CAM Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM overlay error: {e}")


def _render_verdict_banner(authenticity_pct: float, warn_pct: float) -> None:
    if authenticity_pct < warn_pct:
        st.markdown(
            '<p class="verdict-banner">Classification: '
            '<span class="label-suspicious">Suspicious</span></p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<p class="verdict-banner">Classification: '
            '<span class="label-authentic">Authentic</span></p>',
            unsafe_allow_html=True,
        )


def _profile_caption(results: dict[str, Any]) -> str:
    profile_cfg = get_sensor_profile_config(results.get("sensor_profile"))
    label = results.get("sensor_profile_label") or get_profile_display_name(
        results.get("sensor_profile")
    )
    text = (
        f"Profile: **{label}** · "
        f"Synthetic gate: raw P(fake) > {profile_cfg['sigmoid_threshold']:.0%} · "
        f"Sensitivity: **{results.get('sensitivity', '—')}%**"
    )
    if results.get("denoising_enabled") and results.get("media_type") in {"image", "video"}:
        text += " · **Denoising:** active"
    return text


def render_metadata_explorer(
    report: MetadataReport | None,
    file_path: str | None = None,
) -> None:
    st.markdown("##### Metadata Explorer")
    st.caption("EXIF · container tags · filesystem timestamps")

    if report is None and file_path:
        try:
            report = extract_file_metadata(Path(file_path))
        except (ForensicsError, ProcessingError) as exc:
            st.error(f"Metadata extraction failed: {exc}")
            return

    if report is None:
        st.info("No metadata available yet.")
        return

    if report.red_flag_summary:
        with st.container(border=True):
            st.markdown("**Forensic red flags**")
            for flag in report.red_flag_summary:
                st.markdown(f"- {flag}")

    df = pd.DataFrame(report.to_table_records())

    def _highlight_red_flags(row: pd.Series) -> list[str]:
        if row.get("Forensic Flag") == "Yes":
            return [
                "background-color: rgba(255, 77, 106, 0.22); color: #ff8fa3; font-weight: 500"
            ] * len(row)
        return [""] * len(row)

    try:
        st.dataframe(
            df.style.apply(_highlight_red_flags, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    except (ValueError, TypeError):
        st.dataframe(df, use_container_width=True, hide_index=True)

    red_count = int((df["Forensic Flag"] == "Yes").sum())
    st.caption(
        f"**{report.file_name}** · {report.media_type.upper()} · "
        f"{red_count} flagged field(s)"
    )


def render_media_analysis_tab(results: dict[str, Any]) -> None:
    authenticity_pct = results["authenticity_pct"]
    warn_pct = float(results.get("warn_pct", 45.0))
    media_type = results["media_type"]

    st.markdown(f"##### Target: `{results['uploaded_name']}`")
    st.caption(_profile_caption(results))
    render_gauge_panel(authenticity_pct, warn_pct)
    _render_verdict_banner(authenticity_pct, warn_pct)

    if media_type == "image":
        if results.get("image_bytes") is not None:
            media_data = results.get("image_bytes")
            if media_data is not None:
                try:
                    import numpy as np
                    safe_media = media_data
                    if hasattr(safe_media, 'detach'):
                        safe_media = safe_media.detach().cpu().numpy()
                    if isinstance(safe_media, np.ndarray):
                        safe_media = np.squeeze(safe_media)
                        if safe_media.dtype in [np.float32, np.float64]:
                            safe_media = (np.clip(safe_media, 0, 1) * 255).astype(np.uint8)
                    st.image(safe_media, caption="Analyzed Media", use_container_width=True)
                except Exception as e:
                    st.error(f"Media format error: {e}")
        else:
            st.info("Visual preview is not available for this file type.")
            
        if results.get("gradcam_meta"):
            display_gradcam_panel(results["gradcam_meta"])
            
    elif media_type == "video":
        frame_scores = results.get("frame_scores") or []
        if frame_scores:
            st.caption(
                f"Scored **{len(frame_scores)}** frames (1/sec) · "
                f"mean={float(np.mean(frame_scores)):.1f}% · "
                f"min={min(frame_scores):.1f}%"
            )
        if results.get("gradcam_meta"):
            display_gradcam_panel(
                results["gradcam_meta"],
                frame_label=f"Key frame ~{results.get('key_frame_index', 0)}s",
            )
    elif media_type == "audio":
        st.info("Open **Spectrum Evidence** for the mel-spectrogram.")

    with st.expander("Raw analysis payload"):
        details = dict(results.get("analysis_details", {}))
        details["detection_factors"] = results.get("detection_factors", [])
        st.json(details)


def render_spectrum_evidence_tab(results: dict[str, Any]) -> None:
    media_type = results["media_type"]
