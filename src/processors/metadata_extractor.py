"""Extract file-system, EXIF, and container metadata for forensic review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image

from src.utils.errors import MediaLoadError, ProcessingError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

# Standard EXIF fields expected from in-camera JPEG exports
JPEG_EXIF_EXPECTED = (
    "Camera Make",
    "Camera Model",
    "Date Taken (EXIF)",
    "Software",
    "Resolution",
)


@dataclass
class MetadataRow:
    field: str
    value: str
    forensic_red_flag: bool = False
    notes: str = ""


@dataclass
class MetadataReport:
    file_name: str
    media_type: str
    rows: list[MetadataRow] = field(default_factory=list)
    red_flag_summary: list[str] = field(default_factory=list)

    def to_table_records(self) -> list[dict[str, str]]:
        return [
            {
                "Field": row.field,
                "Value": row.value,
                "Forensic Flag": "Yes" if row.forensic_red_flag else "—",
                "Notes": row.notes,
            }
            for row in self.rows
        ]

    def get_field_value(self, field_name: str) -> str | None:
        """Return a metadata field value, or None if missing."""
        for row in self.rows:
            if row.field == field_name:
                if row.value and row.value.strip() and row.value.strip() != "—":
                    return row.value.strip()
        return None

    @property
    def camera_make(self) -> str | None:
        return self.get_field_value("Camera Make")

    @property
    def camera_model(self) -> str | None:
        return self.get_field_value("Camera Model")


def _fmt_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.2f} MB"


def _fmt_timestamp(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except (OSError, ValueError, OverflowError):
        return "Unavailable"


def _row(
    name: str,
    value: str | None,
    *,
    red_flag: bool = False,
    notes: str = "",
) -> MetadataRow:
    display = value.strip() if value and str(value).strip() else "—"
    is_missing = display == "—"
    return MetadataRow(
        field=name,
        value=display,
        forensic_red_flag=red_flag or is_missing,
        notes=notes or ("Missing standard EXIF field" if is_missing and red_flag else notes),
    )


def _read_exifread_tags(path: Path) -> dict[str, str]:
    try:
        import exifread
    except ImportError as exc:
        raise ProcessingError(
            "exifread is not installed; add it to requirements.txt", str(exc)
        ) from exc

    tags: dict[str, str] = {}
    try:
        with path.open("rb") as handle:
            raw = exifread.process_file(handle, details=False)
        for key, tag in raw.items():
            if key in ("JPEGThumbnail", "TIFFThumbnail", "Filename", "EXIF MakerNote"):
                continue
            tags[key] = str(tag).strip()
    except OSError as exc:
        raise MediaLoadError("Could not read EXIF data", str(exc)) from exc
    return tags


def _image_resolution(path: Path) -> str | None:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return f"{w} × {h}"
    except (OSError, ValueError):
        return None


def _extract_image_metadata(path: Path, report: MetadataReport) -> None:
    suffix = path.suffix.lower()
    exif_tags: dict[str, str] = {}

    if suffix in {".jpg", ".jpeg", ".tif", ".tiff"}:
        try:
            exif_tags = _read_exifread_tags(path)
        except (MediaLoadError, ProcessingError):
            exif_tags = {}

    camera_make = exif_tags.get("Image Make")
    camera_model = exif_tags.get("Image Model")
    camera = " ".join(p for p in (camera_make, camera_model) if p) or None

    date_taken = (
        exif_tags.get("EXIF DateTimeOriginal")
        or exif_tags.get("Image DateTime")
        or exif_tags.get("EXIF DateTimeDigitized")
    )
    software = exif_tags.get("Image Software")
    resolution = None
    if exif_tags.get("EXIF ExifImageWidth") and exif_tags.get("EXIF ExifImageLength"):
        resolution = f"{exif_tags['EXIF ExifImageWidth']} × {exif_tags['EXIF ExifImageLength']}"
    else:
        resolution = _image_resolution(path)

    is_jpeg = suffix in {".jpg", ".jpeg"}
    report.rows.extend(
        [
            _row("Resolution", resolution, red_flag=is_jpeg),
            _row("Camera Make", camera_make, red_flag=is_jpeg),
            _row("Camera Model", camera_model, red_flag=is_jpeg),
            _row("Camera (Combined)", camera, red_flag=is_jpeg and not camera),
            _row("Date Taken (EXIF)", date_taken, red_flag=is_jpeg),
            _row("Software Used", software, red_flag=is_jpeg),
            _row("EXIF Tag Count", str(len(exif_tags)) if exif_tags else None, red_flag=is_jpeg),
        ]
    )

    if is_jpeg and len(exif_tags) == 0:
        report.red_flag_summary.append(
            "No EXIF block found — file may have been re-encoded or stripped."
        )
    elif is_jpeg:
        for label in JPEG_EXIF_EXPECTED:
            matching = next((r for r in report.rows if r.field == label or label in r.field), None)
            if matching and matching.value == "—":
                report.red_flag_summary.append(f"Missing standard EXIF: {label}")


def _extract_mutagen_audio(path: Path) -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "bitrate": None,
        "duration": None,
        "codec": None,
        "sample_rate": None,
        "channels": None,
        "title": None,
        "artist": None,
        "encoder": None,
    }
    suffix = path.suffix.lower()

    try:
        if suffix == ".mp3":
            from mutagen.mp3 import MP3

            audio = MP3(path)
        elif suffix == ".wav":
            from mutagen.wave import WAVE

            audio = WAVE(path)
        elif suffix in {".flac", ".ogg", ".m4a", ".aac"}:
            from mutagen import File

            audio = File(path)
        else:
            return info

        if audio is None:
            return info

        if getattr(audio, "info", None):
            if getattr(audio.info, "bitrate", None):
                info["bitrate"] = f"{int(audio.info.bitrate / 1000)} kbps"
            if getattr(audio.info, "length", None):
                info["duration"] = f"{audio.info.length:.2f} s"
            if getattr(audio.info, "sample_rate", None):
                info["sample_rate"] = f"{int(audio.info.sample_rate)} Hz"
            if getattr(audio.info, "channels", None):
                info["channels"] = str(audio.info.channels)

        tags = getattr(audio, "tags", None)
        if tags:
            info["title"] = _first_tag(tags, ["TIT2", "title", "\xa9nam"])
            info["artist"] = _first_tag(tags, ["TPE1", "artist", "\xa9ART"])
            info["encoder"] = _first_tag(tags, ["TSSE", "encoder", "\xa9too"])
    except Exception:
        pass

    return info


def _first_tag(tags, keys: list[str]) -> str | None:
    for key in keys:
        if key in tags:
            val = tags[key]
            if hasattr(val, "text"):
                return str(val.text[0]) if val.text else None
            return str(val)
    return None


def _extract_mutagen_mp4(path: Path) -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "bitrate": None,
        "duration": None,
        "codec": None,
        "resolution": None,
        "software": None,
        "creation_date": None,
    }
    try:
        from mutagen.mp4 import MP4

        video = MP4(path)
        if video.info:
            if video.info.bitrate:
                info["bitrate"] = f"{int(video.info.bitrate / 1000)} kbps"
            if video.info.length:
                info["duration"] = f"{video.info.length:.2f} s"
        if video.tags:
            info["software"] = _first_tag(video.tags, ["\xa9too", "\xa9swr"])
            info["creation_date"] = _first_tag(video.tags, ["\xa9day", "\xa9cmt"])
    except Exception:
        pass

    try:
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if w > 0 and h > 0:
                info["resolution"] = f"{w} × {h}"
            if fps and fps > 0 and not info["duration"]:
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames > 0:
                    info["duration"] = f"{frames / fps:.2f} s"
        cap.release()
    except cv2.error:
        pass

    return info


def _extract_audio_metadata(path: Path, report: MetadataReport) -> None:
    meta = _extract_mutagen_audio(path)
    report.rows.extend(
        [
            _row("Audio Bitrate", meta.get("bitrate")),
            _row("Duration", meta.get("duration")),
            _row("Sample Rate", meta.get("sample_rate")),
            _row("Channels", meta.get("channels")),
            _row("Title", meta.get("title")),
            _row("Artist", meta.get("artist")),
            _row("Encoder / Software", meta.get("encoder"), red_flag=True),
        ]
    )
    if not meta.get("encoder") and not meta.get("bitrate"):
        report.red_flag_summary.append(
            "Sparse audio container tags — possible export from editing software."
        )


def _extract_video_metadata(path: Path, report: MetadataReport) -> None:
    meta = _extract_mutagen_mp4(path)
    report.rows.extend(
        [
            _row("Resolution", meta.get("resolution"), red_flag=True),
            _row("Duration", meta.get("duration")),
            _row("Video / Audio Bitrate", meta.get("bitrate")),
            _row("Container Software", meta.get("software"), red_flag=True),
            _row("Creation Date (container)", meta.get("creation_date"), red_flag=True),
        ]
    )
    if not meta.get("software") and not meta.get("creation_date"):
        report.red_flag_summary.append(
            "Missing container metadata (software / creation date) — common after re-encoding."
        )


def extract_file_metadata(file_path: str | Path) -> MetadataReport:
    """
    Extract forensic metadata from an uploaded media file.

    Uses exifread for images and mutagen (+ os.stat / OpenCV) for audio/video.
    """
    try:
        path = Path(file_path)
        if not path.is_file():
            raise MediaLoadError("File not found for metadata extraction", str(path))

        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            media_type = "image"
        elif suffix in VIDEO_SUFFIXES:
            media_type = "video"
        elif suffix in AUDIO_SUFFIXES:
            media_type = "audio"
        else:
            media_type = "unknown"

        stat = path.stat()
        report = MetadataReport(file_name=path.name, media_type=media_type)

        report.rows.extend(
            [
                _row("File Name", path.name),
                _row("File Size", _fmt_bytes(stat.st_size)),
                _row("Creation Date (OS)", _fmt_timestamp(stat.st_ctime)),
                _row("Modified Date (OS)", _fmt_timestamp(stat.st_mtime)),
                _row("Media Type", media_type.upper()),
            ]
        )

        if media_type == "image":
            _extract_image_metadata(path, report)
        elif media_type == "video":
            _extract_video_metadata(path, report)
        elif media_type == "audio":
            _extract_audio_metadata(path, report)
        else:
            raise ProcessingError("Unsupported file type for metadata", suffix)

        for row in report.rows:
            if row.forensic_red_flag and row.notes and row.notes not in report.red_flag_summary:
                report.red_flag_summary.append(f"{row.field}: {row.notes}")

        return report
    except (MediaLoadError, ProcessingError):
        raise
    except OSError as exc:
        raise MediaLoadError("Metadata extraction failed", str(exc)) from exc
