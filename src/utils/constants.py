"""Shared model class indices and labels (no torch / model imports)."""

CLASS_REAL = 0
CLASS_FAKE = 1

CLASS_LABELS: dict[int, str] = {
    CLASS_REAL: "authentic",
    CLASS_FAKE: "fake / synthetic",
}
