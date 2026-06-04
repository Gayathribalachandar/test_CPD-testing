import math
import re

from PySide6.QtGui import QValidator
from PySide6.QtWidgets import QDoubleSpinBox as QtDoubleSpinBox
from PySide6.QtWidgets import QSpinBox as QtSpinBox


_SCIENTIFIC_FULL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?$"
)
_SCIENTIFIC_PARTIAL_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d*)|(?:\d*))?(?:[eE][+-]?\d*)?$"
)


def _strip_affixes(widget, text):
    cleaned = str(text or "").strip()
    try:
        prefix = str(widget.prefix() or "")
    except Exception:
        prefix = ""
    try:
        suffix = str(widget.suffix() or "")
    except Exception:
        suffix = ""
    if prefix and cleaned.startswith(prefix):
        cleaned = cleaned[len(prefix):]
    if suffix and cleaned.endswith(suffix):
        cleaned = cleaned[:-len(suffix)]
    return cleaned.strip().replace(",", "")


def _parse_scientific(text):
    cleaned = str(text or "").strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def parse_numeric_text(text):
    return _parse_scientific(text)


def is_numeric_text(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return _parse_scientific(str(value or "").strip()) is not None


def _validate_scientific_text(widget, text):
    cleaned = _strip_affixes(widget, text)
    if cleaned == "":
        return QValidator.Intermediate
    if _SCIENTIFIC_FULL_RE.fullmatch(cleaned):
        value = _parse_scientific(cleaned)
        if value is not None:
            return QValidator.Acceptable
    if _SCIENTIFIC_PARTIAL_RE.fullmatch(cleaned):
        return QValidator.Intermediate
    return QValidator.Invalid


class ScientificDoubleSpinBox(QtDoubleSpinBox):
    """QDoubleSpinBox that accepts decimal and scientific notation."""

    def validate(self, text, pos):
        return (_validate_scientific_text(self, text), text, pos)

    def valueFromText(self, text):
        value = _parse_scientific(_strip_affixes(self, text))
        if value is None:
            return super().valueFromText(text)
        minimum = float(self.minimum())
        maximum = float(self.maximum())
        return max(minimum, min(maximum, float(value)))


class ScientificSpinBox(QtSpinBox):
    """QSpinBox that accepts integer, decimal, and scientific notation."""

    def validate(self, text, pos):
        return (_validate_scientific_text(self, text), text, pos)

    def valueFromText(self, text):
        value = _parse_scientific(_strip_affixes(self, text))
        if value is None:
            return super().valueFromText(text)
        value = int(round(float(value)))
        minimum = int(self.minimum())
        maximum = int(self.maximum())
        return max(minimum, min(maximum, value))
