"""Rule-based severity assessment from detection geometry and damage types."""

from vision_rag_claims.schemas import Detection, SeverityLevel

# Types that imply structural or safety risk
_STRUCTURAL_TYPES = {"glass shatter", "crack"}

# Per-type weight multiplier for weighted coverage calculation
_TYPE_WEIGHT: dict[str, float] = {
    "glass shatter": 2.0,
    "crack":         1.5,
    "tire flat":     1.3,
    "lamp broken":   1.2,
    "dent":          1.0,
    "scratch":       0.8,
}


def _level_up(level: SeverityLevel) -> SeverityLevel:
    return SeverityLevel.moderate if level is SeverityLevel.minor else SeverityLevel.severe


def assess_severity(
    detections: list[Detection],
    image_height: int,
    image_width: int,
) -> SeverityLevel:
    """
    Return a SeverityLevel for the given detections.

    Rules (first match wins on each axis):
    - weighted_coverage = Σ(mask_area_ratio × type_weight)
    - < 0.02 → MINOR | 0.02–0.10 → MODERATE | > 0.10 → SEVERE
    - Structural types (glass shatter, crack) ensure at least MODERATE
    - ≥ 3 detections bump the result up one level
    """
    if not detections:
        return SeverityLevel.minor

    image_area = image_height * image_width
    has_structural = False
    weighted_coverage = 0.0

    for det in detections:
        ratio = det.mask_area_pixels / image_area
        weight = _TYPE_WEIGHT.get(det.damage_type.lower(), 1.0)
        weighted_coverage += ratio * weight
        if det.damage_type.lower() in _STRUCTURAL_TYPES:
            has_structural = True

    if weighted_coverage < 0.02:
        level = SeverityLevel.minor
    elif weighted_coverage <= 0.10:
        level = SeverityLevel.moderate
    else:
        level = SeverityLevel.severe

    if has_structural and level is SeverityLevel.minor:
        level = SeverityLevel.moderate

    if len(detections) >= 3:
        level = _level_up(level)

    return level
