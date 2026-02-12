"""
Catalog validation for trajmod.

Validates earthquake and SSE catalogs to ensure compatibility with TrajectoryModel.
"""

import datetime
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class CatalogValidationError(Exception):
    """Raised when catalog validation fails."""
    pass


def validate_earthquake_catalog(
    catalog: List[Dict],
    strict: bool = True,
    fix: bool = False
) -> Tuple[bool, List[str], Optional[List[Dict]]]:
    """Validate earthquake catalog format.

    Args:
        catalog: List of earthquake event dictionaries
        strict: If True, enforce all requirements strictly
        fix: If True, attempt to fix common issues

    Returns:
        Tuple of (is_valid, error_messages, fixed_catalog)
        - is_valid: True if catalog passes validation
        - error_messages: List of validation errors/warnings
        - fixed_catalog: Fixed catalog if fix=True, else None

    Example:
        >>> is_valid, errors, fixed = validate_earthquake_catalog(catalog)
        >>> if not is_valid:
        ...     print("Errors:", errors)
        ...     if fixed is not None:
        ...         catalog = fixed  # Use fixed version
    """
    errors = []
    warnings = []
    fixed_catalog = [] if fix else None

    if not isinstance(catalog, list):
        errors.append(f"Catalog must be a list, got {type(catalog).__name__}")
        return False, errors, None

    if len(catalog) == 0:
        warnings.append("Empty catalog")
        return True, warnings, []

    for i, event in enumerate(catalog):
        if not isinstance(event, dict):
            errors.append(f"Event {i}: must be a dict, got {type(event).__name__}")
            continue

        # Make a copy for fixing
        fixed_event = event.copy() if fix else None

        # Check required fields
        time_field, time_error = _validate_earthquake_time(event, i, strict, fixed_event)
        if time_error:
            errors.append(time_error)
            if not fix:
                continue

        lat_error = _validate_latitude(event, i, strict, fixed_event)
        if lat_error:
            errors.append(lat_error)
            if not fix or strict:
                continue

        lon_error = _validate_longitude(event, i, strict, fixed_event)
        if lon_error:
            errors.append(lon_error)
            if not fix or strict:
                continue

        mag_error = _validate_magnitude(event, i, strict, fixed_event)
        if mag_error:
            errors.append(mag_error)
            if not fix or strict:
                continue

        # Check optional fields
        if "depth" in event:
            depth_warning = _validate_depth(event, i, fixed_event)
            if depth_warning:
                warnings.append(depth_warning)

        # Add to fixed catalog
        if fix and fixed_event is not None:
            fixed_catalog.append(fixed_event)

    is_valid = len(errors) == 0
    all_messages = errors + warnings

    if is_valid and warnings:
        logger.info(f"Catalog valid with {len(warnings)} warnings")
    elif not is_valid:
        logger.error(f"Catalog invalid: {len(errors)} errors")

    return is_valid, all_messages, fixed_catalog


def validate_sse_catalog(
    catalog: List[Dict],
    strict: bool = True,
    fix: bool = False
) -> Tuple[bool, List[str], Optional[List[Dict]]]:
    """Validate SSE catalog format.

    Args:
        catalog: List of SSE event dictionaries
        strict: If True, enforce all requirements strictly
        fix: If True, attempt to fix common issues

    Returns:
        Tuple of (is_valid, error_messages, fixed_catalog)

    Example:
        >>> is_valid, errors, fixed = validate_sse_catalog(sse_catalog)
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(error)
    """
    errors = []
    warnings = []
    fixed_catalog = [] if fix else None

    if not isinstance(catalog, list):
        errors.append(f"Catalog must be a list, got {type(catalog).__name__}")
        return False, errors, None

    if len(catalog) == 0:
        warnings.append("Empty catalog")
        return True, warnings, []

    for i, event in enumerate(catalog):
        if not isinstance(event, dict):
            errors.append(f"Event {i}: must be a dict, got {type(event).__name__}")
            continue

        # Make a copy for fixing
        fixed_event = event.copy() if fix else None

        # Check required fields
        start_field, start_error = _validate_sse_start(event, i, strict, fixed_event)
        if start_error:
            errors.append(start_error)
            if not fix:
                continue

        end_field, end_error = _validate_sse_end(event, i, strict, fixed_event)
        if end_error:
            errors.append(end_error)
            if not fix:
                continue

        # Check temporal consistency
        if start_field and end_field:
            start_time = event.get(start_field)
            end_time = event.get(end_field)
            if start_time and end_time:
                if end_time <= start_time:
                    errors.append(f"Event {i}: end time ({end_time}) must be "
                                f"after start time ({start_time})")
                    if not fix or strict:
                        continue

        lat_error = _validate_latitude(event, i, strict, fixed_event)
        if lat_error:
            errors.append(lat_error)
            if not fix or strict:
                continue

        lon_error = _validate_longitude(event, i, strict, fixed_event)
        if lon_error:
            errors.append(lon_error)
            if not fix or strict:
                continue

        mag_error = _validate_magnitude(event, i, strict, fixed_event)
        if mag_error:
            errors.append(mag_error)
            if not fix or strict:
                continue

        # Add to fixed catalog
        if fix and fixed_event is not None:
            fixed_catalog.append(fixed_event)

    is_valid = len(errors) == 0
    all_messages = errors + warnings

    if is_valid and warnings:
        logger.info(f"SSE catalog valid with {len(warnings)} warnings")
    elif not is_valid:
        logger.error(f"SSE catalog invalid: {len(errors)} errors")

    return is_valid, all_messages, fixed_catalog


# ============================================================================
# Helper Functions
# ============================================================================

def _validate_earthquake_time(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Validate earthquake time field.

    Returns:
        (field_name, error_message) or (field_name, None) if valid
    """
    # Try common field names
    time_fields = ["time", "date", "datetime", "eq_day"]
    found_field = None

    for field in time_fields:
        if field in event:
            found_field = field
            break

    if not found_field:
        return None, f"Event {idx}: missing time field (expected: {time_fields})"

    time_value = event[found_field]

    # Check if None
    if time_value is None:
        return found_field, f"Event {idx}: time field '{found_field}' is None"

    # Check type
    if isinstance(time_value, str):
        error = f"Event {idx}: time field '{found_field}' is str, expected datetime.datetime"
        if fixed_event is not None:
            # Try to parse
            try:
                fixed_event[found_field] = datetime.datetime.fromisoformat(
                    time_value.replace('Z', '+00:00')
                )
                return found_field, None  # Fixed!
            except:
                return found_field, error + " (could not parse)"
        return found_field, error

    if isinstance(time_value, datetime.date) and not isinstance(time_value, datetime.datetime):
        error = f"Event {idx}: time field '{found_field}' is datetime.date, expected datetime.datetime"
        if fixed_event is not None:
            fixed_event[found_field] = datetime.datetime.combine(
                time_value, datetime.time()
            )
            return found_field, None  # Fixed!
        return found_field, error

    if not isinstance(time_value, datetime.datetime):
        return found_field, f"Event {idx}: time field '{found_field}' has invalid type {type(time_value)}"

    return found_field, None  # Valid!


def _validate_sse_start(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Validate SSE start time."""
    start_fields = ["start", "start_time", "start_day"]
    found_field = None

    for field in start_fields:
        if field in event:
            found_field = field
            break

    if not found_field:
        return None, f"Event {idx}: missing start time field (expected: {start_fields})"

    start_value = event[found_field]

    if start_value is None:
        return found_field, f"Event {idx}: start field '{found_field}' is None"

    if isinstance(start_value, str):
        error = f"Event {idx}: start field '{found_field}' is str, expected datetime"
        if fixed_event is not None:
            try:
                fixed_event[found_field] = datetime.datetime.fromisoformat(
                    start_value.replace('Z', '+00:00')
                )
                return found_field, None
            except:
                return found_field, error + " (could not parse)"
        return found_field, error

    if isinstance(start_value, datetime.date) and not isinstance(start_value, datetime.datetime):
        error = f"Event {idx}: start is datetime.date, expected datetime.datetime"
        if fixed_event is not None:
            fixed_event[found_field] = datetime.datetime.combine(start_value, datetime.time())
            return found_field, None
        return found_field, error

    if not isinstance(start_value, datetime.datetime):
        return found_field, f"Event {idx}: start has invalid type {type(start_value)}"

    return found_field, None


def _validate_sse_end(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Validate SSE end time."""
    end_fields = ["end", "end_time", "end_day"]
    found_field = None

    for field in end_fields:
        if field in event:
            found_field = field
            break

    if not found_field:
        return None, f"Event {idx}: missing end time field (expected: {end_fields})"

    end_value = event[found_field]

    if end_value is None:
        return found_field, f"Event {idx}: end field '{found_field}' is None"

    if isinstance(end_value, str):
        error = f"Event {idx}: end field '{found_field}' is str, expected datetime"
        if fixed_event is not None:
            try:
                fixed_event[found_field] = datetime.datetime.fromisoformat(
                    end_value.replace('Z', '+00:00')
                )
                return found_field, None
            except:
                return found_field, error + " (could not parse)"
        return found_field, error

    if isinstance(end_value, datetime.date) and not isinstance(end_value, datetime.datetime):
        error = f"Event {idx}: end is datetime.date, expected datetime.datetime"
        if fixed_event is not None:
            fixed_event[found_field] = datetime.datetime.combine(end_value, datetime.time())
            return found_field, None
        return found_field, error

    if not isinstance(end_value, datetime.datetime):
        return found_field, f"Event {idx}: end has invalid type {type(end_value)}"

    return found_field, None


def _validate_latitude(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Optional[str]:
    """Validate latitude field."""
    if "lat" not in event and "latitude" not in event:
        return f"Event {idx}: missing 'lat' field"

    # Handle both 'lat' and 'latitude'
    lat_field = "lat" if "lat" in event else "latitude"
    lat = event[lat_field]

    if lat is None:
        return f"Event {idx}: '{lat_field}' is None"

    try:
        lat_float = float(lat)
    except (TypeError, ValueError):
        return f"Event {idx}: '{lat_field}' cannot be converted to float: {lat}"

    if not -90 <= lat_float <= 90:
        return f"Event {idx}: '{lat_field}' out of range [-90, 90]: {lat_float}"

    # Normalize to 'lat'
    if fixed_event is not None and lat_field == "latitude":
        fixed_event["lat"] = lat_float
        if "latitude" in fixed_event:
            del fixed_event["latitude"]

    return None


def _validate_longitude(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Optional[str]:
    """Validate longitude field."""
    if "lon" not in event and "longitude" not in event:
        return f"Event {idx}: missing 'lon' field"

    # Handle both 'lon' and 'longitude'
    lon_field = "lon" if "lon" in event else "longitude"
    lon = event[lon_field]

    if lon is None:
        return f"Event {idx}: '{lon_field}' is None"

    try:
        lon_float = float(lon)
    except (TypeError, ValueError):
        return f"Event {idx}: '{lon_field}' cannot be converted to float: {lon}"

    # Allow both -180 to 180 and 0 to 360
    if not (-180 <= lon_float <= 360):
        return f"Event {idx}: '{lon_field}' out of range [-180, 360]: {lon_float}"

    # Normalize to 'lon'
    if fixed_event is not None and lon_field == "longitude":
        fixed_event["lon"] = lon_float
        if "longitude" in fixed_event:
            del fixed_event["longitude"]

    return None


def _validate_magnitude(
    event: Dict,
    idx: int,
    strict: bool,
    fixed_event: Optional[Dict] = None
) -> Optional[str]:
    """Validate magnitude field."""
    if "magnitude" not in event and "mag" not in event:
        return f"Event {idx}: missing 'magnitude' field"

    # Handle both 'magnitude' and 'mag'
    mag_field = "magnitude" if "magnitude" in event else "mag"
    mag = event[mag_field]

    if mag is None:
        return f"Event {idx}: '{mag_field}' is None"

    try:
        mag_float = float(mag)
    except (TypeError, ValueError):
        return f"Event {idx}: '{mag_field}' cannot be converted to float: {mag}"

    # Reasonable range check
    if not -2 <= mag_float <= 10:
        return f"Event {idx}: '{mag_field}' unusual value: {mag_float}"

    # Normalize to 'magnitude'
    if fixed_event is not None and mag_field == "mag":
        fixed_event["magnitude"] = mag_float
        if "mag" in fixed_event:
            del fixed_event["mag"]

    return None


def _validate_depth(
    event: Dict,
    idx: int,
    fixed_event: Optional[Dict] = None
) -> Optional[str]:
    """Validate optional depth field."""
    depth = event.get("depth")

    if depth is None:
        return None

    try:
        depth_float = float(depth)
    except (TypeError, ValueError):
        return f"Event {idx}: 'depth' cannot be converted to float: {depth}"

    # Typical range check
    if not 0 <= depth_float <= 1000:
        return f"Event {idx}: unusual 'depth' value: {depth_float} km"

    if fixed_event is not None:
        fixed_event["depth"] = depth_float

    return None


# ============================================================================
# Validation Summary
# ============================================================================

def print_validation_summary(
    catalog: List[Dict],
    catalog_type: str = "earthquake"
) -> None:
    """Print a summary of catalog validation.

    Args:
        catalog: Event catalog to validate
        catalog_type: "earthquake" or "sse"
    """
    print("=" * 80)
    print(f"{catalog_type.upper()} CATALOG VALIDATION SUMMARY")
    print("=" * 80)

    if catalog_type == "earthquake":
        is_valid, messages, _ = validate_earthquake_catalog(catalog, strict=False)
    else:
        is_valid, messages, _ = validate_sse_catalog(catalog, strict=False)

    print(f"\nTotal events: {len(catalog)}")
    print(f"Status: {'✅ VALID' if is_valid else '❌ INVALID'}")

    if messages:
        print(f"\nMessages ({len(messages)}):")
        for msg in messages:
            print(f"  {msg}")
    else:
        print("\n✅ No issues found!")

    print("=" * 80)