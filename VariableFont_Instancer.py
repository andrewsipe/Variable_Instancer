#!/usr/bin/env python3
"""
Variable Font Instancer - Extract static instances from variable fonts

Naming Philosophy
=================
This script supports three naming strategies:

1. STAT-based (default): Uses STAT table AxisValue names
   - Most reliable for modern VFs
   - Respects designer intent

2. fvar-based: Uses fvar instance names
   - Legacy compatibility
   - May have inconsistent naming

3. Hybrid: fvar names with STAT-derived completions
   - Fills missing "Regular" tokens when appropriate
   - Family-aware decisions

Usage:
  python script.py fontfile.ttf              # Generate named instances (default)
  python script.py font1.ttf font2.ttf       # Batch process multiple fonts
  python script.py fonts/                    # Process directory
  python script.py fontfile.ttf --custom     # Create custom instances
  python script.py fontfile.ttf --info       # View font information
  python script.py fontfile.ttf --auto --naming stat    # Auto-generate (STAT)
  python script.py fontfile.ttf --auto --naming fvar    # Auto-generate (fvar hybrid)
  python script.py fontfile.ttf -y   # Skips redundant fvar rows (same output identity) by default
  python script.py fontfile.ttf -y --all-fvar-instance-rows   # Emit every fvar row
"""

import argparse
import json
import re
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from typing import Optional, List, Dict, Tuple, Union, Literal, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Add project root to path for FontCore imports (works for root and subdirectory scripts)
# ruff: noqa: E402
_project_root = Path(__file__).parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from FontCore.core_name_policies import (
    sanitize_postscript,
    strip_variable_tokens,
    normalize_fvar_name,
)
from FontCore.core_gpos_repair import repair_pairpos_second_glyph_order
from FontCore.core_ttx_table_io import deduplicate_namerecords_binary
from FontCore.core_file_collector import collect_font_files
from FontCore.core_error_handling import ErrorTracker, ErrorContext
import FontCore.core_console_styles as cs
from FontCore.core_console_styles import StatusIndicator

logger = cs.get_logger(__name__)
console = cs.get_console()
RICH_AVAILABLE = cs.RICH_AVAILABLE


def _emit_dim(text: str) -> None:
    """Dim line via Rich markup (cs.emit has no style= kwarg)."""
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(f"[dim]{escape(text)}[/dim]")
    else:
        cs.emit(text)


def _emit_menu(text: str) -> None:
    """Menu / prompt line at normal contrast (readable on dark terminals)."""
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(escape(text))
    else:
        cs.emit(text)


def _emit_menu_row(
    label: str, body: str, *, label_col: int = 16, dim_hint: str = ""
) -> None:
    """One menu line: bold label, body starts at a fixed column (aligned)."""
    pad_len = max(1, label_col - len(label))
    pad = " " * pad_len
    if RICH_AVAILABLE:
        from rich.markup import escape

        hint = f" [dim]{escape(dim_hint)}[/dim]" if dim_hint else ""
        cs.emit(f"  [bold]{escape(label)}[/bold]{pad}{escape(body)}{hint}")
    else:
        hint = f" — {dim_hint}" if dim_hint else ""
        cs.emit(f"  {label}{pad}{body}{hint}")


def _raise_if_quit(line: str) -> None:
    """Exit the process if the user entered q / quit (any prompt)."""
    if line.strip().lower() in ("q", "quit"):
        raise SystemExit(0)


def _emit_bold(text: str) -> None:
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(f"[bold]{escape(text)}[/bold]")
    else:
        cs.emit(text)


_COORD_CANONICAL_ORDER = ["wdth", "wght", "slnt", "ital", "obli", "opsz"]


def _coord_sort_key(item: Tuple[str, float]) -> Tuple[int, Union[int, str]]:
    axis = item[0]
    try:
        return (0, _COORD_CANONICAL_ORDER.index(axis))
    except ValueError:
        return (1, axis)


def _sorted_coord_items(coords: Dict[str, float]) -> List[Tuple[str, float]]:
    return sorted(coords.items(), key=_coord_sort_key)


def _format_coord_part(tag: str, value: float) -> str:
    if value == int(value):
        return f"{tag}={int(value)}"
    return f"{tag}={value}"


@dataclass
class _PendingInstance:
    """A queued custom instance not yet generated."""

    coordinates: Dict[str, float]
    name: str

    def display(self) -> str:
        coord_str = "  ".join(
            _format_coord_part(k, v) for k, v in _sorted_coord_items(self.coordinates)
        )
        return f"{coord_str}   →  {self.name}"


# ============================================================================
# Configuration & Constants
# ============================================================================


class NamingMode(Enum):
    """Naming strategy for instance generation."""

    STAT = "stat"
    FVAR_HYBRID = "fvar-hybrid"
    FVAR_RAW = "fvar-raw"


class WeightClass:
    """Standard weight class values."""

    REGULAR = 400
    BOLD = 700

    @staticmethod
    def is_regular(weight: float) -> bool:
        return abs(weight - WeightClass.REGULAR) < 0.5

    @staticmethod
    def is_bold(weight: float) -> bool:
        return abs(weight - WeightClass.BOLD) < 0.5


# Magic value constants
UNKNOWN_FVAR_NAME = "Unknown"
AXIS_VALUE_EPSILON = 0.5
ITALIC_ANGLE_THRESHOLD = 0.1
ITALIC_ANGLE_MIN = 0.01


def instance_coordinate_key(
    coordinates: Dict[str, float], epsilon: float = AXIS_VALUE_EPSILON
) -> Tuple[Tuple[str, float], ...]:
    """Stable fingerprint for comparing fvar instances by axis coordinates (not names)."""
    if not coordinates:
        return ()

    def snap(v: float) -> float:
        return round(v / epsilon) * epsilon

    return tuple((tag, snap(float(val))) for tag, val in sorted(coordinates.items()))


def _collapse_name_key(name: str) -> str:
    """Normalize a resolved output name for duplicate comparison."""
    return re.sub(r"\s+", " ", name.strip()).casefold()


@dataclass
class InstancerConfig:
    """Configuration for instance processing."""

    output_dir: Optional[Path] = None
    keep_stat: bool = False
    naming_mode: NamingMode = NamingMode.STAT
    dry_run: bool = False
    skip_coordinate_duplicates: bool = True


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class AxisInfo:
    """Information about a font axis."""

    tag: str
    name: str
    min_value: float
    default_value: float
    max_value: float

    def format_range(self) -> str:
        """Format range for display (legacy method)."""
        return f"{self.min_value} → {self.default_value} → {self.max_value}"

    def format_variable_range(self) -> str:
        """Format range for variable axes (min — max)."""
        return f"{self.min_value} — {self.max_value}"

    def is_variable(self) -> bool:
        """Check if axis is variable (has a range) or fixed (single value)."""
        return abs(self.max_value - self.min_value) > 0.001

    def is_in_range(self, value: float) -> bool:
        return self.min_value <= value <= self.max_value


@dataclass
class InstanceInfo:
    """Information about a named instance."""

    index: int
    fvar_name: str
    stat_name: str
    coordinates: Dict[str, float]
    is_italic: bool
    is_bold: bool

    def format_coordinates(self) -> str:
        """Format coordinates in canonical order: wdth → wght → slnt/ital."""
        # Define canonical axis order
        CANONICAL_ORDER = ["wdth", "wght", "slnt", "ital", "obli", "opsz"]

        # Sort axes by canonical order, then alphabetically for unknown axes
        def axis_sort_key(item):
            axis = item[0]
            try:
                return (0, CANONICAL_ORDER.index(axis))
            except ValueError:
                return (1, axis)  # Unknown axes go last, alphabetically

        sorted_coords = sorted(self.coordinates.items(), key=axis_sort_key)
        return ", ".join([f"{axis}={value}" for axis, value in sorted_coords])

    def weight_value(self) -> float:
        return self.coordinates.get("wght", WeightClass.REGULAR)


@dataclass
class FontMetadata:
    """Complete font metadata from analysis."""

    axes: List[AxisInfo]
    instances: List[InstanceInfo]
    stat_values: Dict[str, Dict[float, str]]
    source_italic: bool
    family_name: str


# ============================================================================
# STAT Name Parser (Consolidated)
# ============================================================================


class STATNameParser:
    """Handles all STAT table name resolution logic."""

    def __init__(self, font: TTFont):
        self.font = font
        self.stat_values: Dict[str, Dict[float, str]] = {}
        self.index_to_tag: Dict[int, str] = {}
        self._parse_stat()

    def _parse_stat(self) -> None:
        """Extract STAT values and build index mappings."""
        if "STAT" not in self.font:
            return

        stat = self.font["STAT"].table

        # Build axis index mapping
        if hasattr(stat, "DesignAxisRecord") and stat.DesignAxisRecord:
            for i, axis in enumerate(stat.DesignAxisRecord.Axis):
                tag = axis.AxisTag
                self.index_to_tag[i] = tag
                if tag not in self.stat_values:
                    self.stat_values[tag] = {}

        # Extract axis value names
        if hasattr(stat, "AxisValueArray") and stat.AxisValueArray:
            for axis_value in stat.AxisValueArray.AxisValue:
                self._extract_axis_value_name(axis_value)

    def _extract_axis_value_name(self, axis_value) -> None:
        """Extract name from a single AxisValue record.

        Attempts to retrieve the name string for a STAT table AxisValue record
        using the ValueNameID. If extraction fails (nameID doesn't exist or
        name record is missing), logs a debug message and returns silently.

        This can fail when:
        - The nameID references a non-existent name record
        - The name table is incomplete or malformed
        - The font uses platform-specific encoding that getDebugName() can't decode

        Args:
            axis_value: AxisValue record from STAT table with ValueNameID attribute
        """
        name_id = axis_value.ValueNameID
        value_name = self.font["name"].getDebugName(name_id)

        if not value_name:
            logger.debug(f"Could not extract STAT value name for nameID {name_id}")
            return

        # Get axis tag via index
        if hasattr(axis_value, "AxisIndex"):
            axis_tag = self.index_to_tag.get(axis_value.AxisIndex)
            if axis_tag and hasattr(axis_value, "Value"):
                self.stat_values[axis_tag][axis_value.Value] = value_name

    def get_label_for_axis(
        self, axis_tag: str, value: float, epsilon: float = AXIS_VALUE_EPSILON
    ) -> Optional[str]:
        """Get STAT label for specific axis/value."""
        axis_map = self.stat_values.get(axis_tag)
        if not axis_map:
            return None

        # Exact match
        if value in axis_map:
            return axis_map[value]

        # Nearest match within tolerance
        try:
            nearest = min(axis_map.keys(), key=lambda k: abs(k - value))
            if abs(nearest - value) <= epsilon:
                return axis_map[nearest]
        except ValueError:
            pass

        return None

    def build_subfamily_name(
        self, coordinates: Dict[str, float], metadata: Optional["FontMetadata"] = None
    ) -> str:
        """Construct subfamily name from STAT AxisValue matches.

        Canonical order: width → weight → slope
        Uses family context to determine when "Regular" should be included.
        """
        labels_by_tag: Dict[str, str] = {}

        if "STAT" not in self.font or "fvar" not in self.font:
            return "Regular"

        try:
            stat = self.font["STAT"].table

            if not getattr(stat, "AxisValueArray", None):
                return "Regular"

            for av in stat.AxisValueArray.AxisValue:
                axis_index = getattr(av, "AxisIndex", None)
                if axis_index is None:
                    continue

                axis_tag = self.index_to_tag.get(axis_index)
                if not axis_tag:
                    continue

                # Check if axis exists in fvar table (not just STAT)
                fvar_axes = {axis.axisTag for axis in self.font["fvar"].axes}
                if axis_tag not in fvar_axes:
                    continue  # Skip axes that only exist in STAT

                if axis_tag not in coordinates:
                    continue

                coord_value = float(coordinates[axis_tag])

                # Check if this AxisValue matches the coordinate
                if self._axis_value_matches(av, coord_value):
                    label = self._get_name_string(av.ValueNameID)
                    if label and axis_tag not in labels_by_tag:
                        labels_by_tag[axis_tag] = label

        except Exception as e:
            logger.warning(f"Error building STAT name: {e}")
            return "Regular"

        return self._compose_name_parts(labels_by_tag, coordinates, metadata)

    def _axis_value_matches(self, av, coord_value: float) -> bool:
        """Check if AxisValue record matches coordinate value."""
        try:
            fmt = int(getattr(av, "Format", 0))
        except Exception:
            return False

        epsilon = 1e-6

        if fmt == 1:  # Format 1: Single value
            try:
                val = float(getattr(av, "Value", coord_value))
                return abs(val - coord_value) <= epsilon
            except Exception:
                return False

        elif fmt == 2:  # Format 2: Range
            try:
                vmin = float(getattr(av, "RangeMinValue", coord_value))
                vmax = float(getattr(av, "RangeMaxValue", coord_value))
                return vmin <= coord_value <= vmax
            except Exception:
                return False

        elif fmt == 3:  # Format 3: Linked value
            try:
                val = float(getattr(av, "Value", coord_value))
                return abs(val - coord_value) <= epsilon
            except Exception:
                return False

        return False

    def _get_name_string(self, name_id: Optional[int]) -> Optional[str]:
        """Get name string for nameID."""
        if name_id is None:
            return None
        try:
            rec = self.font["name"].getName(int(name_id), 3, 1, 0x409)
            if rec is None:
                return None
            return rec.toUnicode()
        except Exception:
            return None

    def _should_suppress_width(self, width: str) -> bool:
        """Check if width label should be suppressed (contains suppressible terms)."""
        WIDTH_SUPPRESSIBLE = {"regular", "normal", "standard", "roman"}
        width_lower = width.strip().lower()
        return any(term in width_lower for term in WIDTH_SUPPRESSIBLE)

    def _should_suppress_slope(self, slope: str) -> bool:
        """Check if slope label should be suppressed (contains suppressible terms)."""
        SLOPE_SUPPRESSIBLE = {"roman", "upright", "normal", "regular"}
        slope_lower = slope.strip().lower()
        return any(term in slope_lower for term in SLOPE_SUPPRESSIBLE)

    def _determine_family_weight_context(
        self, metadata: Optional["FontMetadata"]
    ) -> Tuple[bool, bool]:
        """Determine if family has heavier or lighter weights than Regular (400).

        Returns:
            Tuple of (has_heavier_weights, has_lighter_weights)
        """
        has_heavier_weights = False
        has_lighter_weights = False

        if metadata:
            for inst in metadata.instances:
                inst_wght = inst.coordinates.get("wght", 400.0)

                if inst_wght > 400.0:
                    has_heavier_weights = True
                if inst_wght < 400.0:
                    has_lighter_weights = True

        return (has_heavier_weights, has_lighter_weights)

    def _process_weight_label(
        self, weight: Optional[str], is_regular_weight: bool, has_other_weights: bool
    ) -> Optional[str]:
        """Process weight label: clean up "Normal" prefix and handle Regular weight.

        Args:
            weight: Weight label from STAT table (may be None)
            is_regular_weight: Whether this instance is at Regular weight (400)
            has_other_weights: Whether family has other weight variants

        Returns:
            Processed weight label, or None if no weight should be added
        """
        if weight:
            # Clean up "Normal" prefix from weight labels like "Normal Thin" -> "Thin"
            weight_cleaned = weight.strip()
            if weight_cleaned.lower().startswith("normal "):
                weight_cleaned = weight_cleaned[7:].strip()  # Remove "Normal " prefix
            elif weight_cleaned.lower() == "normal":
                # Replace standalone "Normal" with "Regular" for weight axis
                weight_cleaned = "Regular"
            return weight_cleaned
        elif is_regular_weight and has_other_weights:
            # No weight label found, but this is Regular weight in a family with other weights
            return "Regular"
        return None

    def _compose_name_parts(
        self,
        labels: Dict[str, str],
        coordinates: Dict[str, float],
        metadata: Optional["FontMetadata"] = None,
    ) -> str:
        """Compose name parts in canonical order with family awareness.

        Decision tree for "Regular" insertion:
        1. If weight label exists, use it (after cleaning "Normal" prefix)
        2. If no weight label but weight=400 and family has other weights, add "Regular"
        3. Width labels are suppressed if they contain suppressible terms
        4. Slope labels are suppressed if they contain suppressible terms
        5. If only slope remains and family has other weights, prefix with "Regular"

        Args:
            labels: Dictionary mapping axis tags to their STAT-derived labels
            coordinates: Dictionary mapping axis tags to coordinate values
            metadata: Optional font metadata for family context

        Returns:
            Composed subfamily name string
        """
        parts: List[str] = []

        # Get coordinate values for context
        wght_value = coordinates.get("wght", 400.0)
        is_regular_weight = wght_value == 400.0

        # Determine family weight context
        has_heavier_weights, has_lighter_weights = (
            self._determine_family_weight_context(metadata)
        )
        has_other_weights = has_heavier_weights or has_lighter_weights

        # Width (suppress specific terms)
        width = labels.get("wdth")
        if width and not self._should_suppress_width(width):
            parts.append(width)

        # Weight processing - NEVER suppress weight terms, but clean up "Normal" prefix
        # Decision: Add weight label if present, or add "Regular" if weight=400 and
        # family has other weights (to distinguish Regular from other weights)
        weight = labels.get("wght")
        processed_weight = self._process_weight_label(
            weight, is_regular_weight, has_other_weights
        )
        if processed_weight:
            parts.append(processed_weight)

        # Slope (suppress specific terms)
        # Only add slope if it doesn't contain suppressible terms like "roman", "upright"
        slope = labels.get("slnt") or labels.get("ital") or labels.get("obli")
        if slope and not self._should_suppress_slope(slope):
            parts.append(slope)

        # Special case: If only slope remains (no width, no weight), and family has
        # other weights, prefix with "Regular" to indicate this is Regular weight
        # Example: "Italic" → "Regular Italic" (when family also has "Bold Italic")
        # (Re-check slope since we may have added it above)
        slope = labels.get("slnt") or labels.get("ital") or labels.get("obli")
        if len(parts) == 1 and slope and has_other_weights:
            if not self._should_suppress_slope(slope):
                parts.insert(0, "Regular")

        return " ".join(parts) if parts else "Regular"


# ============================================================================
# Family Context (for Hybrid Naming)
# ============================================================================


class FamilyContext:
    """Tracks all instances in a family for intelligent naming decisions."""

    def __init__(self, instances: List[InstanceInfo], stat_parser: STATNameParser):
        self.instances = instances
        self.stat_parser = stat_parser
        self._weight_groups = self._build_weight_groups()

    def _build_weight_groups(
        self,
    ) -> Dict[Tuple[Tuple[str, float], ...], List[InstanceInfo]]:
        """Group instances by non-weight coordinates."""
        groups: Dict[Tuple[Tuple[str, float], ...], List[InstanceInfo]] = {}

        for inst in self.instances:
            key = self._group_key(inst.coordinates)
            if key not in groups:
                groups[key] = []
            groups[key].append(inst)

        return groups

    def _group_key(self, coords: Dict[str, float]) -> Tuple[Tuple[str, float], ...]:
        """Create grouping key from non-weight coordinates."""
        items = [(k, float(v)) for k, v in coords.items() if k != "wght"]
        items = [(k, round(v, 4)) for k, v in items]
        return tuple(sorted(items))

    def should_add_regular(self, inst: InstanceInfo) -> bool:
        """Determine if 'Regular' should be added to an fvar name.

        Adds Regular when:
        1. Weight is 400 (Regular weight)
        2. Name doesn't already contain "Regular"
        3. Other instances in same group have different weights (family context)
        """
        weight = inst.weight_value()
        if not WeightClass.is_regular(weight):
            return False

        # Check if name already contains "Regular"
        name = inst.fvar_name.strip()
        if "regular" in name.lower():
            return False

        # Check if siblings have different weights
        # Group by non-weight coordinates (same ital, trap, etc.)
        group_key = self._group_key(inst.coordinates)
        siblings = self._weight_groups.get(group_key, [])

        # Check if any sibling has a different weight value (not just STAT label)
        # Compare by index since we might be comparing a temp instance
        for sibling in siblings:
            if sibling.index == inst.index:
                continue
            sib_weight = sibling.weight_value()
            # If sibling has different weight, we should add Regular
            if not WeightClass.is_regular(sib_weight):
                return True

        return False

    def build_hybrid_name(self, inst: InstanceInfo) -> str:
        """Build hybrid name: fvar base with STAT-derived completions."""
        if inst.fvar_name == UNKNOWN_FVAR_NAME:
            return inst.stat_name

        base = inst.fvar_name.strip()

        # Decision: Should we add "Regular" to this name?
        # This is determined by should_add_regular() which checks:
        # 1. Weight is 400 (Regular weight)
        # 2. Name doesn't already contain "Regular"
        # 3. Other instances in same group have different weights (family context)
        if not self.should_add_regular(inst):
            return base

        # Position logic for "Regular" insertion:
        # We want to maintain canonical order: Width → Weight → Slope
        # If the name ends with a slope term, insert "Regular" before it
        # Otherwise, append "Regular" at the end
        slope_terms = ["italic", "oblique", "slanted"]

        # Split into words to check if last word is a slope term
        words = base.split()
        if not words:
            # Empty name (shouldn't happen, but handle it)
            return "Regular"

        last_word_lower = words[-1].lower()

        # Check if last word is a slope term
        if last_word_lower in slope_terms:
            # Insert "Regular" before the slope term to maintain canonical order
            if len(words) > 1:
                # Name has content before slope: "Inktrap Italic" -> "Inktrap Regular Italic"
                # This preserves any width/other terms that come before the slope
                before_slope = " ".join(words[:-1])
                slope_original = words[-1]
                return f"{before_slope} Regular {slope_original}"
            else:
                # Pure slope name: "Italic" -> "Regular Italic"
                # When only slope exists, prefix with "Regular" to indicate weight
                slope_original = words[0]
                return f"Regular {slope_original}"
        else:
            # No slope term, append "Regular" at the end
            # "Inktrap" -> "Inktrap Regular"
            # This handles cases where name has width/other terms but no slope
            return f"{base} Regular"


# ============================================================================
# Font Analyzer (Pure Analysis)
# ============================================================================


class FontAnalyzer:
    """Analyzes variable font structure and metadata."""

    def __init__(self, font_path: str):
        self.font_path = font_path
        self.font: Optional[TTFont] = None
        self.stat_parser: Optional[STATNameParser] = None

    def load_and_validate(self) -> bool:
        """Load font and validate it's a variable font."""
        try:
            self.font = TTFont(self.font_path)
        except Exception as e:
            StatusIndicator("error").add_message("Error loading font").with_explanation(
                str(e)
            ).emit()
            return False

        if "fvar" not in self.font:
            StatusIndicator("error").add_message(
                "Not a variable font (missing fvar table)"
            ).emit()
            return False

        return True

    def analyze(self) -> FontMetadata:
        """Perform full font analysis."""
        if not self.font:
            raise RuntimeError("Font not loaded")

        self.stat_parser = STATNameParser(self.font)

        axes = self._extract_axes()
        instances = self._extract_instances()
        source_italic = self._detect_source_italic()
        family_name = self._extract_family_name()

        metadata = FontMetadata(
            axes=axes,
            instances=instances,
            stat_values=self.stat_parser.stat_values,
            source_italic=source_italic,
            family_name=family_name,
        )

        # Update STAT names with family context
        self._update_stat_names_with_context(metadata)

        return metadata

    def _extract_axes(self) -> List[AxisInfo]:
        """Extract axis information from fvar table."""
        axes: List[AxisInfo] = []
        fvar = self.font["fvar"]

        for axis in fvar.axes:
            axis_name = self.font["name"].getDebugName(axis.axisNameID) or axis.axisTag
            axes.append(
                AxisInfo(
                    tag=axis.axisTag,
                    name=axis_name,
                    min_value=axis.minValue,
                    default_value=axis.defaultValue,
                    max_value=axis.maxValue,
                )
            )

        return axes

    def _extract_family_name(self) -> str:
        """Typographic family from name IDs 1 / 16 (already-loaded font)."""
        if not self.font or "name" not in self.font:
            return Path(self.font_path).stem.split("-")[0]
        name_table = self.font["name"]
        raw = name_table.getDebugName(1) or name_table.getDebugName(16) or ""
        family = strip_variable_tokens(raw) or raw
        if not family or family == UNKNOWN_FVAR_NAME:
            return Path(self.font_path).stem.split("-")[0]
        return family

    def _extract_instances(self) -> List[InstanceInfo]:
        """Extract instance information from fvar table."""
        instances: List[InstanceInfo] = []
        fvar = self.font["fvar"]

        for i, instance in enumerate(fvar.instances):
            fvar_name = UNKNOWN_FVAR_NAME
            if hasattr(instance, "subfamilyNameID") and instance.subfamilyNameID:
                try:
                    # Try getDebugName first
                    name = self.font["name"].getDebugName(instance.subfamilyNameID)
                    if name:
                        fvar_name = name
                    else:
                        # Fallback: try getName directly with different platforms
                        name_record = self.font["name"].getName(
                            instance.subfamilyNameID,
                            3,
                            1,
                            0x409,  # Windows, Unicode, en-US
                        )
                        if name_record:
                            fvar_name = name_record.toUnicode()
                        else:
                            # Try Mac platform
                            name_record = self.font["name"].getName(
                                instance.subfamilyNameID,
                                1,
                                0,
                                0,  # Mac, Roman, English
                            )
                            if name_record:
                                fvar_name = name_record.toUnicode()
                except Exception as e:
                    logger.debug(f"Failed to extract fvar name for instance {i}: {e}")
                    fvar_name = UNKNOWN_FVAR_NAME

            stat_name = self.stat_parser.build_subfamily_name(instance.coordinates)
            is_italic = self._detect_italic(instance.coordinates)
            weight = instance.coordinates.get("wght", WeightClass.REGULAR)
            is_bold = WeightClass.is_bold(weight)

            instances.append(
                InstanceInfo(
                    index=i,
                    fvar_name=fvar_name,
                    stat_name=stat_name,
                    coordinates=instance.coordinates,
                    is_italic=is_italic,
                    is_bold=is_bold,
                )
            )

        return instances

    def _update_stat_names_with_context(self, metadata: FontMetadata) -> None:
        """Update STAT names with family context after metadata is created."""
        for instance in metadata.instances:
            # Rebuild STAT name with family context
            instance.stat_name = self.stat_parser.build_subfamily_name(
                instance.coordinates, metadata
            )

    def _detect_source_italic(self) -> bool:
        """Detect if source font is italic from post table."""
        if "post" in self.font and hasattr(self.font["post"], "italicAngle"):
            try:
                angle = self.font["post"].italicAngle
                return abs(angle) > ITALIC_ANGLE_THRESHOLD
            except (AttributeError, TypeError):
                pass
        return False

    def _detect_italic(self, coords: Dict[str, float]) -> bool:
        """Detect if coordinates represent an italic style."""
        # Check slant axis
        if "slnt" in coords and coords["slnt"] != 0:
            return True

        # Check italic axis
        if "ital" in coords:
            ital_val = coords["ital"]
            if ital_val in (0, 0.0, 1, 1.0):
                return ital_val != 0
            try:
                fval = float(ital_val)
                if fval > 1.0:
                    return True
                return fval >= 0.5
            except Exception:
                return False

        # Fallback to source font italic detection
        return self._detect_source_italic()


# ============================================================================
# Instance Naming Strategy
# ============================================================================


class InstanceNamingStrategy:
    """Resolves instance names based on configuration."""

    def __init__(
        self, metadata: FontMetadata, stat_parser: STATNameParser, mode: NamingMode
    ):
        self.metadata = metadata
        self.stat_parser = stat_parser
        self.mode = mode
        self.family_context = FamilyContext(metadata.instances, stat_parser)

    def resolve_name(self, inst: InstanceInfo) -> str:
        """Resolve final name for instance based on strategy."""
        if self.mode == NamingMode.STAT:
            return inst.stat_name

        elif self.mode == NamingMode.FVAR_HYBRID:
            if inst.fvar_name != UNKNOWN_FVAR_NAME:
                # Normalize fvar name before applying hybrid logic
                normalized_fvar = normalize_fvar_name(
                    inst.fvar_name,
                    stat_values=self.metadata.stat_values,
                    coordinates=inst.coordinates,
                )
                # Create a temporary instance with normalized fvar name for hybrid logic
                temp_inst = InstanceInfo(
                    index=inst.index,
                    fvar_name=normalized_fvar,
                    stat_name=inst.stat_name,
                    coordinates=inst.coordinates,
                    is_italic=inst.is_italic,
                    is_bold=inst.is_bold,
                )
                return self.family_context.build_hybrid_name(temp_inst)
            return inst.stat_name

        elif self.mode == NamingMode.FVAR_RAW:
            if inst.fvar_name != UNKNOWN_FVAR_NAME:
                # Normalize fvar name for filename generation
                return normalize_fvar_name(
                    inst.fvar_name,
                    stat_values=self.metadata.stat_values,
                    coordinates=inst.coordinates,
                )
            return inst.stat_name

        return inst.stat_name


# ============================================================================
# Instance deduplication (output identity)
# ============================================================================


def default_naming_mode_for_instances(instances: List["InstanceInfo"]) -> NamingMode:
    """fvar-hybrid when fvar names exist; STAT otherwise."""
    has_fvar = any(inst.fvar_name != UNKNOWN_FVAR_NAME for inst in instances)
    return NamingMode.FVAR_HYBRID if has_fvar else NamingMode.STAT


def build_instance_output_name_resolver(
    metadata: "FontMetadata",
    stat_parser: "STATNameParser",
    naming_mode: NamingMode,
) -> Callable[["InstanceInfo"], str]:
    """Return a callable that resolves the subfamily name used for filenames."""
    strategy = InstanceNamingStrategy(metadata, stat_parser, naming_mode)
    return strategy.resolve_name


def instance_output_identity(
    inst: "InstanceInfo",
    resolve_output_name: Callable[["InstanceInfo"], str],
) -> Tuple[Tuple[Tuple[str, float], ...], str]:
    """Fingerprint for redundant fvar rows: snapped coordinates plus resolved output name."""
    return (
        instance_coordinate_key(inst.coordinates),
        _collapse_name_key(resolve_output_name(inst)),
    )


def count_coordinate_duplicate_rows(
    instances: List["InstanceInfo"],
    metadata: "FontMetadata",
    stat_parser: "STATNameParser",
    naming_mode: NamingMode,
) -> int:
    """How many instances would be skipped if keeping only first occurrence per output identity."""
    if len(instances) < 2:
        return 0
    resolve_name = build_instance_output_name_resolver(metadata, stat_parser, naming_mode)
    seen: set[Tuple[Tuple[Tuple[str, float], ...], str]] = set()
    dups = 0
    for inst in instances:
        key = instance_output_identity(inst, resolve_name)
        if key in seen:
            dups += 1
        else:
            seen.add(key)
    return dups


def unique_instances_by_coordinates(
    instances: List["InstanceInfo"],
    metadata: "FontMetadata",
    stat_parser: "STATNameParser",
    naming_mode: NamingMode,
) -> Tuple[List["InstanceInfo"], int]:
    """Drop later fvar rows that would produce the same output as an earlier row."""
    resolve_name = build_instance_output_name_resolver(metadata, stat_parser, naming_mode)
    seen: set[Tuple[Tuple[Tuple[str, float], ...], str]] = set()
    out: List["InstanceInfo"] = []
    skipped = 0
    for inst in instances:
        key = instance_output_identity(inst, resolve_name)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        out.append(inst)
    return out, skipped


def coordinate_first_kept_instance_indices(
    instances: List["InstanceInfo"],
    metadata: "FontMetadata",
    stat_parser: "STATNameParser",
    naming_mode: NamingMode,
) -> set[int]:
    """fvar instance indices kept when deduping: first row per output identity."""
    resolve_name = build_instance_output_name_resolver(metadata, stat_parser, naming_mode)
    seen: set[Tuple[Tuple[Tuple[str, float], ...], str]] = set()
    kept: set[int] = set()
    for inst in instances:
        key = instance_output_identity(inst, resolve_name)
        if key not in seen:
            kept.add(inst.index)
            seen.add(key)
    return kept


def prior_duplicate_coordinate_slot_by_index(
    instances: List["InstanceInfo"],
    metadata: "FontMetadata",
    stat_parser: "STATNameParser",
    naming_mode: NamingMode,
) -> Dict[int, Optional[int]]:
    """Map InstanceInfo.index -> earlier # column value (1-based) with the same output identity."""
    resolve_name = build_instance_output_name_resolver(metadata, stat_parser, naming_mode)
    seen: Dict[Tuple[Tuple[Tuple[str, float], ...], str], int] = {}
    out: Dict[int, Optional[int]] = {}
    for inst in sorted(instances, key=lambda i: i.index):
        key = instance_output_identity(inst, resolve_name)
        slot = inst.index + 1
        if key not in seen:
            seen[key] = slot
            out[inst.index] = None
        else:
            out[inst.index] = seen[key]
    return out


def instances_for_processing(
    metadata: "FontMetadata",
    config: "InstancerConfig",
    stat_parser: "STATNameParser",
    naming_mode: Optional[NamingMode] = None,
) -> Tuple[List["InstanceInfo"], int]:
    """
    Instances to iterate for UI numbering and generation.
    Returns (instances, duplicates_skipped) where duplicates_skipped is 0 unless dedupe is enabled.
    """
    if not config.skip_coordinate_duplicates:
        return metadata.instances, 0
    mode = naming_mode or config.naming_mode
    uniq, skipped = unique_instances_by_coordinates(
        metadata.instances, metadata, stat_parser, mode
    )
    return uniq, skipped


# ============================================================================
# Interactive UI
# ============================================================================


class InteractivePrompt:
    """Handles user interaction and prompts."""

    def __init__(
        self,
        metadata: FontMetadata,
        stat_parser: STATNameParser,
        selection_instances: Optional[List[InstanceInfo]] = None,
        coordinate_dedupe_active: bool = False,
        naming_mode: Optional[NamingMode] = None,
    ):
        self.metadata = metadata
        self.stat_parser = stat_parser
        self.selection_instances = selection_instances or metadata.instances
        self.coordinate_dedupe_active = coordinate_dedupe_active
        self.dedupe_naming_mode = naming_mode or self._default_naming_mode()
        self._kept_coordinate_instance_indices = coordinate_first_kept_instance_indices(
            metadata.instances,
            metadata,
            stat_parser,
            self.dedupe_naming_mode,
        )

    def _table_slot_number(self, inst: InstanceInfo) -> int:
        """1-based # column label (fvar instance order), matching the instance table."""
        return inst.index + 1

    def _instance_for_table_slot(self, slot_number: int) -> Optional[InstanceInfo]:
        """Resolve UI # column (fvar instance order label) to an InstanceInfo."""
        for inst in self.metadata.instances:
            if self._table_slot_number(inst) == slot_number:
                return inst
        return None

    def _max_table_slot_number(self) -> int:
        if not self.metadata.instances:
            return 0
        return max(self._table_slot_number(inst) for inst in self.metadata.instances)

    def _default_naming_mode(self) -> NamingMode:
        """fvar-hybrid when fvar names exist; STAT otherwise."""
        return default_naming_mode_for_instances(self.metadata.instances)

    def _parse_numbers_only(
        self, response: str
    ) -> Optional[List[Tuple[int, NamingMode]]]:
        """Parse space/comma-separated table slot numbers (# column). No DSL.

        Returns (processing_list_index, STAT) tuples; mode is overwritten by caller.
        None on invalid token or validation failure. Dedupes by index (first wins).
        """
        if not response.strip():
            return []
        normalized = response.replace(",", " ")
        tokens = [t for t in normalized.split() if t]
        if not tokens:
            return []
        result: List[Tuple[int, NamingMode]] = []
        seen_idx: set[int] = set()
        mx = self._max_table_slot_number()

        for token in tokens:
            if not token.isdigit():
                _emit_dim(
                    f"  '{token}' is not a valid instance number (use digits only)."
                )
                return None
            num = int(token)
            inst_pick = self._instance_for_table_slot(num)
            if inst_pick is None:
                _emit_dim(f"  Instance #{num} is out of range (1–{mx})")
                return None
            if (
                self.coordinate_dedupe_active
                and inst_pick.index not in self._kept_coordinate_instance_indices
            ):
                _emit_dim(
                    f"  #{num} would repeat the output of an earlier row under "
                    f"{self.dedupe_naming_mode.value} naming (purple row + gold #); "
                    "skipped by default; use --all-fvar-instance-rows to generate it."
                )
                return None
            try:
                idx = self.selection_instances.index(inst_pick)
            except ValueError:
                _emit_dim(f"  #{num} could not be resolved for generation.")
                return None
            if idx not in seen_idx:
                seen_idx.add(idx)
                result.append((idx, NamingMode.STAT))

        return result

    def _classify_axes(self) -> Tuple[List[AxisInfo], List[AxisInfo]]:
        """Separate axes into variable and fixed."""
        variable_axes = [axis for axis in self.metadata.axes if axis.is_variable()]
        fixed_axes = [axis for axis in self.metadata.axes if not axis.is_variable()]
        return variable_axes, fixed_axes

    def _format_stat_values_inline(self, axis_tag: str) -> str:
        """Format STAT values for an axis as an inline string."""
        if axis_tag not in self.metadata.stat_values:
            return ""

        values = self.metadata.stat_values[axis_tag]
        if not values:
            return ""

        # Sort by value and format
        sorted_values = sorted(values.items())
        formatted = ", ".join([f"{val}={name}" for val, name in sorted_values])
        return f"{formatted}"

    def show_info_mode(self) -> None:
        """Display comprehensive font information."""
        self._print_header("Font Information")

        # Introduction
        cs.emit("\nThis font contains the following structure:")

        # Axes section with context
        self._print_axes_table()

        # Instances section with context
        cs.emit("\n" + "─" * 70)
        cs.emit("NAMED INSTANCES")
        cs.emit("─" * 70)
        cs.emit("The font defines the following pre-configured instances:")
        self._print_instances_table_with_naming(show_naming_comparison=True)

        # Naming explanation if fvar names exist
        has_fvar_names = any(
            inst.fvar_name != UNKNOWN_FVAR_NAME for inst in self.metadata.instances
        )
        if has_fvar_names:
            cs.emit("\nNaming Note:")
            cs.emit(
                "  • STAT names are derived from the STAT table (canonical, recommended)"
            )
            cs.emit(
                "  • fvar names come from fvar instance records (legacy compatibility)"
            )

        # Validation
        self._print_validation_notices()

    def _prompt_advanced_selection(
        self, has_fvar_names: bool, default_mode: NamingMode
    ) -> Optional[Tuple[List[Tuple[int, NamingMode]], NamingMode]]:
        """Per-slot naming DSL; loops until valid input or back ([x])."""
        cs.emit("")
        _emit_menu("  Advanced — per-slot naming")
        _emit_menu(
            "    Suffix each slot:  s=STAT   r=raw   f=fvar-hybrid     "
            "(e.g. 1s 2r 3f)"
        )
        _emit_menu(
            "    Or:  stat:1,2 fvar:3     s1 r2 f3     "
            "[x] back to instances   [q] quit"
        )
        while True:
            adv = input("  > ").strip()
            _raise_if_quit(adv)
            tok = adv.strip().lower().split()[0] if adv.strip() else ""
            if not tok or tok in ("x", "cancel", "back"):
                return None
            parsed = self._parse_instance_selection(adv, has_fvar_names)
            if parsed is None:
                _emit_dim(
                    "  Could not parse — try 1s 2r 3f or stat:1,2   "
                    "[x] back to instances"
                )
                continue
            adv_mode = parsed[0][1] if parsed else default_mode
            return (parsed, adv_mode)

    def show_instance_selection(
        self,
    ) -> Optional[Union[Literal["custom"], Tuple[List[Tuple[int, NamingMode]], NamingMode]]]:
        """Show instance table then a grouped three-row action menu.

        Layout (always):
            [table]
            Legend: … (only relevant parts)

            Generate all    [Enter] …   [s] STAT   …     (label bold)
            Pick subset     type slot numbers, e.g. 1 3 7   (label bold)
            Other           [c] … [a] … [?] …            (label bold)
            (spacer line)   [x] skip this font   [q] quit

        Returns:
            None                                        — skip this font ([x] or skip)
            "custom"                                    — route to custom mode
            ([], mode)                                  — generate all with mode
            ([(processing_idx, mode), …], mode)         — generate specific instances
        Raises:
            SystemExit(0) when the user chooses quit.
        """
        has_fvar_names = any(
            inst.fvar_name != UNKNOWN_FVAR_NAME for inst in self.metadata.instances
        )
        default_mode = self._default_naming_mode()

        while True:
            self._print_header("Named Instances")
            any_names_differ, has_dup_coords = self._print_instances_table_with_naming(
                show_naming_comparison=has_fvar_names,
                show_legend_banners=False,
            )

            # Compact legend — one line, only what applies to this font
            self._print_table_legend(any_names_differ, has_dup_coords, has_fvar_names)

            # --- Grouped three-row action menu ---
            cs.emit("")

            if has_fvar_names:
                if default_mode == NamingMode.FVAR_HYBRID:
                    gen_line = "[Enter] fvar-hybrid   [s] STAT   [r] raw"
                else:
                    gen_line = "[Enter] STAT   [f] fvar-hybrid   [r] raw"
            else:
                gen_line = "[Enter] STAT"

            _emit_menu_row("Generate all", gen_line)
            _emit_menu_row(
                "Pick subset",
                "type slot numbers, e.g.  1  3  7",
                dim_hint="then choose naming",
            )
            _emit_menu_row(
                "Other",
                "[c] custom   [a] advanced   [?] help",
            )
            cs.emit("")
            _emit_menu("                  [x] skip this font   [q] quit")

            response = input("  > ").strip().lower()

            # --- Navigation / escape ---
            if response in ("q", "quit"):
                raise SystemExit(0)
            rsp0 = response.split()[0] if response else ""
            if rsp0 in ("x", "skip"):
                return None
            if response in ("?", "help"):
                self._show_help_panel(has_fvar_names, default_mode)
                continue  # redraw table + menu
            if response == "c":
                return "custom"

            # --- Advanced DSL (per-slot naming) ---
            if response in ("a", "advanced"):
                adv_result = self._prompt_advanced_selection(
                    has_fvar_names, default_mode
                )
                if adv_result is None:
                    continue
                return adv_result

            # --- Generate-all shortcut keys ---
            if has_fvar_names:
                if response in ("s", "stat"):
                    return ([], NamingMode.STAT)
                if response in ("f", "fvar"):
                    return ([], NamingMode.FVAR_HYBRID)
                if response in ("r", "raw", "fn"):
                    return ([], NamingMode.FVAR_RAW)

            if response == "":
                return ([], default_mode)

            # --- Slot-number picker ---
            indices = self._parse_numbers_only(response)
            if indices is None:
                cs.emit("")
                _emit_menu(
                    "    Enter slot numbers (e.g. 1 3 7), a mode key, [?], "
                    "[x] back, or type skip to skip this font."
                )
                response2 = input("  > ").strip().lower()
                if response2 in ("q", "quit"):
                    raise SystemExit(0)
                r20 = response2.split()[0] if response2 else ""
                if r20 == "x" or response2 == "":
                    continue
                if r20 == "skip" or response2 == "skip":
                    return None
                if response2 in ("?", "help"):
                    self._show_help_panel(has_fvar_names, default_mode)
                    continue
                indices = self._parse_numbers_only(response2)
                if indices is None:
                    StatusIndicator("warning").add_message(
                        "Could not parse selection. Skipping font."
                    ).emit()
                    return None

            if not indices:
                return None

            # Naming mode follow-up (only when font has fvar names)
            mode = default_mode
            if has_fvar_names:
                if default_mode == NamingMode.FVAR_HYBRID:
                    naming_prompt = "[Enter] fvar-hybrid   [s] STAT   [r] raw"
                else:
                    naming_prompt = "[Enter] STAT   [f] fvar-hybrid   [r] raw"
                cs.emit("")
                _emit_menu("  Naming (applies to selected instances)")
                _emit_menu(f"    {naming_prompt}   [q] quit")
                naming_response = input("  > ").strip().lower()
                _raise_if_quit(naming_response)
                if naming_response in ("s", "stat"):
                    mode = NamingMode.STAT
                elif naming_response in ("f", "fvar"):
                    mode = NamingMode.FVAR_HYBRID
                elif naming_response in ("r", "raw", "fn"):
                    mode = NamingMode.FVAR_RAW
                # else: keep default_mode (Enter)

            indices_with_mode = [(idx, mode) for idx, _ in indices]
            return (indices_with_mode, mode)

    def _build_hybrid_name_for_display(self, inst: InstanceInfo) -> str:
        """Build hybrid name for display with bold for added words."""
        if inst.fvar_name == UNKNOWN_FVAR_NAME:
            return "N/A"

        # First normalize the fvar name to remove suppressible terms
        normalized_fvar = normalize_fvar_name(
            inst.fvar_name,
            stat_values=self.metadata.stat_values,
            coordinates=inst.coordinates,
        )

        # Then apply hybrid logic (same as in InstanceNamingStrategy.resolve_name)
        # Create a temporary instance with normalized fvar name for hybrid logic
        temp_inst = InstanceInfo(
            index=inst.index,
            fvar_name=normalized_fvar,
            stat_name=inst.stat_name,
            coordinates=inst.coordinates,
            is_italic=inst.is_italic,
            is_bold=inst.is_bold,
        )
        family_context = FamilyContext(self.metadata.instances, self.stat_parser)
        hybrid = family_context.build_hybrid_name(temp_inst)

        # If modified, bold the added "Regular" word
        if hybrid != normalized_fvar:
            # Check if "Regular" was added (case-insensitive, whole word match)
            normalized_lower = normalized_fvar.lower()
            hybrid_lower = hybrid.lower()

            normalized_count = len(re.findall(r"\bregular\b", normalized_lower))
            hybrid_count = len(re.findall(r"\bregular\b", hybrid_lower))

            if hybrid_count > normalized_count:
                # "Regular" was added - bold it in the hybrid string
                # Replace "Regular" (case-sensitive) with bolded version
                if RICH_AVAILABLE:
                    # Use regex to replace "Regular" as whole word, preserving case
                    bolded_hybrid = re.sub(
                        r"\bRegular\b",
                        "[bold pale_green1]Regular[/bold pale_green1]",
                        hybrid,
                    )
                    return bolded_hybrid
                else:
                    # Non-rich: use asterisks
                    bolded_hybrid = re.sub(r"\bRegular\b", "*Regular*", hybrid)
                    return bolded_hybrid

        return normalized_fvar if normalized_fvar != UNKNOWN_FVAR_NAME else "N/A"

    def _print_instances_table_with_naming(
        self,
        show_naming_comparison: bool,
        show_legend_banners: bool = True,
    ) -> Tuple[bool, bool]:
        """Print instances table with optional naming comparison.

        Args:
            show_naming_comparison: Include the fvar Name column.
            show_legend_banners:    When True (default, used by info mode) fire the
                                    verbose StatusIndicator banners after the table.
                                    Pass False in selection mode to use the compact
                                    one-line legend instead.

        Returns:
            (any_names_differ, has_dup_coords) so the caller can build a legend.
        """
        rows = self.metadata.instances
        if not rows:
            cs.emit("\nNo named instances found")
            return (False, False)

        dup_coord_count = count_coordinate_duplicate_rows(
            self.metadata.instances,
            self.metadata,
            self.stat_parser,
            self.dedupe_naming_mode,
        )
        prior_duplicate_slot = prior_duplicate_coordinate_slot_by_index(
            rows,
            self.metadata,
            self.stat_parser,
            self.dedupe_naming_mode,
        )
        has_dup_coords = dup_coord_count > 0
        dup_row_bg = "on #4b3652"
        any_names_differ = False

        table = cs.create_table(
            show_header=True, row_styles=["on #282a39", "on #1d1f30"]
        )
        if table:
            # Set table width to match console width
            table.width = console.size.width
            table.add_column("#", style="dim", overflow="fold", no_wrap=False)
            table.add_column("STAT Name", style="pale_green1")

            if show_naming_comparison:
                table.add_column("fvar Name", style="dim")

            table.add_column("Style", style="medium_turquoise")
            table.add_column("Coordinates", style="turquoise2")

            for inst in rows:
                ribbi = self._get_ribbi_label(inst)
                prior_same = prior_duplicate_slot.get(inst.index)
                coord_dup_row = prior_same is not None

                # Check if names differ (compare normalized versions)
                normalized_fvar = (
                    normalize_fvar_name(
                        inst.fvar_name,
                        stat_values=self.metadata.stat_values,
                        coordinates=inst.coordinates,
                    )
                    if inst.fvar_name != UNKNOWN_FVAR_NAME
                    else UNKNOWN_FVAR_NAME
                )

                names_differ = (
                    normalized_fvar != inst.stat_name
                    and normalized_fvar != UNKNOWN_FVAR_NAME
                )
                if names_differ:
                    any_names_differ = True

                # Apply highlighting when names differ; skip when row repeats coordinates
                if names_differ and RICH_AVAILABLE and not coord_dup_row:
                    stat_display = (
                        f"[#282a39 on #ffdf80]{inst.stat_name}[/#282a39 on #ffdf80]"
                    )
                    hybrid_display = (
                        f"[#282a39 on #ffdf80]"
                        f"{self._build_hybrid_name_for_display(inst)}"
                        f"[/#282a39 on #ffdf80]"
                    )
                else:
                    stat_display = inst.stat_name
                    hybrid_display = self._build_hybrid_name_for_display(inst)

                # # column: ◆ for duplicate-coord rows, ■ for name-differ rows.
                # Icons are the same markers used in the compact legend below
                # the table, giving the user a direct visual link.
                if RICH_AVAILABLE:
                    if coord_dup_row:
                        num_cell = f"[bold #fecf82]◆ {inst.index + 1}[/bold #fecf82]"
                    elif names_differ and show_naming_comparison:
                        num_cell = f"[#ffdf80]■ {inst.index + 1}[/#ffdf80]"
                    else:
                        num_cell = str(inst.index + 1)
                else:
                    if coord_dup_row:
                        num_cell = f"◆ {inst.index + 1}"
                    elif names_differ and show_naming_comparison:
                        num_cell = f"■ {inst.index + 1}"
                    else:
                        num_cell = str(inst.index + 1)

                row_data = [num_cell, stat_display]

                if show_naming_comparison:
                    row_data.append(hybrid_display)

                row_data.extend([ribbi, inst.format_coordinates()])

                if coord_dup_row:
                    table.add_row(*row_data, style=dup_row_bg)
                else:
                    table.add_row(*row_data)

            console.print(table)
        else:
            # Fallback for non-Rich
            cs.emit(f"\nInstances ({len(rows)}):")
            for inst in rows:
                ribbi = self._get_ribbi_label(inst)
                prior_same = prior_duplicate_slot.get(inst.index)
                coord_dup_row = prior_same is not None

                normalized_fvar = (
                    normalize_fvar_name(
                        inst.fvar_name,
                        stat_values=self.metadata.stat_values,
                        coordinates=inst.coordinates,
                    )
                    if inst.fvar_name != UNKNOWN_FVAR_NAME
                    else UNKNOWN_FVAR_NAME
                )
                names_differ = (
                    normalized_fvar != inst.stat_name
                    and normalized_fvar != UNKNOWN_FVAR_NAME
                )

                prefix = "◆ " if coord_dup_row else ("■ " if names_differ and show_naming_comparison else "  ")
                cs.emit(f"{prefix}{inst.index + 1:2}. {inst.stat_name:25} [{ribbi:12}]")
                if show_naming_comparison and inst.fvar_name != UNKNOWN_FVAR_NAME:
                    hybrid = self._build_hybrid_name_for_display(inst)
                    cs.emit(f"      fvar: {hybrid}")
                mark = "  [dup coords]" if coord_dup_row else ""
                cs.emit(f"      {inst.format_coordinates()}{mark}")

        # Verbose banners — used in info mode; suppressed in selection mode
        # (selection mode uses the compact one-line legend instead).
        if show_legend_banners:
            if has_dup_coords:
                cs.emit("")
                if self.coordinate_dedupe_active:
                    StatusIndicator("info").add_message(
                        "[#fecf82 on #4b3652]◆ Purple-tinted rows[/#fecf82 on #4b3652]"
                        " would produce the same output as an earlier row under "
                        f"{self.dedupe_naming_mode.value} naming; "
                        "those rows are omitted from generation by default "
                        "(use --all-fvar-instance-rows to emit them)."
                    ).emit()
                else:
                    StatusIndicator("info").add_message(
                        "[#fecf82 on #4b3652]◆ Purple-tinted rows[/#fecf82 on #4b3652]"
                        " are redundant under the active naming mode "
                        f"({self.dedupe_naming_mode.value}). "
                        "With --all-fvar-instance-rows they generate like any other when selected."
                    ).emit()

            if table and show_naming_comparison:
                cs.emit("")
                StatusIndicator("info").add_message(
                    "[#282a39 on #ffdf80]■ Highlighted cells[/#282a39 on #ffdf80]"
                    " indicate STAT and fvar names differ"
                ).emit()

                has_modifications = any(
                    self._build_hybrid_name_for_display(inst) != inst.fvar_name
                    for inst in rows
                    if inst.fvar_name != UNKNOWN_FVAR_NAME
                )
                if has_modifications:
                    cs.emit("")
                    StatusIndicator("info").add_message(
                        "[bold]Bold[/bold] words in fvar names indicate"
                        " hybrid additions (use [raw] to omit)"
                    ).emit()

        return (any_names_differ, has_dup_coords)

    def _print_table_legend(
        self,
        has_name_diffs: bool,
        has_dup_coords: bool,
        show_naming_comparison: bool,
    ) -> None:
        """Single compact legend line between the table and the action menu.

        Only emits parts that are relevant to the current font — no legend
        is shown when neither condition applies.  The ■ / ◆ icons match the
        markers already present in the # column of the table above.
        """
        parts: List[str] = []

        if has_name_diffs and show_naming_comparison:
            if RICH_AVAILABLE:
                parts.append(
                    "[#ffdf80]■[/#ffdf80] "
                    "[#282a39 on #ffdf80] Yellow [/#282a39 on #ffdf80]"
                    " STAT/fvar names differ"
                )
            else:
                parts.append("■ Yellow = STAT/fvar names differ")

        if has_dup_coords:
            if RICH_AVAILABLE:
                parts.append(
                    "[#fecf82]◆[/#fecf82] "
                    "[#fecf82 on #4b3652] Purple [/#fecf82 on #4b3652]"
                    " redundant output (skipped)"
                )
            else:
                parts.append("◆ Purple = redundant output (skipped)")

        if not parts:
            return

        separator = "     "
        if RICH_AVAILABLE:
            cs.emit(f"[dim]  Legend:  [/dim]" + separator.join(parts))
        else:
            cs.emit("  Legend:  " + separator.join(parts))

    def _show_help_panel(self, has_fvar_names: bool, default_mode: NamingMode) -> None:
        """On-demand help panel — shown when the user types [?] at the action prompt.

        Explains naming modes, how to use each action, and the table legend.
        After displaying, control returns to the caller which will redraw the menu.
        """
        def _default_tag(mode: NamingMode) -> str:
            return "  [dim](default for this font)[/dim]" if RICH_AVAILABLE else "  (default)"

        if RICH_AVAILABLE:
            from rich.panel import Panel

            naming_section: List[str] = [
                "[bold]NAMING MODES[/bold]",
            ]
            if has_fvar_names:
                naming_section += [
                    f"  [pale_green1]fvar-hybrid[/pale_green1]  "
                    f"fvar instance names; adds \"Regular\" where needed for RIBBI"
                    f"{_default_tag(NamingMode.FVAR_HYBRID) if default_mode == NamingMode.FVAR_HYBRID else ''}",
                    f"  [pale_green1]STAT[/pale_green1]         "
                    f"Names from STAT table axis values (canonical)"
                    f"{_default_tag(NamingMode.STAT) if default_mode == NamingMode.STAT else ''}",
                    f"  [pale_green1]raw[/pale_green1]          "
                    f"fvar names exactly as stored, no normalization",
                ]
            else:
                naming_section += [
                    f"  [pale_green1]STAT[/pale_green1]  "
                    f"Names from STAT table axis values (canonical)"
                    f"{_default_tag(NamingMode.STAT)}",
                    "  [dim]fvar-hybrid and raw are unavailable — this font has no fvar instance names.[/dim]",
                ]

            actions_section: List[str] = [
                "",
                "[bold]GENERATE ALL[/bold]   [dim]Enter · s · f · r[/dim]",
                "  All instances shown in the table, with the chosen naming mode.",
                "",
                "[bold]PICK SUBSET[/bold]   [dim]type slot numbers[/dim]",
                "  Enter [turquoise2]#[/turquoise2] column values, space or comma separated —"
                "  e.g. [turquoise2]1 3 7[/turquoise2] [dim](then choose naming)[/dim]",
                "",
                "[bold]ADVANCED[/bold]   [dim]a[/dim]",
                "  Per-instance naming mode: [turquoise2]1s 2f 3r[/turquoise2]"
                "   or   [turquoise2]stat:1,2 fvar:3[/turquoise2]",
                "  Useful when different instances need different naming modes.",
                "  [dim]Empty line or[/dim] [turquoise2]x[/turquoise2] "
                "[dim]back to the instance table.[/dim]",
                "",
                "[bold]CUSTOM[/bold]   [dim]c[/dim]",
                "  Build instances with exact axis coordinates. Queue as many as you need,",
                "  then [turquoise2]g[/turquoise2] to generate all at once.",
                "  [turquoise2]n[/turquoise2] new from scratch  · "
                "[turquoise2]f[/turquoise2] copy and adjust an existing fvar instance  · "
                "[turquoise2]r[/turquoise2] remove one  ·  [turquoise2]c[/turquoise2] clear queue  ·  "
                "[turquoise2]x[/turquoise2] back without generating",
                "",
                "[bold]SKIP FONT[/bold]   [dim]x[/dim] (instance table)",
                "  Skip this file in a batch, or leave without generating instances.",
                "  In axis entry, [turquoise2]x[/turquoise2] only cancels that custom instance.",
            ]

            legend_section: List[str] = [
                "",
                "[bold]TABLE LEGEND[/bold]",
                "  [#ffdf80]■[/#ffdf80]  Yellow cells — STAT and fvar names differ for this instance",
                "  [#fecf82]◆[/#fecf82]  Purple rows  — Same coordinates as an earlier row;",
                "             skipped by default (--all-fvar-instance-rows to include all)",
                "  [bold]Bold[/bold] text  — \"Regular\" was added by fvar-hybrid normalization",
            ]

            content = "\n".join(naming_section + actions_section + legend_section)
            panel = Panel(
                content,
                title="[bold dodger_blue1] Help [/bold dodger_blue1]",
                border_style="dodger_blue1",
                padding=(1, 2),
            )
            console.print()
            console.print(panel)
            console.print()

        else:
            # Plain-text fallback
            cs.emit("\n=== HELP ===")
            cs.emit("NAMING MODES")
            if has_fvar_names:
                dflt = default_mode.value
                cs.emit(f"  fvar-hybrid  fvar names; adds Regular for RIBBI{' (default)' if dflt == 'fvar-hybrid' else ''}")
                cs.emit(f"  STAT         Names from STAT table axis values{' (default)' if dflt == 'stat' else ''}")
                cs.emit("  raw          fvar names exactly as stored")
            else:
                cs.emit("  STAT  Names from STAT table axis values (default)")
            cs.emit("")
            cs.emit("ACTIONS")
            cs.emit("  Enter (or s/f/r)  Generate all instances with chosen mode")
            cs.emit("  1 3 7             Pick instances by # column (then choose naming)")
            cs.emit("  a                 Advanced (empty / x = back to table)")
            cs.emit("  c                 Custom builder (n/f/g/r/c; x back without generating)")
            cs.emit("  x / skip          Skip this font")
            cs.emit("  q                 Quit program")
            cs.emit("")
            cs.emit("TABLE LEGEND")
            cs.emit("  ■  Yellow = STAT/fvar names differ")
            cs.emit("  ◆  Purple = redundant output (skipped by default)")
            cs.emit("  Bold text = Regular added by fvar-hybrid normalization")
            cs.emit("============\n")

    def _parse_instance_selection(
        self, response: str, allow_naming: bool
    ) -> Optional[List[Tuple[int, NamingMode]]]:
        """Parse instance selection with optional naming prefixes.

        Parses user input for selecting specific instances with optional naming mode
        specifications. Returns None if parsing fails or input is invalid.

        Supported formats:
        - Simple numbers: "1,2,3" or "1 2 3" (uses default/current mode)
        - Mode prefixes: "stat:1,2,3" or "fvar:4,5,6" (sets mode for those numbers)
        - Per-instance suffixes: "1s 2f 3r" (s=STAT, f=fvar-hybrid, r=fvar-raw)
        - Per-instance prefixes: "s1 f2 r3" (same as above, different order)
        - Mode change tokens: "s 1 2" or "fvar 3 4" (sets current mode, then numbers)
        - Mixed: "stat:1,2 fvar:3 4s" (combines multiple formats)

        Examples:
            "1,2,3" → [(0, STAT), (1, STAT), (2, STAT)]
            "stat:1,2 fvar:3" → [(0, STAT), (1, STAT), (2, FVAR_HYBRID)]
            "1s 2f 3r" → [(0, STAT), (1, FVAR_HYBRID), (2, FVAR_RAW)]
            "s 1 2 3" → [(0, STAT), (1, STAT), (2, STAT)]

        Args:
            response: User input string to parse
            allow_naming: If False, ignores naming mode specs and uses STAT for all

        Returns:
            List of (processing_list_index, naming_mode) tuples, or None if parsing fails.
            Numbers refer to the table `#` column (`InstanceInfo.index + 1`). With
            default duplicate-coordinate skipping, visually marked duplicate rows
            (purple band, gold slot #) are rejected unless --all-fvar-instance-rows is set.
        """
        # Normalize input
        normalized = response.replace(",", " ")
        tokens = normalized.split()

        result = []
        current_mode = NamingMode.STAT

        for token in tokens:
            mode = None
            num = None

            # Check for "mode:number" format (e.g., "stat:1", "fvar:2-4")
            if ":" in token:
                mode_str, num_str = token.split(":", 1)

                # Parse mode
                if mode_str in ("s", "stat"):
                    mode = NamingMode.STAT
                elif mode_str in ("f", "fvar"):
                    mode = NamingMode.FVAR_HYBRID
                elif mode_str in ("r", "raw", "fn"):
                    mode = NamingMode.FVAR_RAW
                else:
                    _emit_dim(
                        f"  Invalid mode '{mode_str}'. Use: stat, fvar, or raw"
                    )
                    return None

                # Parse number(s)
                if num_str.isdigit():
                    num = int(num_str)
                else:
                    # Could support ranges here: "1-3" → [1, 2, 3]
                    _emit_dim(f"  Invalid number format '{num_str}'")
                    return None

            # Check for "numbermode" format (e.g., "1s", "2f", "3r")
            elif token[-1] in ("s", "f", "r") and token[:-1].isdigit():
                num = int(token[:-1])
                if token[-1] == "s":
                    mode = NamingMode.STAT
                elif token[-1] == "f":
                    mode = NamingMode.FVAR_HYBRID
                elif token[-1] == "r":
                    mode = NamingMode.FVAR_RAW

            # Check for "modenumber" format (e.g., "s1", "f2", "r3")
            elif token[0] in ("s", "f", "r") and token[1:].isdigit():
                num = int(token[1:])
                if token[0] == "s":
                    mode = NamingMode.STAT
                elif token[0] == "f":
                    mode = NamingMode.FVAR_HYBRID
                elif token[0] == "r":
                    mode = NamingMode.FVAR_RAW

            # Check for mode change tokens
            elif token in ("s", "stat"):
                current_mode = NamingMode.STAT
                continue
            elif token in ("f", "fvar"):
                current_mode = NamingMode.FVAR_HYBRID
                continue
            elif token in ("r", "raw", "fn"):
                current_mode = NamingMode.FVAR_RAW
                continue

            # Plain number
            elif token.isdigit():
                num = int(token)
                mode = current_mode

            else:
                _emit_dim(f"  Invalid input: '{token}'")
                _emit_dim(
                    "  Valid formats: 1,2,3  or  stat:1,2  or  1s 2f  or  s 1 2"
                )
                return None

            # Add to results if we parsed a number
            if num is not None:
                inst_pick = self._instance_for_table_slot(num)
                if inst_pick is None:
                    mx = self._max_table_slot_number()
                    _emit_dim(f"  Instance number {num} out of range (1–{mx})")
                    return None
                if (
                    self.coordinate_dedupe_active
                    and inst_pick.index not in self._kept_coordinate_instance_indices
                ):
                    _emit_dim(
                        f"  Slot #{num} would repeat the output of an earlier row under "
                        f"{self.dedupe_naming_mode.value} naming (purple row + "
                        "gold # in the table). Skipped by default; use "
                        "--all-fvar-instance-rows to include."
                    )
                    return None
                try:
                    idx = self.selection_instances.index(inst_pick)
                except ValueError:
                    _emit_dim(f"  Instance #{num} could not be resolved for generation.")
                    return None

                final_mode = mode if allow_naming else NamingMode.STAT
                result.append((idx, final_mode))

        if not result:
            _emit_dim("  No valid instance numbers provided")
            return None

        return result

    def _prompt_axis_values(
        self,
        variable_axes: List[AxisInfo],
        base_coords: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, float]]:
        """Walk each variable axis; Enter keeps value; [p] previous axis; [x] cancel draft."""
        def _fmt(v: float) -> str:
            if v == int(v):
                return str(int(v))
            return str(v)

        _emit_dim(
            "    [Enter] = keep  ·  [p] previous axis  ·  [x] cancel this instance"
        )

        coords: Dict[str, float] = {}
        i = 0
        while i < len(variable_axes):
            axis = variable_axes[i]
            fallback = (base_coords or {}).get(axis.tag, axis.default_value)
            current = coords.get(axis.tag, fallback)
            label = "currently" if base_coords else "default"

            prompt = (
                f"    {axis.tag:6}  [{_fmt(axis.min_value)}–{_fmt(axis.max_value)}, "
                f"{label} {_fmt(current)}] > "
            )
            while True:
                raw = input(prompt).strip()
                _raise_if_quit(raw)
                low = raw.lower()
                first_tok = low.split()[0] if low else ""
                if first_tok in ("x", "cancel"):
                    return None
                if first_tok in ("p", "prev", "previous"):
                    if i == 0:
                        _emit_dim(
                            "      Already on the first axis — [x] cancels this instance."
                        )
                        continue
                    prev = variable_axes[i - 1]
                    coords.pop(prev.tag, None)
                    i -= 1
                    break
                if raw == "":
                    coords[axis.tag] = current
                    i += 1
                    break
                try:
                    val = float(raw)
                except ValueError:
                    _emit_dim(
                        f"      Not a number. Range: {axis.min_value}–{axis.max_value}"
                    )
                    continue
                if not axis.is_in_range(val):
                    _emit_dim(
                        f"      Out of range ({axis.min_value}–{axis.max_value})"
                    )
                    continue
                coords[axis.tag] = val
                i += 1
                break
        return coords

    def _queue_one_custom_instance(
        self,
        queue: List[_PendingInstance],
        stat_parser: STATNameParser,
        variable_axes: List[AxisInfo],
        fixed_axes: List[AxisInfo],
        base_coords: Optional[Dict[str, float]],
    ) -> None:
        """Prompt for coordinates + name and append to queue, or return if cancelled."""
        while True:
            coords = self._prompt_axis_values(variable_axes, base_coords)
            if coords is None:
                return

            for ax in fixed_axes:
                coords[ax.tag] = ax.min_value

            stat_name = stat_parser.build_subfamily_name(coords, self.metadata)
            has_non_stat = any(
                tag in stat_parser.stat_values
                and val not in stat_parser.stat_values[tag]
                for tag, val in coords.items()
            )
            coord_preview = "  ".join(
                _format_coord_part(k, v) for k, v in _sorted_coord_items(coords)
            )
            cs.emit(f"\n  → {coord_preview}")
            if RICH_AVAILABLE:
                from rich.markup import escape

                hint = "  [dim](between STAT values)[/dim]" if has_non_stat else ""
                cs.emit(
                    f"  → STAT name: [pale_green1]{escape(str(stat_name))}[/pale_green1]"
                    f"{hint}"
                )
            else:
                h = "  (between STAT values)" if has_non_stat else ""
                cs.emit(f"  → STAT name: {stat_name}{h}")

            name_raw = input(
                f"  Name [Enter={stat_name}, or type  ·  e=re-edit  ·  x=cancel] > "
            ).strip()
            _raise_if_quit(name_raw)
            nl = name_raw.lower()

            if nl in ("x", "cancel"):
                return
            if nl in ("e", "edit", "re-edit", "reedit"):
                continue
            if name_raw == "":
                final_name = stat_name
            else:
                final_name = name_raw

            family_name = self.metadata.family_name

            _emit_dim(
                f"  Filename: {family_name}-{final_name.replace(' ', '')}",
            )

            queue.append(_PendingInstance(coordinates=dict(coords), name=final_name))
            _emit_dim(f"  Added ({len(queue)} pending).")
            return

    def _emit_custom_menu(self, queue: List[_PendingInstance]) -> None:
        """Two-line custom builder menu; dim g/r/c when queue is empty (Rich).

        Rich treats ``[letter]`` as markup. Shortcut keys must be passed through
        ``rich.markup.escape`` (or use ``_emit_menu`` on lines with no raw markup).
        """
        if not RICH_AVAILABLE:
            _emit_menu(
                "  [n] new   [f] from fvar #   [g] generate all   "
                "[r] remove #   [c] clear"
            )
            _emit_menu("  [x] back   [q] quit")
            if not queue:
                _emit_dim("  ([g] / [r] / [c] after you queue instances.)")
            return

        from rich.markup import escape

        lead = f"  {escape('[n]')} new   {escape('[f]')} from fvar #   "
        tail = escape("[g] generate all   [r] remove #   [c] clear")
        if queue:
            cs.emit(lead + tail)
        else:
            cs.emit(lead + f"[dim]{tail}[/dim]")
        _emit_menu("  [x] back   [q] quit")

    def show_custom_mode(
        self, font_path: str, axes: List[AxisInfo], stat_parser: STATNameParser
    ) -> Optional[List[Tuple[Dict[str, float], str]]]:
        """Queue-based custom builder.

        Returns a list of (coordinates, name) to generate, or None if the user
        backs out without generating (``[x]``).
        """
        _ = axes  # signature compatibility; axes come from self._classify_axes()
        variable_axes, fixed_axes = self._classify_axes()

        if not variable_axes:
            self._print_header("Custom Instance Builder")
            self._print_axes_table()
            StatusIndicator("warning").add_message(
                "This font has no variable axes — all coordinates are fixed."
            ).emit()
            input("\nPress Enter to continue...")
            # Return [] (not None) — distinguishable from user cancel (None) by run_custom_mode,
            # but both result in no generation. The caller's `if not batch:` handles both.
            return []

        self._print_header("Custom Instance Builder")
        self._print_axes_table()
        cs.emit("")

        queue: List[_PendingInstance] = []

        while True:
            if queue:
                noun = "instance" if len(queue) == 1 else "instances"
                cs.emit(f"\n  Pending ({len(queue)} {noun}):")
                for i, item in enumerate(queue, 1):
                    cs.emit(f"    {i}.  {item.display()}")
            else:
                _emit_dim("\n  Pending:  (none)")

            cs.emit("")
            self._emit_custom_menu(queue)

            choice_line = input("  > ").strip()
            _raise_if_quit(choice_line)
            choice = choice_line.lower()
            key = choice.split()[0] if choice else ""

            if key in ("x", "cancel", "back"):
                return None

            if key in ("g", "go", "generate"):
                if queue:
                    return [(p.coordinates, p.name) for p in queue]
                _emit_dim(
                    "  Nothing queued — add an instance with [n] or [f] first."
                )
                continue

            if key in ("c", "clear"):
                if queue:
                    queue.clear()
                else:
                    _emit_dim("  Queue is already empty.")
                continue

            if key in ("r", "remove"):
                if not queue:
                    _emit_dim("  Queue is empty — nothing to remove.")
                    continue
                raw = input(f"  Remove from queue — # (1–{len(queue)}) > ").strip()
                _raise_if_quit(raw)
                try:
                    idx = int(raw) - 1
                    if 0 <= idx < len(queue):
                        removed = queue.pop(idx)
                        _emit_dim(f"  Removed: {removed.name}")
                    else:
                        _emit_dim("  Out of range")
                except ValueError:
                    _emit_dim("  Invalid number")
                continue

            if key in ("f", "from", "copy"):
                mx = self._max_table_slot_number()
                if mx == 0:
                    _emit_dim("  No fvar instances in this font.")
                    continue
                cs.emit("")
                cs.emit("  Copy from fvar:")
                for inst in self.metadata.instances:
                    coord = "  ".join(
                        _format_coord_part(k, v)
                        for k, v in _sorted_coord_items(inst.coordinates)
                    )
                    label = inst.stat_name or inst.fvar_name
                    slot = self._table_slot_number(inst)
                    cs.emit(f"    {slot:<3} {coord}  ({label})")
                raw = input(
                    f"  Which # (1–{mx})?  [x] cancel > "
                ).strip()
                _raise_if_quit(raw)
                tok = raw.lower().split()[0] if raw.strip() else ""
                if tok in ("x", "cancel"):
                    continue
                try:
                    slot = int(raw)
                except ValueError:
                    _emit_dim("  Not a number")
                    continue
                base_inst = self._instance_for_table_slot(slot)
                if base_inst is None:
                    _emit_dim(f"  No instance #{slot}")
                    continue
                base_coords = dict(base_inst.coordinates)
                base_label = base_inst.stat_name or base_inst.fvar_name
                base_preview = "  ".join(
                    _format_coord_part(k, v)
                    for k, v in _sorted_coord_items(base_coords)
                )
                cs.emit(f"\n  Base: {base_preview}  ({base_label})")
                cs.emit("  Adjust each axis ([Enter] keeps the value shown).")
                self._queue_one_custom_instance(
                    queue,
                    stat_parser,
                    variable_axes,
                    fixed_axes,
                    base_coords,
                )
            elif key in ("n", "new"):
                cs.emit(
                    "\n  New instance — axis values ([Enter] = default for each axis)."
                )
                self._queue_one_custom_instance(
                    queue,
                    stat_parser,
                    variable_axes,
                    fixed_axes,
                    None,
                )
            else:
                if not key:
                    _emit_dim(
                        "  Choose n / f / g / …   [x] back without generating."
                    )
                else:
                    _emit_dim("  Unknown option")
                continue

    def _print_header(self, title: str) -> None:
        """Print section header."""
        cs.fmt_header(title, console)

    def _print_axes_table(self) -> None:
        """Print axes information as table with variable/fixed distinction."""
        variable_axes, fixed_axes = self._classify_axes()

        if variable_axes:
            cs.emit("\nVariable Axes:")
            var_table = cs.create_table(
                show_header=True, row_styles=["on #282a39", "on #1d1f30"]
            )
            if var_table:
                var_table.width = console.size.width
                var_table.add_column("Tag", style="cyan1")
                var_table.add_column("Name", style="pale_green1")
                var_table.add_column("Range", style="turquoise2")
                var_table.add_column("STAT Values", style="dim", no_wrap=False)

                for axis in variable_axes:
                    stat_values = self._format_stat_values_inline(axis.tag)
                    var_table.add_row(
                        axis.tag,
                        axis.name,
                        axis.format_variable_range(),
                        stat_values,
                    )

                console.print(var_table)
            else:
                for axis in variable_axes:
                    stat_values = self._format_stat_values_inline(axis.tag)
                    cs.emit(
                        f"  {axis.tag:6} {axis.name:12} "
                        f"{axis.format_variable_range():20} {stat_values}"
                    )

        if fixed_axes:
            cs.emit("\nFixed Axes:")
            fix_table = cs.create_table(
                show_header=True, row_styles=["on #282a39", "on #1d1f30"]
            )
            if fix_table:
                fix_table.width = console.size.width
                fix_table.add_column("Tag", style="cyan1")
                fix_table.add_column("Name", style="pale_green1")
                fix_table.add_column("Value", style="turquoise2")
                fix_table.add_column("Note", style="dim", no_wrap=False)

                for axis in fixed_axes:
                    stat_values = self._format_stat_values_inline(axis.tag)
                    note = (
                        f"Fixed at {stat_values if stat_values else str(axis.min_value)}"
                    )
                    fix_table.add_row(
                        axis.tag,
                        axis.name,
                        f"{axis.min_value}",
                        note,
                    )

                console.print(fix_table)
            else:
                for axis in fixed_axes:
                    stat_values = self._format_stat_values_inline(axis.tag)
                    cs.emit(
                        f"  {axis.tag:6} {axis.name:12} {axis.min_value} [FIXED]"
                    )
                    cs.emit(
                        "    → All instances will use "
                        f"{stat_values if stat_values else 'this value'}"
                    )

    def _print_validation_notices(self) -> None:
        """Print validation notices about the font."""
        notices = []

        dup_skipped = count_coordinate_duplicate_rows(
            self.metadata.instances,
            self.metadata,
            self.stat_parser,
            self.dedupe_naming_mode,
        )
        if dup_skipped > 0:
            notices.append(
                f"{dup_skipped} named instance row(s) would produce the same output as an "
                f"earlier row under {self.dedupe_naming_mode.value} naming "
                f"(axis values compared with epsilon {AXIS_VALUE_EPSILON}); "
                "only the first row per output identity is generated by default. Use "
                "--all-fvar-instance-rows to emit every fvar record including repeats."
            )

        # Check for instances with no STAT mapping
        for inst in self.metadata.instances:
            if inst.stat_name == "Regular" and inst.fvar_name != UNKNOWN_FVAR_NAME:
                # Check if this is a fallback due to missing STAT values
                has_non_default = any(
                    coord != axis.default_value
                    for axis in self.metadata.axes
                    for tag, coord in inst.coordinates.items()
                    if tag == axis.tag
                )
                if has_non_default:
                    notices.append(
                        f"Instance #{inst.index + 1} '{inst.fvar_name}' may have incomplete STAT mapping"
                    )

        # Check for high nameIDs (>255) that might need cleanup
        # Note: This would require access to the font's name table, which we don't have here
        # So we'll skip this check for now

        if notices:
            cs.emit("\n" + "─" * 70)
            cs.emit("VALIDATION NOTICES")
            cs.emit("─" * 70)
            status = StatusIndicator("info").add_message("Items requiring attention")
            for notice in notices:
                status.add_item(notice, indent_level=1)
            status.emit()

            _emit_bold("\nSuggested actions:")
            cs.emit("  1. Check STAT table completeness")
            cs.emit("  2. Verify instance coordinates match STAT values")
            cs.emit("  3. Consider using [stat] naming mode (default)")
        else:
            cs.emit("\n" + "─" * 70)
            cs.emit("")
            StatusIndicator("success").add_message(
                "Font structure validated - no issues detected"
            ).emit()

    def _get_ribbi_label(self, inst: InstanceInfo) -> str:
        """Get RIBBI classification label."""
        if inst.is_bold and inst.is_italic:
            return "Bold Italic"
        elif inst.is_bold:
            return "Bold"
        elif inst.is_italic:
            return "Italic"
        else:
            return "Regular"


# ============================================================================
# Instance Generator
# ============================================================================


class InstanceGenerator:
    """Generates static font instances."""

    def __init__(
        self,
        analyzer: FontAnalyzer,
        stat_parser: STATNameParser,
        config: InstancerConfig,
        error_tracker: Optional[ErrorTracker] = None,
        metadata: Optional["FontMetadata"] = None,
    ):
        self.analyzer = analyzer
        self.stat_parser = stat_parser
        self.config = config
        self.error_tracker = error_tracker or ErrorTracker()
        self.metadata = metadata
        self.successful_count = 0

    def generate_instance(
        self, coordinates: Dict[str, float], subfamily_name: str
    ) -> Optional[str]:
        """Generate a single static instance.

        Returns: Output path on success, None on failure.
        """
        try:
            # Collect VF nameIDs before instancing
            vf_nameids = self._collect_vf_name_ids(self.analyzer.font)

            # Create fresh font copy
            instance_font = TTFont(self.analyzer.font_path)

            pairpos_fixes = repair_pairpos_second_glyph_order(instance_font)
            if pairpos_fixes:
                logger.debug(
                    "Reordered PairPos targets before instancing: %s",
                    pairpos_fixes,
                )

            # Instantiate
            instancer.instantiateVariableFont(
                instance_font, coordinates, inplace=True, updateFontNames=False
            )

            # Detect italic from result
            italic_angle = instance_font["post"].italicAngle
            is_italic = abs(italic_angle) > ITALIC_ANGLE_THRESHOLD

            # Clean up variable font data
            self._remove_vf_tables(instance_font)
            self._remove_vf_name_ids(instance_font, vf_nameids)

            # Update naming
            self._update_names(instance_font, subfamily_name, is_italic, coordinates)

            # Clean up Mac names if requested
            # Always remove Mac platform records
            self._remove_mac_names(instance_font)

            # Update metrics and style bits
            self._update_metrics_and_bits(instance_font, is_italic, coordinates)

            # Correct italic angle if needed
            if not is_italic and abs(italic_angle) > ITALIC_ANGLE_MIN:
                instance_font["post"].italicAngle = 0.0

            # Save
            output_path = self._save_instance(instance_font, subfamily_name)

            self.successful_count += 1
            return output_path

        except Exception as e:
            logger.error(f"Failed to generate {subfamily_name}: {e}")
            self.error_tracker.add_from_exception(
                context=ErrorContext.CONSTRUCTION,
                exception=e,
                filepath=str(self.analyzer.font_path),
                message=f"Failed to generate instance: {subfamily_name}",
                additional_info={
                    "coordinates": coordinates,
                    "subfamily": subfamily_name,
                },
            )
            StatusIndicator("error").add_message(f"{subfamily_name}").with_explanation(
                str(e)
            ).emit()
            return None

    def _remove_vf_tables(self, font: TTFont) -> None:
        """Remove variable font tables."""
        tables_to_remove = ["fvar", "gvar", "avar", "MVAR", "HVAR", "VVAR", "cvar"]
        if not self.config.keep_stat:
            tables_to_remove.append("STAT")

        for table in tables_to_remove:
            if table in font:
                del font[table]

    def _collect_vf_name_ids(self, font: TTFont) -> set:
        """Collect nameIDs used by variable font tables."""
        vf_nameids = set()
        stat_nameids = set()

        # Collect from fvar
        if "fvar" in font:
            for axis in font["fvar"].axes:
                vf_nameids.add(axis.axisNameID)
            for inst in font["fvar"].instances:
                vf_nameids.add(inst.subfamilyNameID)
                if (
                    hasattr(inst, "postscriptNameID")
                    and inst.postscriptNameID != 0xFFFF
                ):
                    vf_nameids.add(inst.postscriptNameID)

        # Collect from STAT
        if "STAT" in font:
            stat = font["STAT"].table

            if hasattr(stat, "DesignAxisRecord") and stat.DesignAxisRecord:
                for axis in stat.DesignAxisRecord.Axis:
                    stat_nameids.add(axis.AxisNameID)

            if hasattr(stat, "AxisValueArray") and stat.AxisValueArray:
                for value in stat.AxisValueArray.AxisValue:
                    stat_nameids.add(value.ValueNameID)

            if hasattr(stat, "ElidedFallbackNameID") and stat.ElidedFallbackNameID:
                stat_nameids.add(stat.ElidedFallbackNameID)

        # If keeping STAT, don't remove STAT-related IDs
        targets = (
            vf_nameids.union(stat_nameids) if not self.config.keep_stat else vf_nameids
        )

        # Only return nameIDs > 255 (OpenType spec reserves 0-255)
        return {nid for nid in targets if nid > 255}

    def _remove_vf_name_ids(self, font: TTFont, vf_nameids: set) -> None:
        """Remove collected VF nameIDs from instance font."""
        if "name" not in font or not vf_nameids:
            return

        name_table = font["name"]
        name_table.names = [r for r in name_table.names if r.nameID not in vf_nameids]

        # Always remove nameID 25 from static instances
        try:
            name_table.names = [r for r in name_table.names if r.nameID != 25]
        except Exception:
            pass

    def _remove_mac_names(self, font: TTFont) -> None:
        """Remove Macintosh platform name records."""
        if "name" not in font:
            return
        name_table = font["name"]
        name_table.names = [r for r in name_table.names if r.platformID != 1]

    def _update_names(
        self,
        font: TTFont,
        subfamily_name: str,
        is_italic: bool,
        coordinates: Dict[str, float],
    ) -> None:
        """Update font name table."""
        name_table = font["name"]

        family_name = name_table.getDebugName(1) or UNKNOWN_FVAR_NAME
        family_name = strip_variable_tokens(family_name) or family_name

        weight = coordinates.get("wght", WeightClass.REGULAR)

        # Get semantic Bold from STAT (not just weight == 700)
        weight_label = self.stat_parser.get_label_for_axis("wght", weight)
        is_bold = False
        if weight_label:
            label_lower = weight_label.lower()
            if "bold" in label_lower:
                # Exclude compound bold weights
                if not any(
                    prefix in label_lower
                    for prefix in ["extra", "semi", "demi", "ultra", "super"]
                ):
                    is_bold = True

        # Build ID17 (keeps Regular)
        id17 = subfamily_name

        # RIBBI for ID2
        if is_bold and is_italic:
            ribbi = "Bold Italic"
        elif is_bold:
            ribbi = "Bold"
        elif is_italic:
            ribbi = "Italic"
        else:
            ribbi = "Regular"

        # ID1/ID4: Strip "Regular" from style
        style_clean = (
            id17.replace(" Regular ", " ")
            .replace(" Regular", "")
            .replace("Regular ", "")
        )
        if style_clean == "Regular":
            style_clean = ""
        style_clean = " ".join(style_clean.split())

        id1 = f"{family_name} {style_clean}" if style_clean else family_name
        id4 = f"{family_name} {style_clean}" if style_clean else family_name

        # ID6: PostScript
        ps_name = sanitize_postscript(f"{family_name}-{id17}")

        id16 = family_name

        self._set_name(name_table, 1, id1)
        self._set_name(name_table, 2, ribbi)
        self._set_name(name_table, 4, id4)
        self._set_name(name_table, 6, ps_name)
        self._set_name(name_table, 16, id16)
        self._set_name(name_table, 17, id17)

        # WWS naming removed - no longer supported

        # Deduplicate core records
        try:
            core_ids = [1, 2, 4, 6, 16, 17]
            for nid in core_ids:
                deduplicate_namerecords_binary(name_table, nid)
        except Exception as e:
            logger.warning(f"Failed to deduplicate name records: {e}")

    def _compute_width_class(self, wdth_value: float) -> int:
        """Compute a relative usWidthClass (1–9) from a wdth axis value.

        The OS/2 spec defines usWidthClass as an integer 1–9 where 5 is
        "Normal / Regular".  Because the axis range is designer-defined and
        has no fixed relationship to those nine slots, we derive the value
        proportionally:

          • The axis *default* always maps to 5 (Normal).
          • Values below the default are mapped linearly from the axis minimum
            (→ 1, Ultra-Condensed) up to the default (→ 5, Normal).
          • Values above the default are mapped linearly from the default
            (→ 5) up to the axis maximum (→ 9, Ultra-Expanded).

        If the font has no wdth axis the current OS/2 value is left unchanged
        (this method returns None in that case, handled by the caller).

        The result is clamped to [1, 9] for safety.
        """
        metadata = self.metadata
        if metadata is None:
            return None

        wdth_axis = next(
            (a for a in metadata.axes if a.tag == "wdth"), None
        )
        if wdth_axis is None:
            return None

        default = wdth_axis.default_value
        min_val = wdth_axis.min_value
        max_val = wdth_axis.max_value

        # Axis with no real range — treat everything as Normal (5)
        if min_val >= max_val:
            return 5

        if abs(wdth_value - default) < 0.001:
            return 5

        if wdth_value < default:
            # Condensed side: [min_val, default] → [1, 5]
            span = default - min_val
            if span < 0.001:
                return 5
            ratio = (wdth_value - min_val) / span   # 0.0 at min → 1.0 at default
            width_class = 1.0 + ratio * 4.0          # 1 at min → 5 at default
        else:
            # Expanded side: [default, max_val] → [5, 9]
            span = max_val - default
            if span < 0.001:
                return 5
            ratio = (wdth_value - default) / span    # 0.0 at default → 1.0 at max
            width_class = 5.0 + ratio * 4.0          # 5 at default → 9 at max

        return max(1, min(9, round(width_class)))

    def _update_metrics_and_bits(
        self, font: TTFont, is_italic: bool, coordinates: Dict[str, float]
    ) -> None:
        """Update font metrics and style bits."""
        weight = coordinates.get("wght", WeightClass.REGULAR)

        # Get semantic Bold from STAT
        weight_label = self.stat_parser.get_label_for_axis("wght", weight)
        is_bold = False
        if weight_label:
            label_lower = weight_label.lower()
            if "bold" in label_lower:
                if not any(
                    prefix in label_lower
                    for prefix in ["extra", "semi", "demi", "ultra", "super"]
                ):
                    is_bold = True

        # Update OS/2 weight class and width class
        if "OS/2" in font:
            font["OS/2"].usWeightClass = int(round(weight))

            # Set usWidthClass relative to the wdth axis range so that the
            # axis default always maps to 5 (Normal), condensed instances
            # count down toward 1, and expanded instances count up toward 9.
            wdth_value = coordinates.get("wdth")
            if wdth_value is not None:
                computed_width_class = self._compute_width_class(wdth_value)
                if computed_width_class is not None:
                    font["OS/2"].usWidthClass = computed_width_class
                    logger.debug(
                        "Set usWidthClass=%d for wdth=%.2f", computed_width_class, wdth_value
                    )

            # Update fsSelection
            selection = font["OS/2"].fsSelection
            selection &= ~((1 << 0) | (1 << 5) | (1 << 6))  # Clear bits

            if not is_bold and not is_italic:
                selection |= 1 << 6  # Regular
            else:
                if is_bold:
                    selection |= 1 << 5
                if is_italic:
                    selection |= 1 << 0

            # WWS bit setting removed - no longer supported

            font["OS/2"].fsSelection = selection

        # Update head macStyle
        if "head" in font:
            mac_style = 0
            if is_bold:
                mac_style |= 0b01
            if is_italic:
                mac_style |= 0b10
            font["head"].macStyle = mac_style

    def _set_name(self, name_table, name_id: int, value: str) -> None:
        """Set name on Windows and Mac platforms."""
        name_table.setName(value, name_id, 3, 1, 0x409)
        name_table.setName(value, name_id, 1, 0, 0x409)

    def _get_output_extension(self, font: TTFont) -> str:
        """Determine output extension based on outline format."""
        if "CFF " in font or "CFF2" in font:
            return ".otf"
        return ".ttf"

    def _save_instance(self, font: TTFont, subfamily_name: str) -> str:
        """Save instance font to file.

        Generates filename from PostScript name (nameID 6) or constructs it from
        family name and subfamily name. Handles duplicate filenames by appending
        a tilde-prefixed counter (e.g., "Font~001.ttf", "Font~002.ttf").

        Duplicate handling strategy:
        1. Check if base filename exists
        2. If exists, increment counter and try again (up to 1000 attempts)
        3. Handle race conditions where file is created between exists() check
           and save() call by catching OSError/FileExistsError and retrying

        In dry-run mode, logs the path that would be used without actually saving.

        Args:
            font: TTFont object to save
            subfamily_name: Subfamily name for filename construction if PostScript
                name is unavailable

        Returns:
            Absolute path string of saved file (or would-be path in dry-run mode)

        Raises:
            RuntimeError: If unable to save after 1000 attempts (shouldn't happen
                in normal operation)
        """
        # Use PostScript name (ID6)
        ps_name = font["name"].getDebugName(6)
        if not ps_name:
            family = font["name"].getDebugName(16) or UNKNOWN_FVAR_NAME
            ps_name = sanitize_postscript(f"{family}-{subfamily_name}")

        # Detect correct extension
        extension = self._get_output_extension(font)
        filename = f"{ps_name}{extension}"

        output_dir = self.config.output_dir or Path(self.analyzer.font_path).parent
        output_path = output_dir / filename

        # Handle duplicates and race conditions
        counter = 1
        max_attempts = 1000  # Prevent infinite loops
        attempts = 0

        while attempts < max_attempts:
            # Check if file exists (for initial duplicate detection)
            if output_path.exists():
                base = ps_name.rsplit(".", 1)[0] if "." in ps_name else ps_name
                filename = f"{base}~{counter:03d}{extension}"
                output_path = output_dir / filename
                counter += 1
                attempts += 1
                continue

            # Dry-run mode: log what would be saved without actually saving
            if self.config.dry_run:
                logger.info(f"Would save: {output_path}")
                return str(output_path)

            # Attempt to save with race condition handling
            try:
                font.save(str(output_path))
                return str(output_path)
            except (OSError, FileExistsError) as e:
                # Race condition: file was created between exists() check and save()
                # Retry with incremented counter
                base = ps_name.rsplit(".", 1)[0] if "." in ps_name else ps_name
                filename = f"{base}~{counter:03d}{extension}"
                output_path = output_dir / filename
                counter += 1
                attempts += 1
                logger.debug(
                    f"Race condition detected ({type(e).__name__}: {e}), retrying with filename: {filename}"
                )
                continue

        # If we've exhausted attempts, raise an error
        raise RuntimeError(
            f"Could not save file after {max_attempts} attempts. Last attempted path: {output_path}"
        )


# ============================================================================
# Main Processor
# ============================================================================


class FontProcessor:
    """Main font processing coordinator."""

    def __init__(
        self,
        font_path: str,
        config: InstancerConfig,
        error_tracker: Optional[ErrorTracker] = None,
    ):
        self.font_path = font_path
        self.config = config
        self.error_tracker = error_tracker or ErrorTracker()
        self.analyzer = FontAnalyzer(font_path)
        self.metadata: Optional[FontMetadata] = None
        self.stat_parser: Optional[STATNameParser] = None
        self.last_generator: Optional[InstanceGenerator] = None

    def run_info_mode(self, json_output: bool = False) -> None:
        """Run information-only mode."""
        if not self.analyzer.load_and_validate():
            if json_output:
                print(json.dumps({"error": "Failed to load or validate font"}))
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        if json_output:
            # Output JSON format
            output = {
                "family_name": self.metadata.family_name,
                "axes": [asdict(axis) for axis in self.metadata.axes],
                "instances": [asdict(inst) for inst in self.metadata.instances],
                "stat_values": {
                    axis_tag: {str(val): name for val, name in values.items()}
                    for axis_tag, values in self.metadata.stat_values.items()
                },
                "source_italic": self.metadata.source_italic,
                "coordinate_duplicate_row_count": count_coordinate_duplicate_rows(
                    self.metadata.instances,
                    self.metadata,
                    self.stat_parser,
                    default_naming_mode_for_instances(self.metadata.instances),
                ),
            }
            print(json.dumps(output, indent=2))
        else:
            ui = InteractivePrompt(self.metadata, self.stat_parser)
            ui.show_info_mode()

    def run_custom_mode(self) -> None:
        """Run custom instance creation mode."""
        if not self.analyzer.load_and_validate():
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        ui = InteractivePrompt(self.metadata, self.stat_parser)
        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
            metadata=self.metadata,
        )
        self.last_generator = generator

        batch = ui.show_custom_mode(
            self.font_path, self.metadata.axes, self.stat_parser
        )
        if not batch:
            _emit_dim("  Left custom instance builder — nothing generated.")
            return

        cs.emit(f"\nGenerating {cs.fmt_count(len(batch))} custom instance(s)...")
        for coordinates, custom_name in batch:
            output_path = generator.generate_instance(coordinates, custom_name)
            if output_path:
                StatusIndicator("success", dry_run=self.config.dry_run).add_message(
                    "Generated"
                ).add_file(Path(output_path).name, filename_only=True).emit()

        if generator.successful_count > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            cs.emit(
                f"\nGenerated {generator.successful_count} instance(s) in: {output_dir}"
            )

    def run_instance_mode(self) -> None:
        """Run named instance generation mode."""
        if not self.analyzer.load_and_validate():
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        selection, _ = instances_for_processing(
            self.metadata,
            self.config,
            self.stat_parser,
            default_naming_mode_for_instances(self.metadata.instances),
        )

        ui = InteractivePrompt(
            self.metadata,
            self.stat_parser,
            selection_instances=selection,
            coordinate_dedupe_active=self.config.skip_coordinate_duplicates,
            naming_mode=default_naming_mode_for_instances(self.metadata.instances),
        )

        while True:
            result = ui.show_instance_selection()
            if result is None:
                return
            if result == "custom":
                self.run_custom_mode()
                # run_custom_mode sets last_generator; successful_count reflects this visit.
                gen = self.last_generator
                if gen is not None and gen.successful_count > 0:
                    _emit_dim(
                        f"  ↩ Returned from custom builder "
                        f"({gen.successful_count} instance(s) generated)."
                    )
                continue
            instances_with_modes, default_mode = result
            break

        naming_strategy = InstanceNamingStrategy(
            self.metadata, self.stat_parser, default_mode
        )

        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
            metadata=self.metadata,
        )
        self.last_generator = generator

        if not instances_with_modes:  # Generate all
            cs.emit(
                f"\nGenerating {cs.fmt_count(len(selection))} instances..."
            )

            for inst_num, inst in enumerate(selection, 1):
                final_name = naming_strategy.resolve_name(inst)

                output_path = generator.generate_instance(inst.coordinates, final_name)
                if output_path:
                    filename = Path(output_path).name
                    cs.emit(f"  [{inst_num}] Generated: {filename}")

        else:  # Generate specific instances
            cs.emit(
                f"\nGenerating {cs.fmt_count(len(instances_with_modes))} instance(s)..."
            )

            for idx, mode in instances_with_modes:
                inst = selection[idx]
                inst_num = idx + 1

                # Create strategy for this specific mode
                specific_strategy = InstanceNamingStrategy(
                    self.metadata, self.stat_parser, mode
                )
                final_name = specific_strategy.resolve_name(inst)

                output_path = generator.generate_instance(inst.coordinates, final_name)
                if output_path:
                    filename = Path(output_path).name
                    cs.emit(f"  [{inst_num}] Generated: {filename}")

        if generator.successful_count > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            cs.emit("")
            StatusIndicator("success", dry_run=self.config.dry_run).add_message(
                f"Generated {cs.fmt_count(generator.successful_count)} instance(s)"
            ).add_item(
                f"Output directory: {cs.fmt_file_compact(str(output_dir))}"
            ).emit()

    def run_auto_mode(self, json_output: bool = False, instance_indices: Optional[List[int]] = None) -> None:
        """Run automatic generation mode (no prompts).
        
        Args:
            json_output: If True, output JSON instead of console messages
            instance_indices: If None, generate every processing row from ``instances_for_processing``.
                If a list (possibly empty), 0-based indices into that list ``selection``.
                CLI ``--instance`` values are interpreted in ``main()`` as fvar ``#`` column
                numbers (`InstanceInfo.index + 1`) and converted to these indices.
        """
        if not self.analyzer.load_and_validate():
            if json_output:
                print(json.dumps({"error": "Failed to load or validate font"}))
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        naming_strategy = InstanceNamingStrategy(
            self.metadata, self.stat_parser, self.config.naming_mode
        )

        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
            metadata=self.metadata,
        )
        self.last_generator = generator

        selection, coord_dedupe_skipped = instances_for_processing(
            self.metadata, self.config, self.stat_parser
        )

        instances_to_generate = selection
        if instance_indices is not None:
            instances_to_generate = [
                selection[i]
                for i in instance_indices
                if isinstance(i, int) and 0 <= i < len(selection)
            ]

        if not json_output:
            has_fvar_names = any(
                inst.fvar_name != UNKNOWN_FVAR_NAME
                for inst in self.metadata.instances
            )
            ui = InteractivePrompt(
                self.metadata,
                self.stat_parser,
                selection_instances=selection,
                coordinate_dedupe_active=self.config.skip_coordinate_duplicates,
                naming_mode=self.config.naming_mode,
            )
            ui._print_header("Named Instances")
            any_names_differ, has_dup_coords = ui._print_instances_table_with_naming(
                show_naming_comparison=has_fvar_names,
                show_legend_banners=False,
            )
            ui._print_table_legend(any_names_differ, has_dup_coords, has_fvar_names)
            mode_label = self.config.naming_mode.value
            cs.emit(
                f"\nAuto-generating {cs.fmt_count(len(instances_to_generate))} instances ({mode_label} names)..."
            )
            if coord_dedupe_skipped > 0:
                StatusIndicator("info").add_message(
                    f"Skipped {coord_dedupe_skipped} redundant fvar row(s) that would produce "
                    f"the same output under {self.config.naming_mode.value} naming "
                    "(default; --all-fvar-instance-rows emits all rows)"
                ).emit()

        generated_files = []
        for inst_num, inst in enumerate(instances_to_generate, 1):
            final_name = naming_strategy.resolve_name(inst)

            output_path = generator.generate_instance(inst.coordinates, final_name)
            if output_path:
                filename = Path(output_path).name
                if json_output:
                    generated_files.append({
                        "instance_index": inst.index,
                        "name": final_name,
                        "filename": filename,
                        "path": output_path
                    })
                else:
                    cs.emit(f"  [{inst_num}] Generated: {filename}")

        if json_output:
            output = {
                "success": generator.successful_count > 0,
                "generated_count": generator.successful_count,
                "total_instances": len(self.metadata.instances),
                "processing_rows": len(selection),
                "unique_coordinates_mode": self.config.skip_coordinate_duplicates,
                "skipped_duplicate_coordinate_rows": coord_dedupe_skipped,
                "files": generated_files,
                "output_dir": str(self.config.output_dir or Path(self.font_path).parent)
            }
            print(json.dumps(output, indent=2))
        elif generator.successful_count > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            cs.emit("")
            StatusIndicator("success", dry_run=self.config.dry_run).add_message(
                f"Generated {cs.fmt_count(generator.successful_count)} instance(s)"
            ).add_item(f"Output: {cs.fmt_file_compact(str(output_dir))}").emit()

    def run_instances_json_mode(
        self, instances_list: List[Dict], json_output: bool = False
    ) -> None:
        """Run generation from a list of instance definitions (name + coordinates).

        Each item in instances_list must have "name" and "coordinates".
        Validates axis tags and coordinate ranges; skips invalid entries with a warning.
        """
        if not self.analyzer.load_and_validate():
            if json_output:
                print(json.dumps({"error": "Failed to load or validate font"}))
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        axis_by_tag = {a.tag: a for a in self.metadata.axes}

        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
            metadata=self.metadata,
        )
        self.last_generator = generator

        generated_files: List[Dict] = []
        skipped = 0

        if not json_output:
            cs.emit(
                f"Generating {cs.fmt_count(len(instances_list))} instance(s) from JSON..."
            )

        for item in instances_list:
            name = item.get("name")
            coordinates = item.get("coordinates")
            if not name or coordinates is None:
                skipped += 1
                if not json_output:
                    _emit_dim(
                        f"  [skip] Missing 'name' or 'coordinates': {item}",
                    )
                continue

            if not isinstance(coordinates, dict):
                skipped += 1
                if not json_output:
                    _emit_dim(
                        f"  [skip] Invalid coordinates for '{name}': expected dict",
                    )
                continue

            # Validate: all axis tags exist and values in range
            valid = True
            for tag, val in coordinates.items():
                if tag not in axis_by_tag:
                    if not json_output:
                        _emit_dim(
                            f"  [skip] '{name}': axis '{tag}' not in font",
                        )
                    valid = False
                    break
                try:
                    v = float(val)
                except (TypeError, ValueError):
                    if not json_output:
                        _emit_dim(
                            f"  [skip] '{name}': non-numeric value for {tag}",
                        )
                    valid = False
                    break
                if not axis_by_tag[tag].is_in_range(v):
                    if not json_output:
                        _emit_dim(
                            f"  [skip] '{name}': {tag}={val} outside range "
                            f"{axis_by_tag[tag].min_value}–{axis_by_tag[tag].max_value}",
                        )
                    valid = False
                    break

            if not valid:
                skipped += 1
                continue

            # Ensure all variable axes have a value (fill defaults for missing)
            coords = dict(coordinates)
            for axis in self.metadata.axes:
                if axis.is_variable() and axis.tag not in coords:
                    coords[axis.tag] = axis.default_value

            output_path = generator.generate_instance(coords, name)
            if output_path:
                filename = Path(output_path).name
                if json_output:
                    generated_files.append({
                        "name": name,
                        "filename": filename,
                        "path": output_path,
                        "coordinates": coords,
                    })
                else:
                    cs.emit(f"  Generated: {filename}")

        if json_output:
            output = {
                "success": generator.successful_count > 0,
                "generated_count": generator.successful_count,
                "skipped": skipped,
                "total_requested": len(instances_list),
                "files": generated_files,
                "output_dir": str(
                    self.config.output_dir or Path(self.font_path).parent
                ),
            }
            print(json.dumps(output, indent=2))
        elif generator.successful_count > 0 or skipped > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            cs.emit("")
            StatusIndicator("success", dry_run=self.config.dry_run).add_message(
                f"Generated {cs.fmt_count(generator.successful_count)} instance(s)"
            ).add_item(f"Skipped: {cs.fmt_count(skipped)}").add_item(
                f"Output: {cs.fmt_file_compact(str(output_dir))}"
            ).emit()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    # Check if first argument is a subcommand
    if len(sys.argv) > 1 and sys.argv[1] in ["info", "custom"]:
        # Handle subcommands
        parser = argparse.ArgumentParser(
            description="Extract static instances from variable fonts",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Info subcommand
        info_parser = subparsers.add_parser(
            "info",
            help="Display font information (axes, instances, STAT values)",
            description="Display comprehensive font information including axes, instances, and validation notices",
        )
        info_parser.add_argument("font", help="Variable font file to analyze")
        info_parser.add_argument(
            "--json",
            action="store_true",
            help="Output results as JSON (for app integration)",
        )

        # Custom subcommand
        custom_parser = subparsers.add_parser(
            "custom",
            help="Build custom instances interactively",
            description="Create custom instances by specifying axis coordinates interactively",
        )
        custom_parser.add_argument("font", help="Variable font file to instance")
        # Add global options to custom mode
        custom_parser.add_argument(
            "-o",
            "--output-dir",
            type=str,
            default=None,
            help="Directory for output files (default: same as input font)",
        )
        custom_parser.add_argument(
            "-ks",
            "--keep-stat",
            action="store_true",
            help="Keep STAT table in static instances (removed by default)",
        )
        custom_parser.add_argument(
            "-c",
            "--coords",
            type=str,
            help="Generate instance at specific coordinates (e.g., 'wght=700,slnt=-10')",
        )
    else:
        # Handle default generate mode
        parser = argparse.ArgumentParser(
            description="Extract static instances from variable fonts",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
USAGE EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Basic Usage
───────────
  %(prog)s font.ttf
      Interactive mode - choose instances from a menu

  %(prog)s font.ttf -y
      Generate all instances without prompts (STAT names)

Information & Analysis
──────────────────────
  %(prog)s info font.ttf
      Display font structure (axes, instances, STAT values)
      
  %(prog)s font.ttf --show-info
      Show info then continue to generation

Custom Instances
────────────────
  %(prog)s custom font.ttf
      Interactive coordinate builder
      
  %(prog)s custom font.ttf -i "wght=700,slnt=-10"
      Generate custom instance directly from coordinates

Quick Instance Selection
────────────────────────
  %(prog)s font.ttf -i 7
      Generate only instance #7 (interactive naming choice)
      
  %(prog)s font.ttf -i 1,3,5 -s
      Generate instances 1, 3, and 5 with STAT names
      
  %(prog)s font.ttf -i 2,4,6 -f
      Generate instances 2, 4, and 6 with fvar-hybrid names

Batch Processing
────────────────
  %(prog)s font1.ttf font2.ttf -yes
      Process multiple fonts
      
  %(prog)s fonts/ -yes
      Process all fonts in directory
      
  %(prog)s fonts/ -r -y
      Process directory recursively
      
  %(prog)s fonts/ -jb gd-foundry-styles.json
      Generate instances from JSON (font basename → instance list); implies -y

Naming Strategy Options
───────────────────────
  %(prog)s font.ttf -y -s
      STAT names (default) - from STAT table AxisValues
      
  %(prog)s font.ttf -y -f
      fvar-hybrid - fvar names with smart "Regular" completions
      
  %(prog)s font.ttf -y -r
      fvar-raw - use fvar names without modifications

Output Control
──────────────
  %(prog)s font.ttf -y -os
      Save instances to 'static' subdirectory
      
  %(prog)s font.ttf -y --output-dir ./instances
      Save instances to specific directory
      
  %(prog)s font.ttf -y -ks
      Keep STAT table in generated instances
      
  %(prog)s font.ttf -y --all-fvar-instance-rows
      Emit every fvar instance row, including redundant coordinate/name pairs
      
  %(prog)s font.ttf -dry
      Preview what would be generated (no files created)

Advanced Examples
─────────────────
  %(prog)s fonts/*.ttf -yes -f --output-dir ./static
      Batch process with fvar-hybrid naming to custom directory
      
  %(prog)s font.ttf -i "wght=400" -i "wght=700" -yes
      Generate multiple custom instances in one command
      
  %(prog)s fonts/ -r -y -s -n
      Preview recursive batch processing with STAT names

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAMING STRATEGIES EXPLAINED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STAT Names (-s, default)
  • Derived from STAT table AxisValue records
  • Most reliable and consistent
  • Respects type designer's intent
  • Canonical ordering: Width → Weight → Slope
  Example: "Condensed Bold Italic"

fvar-hybrid Names (-f)
  • Uses fvar instance subfamily names
  • Automatically adds "Regular" when appropriate
  • Family-aware: only adds when other weights exist
  • Good for legacy fonts with incomplete STAT
  Example: "Italic" → "Regular Italic" (if family has Bold)

fvar-raw Names (-r)
  • Uses fvar names exactly as specified
  • No modifications or completions
  • Useful when fvar names are intentionally minimal
  Example: "Italic" → "Italic" (always)
""",
        )

        # Default mode (generate) - fonts as positional argument
    parser.add_argument(
        "fonts",
        nargs="*",
        default=None,
        help="Path(s) to variable font files or directories (default: current directory)",
    )

    # Auto mode flag (applies to default generate mode)
    parser.add_argument(
        "-y",
        "--yes",
        dest="yes",
        action="store_true",
        help="Auto-confirm and generate all instances (skip interactive prompts)",
    )

    # Naming strategy (mutually exclusive)
    naming_group = parser.add_mutually_exclusive_group()
    naming_group.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help="Use STAT table names (default)",
    )
    naming_group.add_argument(
        "-fh",
        "--fvar-hybrid",
        action="store_true",
        help="Use fvar names with completions",
    )
    naming_group.add_argument(
        "-fr",
        "--fvar-raw",
        action="store_true",
        help="Use raw fvar names only",
    )

    # Quick instance generation
    parser.add_argument(
        "-i",
        "--instance",
        type=str,
        help=(
            "Comma-separated fvar instance slots (numbers in the '#' column; "
            "redundant fvar rows—purple row, gold #—are rejected unless "
            "--all-fvar-instance-rows)."
        ),
    )
    parser.add_argument(
        "-c",
        "--coords",
        type=str,
        help="Generate instance at coordinates (e.g., 'wght=400,slnt=0')",
    )

    # Directory processing
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process directories recursively",
    )

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output files (default: same as input font)",
    )
    output_group.add_argument(
        "-os",
        "--output-static",
        action="store_true",
        help="Create and use 'static' subdirectory in source font directory",
    )

    # Font options
    parser.add_argument(
        "-ks",
        "--keep-stat",
        action="store_true",
        help="Keep STAT table in static instances (removed by default)",
    )
    parser.set_defaults(skip_coordinate_duplicates=True)
    parser.add_argument(
        "-uc",
        "--unique-coordinates",
        dest="skip_coordinate_duplicates",
        action="store_true",
        help=(
            "Keep only the first fvar row per output identity (default). "
            f"Coordinates snap to multiples of {AXIS_VALUE_EPSILON}; output names follow "
            "the active naming mode (-s / -fh / -fr). "
            "Same as default; kept for existing scripts."
        ),
    )
    parser.add_argument(
        "--all-fvar-instance-rows",
        dest="skip_coordinate_duplicates",
        action="store_false",
        help=(
            "Generate every fvar instance record, including rows that would produce the same "
            "output as an earlier row under the active naming mode "
            "(disables default redundant-row skipping)."
        ),
    )

    # Display options
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview mode: show what would be done without generating files",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for app integration)",
    )
    parser.add_argument(
        "-jb",
        "--json-batch",
        type=str,
        metavar="FILE",
        default=None,
        help="JSON file mapping font basenames to instance lists (name + coordinates); implies -y",
    )

    args = parser.parse_args()
    
    # Use module-level console, override only when JSON mode
    # (must reference global to avoid UnboundLocalError)
    global console
    console_output = console
    
    # Disable rich console output when --json is present
    if args.json:
        import logging
        logging.basicConfig(level=logging.ERROR)
        cs.RICH_AVAILABLE = False
        # Suppress console output
        console_output = None

    # Handle subcommands (info, custom)
    if hasattr(args, "command") and args.command == "info":
        # Info mode with single font
        if not Path(args.font).exists():
            if args.json:
                print(json.dumps({"error": f"Font file not found: {args.font}"}))
            else:
                StatusIndicator("error").add_message("Font file not found").add_file(
                    args.font, filename_only=False
                ).emit()
            return 1

        if not args.json:
            cs.fmt_header("Variable Font Instancer - Font Information", console_output)

        # Create minimal config for info mode
        config = InstancerConfig()
        processor = FontProcessor(args.font, config)

        try:
            processor.run_info_mode(json_output=args.json)
        except Exception as e:
            if args.json:
                print(json.dumps({"error": f"Failed to analyze font: {str(e)}"}))
            else:
                logger.error(f"Failed to analyze {args.font}: {e}")
                StatusIndicator("error").add_message(
                    "Failed to analyze font"
                ).with_explanation(str(e)).emit()
            return 1

        return 0

    elif hasattr(args, "command") and args.command == "custom":
        # Custom mode with single font
        if not Path(args.font).exists():
            StatusIndicator("error").add_message("Font file not found").add_file(
                args.font, filename_only=False
            ).emit()
            return 1

        cs.fmt_header("Variable Font Instancer - Custom Instance Builder", console_output)

        # Create config for custom mode
        config = InstancerConfig(
            output_dir=Path(args.output_dir) if args.output_dir else None,
            keep_stat=args.keep_stat if hasattr(args, "keep_stat") else False,
        )
        processor = FontProcessor(args.font, config)

        try:
            processor.run_custom_mode()
        except Exception as e:
            logger.error(f"Failed to process {args.font}: {e}")
            StatusIndicator("error").add_message(
                "Failed to process font"
            ).with_explanation(str(e)).emit()
            return 1

        return 0

    # Default generate mode
    # If no paths provided, default to current directory
    if not args.fonts:
        args.fonts = ["."]

    # Collect all font files using core_file_collector
    font_files = collect_font_files(
        args.fonts,
        recursive=getattr(args, "recursive", False),
        allowed_extensions={".ttf", ".otf"},  # Only TTF/OTF for variable fonts
    )

    if not font_files:
        StatusIndicator("error").add_message("No font files found to process").emit()
        return 1

    # Load JSON batch mapping if --json-batch provided
    json_batch_data: Optional[Dict[str, List[Dict]]] = None
    if args.json_batch:
        jb_path = Path(args.json_batch)
        if not jb_path.exists():
            StatusIndicator("error").add_message("JSON batch file not found").add_file(
                args.json_batch, filename_only=False
            ).emit()
            return 1
        try:
            with open(jb_path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                StatusIndicator("error").add_message(
                    "JSON batch file must be an object (font basename → instance list)"
                ).emit()
                return 1
            json_batch_data = raw
            args.yes = True  # No prompts when using JSON batch
        except json.JSONDecodeError as e:
            StatusIndicator("error").add_message("Invalid JSON in batch file").with_explanation(
                str(e)
            ).emit()
            return 1

    # Determine naming strategy from flags (default)
    if args.fvar_hybrid:
        naming_mode = NamingMode.FVAR_HYBRID
    elif args.fvar_raw:
        naming_mode = NamingMode.FVAR_RAW
    else:
        naming_mode = NamingMode.STAT  # default

    # Determine output directory
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output_static:
        # For batch processing, we'll set this per-font in the loop
        # For single font, set it now
        if len(font_files) == 1:
            output_dir = Path(font_files[0]).parent / "static"
            output_dir.mkdir(exist_ok=True)

    # Create configuration
    config = InstancerConfig(
        output_dir=output_dir,
        keep_stat=args.keep_stat,
        naming_mode=naming_mode,
        dry_run=args.dry_run,
        skip_coordinate_duplicates=getattr(
            args, "skip_coordinate_duplicates", True
        ),
    )

    # Show header
    cs.fmt_header("Variable Font Instancer - Extract static instances", console_output)

    if len(font_files) > 1:
        StatusIndicator("info").add_message(
            f"Batch Processing: {cs.fmt_count(len(font_files))} font(s)"
        ).emit()
        cs.emit("=" * 60)

    # Initialize error tracker for batch processing
    error_tracker = ErrorTracker()

    # Process each font
    total_instances = 0
    successful_fonts = 0

    try:
        for idx, font_path in enumerate(font_files, 1):
            if len(font_files) > 1:
                cs.emit(f"\n[{idx}/{len(font_files)}] {cs.fmt_file_compact(font_path)}")

            # If using --output-static, create static dir per font
            if args.output_static and not args.output_dir:
                config.output_dir = Path(font_path).parent / "static"
                config.output_dir.mkdir(exist_ok=True)

            processor = FontProcessor(font_path, config, error_tracker=error_tracker)

            try:
                basename = Path(font_path).name
                if json_batch_data is not None and basename in json_batch_data:
                    processor.run_instances_json_mode(
                        json_batch_data[basename], json_output=args.json
                    )
                    if (
                        processor.last_generator
                        and processor.last_generator.successful_count > 0
                    ):
                        total_instances += processor.last_generator.successful_count
                        successful_fonts += 1
                elif args.yes:
                    # Parse instance indices if --instance specified
                    instance_indices = None
                    if args.instance:
                        # Resolve --instance slots (match interactive '#' column: inst.index + 1)
                        if processor.analyzer.load_and_validate():
                            processor.metadata = processor.analyzer.analyze()
                            processor.stat_parser = processor.analyzer.stat_parser
                            try:
                                if processor.metadata:
                                    slot_nums = [
                                        int(x.strip())
                                        for x in args.instance.split(",")
                                        if x.strip()
                                    ]
                                    processing_rows, _ = instances_for_processing(
                                        processor.metadata, config, processor.stat_parser
                                    )
                                    kept_slots = coordinate_first_kept_instance_indices(
                                        processor.metadata.instances,
                                        processor.metadata,
                                        processor.stat_parser,
                                        config.naming_mode,
                                    )
                                    mx_slot = (
                                        max(
                                            i.index + 1 for i in processor.metadata.instances
                                        )
                                        if processor.metadata.instances
                                        else 0
                                    )
                                    instance_indices = []
                                    for slot in slot_nums:
                                        picked = None
                                        for ti in processor.metadata.instances:
                                            if ti.index + 1 == slot:
                                                picked = ti
                                                break
                                        if picked is None:
                                            StatusIndicator("warning").add_message(
                                                f"--instance slot {slot} out of range (1-{mx_slot})"
                                            ).emit()
                                            continue
                                        if (
                                            config.skip_coordinate_duplicates
                                            and picked.index not in kept_slots
                                        ):
                                            StatusIndicator("warning").add_message(
                                                f"--instance {slot} would repeat the output of an earlier row "
                                                f"under {config.naming_mode.value} naming "
                                                "(skipped by default; pass --all-fvar-instance-rows to include)"
                                            ).emit()
                                            continue
                                        try:
                                            instance_indices.append(
                                                processing_rows.index(picked)
                                            )
                                        except ValueError:
                                            StatusIndicator("warning").add_message(
                                                f"--instance slot {slot} could not be resolved"
                                            ).emit()
                            except (ValueError, AttributeError):
                                pass  # Invalid format; run_auto_mode generates all instances
                    
                    processor.run_auto_mode(json_output=args.json, instance_indices=instance_indices)
                    if hasattr(processor, "metadata") and processor.metadata:
                        # Count successful instances
                        if (
                            processor.last_generator
                            and processor.last_generator.successful_count > 0
                        ):
                            total_instances += processor.last_generator.successful_count
                            successful_fonts += 1
                else:
                    processor.run_instance_mode()
                    if hasattr(processor, "metadata") and processor.metadata:
                        if (
                            processor.last_generator
                            and processor.last_generator.successful_count > 0
                        ):
                            total_instances += processor.last_generator.successful_count
                            successful_fonts += 1

            except Exception as e:
                logger.error(f"Failed to process {font_path}: {e}")
                StatusIndicator("error").add_message("Failed to process font").add_file(
                    font_path, filename_only=False
                ).with_explanation(str(e)).emit()
                continue

    except SystemExit:
        # User chose quit - exit entire program
        cs.emit("")
        StatusIndicator("info").add_message("User requested exit").emit()
        return 0
    except KeyboardInterrupt:
        # Ctrl+C - also exit
        cs.emit("")
        StatusIndicator("info").add_message("Operation cancelled by user").emit()
        return 1

    # Final summary for batch processing
    if len(font_files) > 1 and args.yes:
        cs.emit("\n" + "=" * 60)

        # Show error summary if any errors occurred
        if error_tracker.errors:
            error_summary = error_tracker.get_summary()
            StatusIndicator("warning").add_message(
                f"Encountered {cs.fmt_count(error_summary['total_errors'])} error(s) during processing"
            ).emit()

            # Show errors by context
            for context, count in error_summary["by_context"].items():
                _emit_dim(f"  • {context}: {count} error(s)")

        # Show success summary
        cs.emit("")
        StatusIndicator("success").add_message(
            f"Batch Complete: {cs.fmt_count(successful_fonts)}/{cs.fmt_count(len(font_files))} fonts processed"
        ).add_item(f"Total instances generated: {cs.fmt_count(total_instances)}").emit()

    return 0


if __name__ == "__main__":
    exit(main())