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
"""

import argparse
import re
import sys
from pathlib import Path
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from core.core_name_policies import (
    sanitize_postscript,
    strip_variable_tokens,
    normalize_fvar_name,
)
from core.core_ttx_table_io import deduplicate_namerecords_binary
from core.core_file_collector import collect_font_files
from core.core_error_handling import ErrorTracker, ErrorContext
import core.core_console_styles as cs
from core.core_console_styles import StatusIndicator

logger = cs.get_logger(__name__)
console = cs.get_console()
RICH_AVAILABLE = cs.RICH_AVAILABLE


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


@dataclass
class InstancerConfig:
    """Configuration for instance processing."""

    output_dir: Optional[Path] = None
    keep_stat: bool = False
    naming_mode: NamingMode = NamingMode.STAT
    dry_run: bool = False


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
        """Extract name from a single AxisValue record."""
        name_id = axis_value.ValueNameID
        value_name = self.font["name"].getDebugName(name_id)

        if not value_name:
            return

        # Get axis tag via index
        if hasattr(axis_value, "AxisIndex"):
            axis_tag = self.index_to_tag.get(axis_value.AxisIndex)
            if axis_tag and hasattr(axis_value, "Value"):
                self.stat_values[axis_tag][axis_value.Value] = value_name

    def get_label_for_axis(
        self, axis_tag: str, value: float, epsilon: float = 0.5
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

    def _compose_name_parts(
        self,
        labels: Dict[str, str],
        coordinates: Dict[str, float],
        metadata: Optional["FontMetadata"] = None,
    ) -> str:
        """Compose name parts in canonical order with family awareness."""
        parts: List[str] = []

        # Define suppressible terms per axis type
        WIDTH_SUPPRESSIBLE = {"regular", "normal", "standard", "roman"}
        SLOPE_SUPPRESSIBLE = {"roman", "upright", "normal", "regular"}
        # Weight terms are NEVER suppressed - keep all weight labels

        # Get coordinate values for context
        wght_value = coordinates.get("wght", 400.0)

        # Determine if instance is at Regular weight
        is_regular_weight = wght_value == 400.0

        # Check family context if available
        has_heavier_weights = False
        has_lighter_weights = False

        if metadata:
            for inst in metadata.instances:
                inst_wght = inst.coordinates.get("wght", 400.0)

                if inst_wght > 400.0:
                    has_heavier_weights = True
                if inst_wght < 400.0:
                    has_lighter_weights = True

        # Width (suppress specific terms)
        width = labels.get("wdth")
        if width:
            # Check if width contains suppressible terms
            width_lower = width.strip().lower()
            width_is_suppressible = any(
                term in width_lower for term in WIDTH_SUPPRESSIBLE
            )
            if not width_is_suppressible:
                parts.append(width)

        # Weight processing - NEVER suppress weight terms, but clean up "Normal" prefix
        weight = labels.get("wght")
        if weight:
            # Clean up "Normal" prefix from weight labels like "Normal Thin" -> "Thin"
            weight_cleaned = weight.strip()
            if weight_cleaned.lower().startswith("normal "):
                weight_cleaned = weight_cleaned[7:].strip()  # Remove "Normal " prefix
            elif weight_cleaned.lower() == "normal":
                # Replace standalone "Normal" with "Regular" for weight axis
                weight_cleaned = "Regular"
            parts.append(weight_cleaned)
        elif is_regular_weight and (has_heavier_weights or has_lighter_weights):
            # No weight label found, but this is Regular weight in a family with other weights
            parts.append("Regular")

        # Slope (suppress specific terms)
        slope = labels.get("slnt") or labels.get("ital") or labels.get("obli")
        if slope:
            slope_lower = slope.strip().lower()
            slope_is_suppressible = any(
                term in slope_lower for term in SLOPE_SUPPRESSIBLE
            )
            if not slope_is_suppressible:
                parts.append(slope)

        # If only slope, add "Regular" prefix if family has non-Regular weights
        slope = labels.get("slnt") or labels.get("ital") or labels.get("obli")
        if len(parts) == 1 and slope and (has_heavier_weights or has_lighter_weights):
            slope_is_suppressible = slope.strip().lower() in SLOPE_SUPPRESSIBLE
            if not slope_is_suppressible:
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

    def _build_weight_groups(self) -> Dict[tuple, List[InstanceInfo]]:
        """Group instances by non-weight coordinates."""
        groups: Dict[tuple, List[InstanceInfo]] = {}

        for inst in self.instances:
            key = self._group_key(inst.coordinates)
            if key not in groups:
                groups[key] = []
            groups[key].append(inst)

        return groups

    def _group_key(self, coords: Dict[str, float]) -> tuple:
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
        if inst.fvar_name == "Unknown":
            return inst.stat_name

        base = inst.fvar_name.strip()

        # Check if we should add Regular
        if not self.should_add_regular(inst):
            return base

        # Intelligently insert "Regular" in the correct position
        # Check if name ends with a slope term (Italic/Oblique/Slanted)
        slope_terms = ["italic", "oblique", "slanted"]

        # Split into words to check if last word is a slope term
        words = base.split()
        if not words:
            # Empty name (shouldn't happen, but handle it)
            return "Regular"

        last_word_lower = words[-1].lower()

        # Check if last word is a slope term
        if last_word_lower in slope_terms:
            # Insert "Regular" before the slope term
            if len(words) > 1:
                # Name has content before slope: "Inktrap Italic" -> "Inktrap Regular Italic"
                before_slope = " ".join(words[:-1])
                slope_original = words[-1]
                return f"{before_slope} Regular {slope_original}"
            else:
                # Pure slope name: "Italic" -> "Regular Italic"
                slope_original = words[0]
                return f"Regular {slope_original}"
        else:
            # No slope term, append "Regular" at the end
            # "Inktrap" -> "Inktrap Regular"
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

        metadata = FontMetadata(
            axes=axes,
            instances=instances,
            stat_values=self.stat_parser.stat_values,
            source_italic=source_italic,
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

    def _extract_instances(self) -> List[InstanceInfo]:
        """Extract instance information from fvar table."""
        instances: List[InstanceInfo] = []
        fvar = self.font["fvar"]

        for i, instance in enumerate(fvar.instances):
            fvar_name = "Unknown"
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
                    fvar_name = "Unknown"

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
                return abs(angle) > 0.1
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
            if inst.fvar_name != "Unknown":
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
            if inst.fvar_name != "Unknown":
                # Normalize fvar name for filename generation
                return normalize_fvar_name(
                    inst.fvar_name,
                    stat_values=self.metadata.stat_values,
                    coordinates=inst.coordinates,
                )
            return inst.stat_name

        return inst.stat_name


# ============================================================================
# Interactive UI
# ============================================================================


class InteractivePrompt:
    """Handles user interaction and prompts."""

    def __init__(self, metadata: FontMetadata, stat_parser: STATNameParser):
        self.metadata = metadata
        self.stat_parser = stat_parser

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
            inst.fvar_name != "Unknown" for inst in self.metadata.instances
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

    def show_instance_selection(
        self,
    ) -> Optional[Tuple[List[Tuple[int, NamingMode]], NamingMode]]:
        """Show instances and get user selection."""
        self._print_header("Named Instances")

        # Check if any instances have fvar names (show column if they exist)
        has_fvar_names = any(
            inst.fvar_name != "Unknown" for inst in self.metadata.instances
        )

        # Print table with naming comparison
        self._print_instances_table_with_naming(has_fvar_names)

        # Show simplified options
        print("\n" + "─" * 70)
        print("GENERATION OPTIONS")
        print("─" * 70)

        if has_fvar_names:
            StatusIndicator("info").add_message(
                "This font has fvar names available for comparison"
            ).emit()
            print("  • STAT names:   derived from STAT table (canonical, recommended)")
            print("  • fvar names:   from fvar instance records (legacy)")
            print("  • fvar-hybrid:  fvar names + auto-add 'Regular' when appropriate")
            print()

        print("Generate all instances:")
        print("  [Enter]      STAT names (default)")
        if has_fvar_names:
            print("  [fvar]       fvar-hybrid names")
            print("  [raw]        fvar names (no modifications)")

        print("\nGenerate specific instances:")
        print("  [1,2,3]      Instance numbers (comma or space separated)")
        if has_fvar_names:
            print("  [stat:1,2]   Specific instances with naming mode")
            print("  [1s 2f]      Per-instance mode (s=stat, f=fvar, r=raw)")

        print("\n  [i]          Show table again")
        print("  [s]          Skip this font")
        print("  [q]          Quit program")
        print("─" * 70)

        response = input("\nYour choice: ").strip().lower()

        if response in ("q", "quit"):
            # Exit entire program
            raise SystemExit(0)
        elif response in ("s", "skip"):
            # Skip to next font
            return None
        elif response == "i":
            return self.show_instance_selection()  # Recursive call
        elif response in ("stat", ""):
            return ([], NamingMode.STAT)
        elif response in ("f", "fvar"):
            return ([], NamingMode.FVAR_HYBRID)
        elif response in ("fn", "raw"):
            return ([], NamingMode.FVAR_RAW)
        else:
            instances = self._parse_instance_selection(response, has_fvar_names)
            if instances is None:
                return None
            return (instances, NamingMode.STAT)

    def _build_hybrid_name_for_display(self, inst: InstanceInfo) -> str:
        """Build hybrid name for display with bold for added words."""
        if inst.fvar_name == "Unknown":
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

        return normalized_fvar if normalized_fvar != "Unknown" else "N/A"

    def _print_instances_table_with_naming(self, show_naming_comparison: bool) -> None:
        """Print instances table with optional naming comparison."""
        if not self.metadata.instances:
            cs.emit("\nNo named instances found")
            return

        table = cs.create_table(
            show_header=True, row_styles=["on #282a39", "on #1d1f30"]
        )
        if table:
            # Set table width to match console width
            table.width = console.size.width
            table.add_column("#", style="dim")
            table.add_column("STAT Name", style="pale_green1")

            if show_naming_comparison:
                table.add_column("fvar Name", style="dim")

            table.add_column("Style", style="medium_turquoise")
            table.add_column("Coordinates", style="turquoise2")

            for inst in self.metadata.instances:
                ribbi = self._get_ribbi_label(inst)

                # Check if names differ (compare normalized versions)
                normalized_fvar = (
                    normalize_fvar_name(
                        inst.fvar_name,
                        stat_values=self.metadata.stat_values,
                        coordinates=inst.coordinates,
                    )
                    if inst.fvar_name != "Unknown"
                    else "Unknown"
                )

                names_differ = (
                    normalized_fvar != inst.stat_name and normalized_fvar != "Unknown"
                )

                # Apply highlighting to cells when names differ
                if names_differ and RICH_AVAILABLE:
                    # Yellow background for different names
                    stat_display = (
                        f"[#282a39 on #ffdf80]{inst.stat_name}[/#282a39 on #ffdf80]"
                    )
                    hybrid_display = f"[#282a39 on #ffdf80]{self._build_hybrid_name_for_display(inst)}[/#282a39 on #ffdf80]"
                else:
                    # Normal display
                    stat_display = inst.stat_name
                    hybrid_display = self._build_hybrid_name_for_display(inst)

                row_data = [str(inst.index + 1), stat_display]

                if show_naming_comparison:
                    row_data.append(hybrid_display)

                row_data.extend([ribbi, inst.format_coordinates()])
                table.add_row(*row_data)

            console.print(table)

            # Add note about bold formatting if there are any hybrid modifications
            if show_naming_comparison:
                # Add legend explaining highlighting
                cs.emit("")
                StatusIndicator("info").add_message(
                    "[#282a39 on #ffdf80]Highlighted cells[/#282a39 on #ffdf80] indicate STAT and fvar names differ"
                ).emit()

                has_modifications = any(
                    self._build_hybrid_name_for_display(inst) != inst.fvar_name
                    for inst in self.metadata.instances
                    if inst.fvar_name != "Unknown"
                )
                if has_modifications:
                    cs.emit("")
                    StatusIndicator("info").add_message(
                        "[bold]Bold[/bold] words in fvar names indicate hybrid additions (use [raw] to omit)"
                    ).emit()
        else:
            # Fallback for non-Rich
            cs.emit(f"\nInstances ({len(self.metadata.instances)}):")
            for inst in self.metadata.instances:
                ribbi = self._get_ribbi_label(inst)
                cs.emit(f"  {inst.index + 1:2}. {inst.stat_name:25} [{ribbi:12}]")
                if show_naming_comparison and inst.fvar_name != "Unknown":
                    hybrid = self._build_hybrid_name_for_display(inst)
                    cs.emit(f"      fvar: {hybrid}")
                cs.emit(f"      {inst.format_coordinates()}")

    def _parse_instance_selection(
        self, response: str, allow_naming: bool
    ) -> Optional[List[Tuple[int, NamingMode]]]:
        """Parse instance selection with optional naming prefixes.

        Supports multiple formats:
        - Simple numbers: "1,2,3" or "1 2 3"
        - Mode prefixes: "stat:1,2,3" or "fvar:4,5,6"
        - Per-instance: "1s 2f 3r" (s=STAT, f=fvar, r=raw)
        - Mixed: "stat:1,2 fvar:3 4s"
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
                    print(f"❌ Invalid mode '{mode_str}'. Use: stat, fvar, or raw")
                    return None

                # Parse number(s)
                if num_str.isdigit():
                    num = int(num_str)
                else:
                    # Could support ranges here: "1-3" → [1, 2, 3]
                    print(f"❌ Invalid number format '{num_str}'")
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
                print(f"❌ Invalid input: '{token}'")
                print("   Valid formats: 1,2,3  or  stat:1,2  or  1s 2f  or  s 1 2")
                return None

            # Add to results if we parsed a number
            if num is not None:
                if 1 <= num <= len(self.metadata.instances):
                    idx = num - 1
                    final_mode = mode if allow_naming else NamingMode.STAT
                    result.append((idx, final_mode))
                else:
                    print(
                        f"❌ Instance number {num} out of range (1-{len(self.metadata.instances)})"
                    )
                    return None

        if not result:
            print("❌ No valid instance numbers provided")
            return None

        return result

    def show_custom_mode(
        self, font_path: str, axes: List[AxisInfo], stat_parser: STATNameParser
    ) -> Optional[Tuple[Dict[str, float], str]]:
        """Interactive custom coordinate builder."""
        self._print_header("Custom Instance Builder")
        self._print_axes_table()

        variable_axes, fixed_axes = self._classify_axes()

        if not variable_axes:
            StatusIndicator("warning").add_message(
                "This font has no variable axes. All coordinates are fixed."
            ).emit()
            cs.emit("Cannot create custom instances.")
            input("\nPress Enter to continue...")
            return None

        # Initialize coordinates with all fixed values
        coordinates = {}
        for axis in fixed_axes:
            coordinates[axis.tag] = axis.min_value

        # Prompt only for variable axes
        cs.emit("\nEnter coordinates for variable axes (press Enter for default):")

        for axis in variable_axes:
            while True:
                # Format prompt with range and default
                default_display = (
                    f"{axis.default_value:.1f}"
                    if axis.default_value == int(axis.default_value)
                    else f"{axis.default_value}"
                )
                range_display = f"{axis.min_value:.0f}-{axis.max_value:.0f}"
                prompt = f"  {axis.name} ({range_display}) [{default_display}] [b=back, s=skip, q=quit]: "

                response = input(prompt).strip()

                if response.lower() == "b":
                    # Go back to previous axis
                    if len(coordinates) > 0:
                        last_axis = list(coordinates.keys())[-1]
                        del coordinates[last_axis]
                        cs.emit("  Going back to previous axis...", style="dim")
                        return None  # Signal to restart from previous axis
                    else:
                        cs.emit("  Already at first axis", style="dim")
                    continue

                if response.lower() in ("q", "quit"):
                    # Exit entire program
                    raise SystemExit(0)

                if response.lower() in ("s", "skip"):
                    # Skip to next font
                    return None

                if response == "":
                    coordinates[axis.tag] = axis.default_value
                    break

                try:
                    value = float(response)
                    if axis.is_in_range(value):
                        coordinates[axis.tag] = value
                        break
                    else:
                        cs.emit(
                            f"{cs.indent(2)}Value must be between {axis.min_value} and {axis.max_value}"
                        )
                except ValueError:
                    cs.emit(f"{cs.indent(2)}Invalid number")

        # Sort coordinates in canonical order: wdth, wght, slnt/ital
        canonical_order = ["wdth", "wght", "slnt", "ital", "obli"]
        sorted_coords = []
        for tag in canonical_order:
            if tag in coordinates:
                sorted_coords.append((tag, coordinates[tag]))
        # Add any remaining axes not in canonical order
        for tag, val in coordinates.items():
            if tag not in canonical_order:
                sorted_coords.append((tag, val))

        # Build preview
        cs.emit("\nBuilding instance with:")
        coord_str = ", ".join([f"{k}={v}" for k, v in sorted_coords])
        cs.emit(f"{cs.indent(1)}{coord_str}")

        stat_name = stat_parser.build_subfamily_name(coordinates, self.metadata)
        cs.emit(f"{cs.indent(1)}STAT name: {stat_name}")

        if fixed_axes:
            fixed_str = ", ".join(
                [f"{axis.tag}={axis.min_value}" for axis in fixed_axes]
            )
            cs.emit(f"{cs.indent(1)}Fixed values: {fixed_str}")

        # Extract family name from font file
        try:
            from fontTools.ttLib import TTFont

            temp_font = TTFont(font_path)
            family_name = (
                temp_font["name"].getDebugName(1)
                or temp_font["name"].getDebugName(16)
                or "Unknown"
            )
            # Strip variable font suffixes
            from core.core_name_policies import strip_variable_tokens

            family_name = strip_variable_tokens(family_name) or family_name
            temp_font.close()
        except Exception:
            # Fallback to filename
            family_name = Path(font_path).stem.split("-")[0]

        # Determine if we need custom naming (non-STAT values)
        has_non_stat_values = False
        for tag, val in coordinates.items():
            if tag in stat_parser.stat_values:
                # Check if this value exists in STAT
                if val not in stat_parser.stat_values[tag]:
                    has_non_stat_values = True
                    break

            # If using non-STAT values, offer naming options
            final_name = stat_name
        if has_non_stat_values:
            StatusIndicator("warning").add_message(
                "Some coordinates don't match STAT table values"
            ).emit()
            cs.emit("\nNaming options:")

            # Generate axis-based name
            axis_name_parts = []
            for tag, val in sorted_coords:
                val_str = f"{int(val)}" if val == int(val) else f"{val}"
                axis_name_parts.append(f"{tag}{val_str}")
            axis_name = "".join(axis_name_parts)

            cs.emit(f"  1. {stat_name} (STAT-derived, may be generic)")
            cs.emit(f"  2. {axis_name} (descriptive coordinates)")
            cs.emit("  3. Custom name (manual input)")

            choice = input("\nChoice [1]: ").strip()
            if choice == "2":
                final_name = axis_name
            elif choice == "3":
                custom = input("Enter custom subfamily name: ").strip()
                final_name = custom if custom else stat_name

        cs.emit(
            f"\n{cs.indent(1)}Filename: {family_name}-{final_name.replace(' ', '')}"
        )

        confirm = input("\nGenerate this instance? [Y/n]: ").strip().lower()
        if confirm in ["", "y", "yes"]:
            return (coordinates, final_name)
        return None

    def _print_header(self, title: str) -> None:
        """Print section header."""
        cs.fmt_header(title, console)

    def _print_axes_table(self) -> None:
        """Print axes information as table with variable/fixed distinction."""
        variable_axes, fixed_axes = self._classify_axes()

        if variable_axes:
            cs.emit("\nVariable Axes:")
            table = cs.create_table(
                show_header=True, row_styles=["on #282a39", "on #1d1f30"]
            )
        if table:
            # Set table width to match console width
            table.width = console.size.width
            table.add_column("Tag", style="cyan1")
            table.add_column("Name", style="pale_green1")
            table.add_column("Range", style="turquoise2")
            table.add_column("STAT Values", style="dim", no_wrap=False)

            for axis in variable_axes:
                stat_values = self._format_stat_values_inline(axis.tag)
                table.add_row(
                    axis.tag, axis.name, axis.format_variable_range(), stat_values
                )

            console.print(table)
        else:
            for axis in variable_axes:
                stat_values = self._format_stat_values_inline(axis.tag)
                cs.emit(
                    f"  {axis.tag:6} {axis.name:12} {axis.format_variable_range():20} {stat_values}"
                )

        if fixed_axes:
            cs.emit("\nFixed Axes:")
            table = cs.create_table(
                show_header=True, row_styles=["on #282a39", "on #1d1f30"]
            )
        if table:
            # Set table width to match console width
            table.width = console.size.width
            table.add_column("Tag", style="cyan1")
            table.add_column("Name", style="pale_green1")
            table.add_column("Value", style="turquoise2")
            table.add_column("Note", style="dim", no_wrap=False)

            for axis in fixed_axes:
                stat_values = self._format_stat_values_inline(axis.tag)
                note = f"All instances will use {stat_values if stat_values else 'this value'}"
                table.add_row(axis.tag, axis.name, f"{axis.min_value} [FIXED]", note)

            console.print(table)
        else:
            for axis in fixed_axes:
                stat_values = self._format_stat_values_inline(axis.tag)
                cs.emit(f"  {axis.tag:6} {axis.name:12} {axis.min_value} [FIXED]")
                cs.emit(
                    f"    → All instances will use {stat_values if stat_values else 'this value'}"
                )

    def _print_validation_notices(self) -> None:
        """Print validation notices about the font."""
        notices = []

        # Check for instances with no STAT mapping
        for inst in self.metadata.instances:
            if inst.stat_name == "Regular" and inst.fvar_name != "Unknown":
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

            cs.emit("\nSuggested actions:", style="bold")
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
    ):
        self.analyzer = analyzer
        self.stat_parser = stat_parser
        self.config = config
        self.error_tracker = error_tracker or ErrorTracker()
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

            # Instantiate
            instancer.instantiateVariableFont(
                instance_font, coordinates, inplace=True, updateFontNames=False
            )

            # Detect italic from result
            italic_angle = instance_font["post"].italicAngle
            is_italic = abs(italic_angle) > 0.1

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
            if not is_italic and abs(italic_angle) > 0.01:
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

        family_name = name_table.getDebugName(1) or "Unknown"
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

        # Update OS/2 weight class
        if "OS/2" in font:
            font["OS/2"].usWeightClass = int(round(weight))

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
        """Save instance font to file."""
        # Use PostScript name (ID6)
        ps_name = font["name"].getDebugName(6)
        if not ps_name:
            family = font["name"].getDebugName(16) or "Unknown"
            ps_name = sanitize_postscript(f"{family}-{subfamily_name}")

        # Detect correct extension
        extension = self._get_output_extension(font)
        filename = f"{ps_name}{extension}"

        output_dir = self.config.output_dir or Path(self.analyzer.font_path).parent
        output_path = output_dir / filename

        # Handle duplicates
        counter = 1
        while output_path.exists():
            base = ps_name.rsplit(".", 1)[0] if "." in ps_name else ps_name
            filename = f"{base}~{counter:03d}{extension}"
            output_path = output_dir / filename
            counter += 1

        font.save(str(output_path))
        return str(output_path)


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

    def run_info_mode(self) -> None:
        """Run information-only mode."""
        if not self.analyzer.load_and_validate():
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

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
        )
        self.last_generator = generator

        while True:
            result = ui.show_custom_mode(
                self.font_path, self.metadata.axes, self.stat_parser
            )
            if result is None:
                break

            coordinates, custom_name = result
            print(f"\nGenerating: {custom_name}")
            output_path = generator.generate_instance(coordinates, custom_name)

            if output_path:
                filename = Path(output_path).name
                cs.emit("")
                StatusIndicator("success").add_message("Generated").add_file(
                    filename, filename_only=True
                ).emit()

            another = input("\nCreate another instance? [y/N]: ").strip().lower()
            if another not in ["y", "yes"]:
                break

        if generator.successful_count > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            print(
                f"\nGenerated {generator.successful_count} instance(s) in: {output_dir}"
            )

    def run_instance_mode(self) -> None:
        """Run named instance generation mode."""
        if not self.analyzer.load_and_validate():
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        ui = InteractivePrompt(self.metadata, self.stat_parser)

        result = ui.show_instance_selection()
        if result is None:
            return

        instances_with_modes, default_mode = result

        naming_strategy = InstanceNamingStrategy(
            self.metadata, self.stat_parser, default_mode
        )

        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
        )
        self.last_generator = generator

        if not instances_with_modes:  # Generate all
            cs.emit(
                f"\nGenerating {cs.fmt_count(len(self.metadata.instances))} instances..."
            )

            for inst_num, inst in enumerate(self.metadata.instances, 1):
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
                inst = self.metadata.instances[idx]
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
            StatusIndicator("success").add_message(
                f"Generated {cs.fmt_count(generator.successful_count)} instance(s)"
            ).add_item(
                f"Output directory: {cs.fmt_file_compact(str(output_dir))}"
            ).emit()

    def run_auto_mode(self) -> None:
        """Run automatic generation mode (no prompts)."""
        if not self.analyzer.load_and_validate():
            return

        self.metadata = self.analyzer.analyze()
        self.stat_parser = self.analyzer.stat_parser

        # Verbose mode removed - all info shown by default

        naming_strategy = InstanceNamingStrategy(
            self.metadata, self.stat_parser, self.config.naming_mode
        )

        generator = InstanceGenerator(
            self.analyzer,
            self.stat_parser,
            self.config,
            error_tracker=self.error_tracker,
        )
        self.last_generator = generator

        mode_label = self.config.naming_mode.value
        cs.emit(
            f"Auto-generating {cs.fmt_count(len(self.metadata.instances))} instances ({mode_label} names)..."
        )

        for inst_num, inst in enumerate(self.metadata.instances, 1):
            final_name = naming_strategy.resolve_name(inst)

            output_path = generator.generate_instance(inst.coordinates, final_name)
            if output_path:
                filename = Path(output_path).name
                cs.emit(f"  [{inst_num}] Generated: {filename}")

        if generator.successful_count > 0:
            output_dir = self.config.output_dir or Path(self.font_path).parent
            cs.emit("")
            StatusIndicator("success").add_message(
                f"Generated {cs.fmt_count(generator.successful_count)} instance(s)"
            ).add_item(f"Output: {cs.fmt_file_compact(str(output_dir))}").emit()


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
        help="Generate specific instance(s) by index (e.g., 7 or 1,3,5)",
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

    # Display options
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Preview mode: show what would be done without generating files",
    )

    args = parser.parse_args()

    # Handle subcommands (info, custom)
    if hasattr(args, "command") and args.command == "info":
        # Info mode with single font
        if not Path(args.font).exists():
            StatusIndicator("error").add_message("Font file not found").add_file(
                args.font, filename_only=False
            ).emit()
            return 1

        cs.fmt_header("Variable Font Instancer - Font Information", console)

        # Create minimal config for info mode
        config = InstancerConfig()
        processor = FontProcessor(args.font, config)

        try:
            processor.run_info_mode()
        except Exception as e:
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

        cs.fmt_header("Variable Font Instancer - Custom Instance Builder", console)

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

    # Determine naming strategy from flags (default)
    if args.fvar_hybrid:
        naming_mode = NamingMode.FVAR_HYBRID
    elif args.fvar_raw:
        naming_mode = NamingMode.FVAR_RAW
    else:
        naming_mode = NamingMode.STAT  # default

    # Handle batch processing confirmation
    if len(font_files) > 1 and not args.yes:
        cs.emit(f"\n{cs.fmt_count(len(font_files))} fonts found")
        print("\nBatch Processing Options:")
        print("  [y] Auto-generate all instances for all fonts")
        print("  [n] Interactive mode for each font")
        print("  [q] Cancel operation")

        confirm = input("\nYour choice [y/n/q]: ").strip().lower()

        if confirm in ["q", "quit"]:
            StatusIndicator("info").add_message("Operation cancelled").emit()
            return 0
        elif confirm in ["", "y", "yes"]:
            # Ask for naming strategy
            print("\nNaming strategy for batch processing:")
            print("  [s] STAT names (default)")
            print("  [f] fvar-hybrid names")
            print("  [r] fvar-raw names")

            naming_choice = input("\nYour choice [s]: ").strip().lower()

            if naming_choice in ["f", "fvar-hybrid"]:
                naming_mode = NamingMode.FVAR_HYBRID
                cs.emit("\nAuto-generating all instances with fvar-hybrid naming...")
            elif naming_choice in ["r", "fvar-raw"]:
                naming_mode = NamingMode.FVAR_RAW
                cs.emit("\nAuto-generating all instances with fvar-raw naming...")
            else:
                naming_mode = NamingMode.STAT
                cs.emit("\nAuto-generating all instances with STAT naming...")

            # Auto-confirm for all fonts
            args.yes = True
        elif confirm in ["n", "no"]:
            # Interactive mode for each font
            cs.emit("\nEntering interactive mode for each font...")
            # Don't set args.yes - let each font go through interactive mode
        else:
            StatusIndicator("warning").add_message(
                "Invalid choice. Defaulting to auto-generate."
            ).emit()
            args.yes = True

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
    )

    # Show header
    cs.fmt_header("Variable Font Instancer - Extract static instances", console)

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
                if args.yes:
                    processor.run_auto_mode()
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
                cs.emit(f"  • {context}: {count} error(s)", style="dim")

        # Show success summary
        cs.emit("")
        StatusIndicator("success").add_message(
            f"Batch Complete: {cs.fmt_count(successful_fonts)}/{cs.fmt_count(len(font_files))} fonts processed"
        ).add_item(f"Total instances generated: {cs.fmt_count(total_instances)}").emit()

    return 0


if __name__ == "__main__":
    exit(main())
