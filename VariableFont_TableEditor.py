#!/usr/bin/env python3
"""
Variable Font Table Editor — define STAT axis values and fvar named instances.

Runs before VariableFont_Instancer (reads STAT/fvar data this tool writes).
Companion to VariableFont_Instancer.py; kept as a separate script in this folder.

Usage:
  python VariableFont_TableEditor.py font.ttf
  python VariableFont_TableEditor.py font.ttf --info
  python VariableFont_TableEditor.py font.ttf --config axes.yaml
  python VariableFont_TableEditor.py font.ttf --dry-run
  python VariableFont_TableEditor.py font.ttf --ttx --output out.ttf
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fontTools.ttLib import TTFont

# Project root for FontCore (same walk as VariableFont_Instancer.py)
_project_root = Path(__file__).resolve().parent
while (
    not (_project_root / "FontCore").exists() and _project_root.parent != _project_root
):
    _project_root = _project_root.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import FontCore.core_console_styles as cs
from FontCore.core_console_styles import StatusIndicator
from FontCore.core_font_style_dictionaries import ELIDABLE_WEIGHT_NAMES, ELIDABLE_WIDTH_NAMES
from FontCore.core_nameid_allocator import (
    AxisDef,
    AxisValueDef,
    audit_nameids,
    build_allocation_plan,
    check_for_collisions,
    enumerate_instance_names,
)
from FontCore.core_ot_label_scanner import scan_ot_label_nameids
from FontCore.core_stat_builder import (
    apply_table_edits,
    count_instances,
    default_fix_summary,
    generate_ttx_additions,
)
from FontCore.core_variable_font_detection import (
    VariableFontMode,
    analyze_variable_font,
)

logger = cs.get_logger(__name__)
console = cs.get_console()
RICH_AVAILABLE = cs.RICH_AVAILABLE

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


@dataclass
class EditorConfig:
    output_ttf: bool = True
    output_ttx: bool = False
    config_path: Optional[Path] = None
    save_config_path: Optional[Path] = None
    dry_run: bool = False
    fix_fvar_default: bool = True
    info_only: bool = False


def _load_instancer_module():
    path = _script_dir / "VariableFont_Instancer.py"
    spec = importlib.util.spec_from_file_location("variable_font_instancer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load instancer module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vfi = _load_instancer_module()
STATNameParser = _vfi.STATNameParser
AxisInfo = _vfi.AxisInfo
ITALIC_ANGLE_THRESHOLD = _vfi.ITALIC_ANGLE_THRESHOLD


def _raise_if_quit(line: str) -> None:
    if line.strip().lower() in ("q", "quit"):
        raise SystemExit(0)


def _emit_bold(text: str) -> None:
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(f"[bold]{escape(text)}[/bold]")
    else:
        cs.emit(text)


def _emit_dim(text: str) -> None:
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(f"[dim]{escape(text)}[/dim]")
    else:
        cs.emit(text)


def _emit_audit_line(text: str) -> None:
    """Emit audit text without Rich interpreting [axis tags] as markup."""
    if RICH_AVAILABLE:
        from rich.markup import escape

        cs.emit(escape(text))
    else:
        cs.emit(text)


def _is_collection_file(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"ttcf"
    except OSError:
        return False


def _detect_source_italic(font: TTFont) -> bool:
    if "post" in font and hasattr(font["post"], "italicAngle"):
        try:
            return abs(float(font["post"].italicAngle)) > ITALIC_ANGLE_THRESHOLD
        except (AttributeError, TypeError, ValueError):
            pass
    return False


def _axis_order(font: TTFont) -> List[str]:
    fvar_tags = [ax.axisTag for ax in font["fvar"].axes]
    if "STAT" in font:
        stat = font["STAT"].table
        if hasattr(stat, "DesignAxisRecord") and stat.DesignAxisRecord:
            ordered = sorted(
                stat.DesignAxisRecord.Axis,
                key=lambda a: getattr(a, "AxisOrdering", 0),
            )
            tags = [a.AxisTag for a in ordered if a.AxisTag in fvar_tags]
            for tag in fvar_tags:
                if tag not in tags:
                    tags.append(tag)
            return tags
    return fvar_tags


def _extract_axes(font: TTFont) -> Dict[str, AxisInfo]:
    axes: Dict[str, AxisInfo] = {}
    for axis in font["fvar"].axes:
        name = font["name"].getDebugName(axis.axisNameID) or axis.axisTag
        axes[axis.axisTag] = AxisInfo(
            tag=axis.axisTag,
            name=name,
            min_value=axis.minValue,
            default_value=axis.defaultValue,
            max_value=axis.maxValue,
        )
    return axes


def _suggest_elidable(value: float, name: str, axis_def: AxisDef) -> bool:
    """
    Default for the Elidable? prompt (Enter accepts the default).

    wght + Regular/Roman → False so instance names stay "Condensed Regular",
    not "Condensed". Other ELIDABLE_WEIGHT_NAMES (Book, Text, …) may still
    default to True on wght; width Normal/Standard/Regular still elide on wdth.
    """
    name_stripped = name.strip()
    if axis_def.tag == "wght" and name_stripped in ("Regular", "Roman"):
        return False
    if abs(value - axis_def.default_value) < 0.001:
        return True
    if axis_def.tag == "wght" and name_stripped in ELIDABLE_WEIGHT_NAMES:
        return True
    if axis_def.tag == "wdth" and name_stripped in ELIDABLE_WIDTH_NAMES:
        return True
    return False


def _stat_label_hint(stat_parser: Optional[STATNameParser], axis_tag: str, value: float) -> str:
    """STAT name hint for prompts; uses instancer parser when available."""
    if not stat_parser:
        return ""
    if hasattr(stat_parser, "get_label_for_axis"):
        return stat_parser.get_label_for_axis(axis_tag, value) or ""
    axis_map = getattr(stat_parser, "stat_values", {}).get(axis_tag, {})
    if value in axis_map:
        return axis_map[value]
    return ""


def _prompt_float(message: str) -> float:
    """Prompt for a float; honors q/quit and re-prompts on bad input."""
    while True:
        line = cs.prompt_input(message).strip()
        _raise_if_quit(line)
        if not line:
            cs.emit("  A number is required.")
            continue
        try:
            return float(line)
        except ValueError:
            cs.emit("  Enter a numeric value.")


def _suggest_stat_format(axis_tag: str) -> int:
    return 2 if axis_tag == "opsz" else 1


def _ot_label_ids(ot_labels) -> Set[int]:
    return {r.name_id for r in ot_labels}


def _display_scan(
    font: TTFont,
    axes: Dict[str, AxisInfo],
    axis_order: List[str],
    stat_parser: STATNameParser,
    used_nameids: Dict[int, str],
    ot_label_id_set: Set[int],
    free_start: int,
) -> None:
    _emit_bold("AXES")
    table = cs.create_table(show_header=True)
    if table is not None:
        table.add_column("Tag", style="cyan")
        table.add_column("Name")
        table.add_column("Range")
        table.add_column("Default", justify="right")
        table.add_column("Existing STAT values")
        for tag in axis_order:
            if tag not in axes:
                continue
            ax = axes[tag]
            existing = stat_parser.stat_values.get(tag, {})
            if existing:
                stat_txt = ", ".join(f"{v:g}={n}" for v, n in sorted(existing.items()))
            else:
                stat_txt = "(none)"
            table.add_row(
                tag,
                ax.name,
                ax.format_variable_range(),
                f"{ax.default_value:g}",
                stat_txt,
            )
        console.print(table)
    else:
        for tag in axis_order:
            ax = axes[tag]
            cs.emit(f"  {tag}  {ax.name}  {ax.format_variable_range()}")

    cs.emit_spacer()
    _emit_bold("EXISTING NAMEIDS ≥ 256")
    for nid in sorted(used_nameids):
        desc = used_nameids[nid]
        if nid in ot_label_id_set:
            _emit_audit_line(f"  ⚠ {nid:4d}  {desc}  ← OT label, protected")
        else:
            _emit_audit_line(f"  {nid:4d}  {desc}")
    cs.emit_spacer()
    _emit_dim(f"New allocations will start at: {free_start}")


def _default_mismatch_warnings(
    font: TTFont, axis_defs: List[AxisDef], fix_fvar_default: bool
) -> None:
    defined_by_tag = {ad.tag: {av.value for av in ad.values} for ad in axis_defs}
    for axis in font["fvar"].axes:
        tag = axis.axisTag
        if tag not in defined_by_tag:
            continue
        if not any(abs(axis.defaultValue - v) < 0.001 for v in defined_by_tag[tag]):
            cs.emit(
                f"⚠ {tag} default ({axis.defaultValue:g}) does not match any "
                "planned named value."
            )
            if fix_fvar_default:
                _emit_dim(
                    "  Mark the intended default value as elidable to auto-correct "
                    "(--no-fix-default to skip)."
                )


def _ital_axis_def_silent(is_italic_font: bool) -> AxisDef:
    if is_italic_font:
        val = AxisValueDef(value=1.0, name="Italic", elidable=False, stat_format=1)
        default = 1.0
    else:
        val = AxisValueDef(value=0.0, name="Roman", elidable=True, stat_format=1)
        default = 0.0
    return AxisDef(
        tag="ital",
        display_name="Italic",
        min_value=0.0,
        default_value=default,
        max_value=1.0,
        values=[val],
    )


def _build_ital_axis_def(is_italic_font: bool) -> AxisDef:
    if is_italic_font:
        display = '1.0 "Italic" (not elidable)'
    else:
        display = '0.0 "Roman" (elidable)'
    cs.emit(f"ital — auto-configured: {display}")
    override = cs.prompt_input("[o] override / Enter to accept: ").strip().lower()
    _raise_if_quit(override)
    if override == "o":
        return _prompt_axis_values_interactive(
            AxisDef(
                tag="ital",
                display_name="Italic",
                min_value=0.0,
                default_value=1.0 if is_italic_font else 0.0,
                max_value=1.0,
                values=[],
            ),
            stat_parser=None,
        )
    return _ital_axis_def_silent(is_italic_font)


def _prompt_axis_values_interactive(
    axis_def: AxisDef,
    stat_parser: Optional[STATNameParser],
) -> AxisDef:
    values: List[AxisValueDef] = list(axis_def.values)

    while True:
        cs.emit_section_separator()
        _emit_bold(f"Defining values for:  {axis_def.tag}  ({axis_def.display_name})")
        cs.emit(
            f"Range: {axis_def.min_value:g} — {axis_def.max_value:g}   "
            f"Current default: {axis_def.default_value:g}"
        )
        if stat_parser:
            existing = stat_parser.stat_values.get(axis_def.tag, {})
            if existing:
                cs.emit(
                    "Existing STAT: "
                    + ", ".join(f"{v:g}={n}" for v, n in sorted(existing.items()))
                )
            else:
                cs.emit("Existing STAT values: (none)")

        line = cs.prompt_input("Enter a value (or 'done' · 'back' · 'q'): ").strip()
        _raise_if_quit(line)
        low = line.lower()
        if low == "done":
            break
        if low == "back":
            if values:
                removed = values.pop()
                cs.emit(f"Removed {axis_def.tag}={removed.value:g} ({removed.name})")
            else:
                cs.emit("No values to remove.")
            continue
        if not line:
            continue

        try:
            value = float(line)
        except ValueError:
            cs.emit("Enter a numeric axis value.")
            continue

        if not (axis_def.min_value <= value <= axis_def.max_value):
            cs.emit(
                f"Value must be within [{axis_def.min_value:g}, {axis_def.max_value:g}]"
            )
            continue

        duplicate = False
        for existing in values:
            if abs(existing.value - value) < 0.001:
                cs.emit(
                    f"{axis_def.tag}={value:g} already defined as {existing.name!r}"
                )
                duplicate = True
                break
        if duplicate:
            continue

        stat_hint = _stat_label_hint(stat_parser, axis_def.tag, value)
        name_prompt = f"  Name for {axis_def.tag}={value:g}"
        if stat_hint:
            name_prompt += f" [{stat_hint}]"
        name_prompt += ": "
        name = cs.prompt_input(name_prompt).strip()
        _raise_if_quit(name)
        if not name:
            cs.emit("Name cannot be empty.")
            continue

        suggest_el = _suggest_elidable(value, name, axis_def)
        elidable = cs.prompt_confirm("Elidable?", default=suggest_el)

        fmt_default = _suggest_stat_format(axis_def.tag)
        fmt_line = cs.prompt_input(
            f"STAT format [1=point / 2=range / 3=linked, Enter={fmt_default}]: "
        ).strip()
        _raise_if_quit(fmt_line)
        if fmt_line:
            try:
                stat_format = int(fmt_line)
            except ValueError:
                cs.emit("Format must be 1, 2, or 3.")
                continue
        else:
            stat_format = fmt_default
        if stat_format not in (1, 2, 3):
            cs.emit("Format must be 1, 2, or 3.")
            continue

        range_min = range_max = linked_value = None
        if stat_format == 2:
            range_min = _prompt_float("  Range min: ")
            range_max = _prompt_float("  Range max: ")
        elif stat_format == 3:
            linked_value = _prompt_float("  Linked value: ")

        av = AxisValueDef(
            value=value,
            name=name,
            elidable=elidable,
            stat_format=stat_format,
            range_min=range_min,
            range_max=range_max,
            linked_value=linked_value,
        )
        values.append(av)
        el_txt = "YES" if elidable else "No"
        cs.emit(
            f"  Added: {axis_def.tag}={value:g}  {name!r}  elidable={el_txt}  "
            f"format={stat_format}"
        )

    axis_def.values = values
    if values:
        cs.emit(f"\n{axis_def.tag} — {len(values)} values defined:")
        for av in values:
            el = "YES" if av.elidable else "No"
            cs.emit(f"  {av.value:g}  {av.name}  elidable={el}")
    return axis_def


def _interactive_axis_defs(
    font: TTFont,
    axes: Dict[str, AxisInfo],
    axis_order: List[str],
    stat_parser: STATNameParser,
    is_italic_font: bool,
) -> List[AxisDef]:
    axis_defs: List[AxisDef] = []
    tags = [t for t in axis_order if t in axes]
    idx = 0

    while idx < len(tags):
        tag = tags[idx]
        if tag == "ital":
            ad = _build_ital_axis_def(is_italic_font)
        else:
            info = axes[tag]
            ad = AxisDef(
                tag=tag,
                display_name=info.name,
                min_value=info.min_value,
                default_value=info.default_value,
                max_value=info.max_value,
                values=[],
            )
            ad = _prompt_axis_values_interactive(ad, stat_parser)

        if idx < len(axis_defs):
            axis_defs[idx] = ad
        else:
            axis_defs.append(ad)

        _default_mismatch_warnings(font, axis_defs, fix_fvar_default=True)

        if idx < len(tags) - 1:
            cont = cs.prompt_input(
                "Continue to next axis? [Enter=yes / b=back / q=quit]: "
            ).strip().lower()
            _raise_if_quit(cont)
            if cont == "b":
                axis_defs.pop()
                if idx > 0:
                    idx -= 1
                continue

        idx += 1

    return axis_defs


def _load_yaml_config(path: Path, axes: Dict[str, AxisInfo]) -> List[AxisDef]:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --config (pip install pyyaml)")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    axis_defs: List[AxisDef] = []
    for entry in data.get("axes", []):
        tag = entry.get("tag")
        if not tag or tag not in axes:
            logger.warning("Config axis %r not in font; skipped", tag)
            continue
        info = axes[tag]
        if tag == "ital" and entry.get("auto"):
            is_italic = str(entry.get("auto", "")).lower() == "italic"
            axis_defs.append(_ital_axis_def_silent(is_italic))
            continue
        values: List[AxisValueDef] = []
        for v in entry.get("values", []):
            value = float(v["value"])
            if not (info.min_value <= value <= info.max_value):
                raise ValueError(
                    f"{tag}={value} outside range "
                    f"[{info.min_value}, {info.max_value}]"
                )
            values.append(
                AxisValueDef(
                    value=value,
                    name=str(v["name"]),
                    elidable=bool(v.get("elidable", False)),
                    stat_format=int(v.get("stat_format", 1)),
                    range_min=v.get("range_min"),
                    range_max=v.get("range_max"),
                    linked_value=v.get("linked_value"),
                )
            )
        axis_defs.append(
            AxisDef(
                tag=tag,
                display_name=info.name,
                min_value=info.min_value,
                default_value=info.default_value,
                max_value=info.max_value,
                values=values,
            )
        )
    return axis_defs


def _save_yaml_config(
    path: Path,
    font_path: Path,
    output_path: Optional[Path],
    axis_defs: List[AxisDef],
    fix_fvar_default: bool,
    is_italic_font: bool,
) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required for --save-config")
    payload = {
        "font": font_path.name,
        "output": output_path.name if output_path else None,
        "fix_fvar_default": fix_fvar_default,
        "axes": [],
    }
    for ad in axis_defs:
        entry: dict = {"tag": ad.tag}
        if ad.tag == "ital" and len(ad.values) == 1:
            v = ad.values[0]
            if (v.value == 0 and v.elidable) or (v.value == 1 and not v.elidable):
                entry["auto"] = "italic" if is_italic_font else "roman"
                payload["axes"].append(entry)
                continue
        entry["values"] = [
            {
                "value": av.value,
                "name": av.name,
                "elidable": av.elidable,
                "stat_format": av.stat_format,
                **({"range_min": av.range_min, "range_max": av.range_max}
                   if av.stat_format == 2 else {}),
                **({"linked_value": av.linked_value} if av.stat_format == 3 else {}),
            }
            for av in ad.values
        ]
        payload["axes"].append(entry)
    path.write_text(yaml.dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _display_plan_summary(plan, fix_lines: List[str]) -> None:
    n_axis_ids = len(plan.axis_value_ids)
    n_inst_ids = len(plan.instance_ids)
    _emit_bold("NameID plan:")
    if n_axis_ids:
        cs.emit(
            f"  Axis value names: {n_axis_ids} new IDs  "
            f"({plan.free_start} – {plan.free_start + n_axis_ids - 1})"
        )
    if n_inst_ids:
        inst_start = plan.free_start + n_axis_ids
        cs.emit(
            f"  Instance names:   {n_inst_ids} new IDs  "
            f"({inst_start} – {plan.free_end})"
        )
    for line in fix_lines:
        cs.emit(f"  fvar default: {line}")


def _resolve_output_path(font_path: Path, output_arg: Optional[str]) -> Path:
    if output_arg:
        return Path(output_arg).expanduser().resolve()
    return font_path.with_name(f"{font_path.stem}-patched{font_path.suffix}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Define STAT axis values and fvar named instances for a variable font.",
    )
    parser.add_argument("fontfile", type=Path, help="Variable font (.ttf / .otf)")
    parser.add_argument("--output", "-o", help="Output path for patched font")
    parser.add_argument("--ttx", action="store_true", help="Also write TTX additions file")
    parser.add_argument("--config", type=Path, help="Load axis definitions from YAML")
    parser.add_argument(
        "--save-config", type=Path, help="Save session definitions to YAML after write"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show plan only; do not write")
    parser.add_argument(
        "--no-fix-default",
        action="store_true",
        help="Do not update fvar axis defaults to elidable values",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show axis/nameID audit only, then exit",
    )
    args = parser.parse_args(argv)

    font_path = args.fontfile.expanduser().resolve()
    if not font_path.is_file():
        StatusIndicator("error").add_message(f"File not found: {font_path}").emit()
        return 1
    if _is_collection_file(font_path):
        StatusIndicator("error").add_message(
            "Font collections (.ttc) are not supported"
        ).emit()
        return 1

    editor = EditorConfig(
        output_ttx=args.ttx,
        config_path=args.config,
        save_config_path=args.save_config,
        dry_run=args.dry_run,
        fix_fvar_default=not args.no_fix_default,
        info_only=args.info,
    )

    try:
        font = TTFont(str(font_path))
    except Exception as e:
        StatusIndicator("error").add_message("Error loading font").with_explanation(
            str(e)
        ).emit()
        return 1

    analysis = analyze_variable_font(font, VariableFontMode.LENIENT)
    if not analysis.has_fvar:
        StatusIndicator("error").add_message("Not a variable font (no fvar)").emit()
        return 1
    if not analysis.has_stat:
        StatusIndicator("error").add_message(
            "Font has no STAT table; add STAT before using this tool"
        ).emit()
        return 1
    if not analysis.has_design_axis_record:
        StatusIndicator("error").add_message(
            "STAT has no DesignAxisRecord; cannot write AxisValues"
        ).emit()
        return 1

    axes = _extract_axes(font)
    axis_order = _axis_order(font)
    stat_parser = STATNameParser(font)
    ot_labels = scan_ot_label_nameids(font)
    used_nameids = audit_nameids(font, ot_labels)
    ot_label_id_set = _ot_label_ids(ot_labels)
    free_start = (max(used_nameids.keys()) + 1) if used_nameids else 256

    _display_scan(
        font, axes, axis_order, stat_parser, used_nameids, ot_label_id_set, free_start
    )
    if editor.info_only:
        return 0

    is_italic_font = _detect_source_italic(font)

    if editor.config_path:
        axis_defs = _load_yaml_config(editor.config_path.resolve(), axes)
        axis_defs.sort(key=lambda ad: axis_order.index(ad.tag) if ad.tag in axis_order else 999)
    else:
        axis_defs = _interactive_axis_defs(
            font, axes, axis_order, stat_parser, is_italic_font
        )

    if not axis_defs or any(not ad.values for ad in axis_defs):
        StatusIndicator("error").add_message("No axis values defined").emit()
        return 1

    elided_fallback = "Regular"
    plan = build_allocation_plan(
        font, ot_labels, axis_defs, elided_fallback_name=elided_fallback
    )
    collisions = check_for_collisions(plan, font)
    if collisions:
        for c in collisions:
            StatusIndicator("error").add_message(c).emit()
        return 1

    fix_lines = default_fix_summary(font, axis_defs) if editor.fix_fvar_default else []
    total = count_instances(axis_defs)
    counts = " × ".join(str(len(ad.values)) for ad in axis_defs)
    _emit_bold(f"INSTANCE PREVIEW — {total} instances ({counts})")
    inst_names = enumerate_instance_names(axis_defs, elided_fallback)
    for name in inst_names[:35]:
        cs.emit(f"  {name}")
    if len(inst_names) > 35:
        _emit_dim(f"  … and {len(inst_names) - 35} more")
    if len(inst_names) != len(set(inst_names)):
        cs.emit("⚠ Duplicate composed instance names in preview.")
    cs.emit_spacer()
    _display_plan_summary(plan, fix_lines)

    if editor.dry_run:
        StatusIndicator("info").add_message("Dry run — no files written").emit()
        return 0

    if not editor.config_path:
        confirm = cs.prompt_input(
            "[Enter] Confirm and write   [e] Edit (restart)   [q] Quit: "
        ).strip().lower()
        _raise_if_quit(confirm)
        if confirm == "e":
            cs.emit("Re-run the script to edit axis definitions.")
            return 0

    apply_table_edits(
        font,
        axis_defs,
        plan,
        elided_fallback_name=elided_fallback,
        fix_fvar_default=editor.fix_fvar_default,
    )

    output_path = _resolve_output_path(font_path, args.output)
    if editor.save_config_path:
        _save_yaml_config(
            editor.save_config_path.resolve(),
            font_path,
            output_path,
            axis_defs,
            editor.fix_fvar_default,
            is_italic_font,
        )
        StatusIndicator("success").add_message(
            f"Config saved: {editor.save_config_path}"
        ).emit()

    if not editor.dry_run:
        font.save(str(output_path))
        StatusIndicator("success").add_message(f"Saved: {output_path}").emit()
        if editor.output_ttx:
            ttx_str = generate_ttx_additions(
                font, axis_defs, plan, elided_fallback_name=elided_fallback
            )
            ttx_path = output_path.with_suffix(".ttx")
            ttx_path.write_text(ttx_str, encoding="utf-8")
            StatusIndicator("success").add_message(f"TTX: {ttx_path}").emit()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except KeyboardInterrupt:
        cs.emit("\nInterrupted.")
        raise SystemExit(130)
