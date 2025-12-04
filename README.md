# Variable Font Instancer

Extract static font instances from variable fonts.

## Overview

Tool for generating static font files from variable fonts by instantiating specific axis values. Supports multiple naming strategies based on STAT table or fvar instance names.

## Scripts

### `VariableFont_Instancer.py`
Extract static instances from variable fonts.

**Naming Strategies:**
1. **STAT-based (default)**: Uses STAT table AxisValue names
   - Most reliable for modern variable fonts
   - Respects designer intent
2. **fvar-based**: Uses fvar instance names
   - Legacy compatibility
   - May have inconsistent naming
3. **Hybrid**: fvar names with STAT-derived completions
   - Fills missing "Regular" tokens when appropriate
   - Family-aware decisions

**Usage:**
```bash
# Generate named instances (default, STAT-based)
python VariableFont_Instancer.py fontfile.ttf

# Batch process multiple fonts
python VariableFont_Instancer.py font1.ttf font2.ttf

# Process directory
python VariableFont_Instancer.py fonts/

# Create custom instances
python VariableFont_Instancer.py fontfile.ttf --custom

# View font information
python VariableFont_Instancer.py fontfile.ttf --info

# Auto-generate instances (STAT naming)
python VariableFont_Instancer.py fontfile.ttf --auto --naming stat

# Auto-generate instances (fvar hybrid naming)
python VariableFont_Instancer.py fontfile.ttf --auto --naming fvar
```

**Options:**
- `--auto` - Auto-generate instances from font's defined instances
- `--naming` - Naming strategy: `stat`, `fvar`, or `hybrid`
- `--custom` - Create custom instances with manual axis values
- `--info` - Display font information (axes, instances, etc.)
- `-R, --recursive` - Process directories recursively
- `--dry-run` - Preview instances without generating files

## How It Works

1. **Analysis**: Reads variable font axes and defined instances
2. **Instance Selection**: Uses auto-detection or custom axis values
3. **Instantiation**: Creates static font files using fonttools varLib
4. **Naming**: Applies naming strategy to generated files
5. **Metadata**: Updates NameID tables for static instances

## Use Cases

- **Static font generation**: Create static fonts from variable font masters
- **Instance extraction**: Extract specific weight/width combinations
- **Legacy compatibility**: Generate static fonts for systems that don't support variable fonts

## Dependencies

See `requirements.txt`:
- Core dependencies (fonttools, rich) provided by included `core/` library
- No additional dependencies required

## Installation

1. Clone this repository:
```bash
git clone https://github.com/andrewsipe/VariableInstancer.git
cd VariableInstancer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Related Tools

- [FileRenamer](https://github.com/andrewsipe/FileRenamer) - Rename generated instances
- [FontNameID](https://github.com/andrewsipe/FontNameID) - Update metadata for generated instances
- [FontMetricsNormalizer](https://github.com/andrewsipe/FontMetricsNormalizer) - Normalize metrics across instances

