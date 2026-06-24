import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Variable_Instancer.VariableFont_Instancer import (  # noqa: E402
    InstanceInfo,
    FontMetadata,
    NamingMode,
    STATNameParser,
    InstancerConfig,
    count_coordinate_duplicate_rows,
    unique_instances_by_coordinates,
    default_naming_mode_for_instances,
)
from fontTools.ttLib import TTFont


def _make_inst(
    index: int,
    fvar_name: str,
    stat_name: str,
    coordinates: dict,
) -> InstanceInfo:
    is_italic = coordinates.get("slnt", 0) != 0
    weight = coordinates.get("wght", 400.0)
    return InstanceInfo(
        index=index,
        fvar_name=fvar_name,
        stat_name=stat_name,
        coordinates=coordinates,
        is_italic=is_italic,
        is_bold=abs(weight - 700) < 0.5,
    )


class InstanceDedupTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
      font_path = Path(
          "/Users/skymacbook/Downloads/_Fonts/TTF/Variable/Blazetype/SizekProStencil-Variable.ttf"
      )
      if not font_path.exists():
          cls.font = None
          return
      cls.font = TTFont(font_path)
      cls.stat_parser = STATNameParser(cls.font)
      cls.metadata = FontMetadata(
          axes=[],
          instances=[],
          stat_values=cls.stat_parser.stat_values,
          source_italic=False,
          family_name="Sizek",
      )

  def setUp(self):
      if self.font is None:
          self.skipTest("Blazetype fixture font not available")

  def test_fvar_hybrid_keeps_misaligned_italic_names(self):
      instances = [
          _make_inst(5, "Compressed Book", "Compressed Book", {"wdth": 50, "wght": 450, "slnt": 0}),
          _make_inst(
              15,
              "Compressed Book Italic",
              "Compressed Book",
              {"wdth": 50, "wght": 450, "slnt": 0},
          ),
      ]
      self.metadata.instances = instances
      dups = count_coordinate_duplicate_rows(
          instances,
          self.metadata,
          self.stat_parser,
          NamingMode.FVAR_HYBRID,
      )
      kept, skipped = unique_instances_by_coordinates(
          instances,
          self.metadata,
          self.stat_parser,
          NamingMode.FVAR_HYBRID,
      )
      self.assertEqual(dups, 0)
      self.assertEqual(skipped, 0)
      self.assertEqual(len(kept), 2)

  def test_stat_mode_dedupes_same_stat_output(self):
      instances = [
          _make_inst(5, "Compressed Book", "Compressed Book", {"wdth": 50, "wght": 450, "slnt": 0}),
          _make_inst(
              15,
              "Compressed Book Italic",
              "Compressed Book",
              {"wdth": 50, "wght": 450, "slnt": 0},
          ),
      ]
      self.metadata.instances = instances
      dups = count_coordinate_duplicate_rows(
          instances,
          self.metadata,
          self.stat_parser,
          NamingMode.STAT,
      )
      kept, skipped = unique_instances_by_coordinates(
          instances,
          self.metadata,
          self.stat_parser,
          NamingMode.STAT,
      )
      self.assertEqual(dups, 1)
      self.assertEqual(skipped, 1)
      self.assertEqual(len(kept), 1)
      self.assertEqual(kept[0].fvar_name, "Compressed Book")

  def test_true_duplicate_same_output_name(self):
      instances = [
          _make_inst(0, "Compressed Semibold", "Compressed SemiBold", {"wdth": 50, "wght": 600, "slnt": 0}),
          _make_inst(1, "Compressed SemiBold", "Compressed SemiBold", {"wdth": 50, "wght": 600, "slnt": 0}),
      ]
      self.metadata.instances = instances
      kept, skipped = unique_instances_by_coordinates(
          instances,
          self.metadata,
          self.stat_parser,
          NamingMode.STAT,
      )
      self.assertEqual(skipped, 1)
      self.assertEqual(len(kept), 1)

  def test_default_mode_prefers_fvar_hybrid_when_names_exist(self):
      instances = [
          _make_inst(0, "Thin", "Thin", {"wght": 100, "slnt": 0}),
      ]
      self.assertEqual(
          default_naming_mode_for_instances(instances),
          NamingMode.FVAR_HYBRID,
      )


if __name__ == "__main__":
    unittest.main()
