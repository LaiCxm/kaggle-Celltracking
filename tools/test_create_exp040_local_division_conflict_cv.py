from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp040_local_division_conflict_cv as generator


class Exp040GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_is_cv_only(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp040_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                self.assertEqual(notebook["cells"][6]["id"], "compact-division-conflict-module")
                all_source = ""
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        source = "".join(cell["source"])
                        compile(source, cell.get("id", "cell"), "exec")
                        all_source += source
                self.assertIn('"compact_10_14"', all_source)
                self.assertIn('"compact_12_14"', all_source)
                self.assertIn('"compact_14_14"', all_source)
                self.assertIn("rewire_occupied_daughter_conflicts", all_source)
                self.assertIn("official patched scorer", all_source)
                self.assertIn("submission_generated\": False", all_source)
                self.assertIn(
                    "global DIVISION_CONFLICT_MODE, FOCUS_STRATEGY_MODE, TEST_DIR",
                    all_source,
                )
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
