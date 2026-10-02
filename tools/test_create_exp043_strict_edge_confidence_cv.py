from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp043_strict_edge_confidence_cv as generator


class Exp043GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_has_strict_gates(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp043_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                source = "".join(
                    "".join(cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                )
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        compile("".join(cell["source"]), cell.get("id", "cell"), "exec")
                self.assertIn('"strict_prob05"', source)
                self.assertIn('"strict_prob10"', source)
                self.assertIn('"strict_prob20"', source)
                self.assertIn("require_edge_probability_present", source)
                self.assertIn("submission_generated\": False", source)
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
