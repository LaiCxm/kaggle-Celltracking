from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp039_precision_gate_sweep as generator


class Exp039GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_contains_gate_sweep(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp039_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                self.assertEqual(notebook["cells"][0]["id"], "embedded-modules")
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source_text = "".join(item["source"])
                        compile(source_text, item.get("id", "cell"), "exec")
                        sources.append(source_text)
                all_source = "\n".join(sources)
                self.assertIn("PARENT_GATES_UM = [8.0, 10.0, 12.0, 14.0, 16.0]", all_source)
                self.assertIn("DAUGHTER_GATES_UM = [10.0, 12.0, 14.0, 16.0, 20.0]", all_source)
                self.assertIn("top1_all_recall", all_source)
                self.assertIn("'official_cv': None", all_source)
                self.assertNotIn("submission.csv", all_source)
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
