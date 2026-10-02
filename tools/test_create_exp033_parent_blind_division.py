from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp033_parent_blind_division as generator


class Exp033GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_removes_parent_oracle(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp033_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source = "".join(item["source"])
                        compile(source, item.get("id", "cell"), "exec")
                        sources.append(source)
                all_source = "\n".join(sources)
                self.assertIn("parent_nodes = [node for node", all_source)
                self.assertIn("parent_oracle_used': False", all_source)
                self.assertIn("no_division_threshold_scan.csv", all_source)
                self.assertIn("logistic_mask_geometry", all_source)
        finally:
            generator.OUT = original

    def test_diagnostic_only(self) -> None:
        source = generator.PARENT_BLIND_RANKER + generator.OUTPUTS
        self.assertNotIn("submission.csv", source)
        self.assertIn("'official_cv': None", source)
        self.assertIn("'submission_generated': False", source)


if __name__ == "__main__":
    unittest.main()
