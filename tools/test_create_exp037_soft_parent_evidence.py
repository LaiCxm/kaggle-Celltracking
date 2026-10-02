from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp037_soft_parent_evidence as generator


class Exp037GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_keeps_full_candidates(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp037_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                self.assertEqual(notebook["cells"][0]["id"], "embedded-modules")
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source = "".join(item["source"])
                        compile(source, item.get("id", "cell"), "exec")
                        sources.append(source)
                all_source = "\n".join(sources)
                self.assertIn("SOFT_PARENT_FEATURES", all_source)
                self.assertIn("No parent Top-K gate is applied.", all_source)
                self.assertIn("'selection': 'soft_parent_features_full_candidate_rank'", all_source)
                self.assertIn("'official_cv': None", all_source)
                self.assertNotIn("submission.csv", all_source)
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
