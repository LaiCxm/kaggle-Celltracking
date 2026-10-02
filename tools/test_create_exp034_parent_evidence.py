from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp034_parent_evidence as generator


class Exp034GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_contains_parent_features(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp034_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                sources = []
                for item in notebook["cells"]:
                    if item["cell_type"] == "code":
                        source = "".join(item["source"])
                        compile(source, item.get("id", "cell"), "exec")
                        sources.append(source)
                all_source = "\n".join(sources)
                self.assertEqual(notebook["cells"][0]["id"], "embedded-modules")
                self.assertEqual(notebook["cells"][1]["id"], "setup-and-event-sampling")
                self.assertIn("parent_center_score", all_source)
                self.assertIn("parent_forward_count", all_source)
                self.assertIn("logistic_parent_evidence", all_source)
                self.assertIn("'experiment': 'EXP034'", all_source)
                self.assertNotIn("submission.csv", all_source)
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
