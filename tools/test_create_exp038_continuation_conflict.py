from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp038_continuation_conflict as generator


class Exp038GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_keeps_full_candidates(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp038_test.ipynb"
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
                self.assertIn("CONTINUATION_FEATURES", all_source)
                self.assertIn("candidate_parent_closest_both", all_source)
                self.assertIn("no candidates are gated out", all_source)
                self.assertIn("'official_cv': None", all_source)
                self.assertNotIn("submission.csv", all_source)
        finally:
            generator.OUT = original


if __name__ == "__main__":
    unittest.main()
