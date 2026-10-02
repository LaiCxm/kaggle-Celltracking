"""Static and self-contained import tests for the EXP027 generator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools import create_exp027_structural_selection as generator


class Exp027NotebookGeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_embedded_package_imports(self) -> None:
        original_out = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / "CELL_exp027_test.ipynb"
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding="utf-8"))
                code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                for notebook_cell in code_cells:
                    compile("".join(notebook_cell["source"]), notebook_cell.get("id", "cell"), "exec")
                embedded_source = "".join(code_cells[0]["source"])
                self.assertIn("global_selection.py", embedded_source)
                local_source = embedded_source.replace("/kaggle/working", Path(directory).as_posix())
                script = Path(directory) / "execute_embedded_cell.py"
                script.write_text(local_source, encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(script)], cwd=directory,
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        finally:
            generator.OUT = original_out

    def test_notebook_is_diagnostic_only(self) -> None:
        source = generator.CANDIDATE_AND_SELECTION + generator.OUTPUTS
        self.assertNotIn("submission.csv", source)
        self.assertIn("'official_cv': None", source)
        self.assertIn("'submission_generated': False", source)
        self.assertIn("select_nonconflicting_divisions", source)

    def test_selection_summary_uses_mode_column_not_dataframe_method(self) -> None:
        source = generator.CANDIDATE_AND_SELECTION
        self.assertIn('event_df[event_df["mode"] == mode]', source)
        self.assertNotIn("event_df[event_df.mode == mode]", source)


if __name__ == "__main__":
    unittest.main()
