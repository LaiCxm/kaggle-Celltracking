from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools import create_exp035_hierarchical_parent_gate as generator


class Exp035GeneratorTests(unittest.TestCase):
    def test_notebook_compiles_and_has_two_stage_gate(self) -> None:
        original = generator.OUT
        try:
            with tempfile.TemporaryDirectory() as directory:
                generator.OUT = Path(directory) / 'CELL_exp035_test.ipynb'
                generator.main()
                notebook = json.loads(generator.OUT.read_text(encoding='utf-8'))
                self.assertEqual(notebook['cells'][0]['id'], 'embedded-modules')
                sources = []
                for item in notebook['cells']:
                    if item['cell_type'] == 'code':
                        source = ''.join(item['source'])
                        compile(source, item.get('id', 'cell'), 'exec')
                        sources.append(source)
                all_source = '\n'.join(sources)
                self.assertIn('K_VALUES = [1, 2, 4, 8, 16, 32, 64]', all_source)
                self.assertIn('parent_top_k_then_candidate_rank', all_source)
                self.assertIn("'official_cv': None", all_source)
                self.assertNotIn('submission.csv', all_source)
        finally:
            generator.OUT = original


if __name__ == '__main__':
    unittest.main()
