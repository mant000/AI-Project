"""A provisional teacher must respect fallback, uncertainty threshold and stable ties."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tsp_value.neural_direct import choose_edge


class DirectChoice(unittest.TestCase):
    def test_threshold_fallback_and_tie(self):
        edges=[(0,1),(0,2),(1,2)];ref=edges[-1]
        self.assertEqual(choose_edge(edges,ref,[.0005,-.1,0],[0,0,0]),ref)
        self.assertEqual(choose_edge(edges,ref,[.02,.01,0],[.03,0,0]),edges[1])
        self.assertEqual(choose_edge(edges,ref,[.02,.02,0],[0,0,0]),edges[0])
        with self.assertRaises(ValueError):choose_edge(edges,ref,[float('nan'),0,0],[0,0,0])


if __name__=='__main__':unittest.main()
