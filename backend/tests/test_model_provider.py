import unittest

from backend import model_provider


class TestModelProvider(unittest.TestCase):
    def test_simple_embedding_provider(self):
        provider = model_provider.get_embedding_provider(dimensions=4)
        vector = provider.embed("test")
        self.assertEqual(provider.name, "simple")
        self.assertEqual(len(vector), 4)
        self.assertAlmostEqual(sum(vector), 1.0, places=6)

    def test_unknown_provider_raises(self):
        with self.assertRaises(ValueError):
            model_provider.get_embedding_provider("unknown-provider")


if __name__ == "__main__":
    unittest.main()
