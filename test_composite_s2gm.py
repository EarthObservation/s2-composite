import unittest
import composite_s2gm as s2gm


class TestCompositeS2(unittest.TestCase):

    def test_round_multiple(self):
        self.assertEqual(s2gm.round_multiple(2152.36, 1500, 50), 2150)
        self.assertEqual(s2gm.round_multiple(2152.36, 2500, 50), 2150)
        self.assertEqual(s2gm.round_multiple(2150, 2500, 50), 2150)

    def test_is_list_uniform(self):
        self.assertEqual(s2gm.is_list_uniform([2, 2], 'txt'), None)
        self.assertEqual(s2gm.is_list_uniform(['a', 'a'], 'txt'), None)

        with self.assertRaises(ValueError):
            s2gm.is_list_uniform([], 'txt')
        with self.assertRaises(Exception):
            s2gm.is_list_uniform([1, 2], "msg")

    def test_get_mask_value(self):
        self.assertEqual(s2gm.get_mask_value(33, "all_bad", 100), "snow")
        self.assertEqual(s2gm.get_mask_value(34, "all_bad", 100), "bad")
        self.assertEqual(s2gm.get_mask_value(100, "all_bad", 100), "valid")
        self.assertEqual(s2gm.get_mask_value(33, "less_than", 35), "snow")
        self.assertEqual(s2gm.get_mask_value(34, "less_than", 35), "bad")
        self.assertEqual(s2gm.get_mask_value(35, "less_than", 35), "valid")

        with self.assertRaises(ValueError):
            s2gm.get_mask_value(33, "random", 100)


if __name__ == "__main__":
    unittest.main()
