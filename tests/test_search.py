"""Tests for the reusable search primitives: multimodal basin detection and
the bracketed golden-section local optimizer.

Real Dirichlet-multinomial pooled evidence on this interval is empirically
unimodal, so multimodal detector behavior is tested on explicitly constructed
sampled functions; the same detector drives fit().
"""

import unittest

from dirichlet_eb import core


class DetectPeaksTests(unittest.TestCase):
    def test_interior_double_peak_with_valley(self):
        # Two equal peaks with a deep valley; monotone edges.
        values = [0.0, 1.0, 0.0, -5.0, 0.0, 1.0, 0.0]
        peaks = core._detect_peaks(values, band=1e-10)
        kinds = [(p[0], p[1], p[2]) for p in peaks]
        self.assertIn(("I", 1, 1), kinds)
        self.assertIn(("I", 5, 5), kinds)
        # no endpoint peaks: edges rise inward
        self.assertNotIn(("L", 0, 0), kinds)
        self.assertNotIn(("R", 6, 6), kinds)

    def test_monotone_inc_gives_right_edge_only(self):
        values = [0.0, 1.0, 2.0, 3.0]
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("R", 3, 3)])

    def test_monotone_dec_gives_left_edge_only(self):
        values = [3.0, 2.0, 1.0, 0.0]
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("L", 0, 0)])

    def test_subband_ripples_are_merged_into_plateau(self):
        # Wiggles of 1e-11 across the whole sampled range must not create
        # interior modes: everything is flat within the band, so both edges
        # are equal maxima.
        values = [-6.93147180564, -6.93147180563, -6.93147180562, -6.93147180563]
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("L", 0, 0), ("R", 3, 3)])

    def test_plateau_at_right_edge_with_prior_rise_gives_right_basin(self):
        # Monotone rise whose last steps are sub-band flat: one right basin.
        values = [-10.0, -9.0, -6.93147180563, -6.93147180562, -6.93147180563]
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("R", 2, 4)])

    def test_clear_separation_beyond_band_is_kept(self):
        values = [0.0, 1e-8, 0.0]  # peak exceeds 1e-10 band
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("I", 1, 1)])

    def test_plateau_in_the_middle(self):
        # rise, flat top, fall
        values = [0.0, 1.0, 2.0, 2.0, 2.0, 1.0, 0.0]
        peaks = core._detect_peaks(values, band=1e-10)
        self.assertEqual(peaks, [("I", 2, 4)])

    def test_fully_constant_gives_both_edges(self):
        peaks = core._detect_peaks([0.5] * 5, band=1e-10)
        self.assertEqual(peaks, [("L", 0, 0), ("R", 4, 4)])

    def test_single_point(self):
        self.assertEqual(core._detect_peaks([1.0], band=1e-10), [("L", 0, 0)])


class GoldenSectionTests(unittest.TestCase):
    def test_finds_known_minimum(self):
        # min of (t - 0.37)^2 on [0, 1] at 0.37
        x, lo, hi, fx, evals, iters = core._golden_minimize(
            lambda t: (t - 0.37) ** 2, 0.0, 1.0, 1e-12, 200
        )
        self.assertLess(abs(x - 0.37), 1e-10)
        self.assertLessEqual(lo, 0.37 + 1e-12)
        self.assertGreaterEqual(hi, 0.37 - 1e-12)

    def test_offcenter_minimum_inside_bracket(self):
        # A clean minimum near, but not at, the bracket center.
        f = lambda t: (t + 0.83) ** 2
        x, lo, hi, fx, _, _ = core._golden_minimize(f, -2.0, 0.0, 1e-12, 200)
        self.assertLess(abs(x - (-0.83)), 1e-10)
        self.assertGreaterEqual(lo, -2.0 - 1e-15)
        self.assertLessEqual(hi, 0.0 + 1e-15)

    def test_budget_exhaustion_raises(self):
        with self.assertRaises(core.SearchBudgetExhaustedError):
            core._golden_minimize(lambda t: (t - 0.5) ** 2, 0.0, 1.0, 1e-15, 2)

    def test_search_never_escapes_bracket(self):
        seen = []

        def f(t):
            seen.append(t)
            return (t - 0.2) ** 2

        core._golden_minimize(f, -1.0, 1.0, 1e-10, 100)
        self.assertTrue(all(-1.0 <= t <= 1.0 for t in seen))


if __name__ == "__main__":
    unittest.main()
