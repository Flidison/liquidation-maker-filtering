import unittest
import numpy as np
import polars as pl
import pandas as pd
import liquidation_followthrough as lf


class OutcomeTests(unittest.TestCase):
    def path(self, returns, ages=None):
        # Last price in second s is available at boundary s+1.
        mid = 100*np.exp(np.r_[0., np.cumsum(returns)])
        return lf.PricePath(pl.DataFrame({"second": np.arange(len(mid)), "mid_close": mid,
            "quote_age_upper_s": np.ones(len(mid), int) if ages is None else ages}))

    def test_return_and_rv_are_different(self):
        p = self.path(np.array([.001, -.001, .002, -.002, 0.]))
        v = p.interval(np.array([1]), np.array([6]))
        self.assertAlmostEqual(v["return_bps"][0], 0., places=8)
        self.assertAlmostEqual(v["rv_bps"][0], np.sqrt(10)*10, places=7)
        self.assertAlmostEqual(v["rms_bps"][0], np.sqrt(2)*10, places=7)

    def test_stale_quote_and_recovery_invalid(self):
        p = self.path(np.full(5, .001), np.array([1, 1, 3, 1, 1, 1]))
        self.assertFalse(p.interval([2], [4])["valid"][0])
        self.assertFalse(p.interval([3], [4])["valid"][0])
        self.assertTrue(p.interval([4], [6])["valid"][0])
        self.assertFalse(p.interval([-1], [2])["valid"][0])
        self.assertFalse(p.interval([4], [7])["valid"][0])

    def test_zero_baseline_directions(self):
        np.testing.assert_equal(lf.change([1, 0, 0, np.nan], [0, 0, 1, 1]),
                                ["increased", "unchanged", "decreased", "missing"])
        self.assertTrue(np.isnan(lf.safe_ratio([1.], [0.])[0]))
        np.testing.assert_equal(lf.direction([1, -1, 0, np.nan]), ["up", "down", "unchanged", "missing"])

    def test_future_append_does_not_change_available_price(self):
        short = self.path(np.array([.001, -.002, .003]))
        long = self.path(np.array([.001, -.002, .003, .1]))
        for key in ("return_bps", "rv_bps"):
            np.testing.assert_allclose(short.interval([1], [4])[key], long.interval([1], [4])[key])
        self.assertGreater(long.interval([4], [5])["return_bps"][0], 900)

    def test_outcome_does_not_cross_validation_boundary(self):
        start = lf.lb.VALIDATION-30
        seconds = pl.DataFrame({"second": np.arange(start, start+60),
            "mid_close": 100*np.exp(np.arange(60)*.001), "quote_age_upper_s": np.ones(60, int)})
        p = lf.PricePath(seconds)
        c = pd.DataFrame({"episode_id": ["test"], "symbol": ["btcusdt"], "exchange": ["bybit"],
            "side": ["sell"], "phase": ["train"], "classification": [lf.CLASSES[1]],
            "onset_timestamp": [(start+10)*1_000_000],
            "first_detection_timestamp": [(start+15)*1_000_000],
            "end_timestamp": [(start+20)*1_000_000],
            "end_recognition_timestamp": [(start+25)*1_000_000],
            "first_cascade_detection_timestamp": pd.array([None], dtype="Int64")})
        c = lf.measure_catalog(c, p)
        o = lf.measure_outcomes(c, p, np.zeros(12, bool))
        at_end = o.loc[o.anchor.eq("episode_end")].set_index("horizon_s")
        self.assertTrue(np.isfinite(at_end.loc[5, "post_return_bps"]))
        self.assertLess(at_end.loc[5, "post_signed_return_bps"], 0)
        self.assertFalse(at_end.loc[20, "target_within_phase"])
        self.assertTrue(np.isnan(at_end.loc[20, "post_return_bps"]))


if __name__ == "__main__":
    unittest.main()
