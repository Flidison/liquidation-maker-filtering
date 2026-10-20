"""Small, independent tests for the detector's temporal contract."""
import unittest
import numpy as np
import pandas as pd
import polars as pl
import liquidation_bursts as lb


class TemporalContract(unittest.TestCase):
    def test_baseline_excludes_current_and_future(self):
        a = np.array([0., 1., 3., 2., 0., 100., 900.])
        np.testing.assert_allclose(lb.prior_mean(a, 3)[3:], [4/3, 2, 5/3, 34])
        self.assertAlmostEqual(lb.prior_quantile(a, 3, .99)[3], np.quantile(a[:3], .99))
        b = a.copy()
        b[5:] *= 1000
        np.testing.assert_allclose(lb.prior_mean(a, 3)[:5], lb.prior_mean(b, 3)[:5], equal_nan=True)

    def test_bybit_delay_and_half_open_boundaries(self):
        raw = np.array([4_900_000, 5_000_000, 4_800_000]) + lb.START * 1_000_000
        f = pd.DataFrame({"side": ["sell", "sell", "buy"],
                          "effective_timestamp": raw + 200_000,
                          "notional": [10., 20., 30.], "suspect_price": False,
                          "duplicate_extra_copy": False})
        c, v, _ = lb.liquidation_windows(f, "sell", 3, 5)
        self.assertEqual(c.tolist(), [0, 2, 0])
        self.assertEqual(v.tolist(), [0., 30., 0.])
        # Exactly 5.0 seconds belongs to [5,10), including the delayed buy record.
        buy, _, _ = lb.liquidation_windows(f, "buy", 3, 5)
        self.assertEqual(buy.tolist(), [0, 1, 0])

    def test_stale_quote_and_append_invariance(self):
        s = pl.DataFrame({"second": np.arange(lb.START, lb.START+40),
                          "trade_notional": np.ones(40)*100,
                          "trade_count": np.ones(40, dtype=int),
                          "mid_close": np.ones(40)*100,
                          "quote_age_upper_s": np.r_[np.ones(10,dtype=int), np.arange(1,11), np.ones(20,dtype=int)],
                          "trade_age_upper_s": np.ones(40,dtype=int)})
        full = lb.market_windows(s, 5, 10)
        self.assertFalse(full.price_valid.iloc[3])
        self.assertFalse(full.price_valid.iloc[4])  # recovery still has a stale starting quote
        self.assertTrue(np.isnan(full.log_return.iloc[3]))
        pd.testing.assert_frame_equal(full.head(5), lb.market_windows(s.head(25), 5, 10))

    def test_zero_baseline_and_cascade_detection_time(self):
        count = np.array([0,0,0,0,8,9,0], dtype=np.int32)
        vol = count.astype(float)*10000
        t = (lb.FREEZE + np.arange(1,8)*5)*1_000_000
        market = pd.DataFrame({"timestamp": t, "history_ready": [False,False,True,True,True,True,True],
                               "log_return": [-.001]*7, "strong_price": [True]*7,
                               "strong_trade": [True]*7})
        features = lb.liquidation_features(count, vol, np.zeros(7), market, 10, 5)
        threshold = {"minimum_count": 3, "minimum_notional": 25000,
                     "ratio_threshold": 5, "minimum_consecutive_windows": 2}
        flags = lb.classify(features, market, threshold, "sell")
        self.assertTrue(np.isnan(features.raw_count_ratio.iloc[4]))
        self.assertEqual(features.count_intensity_ratio.iloc[4], 8)
        self.assertTrue(flags.burst.iloc[4])
        self.assertFalse(flags.potential_cascade.iloc[4])
        # At the next window, the former burst enters the baseline and reduces its ratio.
        # Make the second observation sufficiently intense without changing prior features.
        count[5], vol[5] = 100, 1_000_000
        features = lb.liquidation_features(count, vol, np.zeros(7), market, 10, 5)
        flags = lb.classify(features, market, threshold, "sell")
        self.assertTrue(flags.potential_cascade.iloc[5])
        self.assertFalse(flags.potential_cascade.iloc[4])
        self.assertEqual(flags.classification.iloc[0], "Not evaluated")
        buy_flags = lb.classify(features, market, threshold, "buy")
        self.assertFalse(buy_flags.potential_cascade.any())


if __name__ == "__main__":
    unittest.main()
