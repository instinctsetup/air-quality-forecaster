# Viva questions you should expect, with answers

**Q: Why quantile regression instead of a normal regression?**
A point forecast hides uncertainty. Pinball loss gives a calibrated interval (p10-p90) in one model, which is what decisions actually need - "how bad could the air get tomorrow" is an interval question.

**Q: Derive why tau = 0.9 gives the 90th percentile.**
Underpredicting costs 0.9 per unit, overpredicting 0.1. The loss is minimized where P(y <= q) = 0.9 - you keep raising q until being under is exactly 9x as costly as being over. (This is a standard result: the pinball minimizer is the tau-quantile.)

**Q: How do you stop the interval from crossing?**
The network outputs a median and two positive offsets through softplus; p10 = median - offset_lo, p90 = median + offset_hi. Ordering holds by construction, even on random weights.

**Q: Why is your test split by time?**
Random splits put neighbors of every test hour in training - data leakage. Time-based splitting tests exactly what deployment does: predict the future from the past.

**Q: Your 1-step MAE loses to naive persistence. Isn't the project a failure?**
No - that result is expected in this domain (hourly pollutants are heavily autocorrelated), and reporting it honestly is the point. The model wins where forecasting is actually hard: 24 hours ahead (26% better than flat persistence), and it provides calibrated intervals no baseline offers.

**Q: What is recursive multi-step forecasting and what is its weakness?**
Feed each prediction back as an input for the next step. Weakness: errors compound - the model reads its own mistakes as if they were data. That is exactly why the interval fan widens with horizon, and why we evaluate the 24h recursion explicitly instead of only 1-step.

**Q: What would you add with more time?**
A direct multi-horizon model (24 outputs, no recursion), weather forecast features as *known future* inputs, ensembling across seeds for sharper intervals, and conformal calibration of the interval on a validation slice.
