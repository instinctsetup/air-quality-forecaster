# The math inside air-quality-forecaster, in plain words

## Pinball (quantile) loss - the heart of the project

For quantile level tau, the loss of predicting q when the truth is y:

    L(q, y) =  tau * (y - q)     if y >= q   (underprediction)
    L(q, y) =  (tau - 1)(y - q)  if y <  q   (overprediction)

At tau = 0.5 both penalties are equal and the loss is exactly MAE/2 - the optimal point is the **median**. At tau = 0.9, underpredicting costs 9x more than overpredicting, which pushes the optimal prediction up to the 90th percentile. Train one network with three outputs at taus (0.1, 0.5, 0.9) and you get a median plus an 80% interval in a single forward pass.

The gradient is piecewise constant: `-tau` when underpredicting, `1 - tau` when overpredicting. That robustness (outliers move the gradient by a bounded amount) is why quantile regression is loved in practice.

## Non-crossing intervals by construction

Raw network outputs could produce p90 < p50, which is nonsense. Instead the net outputs `(median, d_lo, d_hi)` and we compute:

    p10 = median - softplus(d_lo)        softplus(x) = log(1 + e^x) >= 0
    p90 = median + softplus(d_hi)

softplus keeps both deltas positive, so p10 <= p50 <= p90 always - even before training. Backprop chains through `d softplus / dx = sigmoid(x)`.

## The network

`input(26) -> 64 ReLU -> 32 ReLU -> 3 outputs`, He initialization, plain mini-batch gradient descent with decaying learning rate. Backprop is the chain rule written out line by line in `model.py::backward`.

## Cyclical time encoding

Hour-of-day as an integer would tell the model that 23:00 and 00:00 are maximally far apart. Encode as `(sin(2 pi h / 24), cos(2 pi h / 24))` and midnight neighbors eleven PM on the circle, as it should.

## Why the split is by time, not random

Hours next to each other are nearly identical. A random split would put neighbors of every test point in training - the model "remembers" instead of forecasting, and the metrics lie. We train on the first 80% of time and test on the last 20%.

## Ablation: exogenous sensors hurt (honest)

Adding the latest NOx, NO2, temperature and humidity as features raised 1-step test MAE from 0.66 to 0.78 - the extra channels are noisy and the model overfit them. They were removed. Keeping a feature because it *should* help, rather than because it *measurably does*, is how forecasts quietly rot.

## Why persistence wins at 1 hour (and why we still ship the model)

CO disperses slowly, so "next hour = this hour" is extraordinarily strong (MAE 0.533). Beating it at 1-step is a known-hard problem in the literature. The model's real value: at 24 hours it beats flat persistence by 26% (0.92 vs 1.25 MAE) and it quantifies uncertainty - a baseline point forecast can never tell you *how unsure* it is.
