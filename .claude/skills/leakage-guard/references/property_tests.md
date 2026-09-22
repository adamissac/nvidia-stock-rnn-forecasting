# The causality property test

For every registered feature function f and random cutoff t:

1. Build market data D.
2. Build D' by perturbing every value strictly after t (all tickers, macro, and
   calendar inputs). Rows at or before t stay identical.
3. Assert `f(D).loc[:t]` equals `f(D').loc[:t]` exactly (NaN positions included).

If a feature ever reads a row after t, some perturbation changes its value at or
before t, and hypothesis finds it. The test lives in
`tests/property/test_feature_causality.py` and iterates over
`nvquant.features.registry.all_specs()`, so a new feature is covered as soon as
it's registered.

The same idea covers:

- labels: every label row has `t_end >= t`, and the label at t is unchanged when
  data after `t_end` is perturbed (`tests/property/test_labels.py`);
- fitted features and the HMM forward filter: output at t is unchanged when data
  after t is perturbed, given parameters fit on data before the cutoff;
- splitters: no training label interval `[t, t_end]` overlaps a test fold, and the
  embargo gap holds (`tests/property/test_cv_purging.py`).

Minimal template:

```python
@given(cutoff=st.integers(min_value=50, max_value=N - 2), seed=st.integers(0, 10_000))
def test_feature_is_causal(spec, cutoff, seed):
    data = make_market_data(seed)
    t = data.index[cutoff]
    perturbed = perturb_after(data, t, seed=seed + 1)
    a = spec.compute(data).loc[:t]
    b = spec.compute(perturbed).loc[:t]
    pd.testing.assert_frame_equal(a, b)
```
