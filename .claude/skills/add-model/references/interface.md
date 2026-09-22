# Forecaster interface

```python
class Forecaster(abc.ABC):
    name: str
    probabilistic: bool = False  # True if predict_dist is implemented

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """X may include rows before y's first index (history for sequence models).
        y.index is a subset of X.index. Fit scalers etc. here, on these rows only."""

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
        """Point forecast for each date in `index`. X holds every row up to max(index);
        use only rows <= each prediction date."""

    def predict_dist(self, X: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Optional: columns `mean`, `std` and/or quantile columns like `q0.1`."""
```

Example (ridge):

```python
class RidgeForecaster(Forecaster):
    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        rows = X.loc[y.index]
        self._pipe = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        self._pipe.fit(rows.to_numpy(), y.to_numpy(), ridge__sample_weight=...)
        return self

    def predict(self, X, index):
        return pd.Series(self._pipe.predict(X.loc[index].to_numpy()), index=index)
```

The harness (`nvquant.experiments.harness.walk_forward`) handles purging,
fitted features, target scaling, and registry logging. Models never see
dates after the fold's training cutoff during `fit`.
