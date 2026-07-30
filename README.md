# NVIDIA Stock Price Forecasting (RNN/LSTM)

A deep learning project forecasting NVIDIA stock prices using a stacked Recurrent Neural Network built with LSTM layers. Trained on over 1,257 days of historical open price data, the model predicts the next day's price from a 60-day rolling window and achieves an RMSE of 0.08 on unseen test data.

## How It Works

Stock prices are inherently sequential â€” the price today depends on what happened over the past weeks. To capture that, I used a sliding window approach: for each day, the model takes the 60 preceding days as input and predicts the 61st. All prices were normalized using MinMaxScaler before training, and the model outputs scaled predictions that are then inverted back to real price values.

## Model Architecture

The network is a 4-layer stacked LSTM with 100 units per layer and 20% Dropout between each layer to reduce overfitting. The final layer is a Dense output node producing a single price prediction.

- Input: 60-day rolling window of normalized open prices
- LSTM Layer 1: 100 units, ReLU, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 100 units, ReLU, return_sequences=True
- Dropout: 0.2
- LSTM Layer 3: 100 units, ReLU, return_sequences=True
- Dropout: 0.2
- LSTM Layer 4: 100 units, ReLU
- Dropout: 0.2
- Output: Dense(1)

Compiled with Adam optimizer and MSE loss.

## Results

The model achieved **RMSE of 0.08** on held-out test data, with strong trend accuracy â€” correctly capturing the direction of price movement across the test period. The model was particularly effective at tracking NVIDIA's significant price run-ups, reflecting the strength of LSTM at learning long-range sequential dependencies.

## What I Learned

Building this project gave me hands-on experience with time series preprocessing, sequence windowing, and the practical tradeoffs of LSTM depth vs. training time. I also learned how critical normalization is for RNN stability â€” early runs without scaling produced unstable gradients and poor convergence.

## Tech Stack

Python, TensorFlow/Keras, Pandas, NumPy, scikit-learn, Matplotlib, Google Colab

## Files

- `nvidia_stock_rnn.ipynb` â€” full notebook with data loading, preprocessing, model training, evaluation, and prediction plots

## How to run

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python download_data.py
jupyter notebook nvidia_stock_rnn.ipynb
```

Requires Python 3.10+ and TensorFlow. `download_data.py` fetches NVDA history via yfinance and writes `data/TRAIN_NVDA.csv` and `data/TEST_NVDA.csv`. Point the notebook at those files instead of Colab Drive paths.

## Disclaimer

Educational time-series project only. Not investment advice. Past RMSE on a historical window does not imply future trading performance.
## Troubleshooting

- **Download fails**: rerun `make data`; mirrors can rate-limit anonymous requests.
- **Notebook paths**: ensure CSVs live under `data/` after the download step.
- **Reproducibility**: pin dependencies with `pip freeze > requirements.lock` when publishing results.
