# Stock Analyzer Pro

A browser-based Streamlit stock analyzer with:

- Buy / hold / sell bias
- Overall score and confidence score
- Technical, valuation, quality, and growth sub-scores
- Candlestick chart with 20/50/200-day moving averages
- RSI, MACD, ATR, volatility, P/E, PEG, margins, ROE, debt/equity, revenue growth, earnings growth, dividend yield, and more
- Recent ticker dropdown
- Metric definitions next to the values

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, and `README.md` to your GitHub repo.
2. Go to Streamlit Community Cloud.
3. Select the repo and set the main file path to `app.py`.
4. Deploy.

This is an educational research tool, not financial advice.
