import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

APP_TITLE = "Stock Analyzer Pro"
RECENT_FILE = Path("recent_tickers.json")

st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")

# ---------- Helpers ----------
def clamp(value, low=0, high=100):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50
    return max(low, min(high, value))


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def fmt_money(x):
    x = safe_float(x)
    if np.isnan(x):
        return "N/A"
    if abs(x) >= 1_000_000_000_000:
        return f"${x/1_000_000_000_000:.2f}T"
    if abs(x) >= 1_000_000_000:
        return f"${x/1_000_000_000:.2f}B"
    if abs(x) >= 1_000_000:
        return f"${x/1_000_000:.2f}M"
    return f"${x:,.2f}"


def fmt_pct(x):
    x = safe_float(x)
    if np.isnan(x):
        return "N/A"
    return f"{x*100:.2f}%"


def fmt_num(x):
    x = safe_float(x)
    if np.isnan(x):
        return "N/A"
    return f"{x:,.2f}"


def load_recent():
    try:
        if RECENT_FILE.exists():
            return json.loads(RECENT_FILE.read_text())[:12]
    except Exception:
        pass
    return ["AAPL", "MSFT", "NVDA", "TSLA", "SPY"]


def save_recent(ticker):
    ticker = ticker.upper().strip()
    recent = [x for x in load_recent() if x != ticker]
    recent.insert(0, ticker)
    try:
        RECENT_FILE.write_text(json.dumps(recent[:12]))
    except Exception:
        pass


@st.cache_data(ttl=900, show_spinner=False)
def get_history(ticker, period="2y"):
    return yf.Ticker(ticker).history(period=period, auto_adjust=False)


@st.cache_data(ttl=900, show_spinner=False)
def get_info(ticker):
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return info


@st.cache_data(ttl=1800, show_spinner=False)
def get_financials(ticker):
    tk = yf.Ticker(ticker)
    data = {}
    for name in ["financials", "balance_sheet", "cashflow"]:
        try:
            df = getattr(tk, name)
            data[name] = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            data[name] = pd.DataFrame()
    return data


def compute_indicators(df):
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = close.rolling(20).mean()
    out["SMA50"] = close.rolling(50).mean()
    out["SMA200"] = close.rolling(200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    tr = pd.concat([
        out["High"] - out["Low"],
        (out["High"] - out["Close"].shift()).abs(),
        (out["Low"] - out["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["VOL20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    return out


def get_latest(ind):
    return ind.dropna(subset=["Close"]).iloc[-1]


def score_technical(ind):
    row = get_latest(ind)
    close = safe_float(row.get("Close"))
    sma20 = safe_float(row.get("SMA20"))
    sma50 = safe_float(row.get("SMA50"))
    sma200 = safe_float(row.get("SMA200"))
    rsi = safe_float(row.get("RSI14"))
    macd = safe_float(row.get("MACD"))
    signal = safe_float(row.get("MACD_SIGNAL"))
    ret_3m = ind["Close"].pct_change(63).iloc[-1] if len(ind) > 70 else np.nan
    ret_6m = ind["Close"].pct_change(126).iloc[-1] if len(ind) > 135 else np.nan

    parts = []
    parts.append(("Price vs 20D trend", 70 if close > sma20 else 35, close > sma20))
    parts.append(("Price vs 50D trend", 72 if close > sma50 else 32, close > sma50))
    parts.append(("Price vs 200D trend", 78 if close > sma200 else 28, close > sma200))
    if not np.isnan(rsi):
        if 45 <= rsi <= 65:
            rsi_score = 68
        elif 35 <= rsi < 45 or 65 < rsi <= 75:
            rsi_score = 55
        elif rsi < 30:
            rsi_score = 42
        else:
            rsi_score = 35
        parts.append(("RSI balance", rsi_score, 35 <= rsi <= 75))
    parts.append(("MACD momentum", 70 if macd > signal else 35, macd > signal))
    if not np.isnan(ret_3m):
        parts.append(("3M price momentum", clamp(50 + ret_3m * 160), ret_3m > 0))
    if not np.isnan(ret_6m):
        parts.append(("6M price momentum", clamp(50 + ret_6m * 120), ret_6m > 0))
    return round(np.mean([p[1] for p in parts]), 1), parts


def score_valuation(info):
    pe = safe_float(info.get("trailingPE"))
    fpe = safe_float(info.get("forwardPE"))
    peg = safe_float(info.get("pegRatio"))
    ps = safe_float(info.get("priceToSalesTrailing12Months"))
    pb = safe_float(info.get("priceToBook"))
    fcf_yield = safe_float(info.get("freeCashflow")) / safe_float(info.get("marketCap")) if safe_float(info.get("marketCap")) else np.nan
    parts = []
    if not np.isnan(pe): parts.append(("Trailing P/E", clamp(80 - pe * 1.2), pe < 30))
    if not np.isnan(fpe): parts.append(("Forward P/E", clamp(82 - fpe * 1.4), fpe < 25))
    if not np.isnan(peg): parts.append(("PEG ratio", clamp(85 - peg * 20), peg < 2))
    if not np.isnan(ps): parts.append(("Price/Sales", clamp(78 - ps * 4), ps < 8))
    if not np.isnan(pb): parts.append(("Price/Book", clamp(75 - pb * 3), pb < 6))
    if not np.isnan(fcf_yield): parts.append(("Free cash flow yield", clamp(45 + fcf_yield * 700), fcf_yield > 0.03))
    if not parts:
        return 50, [("Valuation data unavailable", 50, False)]
    return round(np.mean([p[1] for p in parts]), 1), parts


def score_quality(info):
    roe = safe_float(info.get("returnOnEquity"))
    roa = safe_float(info.get("returnOnAssets"))
    op_margin = safe_float(info.get("operatingMargins"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_eq = safe_float(info.get("debtToEquity"))
    current_ratio = safe_float(info.get("currentRatio"))
    parts = []
    if not np.isnan(roe): parts.append(("Return on equity", clamp(45 + roe * 130), roe > 0.15))
    if not np.isnan(roa): parts.append(("Return on assets", clamp(45 + roa * 220), roa > 0.06))
    if not np.isnan(op_margin): parts.append(("Operating margin", clamp(45 + op_margin * 120), op_margin > 0.12))
    if not np.isnan(profit_margin): parts.append(("Net profit margin", clamp(45 + profit_margin * 130), profit_margin > 0.10))
    if not np.isnan(debt_eq): parts.append(("Debt/equity", clamp(75 - debt_eq * 0.18), debt_eq < 120))
    if not np.isnan(current_ratio): parts.append(("Current ratio", clamp(40 + min(current_ratio, 3) * 15), current_ratio > 1))
    if not parts:
        return 50, [("Quality data unavailable", 50, False)]
    return round(np.mean([p[1] for p in parts]), 1), parts


def score_growth(info):
    rev_g = safe_float(info.get("revenueGrowth"))
    earn_g = safe_float(info.get("earningsGrowth"))
    ebitda_g = safe_float(info.get("ebitdaMargins"))
    parts = []
    if not np.isnan(rev_g): parts.append(("Revenue growth", clamp(45 + rev_g * 150), rev_g > 0.08))
    if not np.isnan(earn_g): parts.append(("Earnings growth", clamp(45 + earn_g * 120), earn_g > 0.08))
    if not np.isnan(ebitda_g): parts.append(("EBITDA margin", clamp(45 + ebitda_g * 100), ebitda_g > 0.15))
    if not parts:
        return 50, [("Growth data unavailable", 50, False)]
    return round(np.mean([p[1] for p in parts]), 1), parts


def final_rating(scores, weights):
    weighted = sum(scores[k] * weights[k] for k in weights) / sum(weights.values())
    if weighted >= 72:
        action = "BUY / ACCUMULATE"
    elif weighted >= 58:
        action = "WATCH / HOLD"
    elif weighted >= 45:
        action = "WEAK HOLD / CAUTION"
    else:
        action = "AVOID / SELL BIAS"
    confidence = abs(weighted - 50) * 1.35 + 35
    confidence = clamp(confidence, 35, 95)
    return round(weighted, 1), action, round(confidence, 1)


def build_chart(ind, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=ind.index, open=ind["Open"], high=ind["High"], low=ind["Low"], close=ind["Close"], name="Price"))
    for col in ["SMA20", "SMA50", "SMA200"]:
        fig.add_trace(go.Scatter(x=ind.index, y=ind[col], mode="lines", name=col))
    fig.update_layout(title=f"{ticker} Price Trend", xaxis_rangeslider_visible=False, height=560, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def metric_explainer():
    return pd.DataFrame([
        ["Overall Score", "Weighted 0–100 blend of technicals, valuation, quality, and growth."],
        ["Confidence", "How far the score is from neutral, adjusted to avoid fake 99% certainty."],
        ["RSI", "Momentum gauge. 30 is often oversold; 70 is often overbought."],
        ["MACD", "Trend/momentum signal. MACD above signal line is usually bullish."],
        ["SMA 20/50/200", "Short, medium, and long-term trend lines."],
        ["P/E", "Price investors pay for each $1 of earnings."],
        ["PEG", "P/E adjusted for growth. Lower can mean cheaper growth."],
        ["FCF Yield", "Free cash flow divided by market cap. Higher is usually better."],
        ["ROE", "Profit generated versus shareholder equity."],
        ["Debt/Equity", "Leverage level. Lower is usually safer."],
    ], columns=["Metric", "Definition"])

# ---------- UI ----------
st.title("📈 Stock Analyzer Pro")
st.caption("Educational stock research tool. Not financial advice. Data from Yahoo Finance via yfinance.")

recent = load_recent()
with st.sidebar:
    st.header("Controls")
    selected_recent = st.selectbox("Recent entries", recent, index=0)
    ticker = st.text_input("Ticker", selected_recent).upper().strip()
    period = st.selectbox("Price history", ["6mo", "1y", "2y", "5y"], index=2)
    st.subheader("Scoring weights")
    w_tech = st.slider("Technical", 0, 100, 35)
    w_val = st.slider("Valuation", 0, 100, 25)
    w_quality = st.slider("Quality", 0, 100, 25)
    w_growth = st.slider("Growth", 0, 100, 15)
    run = st.button("Analyze", type="primary", use_container_width=True)

if not ticker:
    st.info("Enter a ticker to begin.")
    st.stop()

if run or ticker:
    try:
        hist = get_history(ticker, period)
        info = get_info(ticker)
        if hist is None or hist.empty:
            st.error("No price data found. Check the ticker symbol.")
            st.stop()
        save_recent(ticker)
        ind = compute_indicators(hist)
        row = get_latest(ind)
        tech_score, tech_parts = score_technical(ind)
        val_score, val_parts = score_valuation(info)
        qual_score, qual_parts = score_quality(info)
        growth_score, growth_parts = score_growth(info)
        scores = {"Technical": tech_score, "Valuation": val_score, "Quality": qual_score, "Growth": growth_score}
        weights = {"Technical": w_tech, "Valuation": w_val, "Quality": w_quality, "Growth": w_growth}
        overall, action, confidence = final_rating(scores, weights)

        name = info.get("longName") or info.get("shortName") or ticker
        st.subheader(f"{name} ({ticker})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", action)
        c2.metric("Overall Score", f"{overall}/100")
        c3.metric("Confidence", f"{confidence}%")
        c4.metric("Last Price", fmt_money(row.get("Close")))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Technical", f"{tech_score}/100")
        c2.metric("Valuation", f"{val_score}/100")
        c3.metric("Quality", f"{qual_score}/100")
        c4.metric("Growth", f"{growth_score}/100")

        st.plotly_chart(build_chart(ind, ticker), use_container_width=True)

        st.markdown("### Key Metrics")
        metrics = {
            "Market Cap": fmt_money(info.get("marketCap")),
            "Beta": fmt_num(info.get("beta")),
            "Trailing P/E": fmt_num(info.get("trailingPE")),
            "Forward P/E": fmt_num(info.get("forwardPE")),
            "PEG": fmt_num(info.get("pegRatio")),
            "Price/Sales": fmt_num(info.get("priceToSalesTrailing12Months")),
            "Price/Book": fmt_num(info.get("priceToBook")),
            "Dividend Yield": fmt_pct(info.get("dividendYield")),
            "Revenue Growth": fmt_pct(info.get("revenueGrowth")),
            "Earnings Growth": fmt_pct(info.get("earningsGrowth")),
            "Profit Margin": fmt_pct(info.get("profitMargins")),
            "Operating Margin": fmt_pct(info.get("operatingMargins")),
            "ROE": fmt_pct(info.get("returnOnEquity")),
            "Debt/Equity": fmt_num(info.get("debtToEquity")),
            "Current Ratio": fmt_num(info.get("currentRatio")),
            "RSI 14": fmt_num(row.get("RSI14")),
            "ATR 14": fmt_money(row.get("ATR14")),
            "20D Volatility": fmt_pct(row.get("VOL20")),
        }
        st.dataframe(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### Score Breakdown")
            breakdown = []
            for group, parts in [("Technical", tech_parts), ("Valuation", val_parts), ("Quality", qual_parts), ("Growth", growth_parts)]:
                for name_part, score, passed in parts:
                    breakdown.append({"Category": group, "Metric": name_part, "Score": round(score, 1), "Pass?": "✅" if passed else "⚠️"})
            st.dataframe(pd.DataFrame(breakdown), use_container_width=True, hide_index=True)
        with right:
            st.markdown("### Risk / Reward Levels")
            close = safe_float(row.get("Close"))
            atr = safe_float(row.get("ATR14"))
            sma50 = safe_float(row.get("SMA50"))
            sma200 = safe_float(row.get("SMA200"))
            levels = pd.DataFrame([
                ["Current price", fmt_money(close)],
                ["Possible support", fmt_money(min(sma50, sma200) if not np.isnan(sma50) and not np.isnan(sma200) else sma50)],
                ["Volatility stop idea", fmt_money(close - 2 * atr) if not np.isnan(atr) else "N/A"],
                ["Upside target idea", fmt_money(close + 3 * atr) if not np.isnan(atr) else "N/A"],
                ["ATR % of price", fmt_pct(atr / close) if close and not np.isnan(atr) else "N/A"],
            ], columns=["Level", "Value"])
            st.dataframe(levels, use_container_width=True, hide_index=True)
            st.warning("Targets/stops are mechanical reference levels, not predictions.")

        with st.expander("Metric definitions"):
            st.dataframe(metric_explainer(), use_container_width=True, hide_index=True)

        with st.expander("Company summary"):
            st.write(info.get("longBusinessSummary", "No company summary available."))

        st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        st.error(f"Something went wrong: {e}")
