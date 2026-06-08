
import os
import math
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pro Stock Signal Terminal", page_icon="📈", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
:root { --green:#00c853; --red:#ff3b30; --yellow:#ffd60a; --bg:#0b0f14; --panel:#111820; --soft:#1b2530; }
.stApp { background: #0b0f14; color: #e8edf2; }
.block-container { padding-top: 5rem; }
[data-testid="stSidebar"] { background: #080b0f; }
.metric-card {
    background: #111820; border: 1px solid #1f2b38; border-radius: 16px;
    padding: 16px; box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
}
.big-signal { font-size: 42px; font-weight: 900; line-height: 1.05; }
.buy { color: #00c853; }
.sell { color: #ff3b30; }
.hold { color: #ffd60a; }
.news-card {
    background: #111820; border-radius: 14px; padding: 14px; margin-bottom: 10px;
    border-left: 6px solid #4b5563;
}
.news-pos { border-left-color: #00c853; }
.news-neg { border-left-color: #ff3b30; }
.news-neu { border-left-color: #ffd60a; }
.small { color:#9aa7b2; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

# ---------- HELPERS ----------
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

@st.cache_data(ttl=300)
def load_stock(ticker, period="1y", interval="1d"):
    t = yf.Ticker(ticker)
    hist = t.history(period=period, interval=interval, auto_adjust=True)
    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}
    return hist, info

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["RSI"] = rsi(df["Close"])
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["RET20"] = df["Close"].pct_change(20)
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    return df

def signal_engine(df, info):
    latest = df.dropna().iloc[-1]
    prev = df.dropna().iloc[-2]

    score = 0
    reasons = []

    # Trend
    if latest.Close > latest.SMA20 > latest.SMA50:
        score += 18; reasons.append("Bullish short-term trend: price > 20SMA > 50SMA")
    elif latest.Close < latest.SMA20 < latest.SMA50:
        score -= 18; reasons.append("Bearish short-term trend: price < 20SMA < 50SMA")

    if latest.Close > latest.SMA200:
        score += 12; reasons.append("Price above 200SMA: long-term trend supportive")
    else:
        score -= 12; reasons.append("Price below 200SMA: long-term trend weak")

    # Momentum
    if latest.MACD > latest.MACD_SIGNAL and prev.MACD <= prev.MACD_SIGNAL:
        score += 16; reasons.append("Fresh MACD bullish crossover")
    elif latest.MACD < latest.MACD_SIGNAL and prev.MACD >= prev.MACD_SIGNAL:
        score -= 16; reasons.append("Fresh MACD bearish crossover")
    elif latest.MACD > latest.MACD_SIGNAL:
        score += 8; reasons.append("MACD momentum positive")
    else:
        score -= 8; reasons.append("MACD momentum negative")

    if 50 <= latest.RSI <= 68:
        score += 12; reasons.append("RSI in healthy bullish range")
    elif latest.RSI > 75:
        score -= 10; reasons.append("RSI overbought")
    elif latest.RSI < 30:
        score += 6; reasons.append("RSI oversold bounce potential")
    elif latest.RSI < 45:
        score -= 6; reasons.append("RSI below bullish range")

    # Volume confirmation
    if latest.Volume > latest.VOL20 * 1.25 and latest.Close > prev.Close:
        score += 10; reasons.append("Up move confirmed by above-average volume")
    elif latest.Volume > latest.VOL20 * 1.25 and latest.Close < prev.Close:
        score -= 10; reasons.append("Sell pressure confirmed by above-average volume")

    # Fundamentals
    pe = safe_float(info.get("trailingPE"))
    fpe = safe_float(info.get("forwardPE"))
    rev_growth = safe_float(info.get("revenueGrowth"))
    profit_margin = safe_float(info.get("profitMargins"))
    debt_equity = safe_float(info.get("debtToEquity"))

    if not math.isnan(rev_growth):
        if rev_growth > 0.10: score += 8; reasons.append("Revenue growth over 10%")
        elif rev_growth < 0: score -= 8; reasons.append("Negative revenue growth")

    if not math.isnan(profit_margin):
        if profit_margin > 0.12: score += 6; reasons.append("Solid profit margin")
        elif profit_margin < 0: score -= 8; reasons.append("Negative profit margin")

    if not math.isnan(debt_equity):
        if debt_equity < 80: score += 4; reasons.append("Debt/equity appears manageable")
        elif debt_equity > 180: score -= 6; reasons.append("Debt/equity elevated")

    if not math.isnan(pe) and not math.isnan(fpe):
        if fpe < pe: score += 5; reasons.append("Forward P/E below trailing P/E")
        elif fpe > pe * 1.25: score -= 4; reasons.append("Forward P/E materially higher than trailing P/E")

    score = max(-100, min(100, score))
    confidence = min(99, max(1, int(abs(score) * 0.9 + 10)))

    if score >= 35:
        action = "BUY"
    elif score <= -35:
        action = "SELL / AVOID"
    else:
        action = "HOLD / WAIT"

    close = latest.Close
    atr = latest.ATR if not np.isnan(latest.ATR) else close * 0.03
    if action == "BUY":
        entry = close
        stop = close - atr * 1.5
        target1 = close + atr * 2
        target2 = close + atr * 3.5
    elif action.startswith("SELL"):
        entry = close
        stop = close + atr * 1.5
        target1 = close - atr * 2
        target2 = close - atr * 3.5
    else:
        entry = close
        stop = np.nan
        target1 = np.nan
        target2 = np.nan

    return action, score, confidence, reasons, entry, stop, target1, target2

def simple_sentiment(text):
    pos = ["beats", "beat", "raises", "upgrade", "bullish", "growth", "profit", "record", "surge", "strong", "approval", "partnership"]
    neg = ["misses", "miss", "cuts", "downgrade", "bearish", "lawsuit", "probe", "decline", "weak", "loss", "recall", "bankruptcy"]
    t = (text or "").lower()
    s = sum(w in t for w in pos) - sum(w in t for w in neg)
    if s > 0: return "Positive", "news-pos", "🟢"
    if s < 0: return "Negative", "news-neg", "🔴"
    return "Neutral", "news-neu", "🟡"

@st.cache_data(ttl=900)
def get_news_yfinance(ticker):
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        items = []

    rows = []

    for item in items[:10]:
        content = item.get("content", item)

        title = content.get("title") or item.get("title") or "Untitled"

        link = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or item.get("link", "")
        )

        publisher = (
            content.get("provider", {}).get("displayName")
            or item.get("publisher", "")
        )

        summary = content.get("summary") or item.get("summary", "")

        # Published date
        pub_time = (
            content.get("pubDate")
            or item.get("providerPublishTime")
        )

        formatted_date = "Unknown date"

        try:
            if isinstance(pub_time, (int, float)):
                formatted_date = datetime.fromtimestamp(pub_time).strftime("%b %d, %Y %I:%M %p")
            elif isinstance(pub_time, str):
                formatted_date = pd.to_datetime(pub_time).strftime("%b %d, %Y %I:%M %p")
        except Exception:
            pass

        rows.append({
            "title": title,
            "link": link,
            "publisher": publisher,
            "summary": summary,
            "date": formatted_date
        })

    return rows
    return rows

# ---------- SIDEBAR ----------
st.sidebar.title("📈 Signal Terminal")
ticker = st.sidebar.text_input("Ticker", value="NVDA").upper().strip()
period = st.sidebar.selectbox("Chart range", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Candle interval", ["1d", "1h", "30m", "15m"], index=0)
watchlist_raw = st.sidebar.text_area("Watchlist", "AAPL\nMSFT\nNVDA\nTSLA\nAMD\nMETA\nPLTR")
watchlist = [x.strip().upper() for x in watchlist_raw.splitlines() if x.strip()]

st.sidebar.caption("Educational model only. Not financial advice.")

# ---------- MAIN ----------
if not ticker:
    st.stop()

hist, info = load_stock(ticker, period, interval)
if hist.empty or len(hist) < 60:
    st.error("Not enough price data. Try a longer range or daily interval.")
    st.stop()

df = add_indicators(hist)
action, score, confidence, reasons, entry, stop, target1, target2 = signal_engine(df, info)
latest = df.iloc[-1]
prev = df.iloc[-2]
change = latest.Close - prev.Close
change_pct = change / prev.Close * 100

name = info.get("shortName") or info.get("longName") or ticker

top = st.columns([1.4, .8, .8, .8, .8])
with top[0]:
    st.markdown(f"## {ticker} — {name}")
    st.markdown(f"<span class='small'>{info.get('sector','')} · {info.get('industry','')}</span>", unsafe_allow_html=True)
with top[1]:
    st.metric("Price", f"${latest.Close:,.2f}", f"{change:+.2f} / {change_pct:+.2f}%")
with top[2]:
    st.metric("Signal Score", f"{score:+.0f}/100")
    st.markdown(
        "<span style='color:#9aa7b2;font-size:12px;'>Overall bullish/bearish rating from technical + fundamental analysis.</span>",
        unsafe_allow_html=True
    )

with top[3]:
    st.metric("Confidence", f"{confidence}%")
    st.markdown(
        "<span style='color:#9aa7b2;font-size:12px;'>How strongly the indicators agree with the signal.</span>",
        unsafe_allow_html=True
    )
with top[4]:
    cls = "buy" if action == "BUY" else "sell" if action.startswith("SELL") else "hold"
    st.markdown(f"<div class='metric-card'><div class='big-signal {cls}'>{action}</div></div>", unsafe_allow_html=True)

left, right = st.columns([2.25, 1])

with left:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candles"))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"], name="SMA20", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA50"], name="SMA50", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(width=1)))
    fig.update_layout(height=560, template="plotly_dark", margin=dict(l=10,r=10,t=25,b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Trade Plan", "Technicals", "Fundamentals", "Watchlist Scan"])
    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entry area", f"${entry:,.2f}")
        c2.metric("Stop loss", "N/A" if np.isnan(stop) else f"${stop:,.2f}")
        c3.metric("Target 1", "N/A" if np.isnan(target1) else f"${target1:,.2f}")
        c4.metric("Target 2", "N/A" if np.isnan(target2) else f"${target2:,.2f}")
        st.write("**Why this signal:**")
        for r in reasons:
            st.write(f"• {r}")

    with tabs[1]:
        tech = pd.DataFrame({
            "Metric": ["RSI", "MACD", "MACD Signal", "SMA20", "SMA50", "SMA200", "Volume vs 20D Avg", "20-period Return"],
            "Value": [
                round(latest.RSI,2), round(latest.MACD,3), round(latest.MACD_SIGNAL,3),
                round(latest.SMA20,2), round(latest.SMA50,2), round(latest.SMA200,2),
                f"{latest.Volume/latest.VOL20:.2f}x", f"{latest.RET20*100:.2f}%"
            ],
            "Meaning": [
                "Momentum; 50-70 is bullish, >75 stretched",
                "Trend momentum line",
                "MACD trigger line",
                "Short-term trend",
                "Medium-term trend",
                "Long-term trend",
                "Breakouts need volume confirmation",
                "Recent price strength"
            ]
        })
        st.dataframe(tech, use_container_width=True, hide_index=True)

    with tabs[2]:
        fundamentals = {
            "Market Cap": info.get("marketCap"),
            "Trailing P/E": info.get("trailingPE"),
            "Forward P/E": info.get("forwardPE"),
            "Revenue Growth": info.get("revenueGrowth"),
            "Profit Margins": info.get("profitMargins"),
            "Debt/Equity": info.get("debtToEquity"),
            "52W High": info.get("fiftyTwoWeekHigh"),
            "52W Low": info.get("fiftyTwoWeekLow"),
            "Analyst Target": info.get("targetMeanPrice"),
            "Recommendation": info.get("recommendationKey"),
        }
        st.dataframe(pd.DataFrame([fundamentals]).T.rename(columns={0:"Value"}), use_container_width=True)

    with tabs[3]:
        rows = []
        for w in watchlist:
            try:
                h, inf = load_stock(w, "1y", "1d")
                if len(h) > 60:
                    dd = add_indicators(h)
                    a, s, conf, _, _, _, _, _ = signal_engine(dd, inf)
                    rows.append({"Ticker": w, "Signal": a, "Score": s, "Confidence": f"{conf}%", "Price": round(dd.iloc[-1].Close, 2)})
            except Exception:
                pass
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("Score", ascending=False), use_container_width=True, hide_index=True)

with right:
    st.markdown("### Company News Sentiment")
    news = get_news_yfinance(ticker)
    if not news:
        st.info("No recent news found from yfinance.")
    for n in news:
        sent, css, emoji = simple_sentiment(n["title"] + " " + n.get("summary",""))
        url = n.get("link") or "#"
        st.markdown(f"""
        <div class="news-card {css}">
            <b>{emoji} {sent}</b><br>
            <a href="{url}" target="_blank" style="color:#e8edf2;text-decoration:none;"><b>{n['title']}</b></a><br>
            <span class="small">{n.get('publisher','')}</span><br>
            <span class="small">{n.get('summary','')[:180]}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Risk Controls")
    st.write("• Never risk more than 1–2% of account on one trade.")
    st.write("• Avoid entries right before earnings unless that is your strategy.")
    st.write("• A BUY signal still needs confirmation from the actual chart.")
