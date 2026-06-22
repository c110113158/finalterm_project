"""
core.py
Data-fetching and question-answering logic for the BTC sentiment LINE Bot.
Kept separate from app.py (the Flask/webhook layer) so it can be tested directly.
"""
import logging
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Gemini (Google AI Studio) setup — used as a fallback when no keyword
# intent matches, so the bot can still answer free-form questions.
# ----------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_gemini_model = None

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel("gemini-3.5-flash")
else:
    logger.warning("GEMINI_API_KEY is not set. AI fallback replies will be disabled.")

# ----------------------------------------------------------------------
# Simple in-memory cache so we don't re-hit external APIs on every message.
# Each entry: {"data": ..., "fetched_at": datetime}
# ----------------------------------------------------------------------
_CACHE = {}
CACHE_TTL_MINUTES = 15


def _cache_get(key):
    entry = _CACHE.get(key)
    if not entry:
        return None
    if datetime.now() - entry["fetched_at"] > timedelta(minutes=CACHE_TTL_MINUTES):
        return None
    return entry["data"]


def _cache_set(key, data):
    _CACHE[key] = {"data": data, "fetched_at": datetime.now()}


# ----------------------------------------------------------------------
# Data fetchers
# ----------------------------------------------------------------------
def get_price_data(days=120):
    """Fetch BTC-USD and XAUT-USD daily closes via yfinance, with returns."""
    cache_key = f"price_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    end = datetime.today()
    start = end - timedelta(days=days)

    btc = yf.download("BTC-USD", start=start, end=end, progress=False, auto_adjust=False)
    xaut = yf.download("XAUT-USD", start=start, end=end, progress=False, auto_adjust=False)

    if btc.empty or xaut.empty:
        raise RuntimeError("yfinance returned no data for BTC-USD or XAUT-USD")

    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    if isinstance(xaut.columns, pd.MultiIndex):
        xaut.columns = xaut.columns.get_level_values(0)

    btc = btc[["Close"]].rename(columns={"Close": "btc_price"})
    xaut = xaut[["Close"]].rename(columns={"Close": "xaut_price"})

    df = btc.join(xaut, how="inner")
    df["btc_return"] = df["btc_price"].pct_change()
    df["xaut_return"] = df["xaut_price"].pct_change()
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    _cache_set(cache_key, df)
    return df


def get_fear_greed(limit=30):
    """Fetch recent Fear & Greed Index values from alternative.me."""
    cache_key = f"fng_{limit}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = "https://api.alternative.me/fng/"
    resp = requests.get(url, params={"limit": limit, "format": "json"}, timeout=15)
    resp.raise_for_status()
    raw = resp.json()["data"]

    df = pd.DataFrame(raw)[["timestamp", "value", "value_classification"]]
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
    df["fear_greed_index"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "fear_greed_label"})
    df = df[["date", "fear_greed_index", "fear_greed_label"]].sort_values("date").reset_index(drop=True)

    _cache_set(cache_key, df)
    return df


# Maps Fear & Greed's English labels to Traditional Chinese for display
_FNG_LABEL_ZH = {
    "Extreme Fear": "極度恐懼",
    "Fear": "恐懼",
    "Neutral": "中性",
    "Greed": "貪婪",
    "Extreme Greed": "極度貪婪",
}


def get_gdelt_tone(days=14):
    """
    Fetch GDELT news tone timeline for 'bitcoin'.
    GDELT's DOC API only reliably covers a recent window (we ask for `days`,
    capped well under its ~3 month limit), so this is always a recent-tone read,
    not a historical one.
    """
    cache_key = f"gdelt_{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        from gdeltdoc import GdeltDoc, Filters

        end = datetime.today()
        start = end - timedelta(days=days)

        f = Filters(keyword="bitcoin", start_date=start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"))
        gd = GdeltDoc()
        tone_df = gd.timeline_search("timelinetone", f)

        if tone_df is None or tone_df.empty:
            _cache_set(cache_key, None)
            return None

        time_col = "datetime" if "datetime" in tone_df.columns else tone_df.columns[0]
        value_col = [c for c in tone_df.columns if c != time_col][0]
        tone_df["date"] = pd.to_datetime(tone_df[time_col], utc=True).dt.tz_localize(None).dt.normalize()
        tone_df = tone_df.rename(columns={value_col: "gdelt_tone"})
        tone_df = tone_df[["date", "gdelt_tone"]].sort_values("date").reset_index(drop=True)

        _cache_set(cache_key, tone_df)
        return tone_df
    except Exception:
        # GDELT can be flaky; callers handle a None result gracefully.
        _cache_set(cache_key, None)
        return None


# ----------------------------------------------------------------------
# Risk / volatility calculations (same formulas as the Colab notebook)
# ----------------------------------------------------------------------
def annualized_vol(returns):
    return returns.dropna().std() * np.sqrt(365)


def max_drawdown(price_series):
    cumulative_max = price_series.cummax()
    drawdown = (price_series - cumulative_max) / cumulative_max
    return drawdown.min()


def value_at_risk(returns, confidence=0.95):
    return returns.dropna().quantile(1 - confidence)


# ----------------------------------------------------------------------
# Intent detection — simple keyword routing (Traditional Chinese + English)
# ----------------------------------------------------------------------
INTENTS = [
    ("compare_risk", ["風險", "比較", "risk", "compare", "哪個高", "哪個風險"]),
    ("query_price", ["價格", "多少錢", "price", "report價"]),
    ("query_sentiment", ["情緒", "恐懼", "貪婪", "sentiment", "fear", "greed"]),
    ("query_news_tone", ["新聞", "風向", "語氣", "news", "tone"]),
    ("query_volatility", ["波動", "volatility"]),
    ("help", ["help", "幫助", "你可以", "功能", "怎麼用"]),
]


def detect_intent(text):
    t = text.lower()
    for intent, keywords in INTENTS:
        for kw in keywords:
            if kw.lower() in t:
                return intent
    return "unknown"


# ----------------------------------------------------------------------
# Response builders — one per intent, each returns a user-facing string
# ----------------------------------------------------------------------
def respond_price():
    df = get_price_data(days=10)
    last = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    btc_chg = last["btc_return"] * 100
    xaut_chg = last["xaut_return"] * 100
    arrow_btc = "📈" if btc_chg >= 0 else "📉"
    arrow_xaut = "📈" if xaut_chg >= 0 else "📉"
    return (
        f"💰 最新價格（{last_date}）\n\n"
        f"BTC：${last['btc_price']:,.2f} {arrow_btc} {btc_chg:+.2f}%\n"
        f"XAUT：${last['xaut_price']:,.2f} {arrow_xaut} {xaut_chg:+.2f}%"
    )


def respond_sentiment():
    df = get_fear_greed(limit=1)
    last = df.iloc[-1]
    label = last["fear_greed_label"]
    label_zh = _FNG_LABEL_ZH.get(label, label)
    idx = last["fear_greed_index"]

    comment_by_label = {
        "Extreme Fear": "市場情緒偏向恐慌，投資人可能較為謹慎觀望。",
        "Fear": "市場情緒偏向恐懼，投資人態度仍較保守。",
        "Neutral": "市場情緒中性，沒有明顯偏向。",
        "Greed": "市場情緒偏向貪婪，投資人較為樂觀。",
        "Extreme Greed": "市場情緒極度貪婪，需留意過熱風險。",
    }
    comment = comment_by_label.get(label, "市場情緒目前無明顯偏向。")

    return (
        f"😨😊 恐懼貪婪指數（{last['date'].strftime('%Y-%m-%d')}）\n\n"
        f"指數：{idx} / 100\n"
        f"分類：{label_zh}\n\n"
        f"{comment}"
    )


def respond_news_tone():
    tone_df = get_gdelt_tone(days=14)
    if tone_df is None or tone_df.empty:
        return (
            "📰 目前無法取得 GDELT 新聞語氣資料（可能是 GDELT API 暫時無回應），"
            "請稍後再試一次。"
        )
    recent_avg = tone_df["gdelt_tone"].tail(7).mean()
    direction = "偏正面" if recent_avg > 0.5 else "偏負面" if recent_avg < -0.5 else "中性"
    return (
        f"📰 近 7 日 GDELT 新聞語氣分數\n\n"
        f"平均語氣：{recent_avg:.2f}\n"
        f"整體偏向：{direction}\n\n"
        f"（語氣分數 > 0 表示正面，< 0 表示負面，數值越極端代表語氣越鮮明）"
    )


def respond_volatility():
    df = get_price_data(days=60)
    btc_vol = annualized_vol(df["btc_return"])
    xaut_vol = annualized_vol(df["xaut_return"])
    ratio = btc_vol / xaut_vol if xaut_vol else float("nan")
    return (
        f"📊 近 60 日年化波動率\n\n"
        f"BTC：{btc_vol:.2%}\n"
        f"XAUT：{xaut_vol:.2%}\n\n"
        f"BTC 波動率約為 XAUT 的 {ratio:.1f} 倍，顯示比特幣風險明顯較高。"
    )


def respond_compare_risk():
    df = get_price_data(days=90)
    btc_vol = annualized_vol(df["btc_return"])
    xaut_vol = annualized_vol(df["xaut_return"])
    btc_dd = max_drawdown(df["btc_price"])
    xaut_dd = max_drawdown(df["xaut_price"])
    btc_var = value_at_risk(df["btc_return"])
    xaut_var = value_at_risk(df["xaut_return"])

    return (
        f"⚖️ BTC 與 XAUT 風險比較（近 90 日）\n\n"
        f"年化波動率：BTC {btc_vol:.1%}　XAUT {xaut_vol:.1%}\n"
        f"最大跌幅：BTC {btc_dd:.1%}　XAUT {xaut_dd:.1%}\n"
        f"VaR 95%（單日）：BTC {btc_var:.1%}　XAUT {xaut_var:.1%}\n\n"
        f"結論：BTC 在所有風險指標上都高於 XAUT，風險明顯較高。"
    )


def respond_help():
    return (
        "🤖 可以問我這些問題：\n\n"
        "・比特幣現在多少錢？\n"
        "・市場是恐懼還是貪婪？\n"
        "・最近新聞風向如何？\n"
        "・BTC 波動率多少？\n"
        "・BTC 和 XAUT 哪個風險高？"
    )


def ask_gemini(text):
    """
    Fallback responder for messages that don't match any keyword intent.
    Sends the user's question to Gemini (Google AI Studio) and returns
    its reply as plain text, in Traditional Chinese to match the rest
    of the bot's tone.
    """
    if _gemini_model is None:
        return (
            "抱歉，我還不太理解這個問題 🙏\n"
            "輸入「help」可以看看我能回答哪些問題喔！"
        )

    prompt = (
        "你是一個 LINE 聊天機器人，負責協助使用者了解比特幣（BTC）價格、"
        "市場情緒（恐懼貪婪指數）、新聞語氣與波動率等研究主題。"
        "請用繁體中文，簡潔、友善地回答使用者的問題，"
        "若問題與比特幣或金融市場無關，也可以正常回答，但保持簡短（3-4句以內）。\n\n"
        f"使用者問題：{text}"
    )

    try:
        response = _gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 300},
        )
        reply = (response.text or "").strip()
        return reply if reply else respond_unknown()
    except Exception as e:
        logger.exception("Gemini call failed")
        return f"⚠️ AI 回覆時發生錯誤，請稍後再試。\n({type(e).__name__})"


def respond_unknown():
    return (
        "抱歉，我還不太理解這個問題 🙏\n"
        "輸入「help」可以看看我能回答哪些問題喔！"
    )


_HANDLERS = {
    "query_price": respond_price,
    "query_sentiment": respond_sentiment,
    "query_news_tone": respond_news_tone,
    "query_volatility": respond_volatility,
    "compare_risk": respond_compare_risk,
    "help": respond_help,
}


def answer(text):
    """Main entry point: takes raw user text, returns a reply string."""
    intent = detect_intent(text)

    if intent == "unknown":
        return ask_gemini(text)

    handler = _HANDLERS.get(intent, respond_unknown)
    try:
        return handler()
    except Exception as e:
        return f"⚠️ 抱歉，查詢資料時發生錯誤，請稍後再試。\n({type(e).__name__})"
