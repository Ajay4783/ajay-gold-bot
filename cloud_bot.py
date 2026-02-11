import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import feedparser
import numpy as np
import requests
import os
import mysql.connector

st.set_page_config(page_title="Gold AI Quant Pro", page_icon="🧬", layout="wide")

try:
    MY_BOT_TOKEN = st.secrets["MY_BOT_TOKEN"]
    MY_CHANNEL_NAME = st.secrets["MY_CHANNEL_NAME"]
except Exception:
    MY_BOT_TOKEN = "TOKEN_REMOVED_FOR_SAFETY"
    MY_CHANNEL_NAME = "@ajay_gold_2026"

st.title("🧬 Gold Rate AI (Quant Pro Version)")
st.markdown("### 🤖 Strategy: AI Trinity + Eagle Eye + Monte Carlo + India ETF Scanner")

def send_telegram_alert(token, channel_id, message):
    if not token or token == "TOKEN_REMOVED_FOR_SAFETY":
        return False, "No Token"
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": channel_id, "text": message, "parse_mode": "Markdown"}
        r = requests.get(url, params=params, timeout=6)
        if r.status_code == 200:
            return True, "Success"
        return False, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

def get_portfolio_status(name):
    try:
        conn = get_db_connection()
        query = "SELECT * FROM trades WHERE username = %s ORDER BY id DESC LIMIT 1"
        df = pd.read_sql(query, conn, params=(name,))
        conn.close()
        if not df.empty:
            last = df.iloc[0]
            return float(last["cash_bal"]), float(last["gold_bal"]), float(last["avg_buy_price"]), df
        else:
            return 100000.0, 0.0, 0.0, pd.DataFrame()
    except Exception as e:
        return 100000.0, 0.0, 0.0, pd.DataFrame()

def execute_trade(name, action, live_price, grams, verdict):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM trades WHERE username = %s ORDER BY id DESC LIMIT 1", (name,))
        last = cursor.fetchone()
        cash = float(last["cash_bal"]) if last else 100000.0
        gold = float(last["gold_bal"]) if last else 0.0
        avg_price = float(last["avg_buy_price"]) if last else 0.0
        total_amt = live_price * grams
        status = False
        msg = ""
        if action == "BUY":
            if cash < total_amt:
                conn.close()
                return False, "❌ Insufficient Cash!"
            new_cash = cash - total_amt
            current_invested = gold * avg_price
            new_invested = current_invested + total_amt
            new_gold = gold + grams
            new_avg = new_invested / new_gold if new_gold > 0 else 0.0
            msg = f"✅ Bought {grams}g @ ₹{live_price:.0f}"
            status = True
        elif action == "SELL":
            if gold < grams:
                conn.close()
                return False, "❌ Not enough Gold!"
            new_cash = cash + total_amt
            new_gold = gold - grams
            new_avg = avg_price if new_gold > 0 else 0.0
            msg = f"✅ Sold {grams}g @ ₹{live_price:.0f}"
            status = True
        else:
            conn.close()
            return False, "❌ Unknown Action"
        if status:
            sql = """INSERT INTO trades (username, trade_date, action, price, grams, amount, cash_bal, gold_bal, avg_buy_price, ai_verdict) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"""
            val = (date.today(), action, live_price, grams, total_amt, new_cash, new_gold, new_avg, verdict)
            cursor.execute(sql, val)
            conn.commit()
            conn.close()
            return True, msg
        conn.close()
        return False, "Transaction Failed"
    except Exception as e:
        return False, str(e)
    

def get_trade_history(name):
    try:
        conn = get_db_connection()
        df = pd.read_sql("SELECT * FROM trades WHERE username = %s ORDER BY id DESC", conn, params=(name,))
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()
    

def reset_account(name):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM trades WHERE username = %s", (name,))
        sql = """INSERT INTO trades (username, trade_date, action, price, grams, amount, cash_bal, gold_bal, avg_buy_price, ai_verdict) 
                 VALUES (%s, %s, 'INIT', 0.0, 0.0, 0.0, 100000.0, 0.0, 0.0, 'START')"""
        cursor.execute(sql, (name, date.today()))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

st.sidebar.header("⚙️ AI Settings")
years = st.sidebar.slider("📅 AI Learning Period (Years)", 2, 5, 2)
st.sidebar.markdown("---")
st.sidebar.header("🏛️ Tax & Duty Settings")
tax_percentage = st.sidebar.slider("Import Duty + GST (%)", 0.0, 50.0, 9.2, 0.1)
user_name = st.sidebar.text_input("👤 Enter Your Name", value="Guest").strip()

if not user_name:
    st.warning("Please enter your name to start trading!")
    st.stop()

if MY_CHANNEL_NAME:
    st.sidebar.success(f"✅ Connected: **{MY_CHANNEL_NAME}**")


def initialize_user(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM trades WHERE username = %s", (name,))
    if cursor.fetchone()[0] == 0:
        sql = """INSERT INTO trades (username, trade_date, action, price, grams, amount, cash_bal, gold_bal, avg_buy_price, ai_verdict) 
                 VALUES (%s, %s, 'INIT', 0.0, 0.0, 0.0, 100000.0, 0.0, 0.0, 'START')"""
        cursor.execute(sql, (name, date.today()))
        conn.commit()
    conn.close()

initialize_user(user_name)

@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        df = yf.download("INR=X", period="2d", progress=False)
        if not df.empty:
            if 'Close' in df.columns:
                return float(df['Close'].iloc[-1])
            return float(df.iloc[-1, df.columns.get_loc('Close')]) if 'Close' in df.columns else float(df.iloc[-1,0])
        return 86.50
    except Exception:
        return 86.50

current_usd_inr = get_usd_inr_rate()
st.sidebar.info(f"💵 Live USD/INR Rate: **₹{current_usd_inr:.2f}**")

def to_inr(price_usd):
    try:
        return (price_usd * current_usd_inr / 31.1035) * (1 + tax_percentage/100)
    except Exception:
        return price_usd

@st.cache_data(ttl=600)
def get_live_gold_rate():
    try:
        gdf = yf.download("GC=F", period="2d", progress=False)
        if not gdf.empty and 'Close' in gdf.columns:
            price_usd = float(gdf['Close'].iloc[-1])
            return to_inr(price_usd)
    except Exception:
        pass
    return None

live_price_today = get_live_gold_rate()

def ensure_ohlc(df):
    if 'High' not in df.columns:
        df['High'] = df['Gold'] * 1.005
    if 'Low' not in df.columns:
        df['Low'] = df['Gold'] * 0.995
    if 'Open' not in df.columns:
        df['Open'] = df['Gold'].shift(1).fillna(df['Gold'])
    return df

def detect_patterns(df):
    df = ensure_ohlc(df)
    body = (df['Gold'] - df['Open']).abs()
    lower_wick = np.minimum(df['Gold'], df['Open']) - df['Low']
    upper_wick = df['High'] - np.maximum(df['Gold'], df['Open'])
    df['is_Hammer'] = np.where((lower_wick > 2 * body) & (upper_wick < body), 1, 0)
    df['is_ShootingStar'] = np.where((upper_wick > 2 * body) & (lower_wick < body), 1, 0)
    return df

@st.cache_data
def load_data(years):
    today = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    def get_ticker_data(ticker, name):
        try:
            df = yf.download(ticker, start=start_date, end=today, progress=False)
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                if 'Close' in df.columns.get_level_values(0):
                    df = df.xs('Close', axis=1, level=0)
                else:
                    df = df.iloc[:, 0].to_frame()
            else:
                if 'Close' in df.columns:
                    df = df[['Close']]
                else:
                    df = df.iloc[:, 0].to_frame()
            df.columns = [name]
            return df
        except Exception:
            return pd.DataFrame()
    g_df = get_ticker_data('GC=F', 'Gold')
    s_df = get_ticker_data('SI=F', 'Silver')
    o_df = get_ticker_data('CL=F', 'Oil')
    usd_df = get_ticker_data('DX=F', 'USD_Index')
    vix_df = get_ticker_data('^VIX', 'VIX')
    tnx_df = get_ticker_data('^TNX', 'Bond_Yield')
    btc_df = get_ticker_data('BTC-USD', 'Bitcoin')
    if g_df.empty:
        return pd.DataFrame()
    df = pd.concat([g_df, s_df, o_df, usd_df, vix_df, tnx_df, btc_df], axis=1)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    if len(df) < 50:
        return pd.DataFrame()
    df = ensure_ohlc(df)
    df['SMA_15'] = df['Gold'].rolling(window=15).mean()
    df['EMA_20'] = df['Gold'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Gold'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Gold'].ewm(span=200, adjust=False).mean()
    weekly_df = df['Gold'].resample('W').last()
    weekly_ema_50 = weekly_df.ewm(span=50, adjust=False).mean()
    df['Weekly_Trend'] = weekly_ema_50.reindex(df.index, method='ffill')
    df['TR'] = (df['Gold'] - df['Gold'].shift(1)).abs()
    df['ATR'] = df['TR'].rolling(window=14).mean()
    df['SMA_20'] = df['Gold'].rolling(window=20).mean()
    df['std_dev'] = df['Gold'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['std_dev'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['std_dev'])
    df['Roll_High'] = df['Gold'].rolling(window=100).max()
    df['Roll_Low'] = df['Gold'].rolling(window=100).min()
    df['Fib_618'] = df['Roll_High'] - (0.618 * (df['Roll_High'] - df['Roll_Low']))
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss.replace(0, np.nan))
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Gold'].ewm(span=12, adjust=False).mean()
    ema26 = df['Gold'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm = plus_dm.where(plus_dm > 0, 0)
    minus_dm = (-minus_dm).where(minus_dm < 0, 0)
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - df['Gold'].shift(1)).abs(), (df['Low'] - df['Gold'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    df['ADX'] = dx.rolling(14).mean()
    df['Pivot'] = (df['High'] + df['Low'] + df['Gold']) / 3
    df['R1'] = (2 * df['Pivot']) - df['Low']
    df['R2'] = df['Pivot'] + (df['High'] - df['Low'])
    df = detect_patterns(df)
    df.dropna(inplace=True)
    return df

with st.spinner('🦅 Eagle Eye is Scanning Markets...'):
    market_data = load_data(years)

if market_data.empty or len(market_data) < 10:
    st.error("⚠️ Server Error: Not enough data. Please reload.")
    st.stop()

df = market_data.copy().dropna()
df['Target'] = df['Gold'].shift(-1)
df.dropna(inplace=True)

feature_cols = ['Gold', 'Silver', 'Oil', 'USD_Index', 'VIX', 'Bond_Yield', 'Bitcoin',
                'SMA_15', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Fib_618', 'ATR', 'EMA_50', 'EMA_200', 'ADX']

for col in feature_cols:
    if col not in df.columns:
        df[col] = 0.0

X = df[feature_cols].values
y = df['Target'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=0)
gb = GradientBoostingRegressor(n_estimators=100, random_state=42)

voting_model = VotingRegressor(estimators=[('rf', rf), ('xgb', xgb), ('gb', gb)])
voting_model.fit(x_train, y_train)
acc_vote = voting_model.score(x_test, y_test) if len(x_test) > 0 else 0.0

last_row = market_data.iloc[-1]
last_values = np.array([last_row[feature_cols].values])

pred_usd = float(voting_model.predict(last_values)[0])
market_trend_pct = (pred_usd - last_row['Gold']) / last_row['Gold'] if last_row['Gold'] != 0 else 0.0

avg_sentiment = 0.0
news_entries = []
try:
    rss_url = "https://news.google.com/rss/search?q=Gold+Price+Market+Trends&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    entries = feed.entries[:5]
    if entries:
        total = 0.0
        count = 0
        for news in entries:
            blob = TextBlob(news.title)
            polarity = blob.sentiment.polarity
            news_entries.append((news.title, getattr(news, 'link', ''), polarity))
            total += polarity
            count += 1
        if count > 0:
            avg_sentiment = total / count
except Exception:
    avg_sentiment = 0.0

news_impact_pct = 0.002 if avg_sentiment > 0.1 else -0.002 if avg_sentiment < -0.1 else 0.0
total_change_pct = market_trend_pct + news_impact_pct

price_today_usd = last_row['Gold']
price_today_inr = live_price_today if (live_price_today is not None) else to_inr(price_today_usd)
price_tomorrow_inr = price_today_inr * (1 + total_change_pct)
diff_inr = price_tomorrow_inr - price_today_inr

pct_change = (price_tomorrow_inr - price_today_inr) / price_today_inr if price_today_inr != 0 else 0.0

atr_inr = to_inr(last_row['ATR']) if 'ATR' in last_row.index else 0.0
stop_loss_inr = price_today_inr - (2 * atr_inr)
target_r1 = to_inr(last_row['R1']) if 'R1' in last_row.index else price_today_inr
target_r2 = to_inr(last_row['R2']) if 'R2' in last_row.index else price_today_inr

vix_score = 100 - last_row['VIX'] if (('VIX' in last_row.index) and last_row['VIX'] < 100) else 0
rsi_score = last_row['RSI'] if 'RSI' in last_row.index else 50
fear_greed_score = (vix_score + rsi_score) / 2
fear_greed_score = max(0, min(100, fear_greed_score))

pattern_msg = "None"
if last_row.get('is_Hammer', 0) == 1:
    pattern_msg = "Hammer 🔨 (Possible Buy)"
elif last_row.get('is_ShootingStar', 0) == 1:
    pattern_msg = "Shooting Star 🌠 (Possible Sell)"

current_price_usd = last_row['Gold']
bb_lower = last_row.get('BB_Lower', np.nan)
bb_upper = last_row.get('BB_Upper', np.nan)
fib_618 = last_row.get('Fib_618', np.nan)
adx_val = last_row.get('ADX', 0.0)
ema_50 = last_row.get('EMA_50', 0.0)
ema_200 = last_row.get('EMA_200', 0.0)
weekly_trend = last_row.get('Weekly_Trend', 0.0)

daily_bullish = ema_50 > ema_200
weekly_bullish = current_price_usd > weekly_trend

signal = "HOLD"
signal_color = "orange"
icon = "🟠"
advice = "Conflicting signals. Wait."

if daily_bullish and weekly_bullish and total_change_pct > 0:
    signal = "JACKPOT BUY"
    signal_color = "green"
    icon = "🚀"
    advice = "🦅 Eagle Eye: Weekly & Daily Trends Aligned! Strong Buy."
elif weekly_bullish and total_change_pct < 0:
    signal = "BUY THE DIP"
    signal_color = "blue"
    icon = "📉"
    advice = "Long Term Trend is UP. Use this dip to Enter."
elif daily_bullish and not weekly_bullish:
    signal = "CAUTION (TRAP)"
    signal_color = "yellow"
    icon = "⚠️"
    advice = "Daily trend is up, but Weekly is DOWN. Risky."
elif not daily_bullish and not weekly_bullish:
    signal = "STRONG SELL"
    signal_color = "red"
    icon = "💥"
    advice = "Market is Crashing on all timeframes."
elif current_price_usd < bb_lower:
    signal = "SNIPER BUY"
    signal_color = "cyan"
    icon = "🎯"
    advice = "Oversold bounce expected."

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🏠 INR Dashboard", "🦅 Eagle Eye & Charts", "🤖 AI Brains", "📉 Backtest", "📝 Paper Trading", "🛠️ Pro Tools", "🇮🇳 Top 10 ETFs"])

with tab1:
    st.subheader("🔮 Price Prediction (Per Gram)")
    c1, c2, c3 = st.columns(3)
    c1.info("📅 Today Price")
    c1.metric("Live Price", f"₹{price_today_inr:,.0f}")
    c2.success("🚀 Tomorrow (AI)")
    c2.metric("Predicted", f"₹{price_tomorrow_inr:,.0f}", f"₹{diff_inr:.0f}")
    c3.metric("AI Confidence", f"{acc_vote*100:.2f}%")
    st.markdown("---")
    left_col, right_col = st.columns([2, 1.2])
    with left_col:
        st.subheader("⚖️ Final Verdict")
        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 15px; border: 2px solid {signal_color}; text-align: center; margin-bottom: 20px;">
                <h1 style="color: {signal_color}; margin:0;">{icon} {signal}</h1>
                <h3 style="color: white; margin-top:10px;">{advice}</h3>
                <p style="color: gray; font-size: 12px;">Eagle Eye Analysis (Weekly + Daily)</p>
            </div>
        """, unsafe_allow_html=True)
        last_signal_file = "last_signal.txt"
        last_signal_sent = "NONE"
        if os.path.exists(last_signal_file):
            with open(last_signal_file, "r") as f:
                last_signal_sent = f.read().strip()
        if signal != last_signal_sent and signal != "HOLD":
            with st.spinner(f"🚀 Sending Alert..."):
                msg = f"🇮🇳 *Ajay Gold Alert*\n💎 Verdict: {signal}\n💰 Price: ₹{price_today_inr:.0f}\n🎯 Target: ₹{target_r1:.0f}"
                success, info = send_telegram_alert(MY_BOT_TOKEN, MY_CHANNEL_NAME, msg)
                try:
                    with open(last_signal_file, "w") as f:
                        f.write(signal)
                except Exception:
                    pass
                if success:
                    st.success("✅ Alert Sent!")
                else:
                    st.warning(f"Alert not sent ({info})")
    with right_col:
        st.subheader("🛡️ Trade Plan (₹)")
        t1, t2 = st.columns(2)
        t1.metric("🛑 Stop Loss", f"₹{stop_loss_inr:,.0f}")
        t2.metric("🎯 Target 1", f"₹{target_r1:,.0f}")
        st.write("#### 📊 Drivers")
        st.caption(f"Fear/Greed: {fear_greed_score:.0f}/100")
        st.caption(f"Trend Strength (ADX): {adx_val:.1f}")
        st.caption(f"Weekly Trend: {'BULLISH' if weekly_bullish else 'BEARISH'}")

with tab2:
    st.header("🦅 Eagle Eye: Multi-Timeframe Analysis")
    last_rsi = market_data['RSI'].iloc[-1]
    last_adx = market_data['ADX'].iloc[-1]
    btc_corr = market_data['Gold'].corr(market_data['Bitcoin'])
    col_chart, col_side = st.columns([2, 1])
    with col_chart:
        chart_data = market_data.tail(200).copy()
        chart_data['Gold_INR'] = chart_data['Gold'].apply(to_inr)
        chart_data['EMA50_INR'] = chart_data['EMA_50'].apply(to_inr)
        chart_data['Weekly_Trend_INR'] = chart_data['Weekly_Trend'].apply(to_inr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Gold_INR'], mode='lines', name='Price', line=dict(color='#FFD700', width=2)))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['EMA50_INR'], mode='lines', name='Daily Trend (50 EMA)', line=dict(color='cyan', width=1)))
        fig.add_trace(go.Scatter(x=chart_data.index, y=chart_data['Weekly_Trend_INR'], mode='lines', name='🦅 Weekly Trend (Strong)', line=dict(color='#FF4500', width=3)))
        fig.update_layout(title="🦅 Price vs Weekly Trend (Orange Line)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), height=550)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Tip:** If Price is ABOVE the Thick Orange Line, it's a Long Term BULL Market.")
    with col_side:
        st.subheader("📉 Momentum (RSI)")
        rsi_col, adx_col = st.columns(2)
        rsi_col.metric("RSI Score", f"{last_rsi:.1f}")
        adx_col.metric("Trend Strength", f"{last_adx:.1f}")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=chart_data.index, y=chart_data['RSI'], name='RSI', line=dict(color='#DA70D6', width=2)))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#00FF00")
        fig_rsi.update_layout(height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), yaxis=dict(range=[0, 100], showgrid=False), margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.markdown("---")
        st.subheader("₿ Gold vs Bitcoin")
        st.metric("Correlation Score", f"{btc_corr:.2f}")
        norm_df = market_data[['Gold', 'Bitcoin']].tail(100).copy()
        norm_df = norm_df / norm_df.iloc[0] * 100
        fig_btc = px.line(norm_df, title="Performance (100 days)")
        fig_btc.update_layout(height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"), margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig_btc, use_container_width=True)

with tab3:
    st.header("🤖 The AI Trinity Stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("XGBoost", "Fast & Furious")
    c2.metric("Random Forest", "Stable & Reliable")
    c3.metric("Gradient Boost", "Error Corrector")
    st.success(f"🏆 **Ensemble Accuracy:** {acc_vote*100:.2f}% (Approx.)")

with tab4:
    st.header("📉 Strategy Profit Backtester")
    if len(market_data) > 250:
        backtest_df = market_data.tail(250).copy()
        backtest_df['Signal'] = np.where((backtest_df['Gold'] > backtest_df['Weekly_Trend']) & (backtest_df['Gold'] > backtest_df['EMA_50']), 1, 0)
        backtest_df['Returns'] = backtest_df['Gold'].pct_change().fillna(0)
        backtest_df['Strategy_Returns'] = backtest_df['Returns'] * backtest_df['Signal'].shift(1).fillna(0)
        backtest_df['Cum_Market'] = (1 + backtest_df['Returns']).cumprod()
        backtest_df['Cum_Strategy'] = (1 + backtest_df['Strategy_Returns']).cumprod()
        total_market_return = (backtest_df['Cum_Market'].iloc[-1] - 1) * 100
        total_strategy_return = (backtest_df['Cum_Strategy'].iloc[-1] - 1) * 100
        col_b1, col_b2 = st.columns(2)
        col_b1.metric("💰 Buy & Hold Return", f"{total_market_return:.2f}%")
        col_b2.metric("🦅 Eagle Eye Return", f"{total_strategy_return:.2f}%")
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cum_Market'], mode='lines', name='Buy & Hold'))
        fig_bt.add_trace(go.Scatter(x=backtest_df.index, y=backtest_df['Cum_Strategy'], mode='lines', name='Eagle Eye Strategy', line=dict(color='green', width=2)))
        st.plotly_chart(fig_bt)
    else:
        st.warning("Need more data for 1-year backtest.")

with tab5:
    st.header(f"📝 Pro Paper Trading ({user_name}'s Account)")
    
    curr_cash, curr_gold, avg_buy_price, _ = get_portfolio_status(user_name)
    history_df = get_trade_history(user_name) 


    invested_value = curr_gold * avg_buy_price
    current_value = curr_gold * price_today_inr
    pnl = current_value - invested_value
    total_net_worth = curr_cash + current_value

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Cash Available", f"₹{curr_cash:,.0f}")
    col2.metric("🏆 Gold Held", f"{curr_gold:.2f} g")
    col3.metric("📊 Invested Value", f"₹{invested_value:,.0f}")
    
    if pnl >= 0:
        col4.metric("📈 Profit/Loss", f"₹{pnl:,.0f}", "Profit", delta_color="normal")
    else:
        col4.metric("📉 Profit/Loss", f"₹{pnl:,.0f}", "Loss", delta_color="inverse")

    st.markdown(f"**💰 Total Net Worth:** ₹{total_net_worth:,.0f}")
    st.markdown("---")

    lc, rc = st.columns(2)
    
    with lc:
        with st.form("buy_form_db"):
            st.subheader("🟢 Buy Gold")
            buy_grams = st.number_input("Grams to Buy", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, key="b_db")
            if st.form_submit_button("Buy Now"):
              
                success, msg = execute_trade(user_name, "BUY", price_today_inr, buy_grams, signal)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    
    with rc:
        with st.form("sell_form_db"):
            st.subheader("🔴 Sell Gold")
            sell_grams = st.number_input("Grams to Sell", min_value=0.0, max_value=1000.0, value=1.0, step=0.1, key="s_db")
            if st.form_submit_button("Sell Now"):
                success, msg = execute_trade(user_name, "SELL", price_today_inr, sell_grams, signal)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    st.markdown("---")

    if st.button(f"⚠️ Reset Portfolio for {user_name}"):
        if reset_account(user_name):
            st.success(f"Portfolio for {user_name} reset successfully!")
            st.rerun()
        else:
            st.error("Failed to reset portfolio.")

    st.subheader(f"📜 {user_name}'s Transaction History")
    if not history_df.empty:
        st.dataframe(history_df, hide_index=True, use_container_width=True)
    else:
        st.info("No trades yet. Start trading using the forms above!")

with tab6:
    st.header("🛠️ Advanced Pro Tools")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.subheader("💰 Position Size Calculator")
        st.caption("How much should I buy based on Risk?")
        capital = st.number_input("My Capital (₹)", value=100000, step=1000)
        risk_pct = st.slider("Risk per Trade (%)", 1, 5, 2)
        entry = st.number_input("Entry Price (₹)", value=float(price_today_inr))
        sl = st.number_input("Stop Loss (₹)", value=float(stop_loss_inr))
        if entry > sl:
            risk_amt = capital * (risk_pct / 100)
            risk_per_gram = entry - sl
            qty = risk_amt / risk_per_gram if risk_per_gram > 0 else 0
            st.success(f"💎 Recommended Buy: **{qty:.2f} Grams**")
            st.info(f"If Stop Loss hits, you lose only: ₹{risk_amt:.0f} ({risk_pct}%)")
        else:
            st.error("Stop Loss must be lower than Entry Price!")
    with col_p2:
        st.subheader("🎲 Monte Carlo Forecast (30 Days)")
        st.caption("Simulating 1000 possible future paths...")
        if st.button("Run Simulation 🎲"):
            returns = market_data['Gold'].pct_change().dropna()
            mu = returns.mean()
            sigma = returns.std()
            simulations = 1000
            days = 30
            start_price = price_today_inr
            sim_data = np.zeros((days, simulations))
            for i in range(simulations):
                prices = [start_price]
                for d in range(days):
                    shock = np.random.normal(mu, sigma)
                    price = prices[-1] * (1 + shock)
                    prices.append(price)
                sim_data[:, i] = prices[1:]
            fig_mc = go.Figure()
            for i in range(min(50, simulations)):
                fig_mc.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(color='rgba(0,255,0,0.1)'), showlegend=False))
            mean_path = np.mean(sim_data, axis=1)
            fig_mc.add_trace(go.Scatter(y=mean_path, mode='lines', name='Average Path', line=dict(color='white', width=3)))
            fig_mc.update_layout(title="Future Price Probability Cone", xaxis_title="Days Ahead", yaxis_title="Price (₹)")
            st.plotly_chart(fig_mc, use_container_width=True)
            best_case = np.percentile(sim_data[-1], 95)
            worst_case = np.percentile(sim_data[-1], 5)
            c_mc1, c_mc2 = st.columns(2)
            c_mc1.metric("🚀 Best Case (95%)", f"₹{best_case:,.0f}")
            c_mc2.metric("📉 Worst Case (5%)", f"₹{worst_case:,.0f}")

with tab7:
    st.header("🇮🇳 India's Top 10 Gold ETFs (Live AI Forecast)")
    st.markdown("We apply our global AI Trend to Indian ETF prices.")
    etf_tickers = {
        "Nippon India ETF Gold BeES": "GOLDBEES.NS",
        "HDFC Gold ETF": "HDFCMFGETF.NS",
        "SBI Gold ETF": "SETFGOLD.NS",
        "Kotak Gold ETF": "KOTAKGOLD.NS",
        "Axis Gold ETF": "AXISGOLD.NS",
        "ICICI Prudential Gold ETF": "ICICIGOLD.NS",
        "Aditya Birla Sun Life Gold ETF": "BSLGOLDETF.NS",
        "UTI Gold ETF": "GOLDSHARE.NS",
        "Invesco India Gold ETF": "IVZINGOLD.NS",
        "Quantum Gold Fund": "QGOLDHALF.NS"
    }
    col_r, col_l = st.columns([1, 3])
    with col_r:
        if st.button("🔄 Refresh ETF Prices"):
            st.rerun()
    with st.spinner("Fetching Live ETF Prices from NSE..."):
        etf_data = []
        for name, ticker in etf_tickers.items():
            try:
                etf = yf.download(ticker, period="5d", progress=False)
                if not etf.empty:
                    if isinstance(etf.columns, pd.MultiIndex) and 'Close' in etf.columns.get_level_values(0):
                        current_price = float(etf.xs('Close', axis=1, level=0).iloc[-1].item())
                    elif 'Close' in etf.columns:
                        current_price = float(etf['Close'].iloc[-1])
                    else:
                        current_price = float(etf.iloc[-1, 0])
                    predicted_price = current_price * (1 + pct_change)
                    gain_loss = predicted_price - current_price
                    etf_data.append({
                        "ETF Name": name,
                        "Live Price (₹)": f"₹{current_price:.2f}",
                        "AI Predicted (₹)": f"₹{predicted_price:.2f}",
                        "Expected Change": f"{gain_loss:+.2f} ({pct_change*100:.2f}%)"
                    })
            except Exception:
                pass
        if etf_data:
            df_etf = pd.DataFrame(etf_data)
            st.dataframe(df_etf, hide_index=True, use_container_width=True)
            st.markdown("---")
            if pct_change > 0:
                st.success(f"🚀 **AI Prediction:** ETFs are likely to **RISE** by approx **{pct_change*100:.2f}%** tomorrow.")
            else:
                st.error(f"📉 **AI Prediction:** ETFs are likely to **FALL** by approx **{pct_change*100:.2f}%** tomorrow.")
            st.caption("Note: Prediction depends on global gold price movement.")
        else:
            st.warning("⚠️ Could not fetch ETF data. Market might be closed or API issue.")
