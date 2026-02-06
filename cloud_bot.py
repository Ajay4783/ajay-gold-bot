import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import feedparser
import numpy as np
import requests
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Gold AI Pro Max", page_icon="🏆", layout="wide")

# ==========================================
# 🔐 ADMIN SETTINGS (SECURE MODE)
# ==========================================
# Ippo Token Code-la irukkaadhu. Streamlit Cloud "Secrets"-la irundhu edukkum.
try:
    MY_BOT_TOKEN = st.secrets["MY_BOT_TOKEN"]
    MY_CHANNEL_NAME = st.secrets["MY_CHANNEL_NAME"]
except:
    # Local Laptop-la run panna mattum idhu work aagum
    # (Safety-kaga inga dummy or empty-a viduvathu nallathu)
    MY_BOT_TOKEN = "TOKEN_REMOVED_FOR_SAFETY" 
    MY_CHANNEL_NAME = "@ajay_gold_2026"
    st.warning("⚠️ Running in Local Mode. Token not found in Secrets.")
# ==========================================

# --- HEADER ---
st.title("🏆 Gold Rate AI Predictor (Pro Version)")
st.markdown("### 🤖 Advanced Analysis: RSI, MACD, Heatmaps & XGBoost")

# --- TELEGRAM FUNCTION ---
def send_telegram_alert(token, channel_id, message):
    # Token illana function work aagaadhu (Crash aagaama thadukkum)
    if not token or token == "TOKEN_REMOVED_FOR_SAFETY":
        return False, "❌ Token Missing in Secrets!"
        
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        params = {"chat_id": channel_id, "text": message, "parse_mode": "Markdown"}
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return True, "✅ Success!"
        else:
            return False, f"❌ Telegram Error: {response.text}"
    except Exception as e:
        return False, f"⚠️ Network Error: {e}"

# --- MEMORY FUNCTIONS ---
def get_last_signal():
    if os.path.exists("last_signal.txt"):
        with open("last_signal.txt", "r") as f:
            return f.read().strip()
    return "NONE"

def save_new_signal(sig):
    with open("last_signal.txt", "w") as f:
        f.write(sig)

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ AI Settings")
years = st.sidebar.slider("📅 AI Learning Period (Years)", 1, 5, 2)
st.sidebar.markdown("---")
st.sidebar.header("🏛️ Tax & Duty Settings")
tax_percentage = st.sidebar.slider("Import Duty + GST (%)", 0.0, 50.0, 9.2, 0.1)

if MY_CHANNEL_NAME:
    clean_link = MY_CHANNEL_NAME.replace("@", "")
    st.sidebar.success(f"✅ Connected to: **{MY_CHANNEL_NAME}**")
    st.sidebar.markdown(f"🔗 [Join Channel Here](https://t.me/{clean_link})")

st.sidebar.caption("Developed by **Ajay S** | Powered by Python & ML")

# --- STEP 1: SCRAPE LIVE GOLD RATE ---
@st.cache_data(ttl=3600)
def get_live_gold_rate():
    try:
        url = "https://www.bankbazaar.com/gold-rate-tamil-nadu.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        requests.get(url, headers=headers, timeout=5)
        return None 
    except Exception:
        return None

live_price_today = get_live_gold_rate()

# --- STEP 2: LOAD DATA (FIXED & ROBUST) ---
@st.cache_data
def load_data(years):
    today = date.today().strftime("%Y-%m-%d")
    start_date = (date.today() - timedelta(days=years*365)).strftime("%Y-%m-%d")
    
    def get_ticker_data(ticker, name):
        try:
            df = yf.download(ticker, start=start_date, end=today, progress=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs('Close', axis=1, level=0)
                except: df = df.iloc[:, 0].to_frame()
            else:
                df = df[['Close']] if 'Close' in df.columns else df.iloc[:, 0].to_frame()
            df.columns = [name]
            return df
        except:
            return pd.DataFrame()

    g_df = get_ticker_data('GC=F', 'Gold')
    s_df = get_ticker_data('SI=F', 'Silver')
    o_df = get_ticker_data('CL=F', 'Oil')

    if g_df.empty: return pd.DataFrame()
    
    df = pd.concat([g_df, s_df, o_df], axis=1)
    df.ffill(inplace=True) 
    df.bfill(inplace=True)
    df.dropna(inplace=True)

    if len(df) < 50: return pd.DataFrame()

    df['SMA_15'] = df['Gold'].rolling(window=15).mean()
    
    delta = df['Gold'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Gold'].ewm(span=12, adjust=False).mean()
    ema26 = df['Gold'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    df.dropna(inplace=True) 
    return df

with st.spinner('🤖 AI is collecting data & checking XGBoost...'):
    market_data = load_data(years)

if market_data.empty or len(market_data) < 10:
    st.error("⚠️ **Server Error:** Not enough data. Please reload.")
    st.stop()

# --- STEP 3: ML TRAINING (RF vs XGBOOST) ---
df = market_data.copy()
df['Target'] = df['Gold'].shift(-1)
df.dropna(inplace=True)

X = df[['Gold', 'Silver', 'Oil', 'SMA_15', 'RSI', 'MACD', 'Signal_Line']].values
y = df['Target'].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(x_train, y_train)
acc_rf = model_rf.score(x_test, y_test)

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
model_xgb.fit(x_train, y_train)
acc_xgb = model_xgb.score(x_test, y_test)

best_model = model_rf if acc_rf > acc_xgb else model_xgb
best_model_name = "RandomForest" if acc_rf > acc_xgb else "XGBoost"

last_row = market_data.iloc[-1]
last_values = np.array([[
    last_row['Gold'], last_row['Silver'], last_row['Oil'], 
    last_row['SMA_15'], last_row['RSI'], last_row['MACD'], last_row['Signal_Line']
]])
pred_usd = best_model.predict(last_values)[0]
market_trend_pct = (pred_usd - last_row['Gold']) / last_row['Gold']

# --- STEP 4: NEWS SENTIMENT ---
avg_sentiment = 0
news_entries = []
try:
    rss_url = "https://news.google.com/rss/search?q=Gold+Price+Market+Trends&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    sentiments = []
    if len(feed.entries) > 0:
        for news in feed.entries[:5]:
            blob = TextBlob(news.title)
            sentiments.append(blob.sentiment.polarity)
            news_entries.append((news.title, news.link, blob.sentiment.polarity))
        if sentiments: avg_sentiment = np.mean(sentiments)
except: pass

news_impact_pct = 0.002 if avg_sentiment > 0.1 else -0.002 if avg_sentiment < -0.1 else 0
total_change_pct = market_trend_pct + news_impact_pct

# --- STEP 5: CALCULATION ---
try: usdinr = yf.download('INR=X', period='1d', progress=False)['Close'].iloc[-1].item()
except: usdinr = 86.50

tax_factor = 1 + (tax_percentage / 100)
calculated_price = ((last_row['Gold'] * usdinr) / 31.1035) * tax_factor
price_today = live_price_today if live_price_today else calculated_price
price_tomorrow = price_today * (1 + total_change_pct)
diff = price_tomorrow - price_today

# ==========================================
# 🆕 SIDEBAR FEATURES
# ==========================================
st.sidebar.markdown("---")
st.sidebar.header("💍 Pavan (Sovereign) Rate")
one_gram = price_today
one_pavan = one_gram * 8
mc_pct = st.sidebar.slider("Making Charges (MC) %", 0, 25, 12)
pavan_with_mc = one_pavan * (1 + mc_pct/100)
col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("1 Gram", f"₹{one_gram:.0f}")
col_s2.metric("1 Pavan", f"₹{one_pavan:.0f}")
st.sidebar.success(f"🏷️ With {mc_pct}% MC: **₹{pavan_with_mc:,.0f}**")

st.sidebar.markdown("---")
st.sidebar.header("💰 My Portfolio")
st.sidebar.caption("Profit/Loss Calculator")
buy_price = st.sidebar.number_input("Bought Price (Per Gram)", min_value=0.0, value=6500.0, step=100.0)
qty = st.sidebar.number_input("Quantity (Grams)", min_value=0.0, value=8.0, step=1.0)
current_value = price_today * qty
invested_value = buy_price * qty
pnl = current_value - invested_value

if pnl > 0:
    st.sidebar.success(f"🎉 Profit: ₹{pnl:,.2f}")
elif pnl < 0:
    st.sidebar.error(f"📉 Loss: ₹{abs(pnl):,.2f}")
else:
    st.sidebar.info("No Profit / No Loss")

# ==========================================
# --- DECISION LOGIC ---
rsi_val = last_row['RSI']
signal, signal_color, icon = "HOLD", "orange", "🟠"
advice = "Market Stable. Hold."

if total_change_pct > 0.001:
    if rsi_val < 70: 
        signal, signal_color, icon = "BUY", "green", "🟢"
        advice = "Strong Upside & RSI Healthy! Accumulate."
    else:
        advice = "Price rising, but RSI High (Risky)."
elif total_change_pct < -0.001:
    if rsi_val > 30: 
        signal, signal_color, icon = "SELL", "red", "🔴"
        advice = "Prices dropping. Wait."
    else:
        advice = "Price dropping, but RSI Low (Reversal possible)."

# ================= TABS LOGIC =================
tab1, tab2, tab3 = st.tabs(["🏠 Home Dashboard", "📊 Analysis & Insights", "🤖 Model Battle"])

with tab1:
    left_col, right_col = st.columns([2, 1.2])
    with left_col:
        st.subheader("📊 Price Dashboard")
        c1, c2, c3 = st.columns(3)
        c1.metric("Active Model", best_model_name)
        c2.metric("Market Mood", "Bullish 🐂" if avg_sentiment > 0 else "Bearish 🐻")
        c3.metric("Trend", "UP 📈" if diff > 0 else "DOWN 📉", f"₹{diff:.2f}")

        st.markdown(f"""
            <div style="background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 2px solid {signal_color}; text-align: center; margin-bottom: 20px;">
                <h2 style="color: {signal_color}; margin:0;">{icon} SIGNAL: {signal}</h2>
                <p style="color: white; margin:0;">{advice}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Telegram Logic (With Safety Check)
        last_signal_sent = get_last_signal()
        if signal != last_signal_sent:
            with st.spinner(f"🚀 Signal Changed. Sending Auto-Alert..."):
                msg = f"""
🏆 *Ajay Gold Tips Update* 🏆

📡 *Signal:* {icon} *{signal}*
💰 *Price:* ₹{price_tomorrow:.2f}
🤖 *Model:* {best_model_name}
📉 *Advice:* {advice}
"""
                success, status_msg = send_telegram_alert(MY_BOT_TOKEN, MY_CHANNEL_NAME, msg)
                if success:
                    save_new_signal(signal)
                    st.success("✅ Auto-Alert Sent!")
                elif "Token Missing" in status_msg:
                    st.warning("⚠️ Alert NOT sent. Deploy to Cloud to enable Telegram.")
        
        p1, p2 = st.columns(2)
        p1.info(f"**📅 Today (Est):** ₹{price_today:.2f}")
        p2.success(f"**🔮 Tomorrow:** ₹{price_tomorrow:.2f}")

        fig = go.Figure()
        chart_data = market_data.tail(60)
        dates = chart_data.index
        scaling_factor = price_today / chart_data['Gold'].iloc[-1]
        fig.add_trace(go.Scatter(x=dates, y=chart_data['Gold'] * scaling_factor, mode='lines', name='Gold Price', line=dict(color='#FFD700', width=3)))
        tomorrow_date = date.today() + timedelta(days=1)
        fig.add_trace(go.Scatter(x=[tomorrow_date], y=[price_tomorrow], mode='markers', name='Prediction', marker=dict(color=signal_color, size=14, symbol='star')))
        fig.update_layout(title="📈 Price Trend", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.subheader("🧠 Technical Indicators")
        tech_df = pd.DataFrame({
            "Indicator": ["RSI (14)", "MACD", "Signal Line"],
            "Value": [f"{last_row['RSI']:.2f}", f"{last_row['MACD']:.2f}", f"{last_row['Signal_Line']:.2f}"],
            "Status": [
                "Overbought 🔴" if last_row['RSI'] > 70 else "Oversold 🟢" if last_row['RSI'] < 30 else "Neutral ⚪",
                "Bullish 🟢" if last_row['MACD'] > last_row['Signal_Line'] else "Bearish 🔴", "-"
            ]
        })
        st.dataframe(tech_df, hide_index=True)
        
        st.markdown("---")
        st.write("##### 📰 Live News")
        if len(news_entries) == 0:
             st.write("No recent news.")
        else:
            for title, link, score in news_entries:
                if score > 0.05:
                    icon = "🟢" 
                    sentiment_label = "Positive"
                    color = "green"
                elif score < -0.05:
                    icon = "🔴" 
                    sentiment_label = "Negative"
                    color = "red"
                else:
                    icon = "⚪"
                    sentiment_label = "Neutral"
                    color = "gray"
                st.markdown(f"{icon} [**Read**]({link}) : {sentiment_label} - :{color}[{title[:60]}...]")

with tab2:
    st.header("📊 Market Correlation Heatmap")
    corr_data = market_data[['Gold', 'Silver', 'Oil', 'SMA_15', 'RSI', 'MACD']].tail(100).corr()
    fig_corr = px.imshow(corr_data, text_auto=True, color_continuous_scale='RdBu_r', title="100-Day Market Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.info("💡 **Tip:** Gold & Silver usually move together (Red). Gold & Oil might be opposite (Blue).")

with tab3:
    st.header("🤖 Model Comparison: RF vs XGBoost")
    m1, m2 = st.columns(2)
    m1.metric("Random Forest Accuracy", f"{acc_rf*100:.2f}%")
    m2.metric("XGBoost Accuracy", f"{acc_xgb*100:.2f}%")
    
    if acc_xgb > acc_rf:
        st.success(f"🏆 **XGBoost** wins today! It is being used for prediction.")
    else:
        st.success(f"🏆 **Random Forest** wins today! It is being used for prediction.")

    pred_rf = model_rf.predict(last_values)[0]
    pred_xgb = model_xgb.predict(last_values)[0]
    
    chart_df = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost"],
        "Predicted Gold Price ($)": [pred_rf, pred_xgb]
    })
    fig_battle = px.bar(chart_df, x="Model", y="Predicted Gold Price ($)", color="Model", title="Model Price Prediction Comparison")
    st.plotly_chart(fig_battle, use_container_width=True)

