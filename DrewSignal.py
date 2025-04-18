import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pycoingecko import CoinGeckoAPI
import requests
from datetime import datetime, timedelta
import time
import pytz
from backtesting import Backtest, Strategy
import json
import os

# Configuration
STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'BAC', 'WFC', 'JNJ', 'PFE', 'MRNA', 'PG', 'KO', 'PEP', 'WMT', 'TGT', 'XOM', 'CVX', 'BA', 'LMT']
CRYPTOS = ['solana', 'matic-network', 'cosmos', 'avalanche-2', 'fantom', 'algorand', 'near', 'hedera', 'arbitrum', 'optimism']
INTERVAL = '5m'
LOOKBACK_DAYS = 30
TIMEZONE = pytz.timezone('US/Eastern')
ALPHA_VANTAGE_API_KEY = '3KK994PVSFRX915A'
NEWSAPI_KEY = '7b813b18aa254bf6aaaeff96f8f64ef2'
CACHE_FILE = 'market_cache.json'
CACHE_TIMEOUT = 300
FUNDAMENTALS_TIMEOUT = 86400
NEWS_CACHE_TIMEOUT = 3600

POSITIVE_WORDS = {'bullish', 'gain', 'rise', 'growth', 'surge', 'positive', 'strong', 'up', 'success', 'profit'}
NEGATIVE_WORDS = {'bearish', 'loss', 'drop', 'decline', 'crash', 'negative', 'weak', 'down', 'fail', 'risk'}

# Cache management
def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            cache['starred_assets'] = set(cache.get('starred_assets', []))
            return cache
    return {'starred_assets': set()}

def save_cache(cache):
    cache_copy = cache.copy()
    cache_copy['starred_assets'] = list(cache_copy['starred_assets'])
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache_copy, f)

def is_stock_market_open():
    now = datetime.now(TIMEZONE)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

def calculate_price_change(symbol, is_crypto=False, cache=None):
    cache_key = f"{symbol}_price_change"
    now = time.time()
    if cache and cache_key in cache and (now - cache[cache_key]['timestamp'] < CACHE_TIMEOUT):
        return cache[cache_key]['value']
    try:
        if is_crypto:
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_by_id(id=symbol, vs_currency='usd', days=1)
            prices = data['prices']
            if len(prices) < 2:
                return 0.0
            latest_price = prices[-1][1]
            prev_price = prices[0][1]
        else:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2d', interval='1h')
            if len(data) < 2:
                return 0.0
            latest_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[0]
        price_change = ((latest_price - prev_price) / prev_price) * 100
        if cache is not None:
            cache[cache_key] = {'value': price_change, 'timestamp': now}
        return price_change
    except:
        return 0.0

def fetch_fundamentals(symbol, cache=None):
    cache_key = f"{symbol}_fundamentals"
    now = time.time()
    if cache and cache_key in cache and (now - cache[cache_key]['timestamp'] < FUNDAMENTALS_TIMEOUT):
        return cache[cache_key]['pe_ratio'], cache[cache_key]['eps']
    try:
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}'
        response = requests.get(url)
        data = response.json()
        pe_ratio = float(data.get('PERatio', 0) or 0)
        eps = float(data.get('EPS', 0) or 0)
        if cache is not None:
            cache[cache_key] = {'pe_ratio': pe_ratio, 'eps': eps, 'timestamp': now}
        return pe_ratio, eps
    except:
        return 0.0, 0.0

def fetch_news_sentiment(symbol, is_crypto=False, cache=None):
    cache_key = f"{symbol}_news_sentiment"
    now = time.time()
    if cache and cache_key in cache and (now - cache[cache_key]['timestamp'] < NEWS_CACHE_TIMEOUT):
        return cache[cache_key]['value']
    try:
        query = symbol if not is_crypto else symbol.replace('-', ' ')
        url = f'https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWSAPI_KEY}'
        response = requests.get(url)
        articles = response.json().get('articles', [])[:5]
        if not articles:
            return 0.0
        sentiment_score = 0.0
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            positive_count = sum(1 for word in POSITIVE_WORDS if word in text)
            negative_count = sum(1 for word in NEGATIVE_WORDS if word in text)
            if positive_count + negative_count > 0:
                sentiment_score += (positive_count - negative_count) / (positive_count + negative_count)
        sentiment_score = sentiment_score / max(len(articles), 1)
        if cache is not None:
            cache[cache_key] = {'value': sentiment_score, 'timestamp': now}
        return sentiment_score
    except:
        return 0.0

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    exp1 = data['Close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_volatility(data, periods=14):
    returns = data['Close'].pct_change()
    return returns.rolling(window=periods).std() * np.sqrt(periods)

def fetch_data(symbol, is_crypto=False):
    for attempt in range(3):
        try:
            if is_crypto:
                cg = CoinGeckoAPI()
                data = cg.get_coin_ohlc_by_id(id=symbol, vs_currency='usd', days=LOOKBACK_DAYS)
                df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close'])
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
                df.set_index('Timestamp', inplace=True)
                df['Volume'] = 0
            else:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=LOOKBACK_DAYS)
                df = yf.download(symbol, start=start_date, end=end_date, interval=INTERVAL, auto_adjust=False)
                if df.empty:
                    time.sleep(2)
                    continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            df['RSI'] = calculate_rsi(df)
            df['MACD'], df['Signal'] = calculate_macd(df)
            df['Volatility'] = calculate_volatility(df)
            df['Returns'] = df['Close'].pct_change().shift(-1)
            df['Target'] = (df['Returns'] > 0).astype(int)
            return df.dropna()
        except Exception as e:
            time.sleep(2)
    return None

def train_model(data):
    try:
        features = ['RSI', 'MACD', 'Signal', 'Volatility', 'PE_Ratio', 'EPS', 'News_Sentiment']
        X = data[features]
        y = data['Target']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        return model, scaler, features
    except:
        return None, None, None

def predict_signal(model, scaler, data, features, symbol, is_crypto):
    try:
        latest = data[features].iloc[-1:]
        latest_scaled = scaler.transform(latest)
        buy_prob = model.predict_proba(latest_scaled)[0][1]
        if buy_prob > 0.5:
            signal = "Buy"
            action = "Buy now"
            reason = "Price is likely to rise based on strong market signals."
        else:
            signal = "Sell/Hold"
            action = "Consider selling or holding"
            reason = "Price may fall or stay stable based on current market signals."
            if buy_prob < 0.3:
                reason += " Selling is more likely."
            else:
                reason += " Holding is more likely."
        return signal, action, reason, buy_prob
    except:
        return "Error", "Error", "Something went wrong.", 0.0

class MLStrategy(Strategy):
    def init(self):
        self.model, self.scaler, self.features = train_model(self.data.df)
    def next(self):
        if self.model is None or self.scaler is None:
            return
        latest_data = self.data.df.iloc[-1:]
        signal, _, _, _ = predict_signal(self.model, self.scaler, latest_data, self.features, self.data._name, False)
        if signal == "Buy" and not self.position:
            self.buy()
        elif signal == "Sell/Hold" and self.position:
            self.position.close()

# Streamlit App
st.title("DrewSignal: Market Analysis")
st.write("Monitor stocks and cryptos with AI-driven signals. Use on mobile for real-time insights.")

# Session state for cache and refresh
if 'cache' not in st.session_state:
    st.session_state.cache = load_cache()
if 'force_update' not in st.session_state:
    st.session_state.force_update = False

# Force update button and refresh interval
col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Force Update"):
        st.session_state.force_update = True
        st.session_state.cache = {'starred_assets': st.session_state.cache.get('starred_assets', set())}
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
        st.write("Force update initiated...")
with col2:
    refresh_interval = st.number_input("Refresh Interval (seconds)", min_value=30, max_value=300, value=30, step=10)

# Market status
stock_market_open = is_stock_market_open()
last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Process assets
stock_data = []
crypto_data = []
latest_data = {}

for symbol in STOCKS:
    star = True if symbol in st.session_state.cache.get('starred_assets', set()) else False
    data = fetch_data(symbol, is_crypto=False)
    if data is None:
        stock_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': 'N/A', 'Signal': 'Data Unavailable', 'Reason': 'Failed to fetch data. Try force update.', '24h Change (%)': 0.0, 'Backtest Return (%)': 0.0, 'Details': symbol})
        continue
    data['PE_Ratio'], data['EPS'] = fetch_fundamentals(symbol, st.session_state.cache)
    data['News_Sentiment'] = fetch_news_sentiment(symbol, is_crypto=False, cache=st.session_state.cache)
    model, scaler, features = train_model(data)
    if model is None:
        stock_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': 'N/A', 'Signal': 'Model Error', 'Reason': 'Failed to train model.', '24h Change (%)': 0.0, 'Backtest Return (%)': 0.0, 'Details': symbol})
        continue
    signal, action, reason, buy_prob = predict_signal(model, scaler, data, features, symbol, is_crypto=False)
    price = data['Close'].iloc[-1]
    price_change = calculate_price_change(symbol, is_crypto=False, cache=st.session_state.cache)
    try:
        bt = Backtest(data, MLStrategy, cash=1000000, commission=0.001)
        stats = bt.run()
        backtest_return = stats['Return [%]']
    except:
        backtest_return = 0.0
    data['Buy_Prob'] = buy_prob
    latest_data[symbol] = data
    stock_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': f"{price:.2f}", 'Signal': signal, 'Reason': reason, '24h Change (%)': price_change, 'Backtest Return (%)': backtest_return, 'Details': symbol})

for symbol in CRYPTOS:
    star = True if symbol in st.session_state.cache.get('starred_assets', set()) else False
    data = fetch_data(symbol, is_crypto=True)
    if data is None:
        crypto_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': 'N/A', 'Signal': 'Data Unavailable', 'Reason': 'Failed to fetch data. Try force update.', '24h Change (%)': 0.0, 'Backtest Return (%)': 0.0, 'Details': symbol})
        continue
    data['PE_Ratio'], data['EPS'] = 0, 0
    data['News_Sentiment'] = fetch_news_sentiment(symbol, is_crypto=True, cache=st.session_state.cache)
    model, scaler, features = train_model(data)
    if model is None:
        crypto_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': 'N/A', 'Signal': 'Model Error', 'Reason': 'Failed to train model.', '24h Change (%)': 0.0, 'Backtest Return (%)': 0.0, 'Details': symbol})
        continue
    signal, action, reason, buy_prob = predict_signal(model, scaler, data, features, symbol, is_crypto=True)
    price = data['Close'].iloc[-1]
    price_change = calculate_price_change(symbol, is_crypto=True, cache=st.session_state.cache)
    try:
        bt = Backtest(data, MLStrategy, cash=1000000, commission=0.001)
        stats = bt.run()
        backtest_return = stats['Return [%]']
    except:
        backtest_return = 0.0
    data['Buy_Prob'] = buy_prob
    latest_data[symbol] = data
    crypto_data.append({'Star': star, 'Symbol': symbol, 'Price ($)': f"{price:.2f}", 'Signal': signal, 'Reason': reason, '24h Change (%)': price_change, 'Backtest Return (%)': backtest_return, 'Details': symbol})

save_cache(st.session_state.cache)
st.session_state.force_update = False

# Display stocks
st.subheader("Stocks")
stock_df = pd.DataFrame(stock_data)
stock_df = stock_df.sort_values(by=["Star", "Symbol"], ascending=[False, True])
edited_stock_df = st.data_editor(
    stock_df,
    column_config={
        "Star": st.column_config.CheckboxColumn("Star", default=False),
        "Details": st.column_config.TextColumn("Details", disabled=True)
    },
    hide_index=True,
    use_container_width=True
)

# Update starred assets for stocks
for index, row in edited_stock_df.iterrows():
    symbol = row['Symbol']
    if row['Star'] and symbol not in st.session_state.cache['starred_assets']:
        st.session_state.cache['starred_assets'].add(symbol)
    elif not row['Star'] and symbol in st.session_state.cache['starred_assets']:
        st.session_state.cache['starred_assets'].discard(symbol)

# Expanders for stock details
for index, row in stock_df.iterrows():
    symbol = row['Details']
    with st.expander(f"Details for {symbol}"):
        data = latest_data.get(symbol)
        price_change = row['24h Change (%)']
        signal = row['Signal']
        pe_ratio = data['PE_Ratio'].iloc[-1] if data is not None and 'PE_Ratio' in data else 0.0
        eps = data['EPS'].iloc[-1] if data is not None and 'EPS' in data else 0.0
        news_sentiment = data['News_Sentiment'].iloc[-1] if data is not None and 'News_Sentiment' in data else 0.0
        buy_prob = data.get('Buy_Prob', 0.0) if data is not None else 0.0
        market_status = "open" if stock_market_open else "closed"

        if signal == "Buy":
            action_text = f"Buy now"
            details = f"Price is likely to rise based on strong market signals."
        elif signal == "Sell/Hold":
            action_text = f"Consider selling or holding"
            details = f"Price may fall or stay stable based on current market signals."
            if buy_prob < 0.3:
                details += f" Selling is more likely."
            else:
                details += f" Holding is more likely."
        else:
            action_text = f"Error"
            details = f"Unable to provide a recommendation."
        
        if price_change > 0:
            details += f" Price is up recently."
        else:
            details += f" Price is down recently."
        if news_sentiment > 0:
            details += f" News is positive."
        else:
            details += f" News is neutral or negative."
        if pe_ratio > 0 and eps > 0:
            details += f" Company metrics are solid."
        
        details += f" Markets are {market_status}. This is an estimate—be cautious!"
        
        st.markdown(f"**{action_text}**")
        st.write(details)

st.markdown(f"**Stock Market {'Open' if stock_market_open else 'Closed'}**")

# Display cryptocurrencies
st.subheader("Cryptocurrencies")
crypto_df = pd.DataFrame(crypto_data)
crypto_df = crypto_df.sort_values(by=["Star", "Symbol"], ascending=[False, True])
edited_crypto_df = st.data_editor(
    crypto_df,
    column_config={
        "Star": st Manny.column_config.CheckboxColumn("Star", default=False),
        "Details": st.column_config.TextColumn("Details", disabled=True)
    },
    hide_index=True,
    use_container_width=True
)

# Update starred assets for cryptos
for index, row in edited_crypto_df.iterrows():
    symbol = row['Symbol']
    if row['Star'] and symbol not in st.session_state.cache['starred_assets']:
        st.session_state.cache['starred_assets'].add(symbol)
    elif not row['Star'] and symbol in st.session_state.cache['starred_assets']:
        st.session_state.cache['starred_assets'].discard(symbol)

# Expanders for crypto details
for index, row in crypto_df.iterrows():
    symbol = row['Details']
    with st.expander(f"Details for {symbol}"):
        data = latest_data.get(symbol)
        price_change = row['24h Change (%)']
        signal = row['Signal']
        news_sentiment = data['News_Sentiment'].iloc[-1] if data is not None and 'News_Sentiment' in data else 0.0
        buy_prob = data.get('Buy_Prob', 0.0) if data is not None else 0.0
        market_status = "open"

        if signal == "Buy":
            action_text = f"Buy now"
            details = f"Price is likely to rise based on strong market signals."
        elif signal == "Sell/Hold":
            action_text = f"Consider selling or holding"
            details = f"Price may fall or stay stable based on current market signals."
            if buy_prob < 0.3:
                details += f" Selling is more likely."
            else:
                details += f" Holding is more likely."
        else:
            action_text = f"Error"
            details = f"Unable to provide a recommendation."
        
        if price_change > 0:
            details += f" Price is up recently."
        else:
            details += f" Price is down recently."
        if news_sentiment > 0:
            details += f" News is positive."
        else:
            details += f" News is neutral or negative."
        
        details += f" Markets are {market_status}. This is an estimate—be cautious!"
        
        st.markdown(f"**{action_text}**")
        st.write(details)

st.markdown("**Crypto Markets Open**")

st.write(f"Last Updated: {last_updated}")
st.write("Note: 'Sell/Hold' signals suggest checking the reason for context. Adjust refresh interval for updates.")

# Auto-refresh
time.sleep(refresh_interval)
st.rerun()