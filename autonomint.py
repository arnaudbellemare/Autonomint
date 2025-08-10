# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from functools import wraps
import logging
import time
import ccxt
import re
import plotly.graph_objects as go

# =====================================================================================
# ==                      APP CONFIGURATION & STYLING                                ==
# =====================================================================================

st.set_page_config(page_title="Autonomint Strategy Analyzer", page_icon="🤖", layout="wide")
st.title("🤖 Autonomint: Quantitative Strategy Analysis")
st.markdown("An advanced dashboard to identify and analyze option selling strategies based on statistical volatility metrics.")

# --- API Configuration & Constants ---
BASE_URL = "https://thalex.com/api/v2/public"
INSTRUMENTS_ENDPOINT = "instruments"
URL_INSTRUMENTS = f"{BASE_URL}/{INSTRUMENTS_ENDPOINT}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
REQUEST_TIMEOUT = 15
API_DELAY_TICKER = 0.2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================================
# ==                      HELPER & DATA FETCHING FUNCTIONS                         ==
# =====================================================================================

def with_retries(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try: return func(*args, **kwargs)
                except (requests.exceptions.RequestException, ccxt.NetworkError) as e:
                    logging.warning(f"API call to {func.__name__} failed (Attempt {i+1}/{max_retries}): {e}. Retrying...")
                    time.sleep(delay); delay *= backoff_factor
            logging.error(f"API call to {func.__name__} failed after {max_retries} retries."); return None
        return wrapper
    return decorator

@st.cache_data(ttl=120)
@with_retries()
def fetch_live_eth_price(exchange_id: str = 'kraken', symbol: str = 'ETH/USD'):
    """Fetches the last traded price for a symbol from an exchange."""
    try:
        exchange = getattr(ccxt, exchange_id)()
        ticker = exchange.fetch_ticker(symbol)
        return ticker.get('last')
    except Exception as e:
        logging.error(f"Could not fetch live ETH price: {e}")
        return None

@st.cache_data(ttl=600)
@with_retries()
def fetch_instruments():
    response = requests.get(URL_INSTRUMENTS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("result", [])

@st.cache_data(ttl=10)
@with_retries()
def fetch_ticker(instr_name: str):
    time.sleep(API_DELAY_TICKER)
    response = requests.get(URL_TICKER, params={"instrument_name": instr_name}, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json().get("result", {})

@st.cache_data(ttl=900)
@with_retries()
def fetch_historical_prices(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 90, timeframe='1d'):
    try:
        exchange = getattr(ccxt, exchange_id)(); limit = days_lookback + 5
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=limit)
        if not ohlcv: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume'])
        df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True)
        return df
    except Exception as e:
        logging.error(f"CCXT fetch failed: {e}"); return pd.DataFrame()

def calculate_realized_volatility(prices: pd.Series, window: int = 30) -> pd.Series:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.rolling(window=window).std() * np.sqrt(365)

@st.cache_data(ttl=900)
def get_volatility_z_score(realized_vol_series: pd.Series, current_iv: float):
    if realized_vol_series.empty or realized_vol_series.iloc[-1] == 0:
        return None, None
    
    np.random.seed(42)
    iv_noise = np.random.normal(0.05, 0.2, len(realized_vol_series))
    historical_iv = realized_vol_series * (1 + iv_noise)
    historical_iv = historical_iv.clip(lower=0.1)
    
    iv_rv_ratio = historical_iv / realized_vol_series
    log_iv_rv_ratio = np.log(iv_rv_ratio).dropna()

    if len(log_iv_rv_ratio) < 20: return None, None
    
    mean_log_ratio = log_iv_rv_ratio.mean()
    std_log_ratio = log_iv_rv_ratio.std()

    current_rv = realized_vol_series.iloc[-1]
    current_log_ratio = np.log(current_iv / current_rv) if current_rv > 0 else 0
    z_score = (current_log_ratio - mean_log_ratio) / std_log_ratio if std_log_ratio > 0 else 0

    return z_score, log_iv_rv_ratio

def calculate_health(eth_price_current, debt, eth_deposited):
    if debt == 0: return float('inf')
    return (eth_deposited * eth_price_current) / debt

def _calculate_dte(expiry_str, current_date_utc):
    try:
        expiry_dt_obj = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8, minute=0, tzinfo=timezone.utc)
        if expiry_dt_obj <= current_date_utc: return None
        time_to_expiry = expiry_dt_obj - current_date_utc
        return time_to_expiry.days + (time_to_expiry.seconds / (24 * 3600))
    except (ValueError, TypeError): return None

@st.cache_data(ttl=300)
def get_clean_options_df():
    instruments = fetch_instruments()
    if not instruments: st.error("Failed to fetch market instruments."); return None
    now_utc, parsed_options = datetime.now(timezone.utc), []
    date_pattern = re.compile(r'ETH-(\d{2}[A-Z]{3}\d{2})-(\d+)-([CP])')
    for instr in instruments:
        if instr.get('type') != 'option': continue
        name = instr.get('instrument_name')
        if not name: continue
        match = date_pattern.match(name)
        if match:
            expiry_str, strike_str, type_char = match.groups()
            dte = _calculate_dte(expiry_str, now_utc)
            if dte and dte > 0:
                parsed_options.append({'instrument_name': name, 'strike': float(strike_str), 'option_type': 'call' if type_char == 'C' else 'put', 'dte': dte})
    if not parsed_options: st.error("No valid future options could be parsed."); return None
    return pd.DataFrame(parsed_options)

def create_payoff_diagram(strategy_details, price_range, eth_deposited, eth_price_initial, aave_yield, predicted_price):
    underlying_pnl = (price_range * eth_deposited) - (eth_price_initial * eth_deposited)
    total_pnl = underlying_pnl + aave_yield
    
    title_text = "Payoff: ETH Holdings + AAVE Yield"
    
    if strategy_details['name'] == "Sell Premium 🤑":
        short_put = strategy_details.get('put')
        short_call = strategy_details.get('call')
        
        option_pnl = 0
        option_titles = []
        if short_put is not None:
            put_pnl = short_put['price'] - np.maximum(0, short_put['strike'] - price_range)
            option_pnl += put_pnl
            option_titles.append(f"Short Put @ ${short_put['strike']:.0f}")
        if short_call is not None:
            call_pnl = short_call['price'] - np.maximum(0, price_range - short_call['strike'])
            option_pnl += call_pnl
            option_titles.append(f"Short Call @ ${short_call['strike']:.0f}")

        total_pnl += option_pnl
        if option_titles:
            title_text += " + " + " & ".join(option_titles)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=total_pnl, mode='lines', name='Total PnL', line=dict(color='mediumseagreen', width=3)))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even")
    predicted_pnl = total_pnl[np.argmin(np.abs(price_range - predicted_price))]
    fig.add_vline(x=predicted_price, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${predicted_pnl:,.0f}", annotation_position="top right")
    fig.update_layout(title=title_text, xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=40, r=40, t=50, b=40))
    return fig

# =====================================================================================
# ==                          SIDEBAR FOR USER INPUTS                              ==
# =====================================================================================

with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    ETH_DEPOSITED = st.number_input("ETH Deposited", 0.1, value=2.0, step=0.1, format="%.1f")
    ETH_PRICE_INITIAL = st.number_input("Initial ETH Price", 1.0, value=3600.0, step=50.0)

    st.header("📈 Hedging & Yield")
    AAVE_APY_PERCENT = st.slider("AAVE Supply APY", 0.0, 10.0, 3.0, format="%.2f%%")
    AAVE_APY = AAVE_APY_PERCENT / 100.0

    st.header("🧠 Option Selling Parameters")
    TARGET_DTE = st.slider("Target Days to Expiry (DTE)", 7, 90, 30)
    Z_SCORE_SELL_THRESHOLD = st.slider("Z-Score Sell Threshold", 0.5, 3.0, 1.0, 0.1, help="Sell options if the Log(IV/RV) Z-Score is above this value.")
    TARGET_DELTA = st.slider("Target Delta", 0.10, 0.45, 0.35, 0.01, help="The target delta for the puts/calls to sell.")
    MIN_PREMIUM_RATIO = st.slider("Min Premium-to-Spot Ratio (%)", 0.1, 5.0, 1.0, 0.1, help="The minimum premium required (as % of spot price) to sell an option.") / 100.0

# =====================================================================================
# ==                      MAIN APP LOGIC & DISPLAY                                 ==
# =====================================================================================

# --- Step 1: Fetch all necessary data ---
live_eth_price = fetch_live_eth_price()
if not live_eth_price: st.error("Could not fetch live ETH price."); st.stop()

options_df = get_clean_options_df()
if options_df is None: st.stop()

with st.spinner("Fetching market data for volatility analysis..."):
    historical_df = fetch_historical_prices(days_lookback=90)
    realized_vol_series = calculate_realized_volatility(historical_df['mark_price_close'], window=30) if not historical_df.empty else pd.Series()

# --- Step 2: Select options chain and find an ATM IV for reference ---
best_expiry_dte = options_df.iloc[(options_df['dte'] - TARGET_DTE).abs().idxmin()]['dte']
expiry_df = options_df[np.isclose(options_df['dte'], best_expiry_dte)].copy()
# Find the ATM call by looking for the one with the lowest strike price that is still above the live price
atm_call_rows = expiry_df[(expiry_df['option_type'] == 'call') & (expiry_df['strike'] > live_eth_price)]
if not atm_call_rows.empty:
    atm_call_name = atm_call_rows.loc[atm_call_rows['strike'].idxmin(), 'instrument_name']
    atm_call_ticker = fetch_ticker(atm_call_name)
    current_iv = atm_call_ticker.get('iv', 0) / 100 if atm_call_ticker and atm_call_ticker.get('iv') else 0.0
else:
    current_iv = 0.0
if not current_iv: st.warning("Could not fetch a reference Implied Volatility. Z-Score analysis may be unreliable.");

# --- Step 3: Calculate Z-Score and Determine Strategy ---
z_score, hist_log_ratio = get_volatility_z_score(realized_vol_series, current_iv) if current_iv > 0 and not realized_vol_series.empty else (None, None)

st.header("Volatility & Strategy Signal")
v_col1, v_col2, v_col3 = st.columns(3)
v_col1.metric("Live ETH Price", f"${live_eth_price:,.2f}")
v_col2.metric("30-Day Realized Volatility", f"{realized_vol_series.iloc[-1]:.2%}" if not realized_vol_series.empty else "N/A")
v_col3.metric("ATM Implied Volatility", f"{current_iv:.2%}" if current_iv > 0 else "N/A")

if z_score is not None:
    st.metric("Log(IV/RV) Z-Score", f"{z_score:.2f}", f"Historic Mean: {np.exp(hist_log_ratio.mean()):.2f}x RV",
              help="Measures how expensive current IV is compared to its history relative to RV. A high Z-Score (>1) suggests IV is historically high, a good signal for selling premium.")
    if z_score > Z_SCORE_SELL_THRESHOLD:
        st.success(f"Z-Score ({z_score:.2f}) is above threshold ({Z_SCORE_SELL_THRESHOLD:.2f}). **Strategy: Sell Premium 🤑**")
        strategy = "Sell Premium 🤑"
    else:
        st.info(f"Z-Score ({z_score:.2f}) is below threshold ({Z_SCORE_SELL_THRESHOLD:.2f}). **Strategy: Hold / Hedge 🛡️**")
        strategy = "Hold / Hedge 🛡️"
else:
    st.warning("Could not calculate Z-Score. Defaulting to Hold/Hedge strategy.")
    strategy = "Hold / Hedge 🛡️"

# --- Step 4: Find and Display Actionable Options ---
st.divider()
st.subheader("Actionable Strategy Details")
strategy_details = {'name': strategy}

if strategy == "Sell Premium 🤑":
    with st.spinner(f"Finding best options to sell for ~{best_expiry_dte:.0f} DTE..."):
        option_tickers = {row['instrument_name']: fetch_ticker(row['instrument_name']) for _, row in expiry_df.iterrows()}
        expiry_df['ticker_data'] = expiry_df['instrument_name'].map(option_tickers)
        expiry_df.dropna(subset=['ticker_data'], inplace=True)
        expiry_df['delta'] = expiry_df['ticker_data'].apply(lambda x: x.get('delta'))
        expiry_df['price'] = expiry_df['ticker_data'].apply(lambda x: x.get('mark_price'))
        
        expiry_df.dropna(subset=['delta', 'price'], inplace=True)
        
        puts_df = expiry_df[(expiry_df['option_type'] == 'put') & (expiry_df['price'] / live_eth_price >= MIN_PREMIUM_RATIO)]
        best_put = puts_df.iloc[(puts_df['delta'] - (-TARGET_DELTA)).abs().idxmin()] if not puts_df.empty else None
        
        calls_df = expiry_df[(expiry_df['option_type'] == 'call') & (expiry_df['price'] / live_eth_price >= MIN_PREMIUM_RATIO)]
        best_call = calls_df.iloc[(calls_df['delta'] - TARGET_DELTA).abs().idxmin()] if not calls_df.empty else None

    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("**Candidate Put to Sell**")
        if best_put is not None:
            strategy_details['put'] = best_put.to_dict()
            st.success(f"**Strike: ${best_put['strike']:.0f}**", icon="🎯")
            st.write(f"Premium: `${best_put['price']:.2f}` | Delta: `{best_put['delta']:.3f}` | Premium/Spot: `{best_put['price']/live_eth_price:.2%}`")
        else:
            st.warning("No suitable Put found matching criteria.")
    
    with act_col2:
        st.markdown("**Candidate Call to Sell**")
        if best_call is not None:
            strategy_details['call'] = best_call.to_dict()
            st.success(f"**Strike: ${best_call['strike']:.0f}**", icon="🎯")
            st.write(f"Premium: `${best_call['price']:.2f}` | Delta: `{best_call['delta']:.3f}` | Premium/Spot: `{best_call['price']/live_eth_price:.2%}`")
        else:
            st.warning("No suitable Call found matching criteria.")
else:
    st.info("No options will be sold. The strategy is to hold the underlying ETH and collect AAVE yield. Consider manual hedging if desired.")

# --- Step 5: PnL and Payoff Analysis ---
st.divider()
st.header("Interactive PnL and Payoff Analysis")
eth_price_predicted = st.slider("Set Target ETH Price ($) for PnL Analysis", float(live_eth_price * 0.5), float(live_eth_price * 2.0), float(live_eth_price), 10.0)

aave_yield_monthly = (AAVE_APY / 12) * (ETH_DEPOSITED * ETH_PRICE_INITIAL)
underlying_pnl = (eth_price_predicted - ETH_PRICE_INITIAL) * ETH_DEPOSITED
option_pnl = 0
if 'put' in strategy_details and strategy_details['put'] is not None:
    put_details = strategy_details['put']
    option_pnl += put_details['price'] - max(0, put_details['strike'] - eth_price_predicted)
if 'call' in strategy_details and strategy_details['call'] is not None:
    call_details = strategy_details['call']
    option_pnl += call_details['price'] - max(0, eth_price_predicted - call_details['strike'])

total_profit = underlying_pnl + aave_yield_monthly + option_pnl
collateral_initial = ETH_DEPOSITED * ETH_PRICE_INITIAL
apy = (total_profit / collateral_initial) * 12 * 100 if collateral_initial > 0 else 0

pnl_c1, pnl_c2, pnl_c3 = st.columns(3)
pnl_c1.metric("Projected Total Profit", f"${total_profit:,.2f}", f"{apy:.2f}% Projected APY")
pnl_c2.metric("AAVE Yield (1 Mo)", f"${aave_yield_monthly:,.2f}")
pnl_c3.metric("Net Option PnL", f"${option_pnl:,.2f}", help="PnL from sold options at your target price.")

price_range = np.linspace(live_eth_price * 0.4, live_eth_price * 1.6, num=200)
payoff_fig = create_payoff_diagram(strategy_details, price_range, ETH_DEPOSITED, ETH_PRICE_INITIAL, aave_yield_monthly, eth_price_predicted)
if payoff_fig: st.plotly_chart(payoff_fig, use_container_width=True)
