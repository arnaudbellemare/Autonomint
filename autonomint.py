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
from scipy.stats import norm # Import at the top now that it will be installed

# =====================================================================================
# ==                      APP CONFIGURATION & STYLING                                ==
# =====================================================================================

st.set_page_config(page_title="Autonomint Quant Strategy", page_icon="📈", layout="wide")
st.title("📈 Autonomint: Delta-Neutral Yield Strategy Engine")
st.markdown("An implementation of the delta-neutral strategy combining dCDS protection, option selling, and dynamic perpetual future hedging based on live market conditions.")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =====================================================================================
# ==                      HELPER & DATA FETCHING FUNCTIONS                         ==
# =====================================================================================

def with_retries(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay;
            for i in range(max_retries):
                try: return func(*args, **kwargs)
                except Exception as e:
                    logging.warning(f"API call to {func.__name__} failed (Attempt {i + 1}/{max_retries}): {e}. Retrying...")
                    time.sleep(delay); delay *= backoff_factor
            logging.error(f"API call to {func.__name__} failed after {max_retries} retries."); return None
        return wrapper
    return decorator

@st.cache_data(ttl=120)
@with_retries()
def fetch_live_eth_price(exchange_id: str = 'kraken', symbol: str = 'ETH/USD'):
    exchange = getattr(ccxt, exchange_id)(); ticker = exchange.fetch_ticker(symbol); return ticker.get('last')

@st.cache_data(ttl=10)
@with_retries()
def get_instrument_ticker(instrument_name: str):
    BASE_URL = "https://thalex.com/api/v2/public"; URL_TICKER = f"{BASE_URL}/ticker"; API_DELAY_TICKER = 0.2
    time.sleep(API_DELAY_TICKER)
    response = requests.get(URL_TICKER, params={"instrument_name": instrument_name}, timeout=15)
    response.raise_for_status(); return response.json().get("result", {})

@st.cache_data(ttl=300)
def get_thalex_actual_daily_funding_rate(coin_symbol: str) -> float:
    instrument_name = f"{coin_symbol.upper()}-PERPETUAL"; ticker_data = get_instrument_ticker(instrument_name)
    if ticker_data:
        if pd.notna(ticker_data.get('average_funding_rate_24h')):
            logging.info(f"Using average_funding_rate_24h for {instrument_name}"); return float(ticker_data['average_funding_rate_24h'])
        elif pd.notna(ticker_data.get('funding_rate')):
            logging.info(f"Estimating daily funding rate from 8h rate for {instrument_name}"); return float(ticker_data['funding_rate']) * 3
        else: logging.warning(f"No funding rate found for {instrument_name}"); return 0.0
    else: logging.warning(f"Could not fetch ticker for {instrument_name} funding rate."); return 0.0

@st.cache_data(ttl=600)
@with_retries()
def get_all_options_data():
    """Fetches and parses all option instruments from Thalex."""
    BASE_URL = "https://thalex.com/api/v2/public"; URL_INSTRUMENTS = f"{BASE_URL}/instruments"
    response = requests.get(URL_INSTRUMENTS, timeout=15); response.raise_for_status()
    instruments = response.json().get("result", [])
    if not instruments: return pd.DataFrame()
    now_utc, parsed_options = datetime.now(timezone.utc), []
    date_pattern = re.compile(r'ETH-(\d{2}[A-Z]{3}\d{2})-(\d+)-([CP])')
    for instr in instruments:
        if instr.get('type') != 'option': continue
        name = instr.get('instrument_name')
        if not name: continue
        match = date_pattern.match(name)
        if match:
            expiry_str, strike_str, type_char = match.groups()
            try:
                expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8, minute=0, tzinfo=timezone.utc)
                if expiry_dt > now_utc:
                    dte = (expiry_dt - now_utc).total_seconds() / (24 * 3600)
                    parsed_options.append({'instrument_name': name, 'strike': float(strike_str), 'option_type': 'call' if type_char == 'C' else 'put', 'dte': dte})
            except (ValueError, TypeError): continue
    return pd.DataFrame(parsed_options)

# --- NEW: Function to automatically fetch ATM Implied Volatility ---
@st.cache_data(ttl=300)
def fetch_atm_iv(options_df: pd.DataFrame, target_dte: int, live_price: float) -> float:
    """Finds the closest-to-money option for a target expiry and returns its IV."""
    if options_df is None or options_df.empty: return 0.0
    # 1. Find the expiry date closest to the user's target DTE
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs()
    closest_expiry_row = options_df.loc[options_df['dte_diff'].idxmin()]
    expiry_df = options_df[np.isclose(options_df['dte'], closest_expiry_row['dte'])].copy()
    # 2. Find the ATM call option (strike closest to live price)
    calls_df = expiry_df[expiry_df['option_type'] == 'call']
    if calls_df.empty: logging.warning("No call options found for the selected expiry."); return 0.0
    calls_df['strike_diff'] = (calls_df['strike'] - live_price).abs()
    atm_call_instrument = calls_df.loc[calls_df['strike_diff'].idxmin()]['instrument_name']
    # 3. Fetch its ticker data and extract the IV
    ticker_data = get_instrument_ticker(atm_call_instrument)
    if ticker_data and pd.notna(ticker_data.get('iv')):
        iv = float(ticker_data['iv']); return iv / 100.0  # Convert from 67.0 to 0.67
    logging.warning(f"Could not fetch IV for ATM instrument {atm_call_instrument}."); return 0.0

@st.cache_data(ttl=900)
@with_retries()
def fetch_historical_prices(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 40, timeframe='1d'):
    exchange = getattr(ccxt, exchange_id)(); limit = days_lookback + 5; since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv: return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume']); df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True); return df

def calculate_rv(prices: pd.Series, window: int = 30) -> float:
    log_returns = np.log(prices / prices.shift(1)); return log_returns.rolling(window=window).std().iloc[-1] * np.sqrt(365)
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0); avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0: return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]; return 100 - (100 / (1 + rs))

# (Simulation & Logic functions remain the same)
def simulate_black_scholes_premium(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)); d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call': price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else: price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return price

def determine_hedging_strategy(iv, rv, rsi, daily_funding_rate, ltv, thresholds):
    reasons = [];
    if iv > thresholds['iv_high']: reasons.append(f"High IV ({iv:.1%}) > threshold ({thresholds['iv_high']:.1%})")
    if iv > rv * (1 + thresholds['iv_rv_spread']): reasons.append(f"IV/RV spread > {thresholds['iv_rv_spread']:.0%} (IV:{iv:.1%}, RV:{rv:.1%})")
    if rsi > thresholds['rsi_high'] or rsi < thresholds['rsi_low']: reasons.append(f"Strong trend (RSI:{rsi:.1f})")
    if abs(daily_funding_rate) > thresholds['funding_rate_high']: reasons.append(f"High daily funding ({daily_funding_rate:.4%})")
    if ltv > thresholds['ltv_high']: reasons.append(f"High LTV ({ltv:.1%})")
    if reasons: return {'hedge_with_perp': True, 'reason': "Perp hedge is a MUST. Trigger(s): " + " | ".join(reasons)}
    reason_skip = "Perp hedge not necessary. ";
    if rv < thresholds['rv_low'] and iv < thresholds['iv_low']: reason_skip += f"Low volatility regime (RV:{rv:.1%}, IV:{iv:.1%})."
    elif thresholds['rsi_low'] <= rsi <= thresholds['rsi_high']: reason_skip += "Market is range-bound."
    else: reason_skip += "No strong signal met."
    return {'hedge_with_perp': False, 'reason': reason_skip}

def calculate_final_pnl(eth_price_final, params, strategy):
    aave_yield = params['eth_deposited'] * params['eth_price_initial'] * (params['aave_apy'] / 12); dcds_payout = max(0, params['eth_deposited'] * params['eth_price_initial'] * params['dcds_coverage'] - (params['eth_deposited'] * (params['eth_price_initial'] - eth_price_final))); dcds_cost = params['eth_deposited'] * params['eth_price_initial'] * params['dcds_coverage'] * params['dcds_premium']; net_dcds_pnl = dcds_payout - dcds_cost
    if strategy['option_type'] == 'put': option_pnl = strategy['premium'] - max(0, strategy['strike'] - eth_price_final)
    else: option_pnl = strategy['premium'] - max(0, eth_price_final - strategy['strike'])
    perp_pnl = 0
    if strategy['hedge_with_perp']: funding_pnl = params['eth_price_initial'] * strategy['daily_funding_rate'] * 30; price_change_pnl = (params['eth_price_initial'] - eth_price_final); perp_pnl = funding_pnl + price_change_pnl
    underlying_pnl = (eth_price_final - params['eth_price_initial']) * params['eth_deposited']; total_pnl = underlying_pnl + aave_yield + net_dcds_pnl + option_pnl + perp_pnl
    return total_pnl, aave_yield, net_dcds_pnl, option_pnl, perp_pnl

# =====================================================================================
# ==                          SIDEBAR FOR USER INPUTS                              ==
# =====================================================================================
with st.sidebar:
    st.header("⚙️ Core Configuration"); ETH_DEPOSITED = st.number_input("ETH Deposited", 1.0, 10.0, 2.0, 0.5); ETH_PRICE_INITIAL = st.number_input("Initial ETH Price ($)", 1000.0, 10000.0, 2000.0, 100.0)
    st.header("📜 Protocol Parameters (Simulated)"); AAVE_APY = st.slider("AAVE Supply APY (%)", 0.1, 10.0, 3.0, 0.1) / 100.0; DCDS_COVERAGE = st.slider("dCDS Downside Coverage (%)", 10.0, 50.0, 20.0, 1.0) / 100.0; DCDS_PREMIUM = st.slider("dCDS Premium (% of covered amount)", 1.0, 15.0, 6.0, 0.5) / 100.0; LTV = st.slider("Loan-to-Value (LTV) (%)", 50.0, 95.0, 80.0, 1.0) / 100.0
    st.header("🧠 Strategy Decision Thresholds"); option_type = st.radio("Option to Sell", ["Put (Upside Retention)", "Call (Capped Upside)"], horizontal=True); TARGET_DTE = st.slider("Target Days to Expiry (DTE) for IV", 7, 60, 30, 1)
    col1, col2 = st.columns(2)
    with col1: st.markdown("**Volatility**"); IV_HIGH = col1.slider("High IV Threshold (%)", 50., 100., 70., 1.) / 100.0; IV_LOW = col1.slider("Low IV Threshold (%)", 20., 60., 50., 1.) / 100.0; RV_LOW = col1.slider("Low RV Threshold (%)", 10., 50., 40., 1.) / 100.0; IV_RV_SPREAD = col1.slider("IV/RV Spread Trigger (%)", 5., 50., 20., 1.) / 100.0
    with col2: st.markdown("**Trend & Funding**"); RSI_HIGH = col2.slider("High RSI (Overbought)", 60, 80, 70); RSI_LOW = col2.slider("Low RSI (Oversold)", 20, 40, 30); FUNDING_HIGH = col2.slider("High Daily Funding Rate Trigger (%)", 0.05, 0.3, 0.15) / 100.0; LTV_HIGH = col2.slider("High LTV Trigger (%)", 75, 95, 80) / 100.0
    thresholds = {'iv_high': IV_HIGH, 'iv_low': IV_LOW, 'rv_low': RV_LOW, 'iv_rv_spread': IV_RV_SPREAD, 'rsi_high': RSI_HIGH, 'rsi_low': RSI_LOW, 'funding_rate_high': FUNDING_HIGH, 'ltv_high': LTV_HIGH}

# =====================================================================================
# ==                      MAIN APP LOGIC & DISPLAY                                 ==
# =====================================================================================

with st.spinner("Fetching all live market data..."):
    live_eth_price = fetch_live_eth_price()
    daily_funding_rate = get_thalex_actual_daily_funding_rate('ETH')
    historical_df = fetch_historical_prices(days_lookback=40)
    all_options = get_all_options_data()

if not live_eth_price or daily_funding_rate is None or historical_df.empty or all_options.empty:
    st.error("Failed to fetch critical market data. Please try again later."); st.stop()

rv = calculate_rv(historical_df['mark_price_close'])
rsi = calculate_rsi(historical_df['mark_price_close'])
# --- IV is now fetched automatically ---
iv = fetch_atm_iv(all_options, TARGET_DTE, live_eth_price)
if iv == 0.0:
    st.warning("Could not fetch a valid Implied Volatility. The decision engine may be impaired.")

st.header("📊 Live Market Conditions")
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Live ETH Price", f"${live_eth_price:,.2f}")
mcol2.metric("30D Realized Volatility (RV)", f"{rv:.2%}")
# --- DISPLAY the fetched IV ---
mcol3.metric(f"~{TARGET_DTE}-Day ATM IV", f"{iv:.2%}")
mcol4.metric("Daily Funding Rate", f"{daily_funding_rate:.4%}", help=f"Annualized estimate: {daily_funding_rate * 365:.2%}")
mcol5.metric("14D RSI", f"{rsi:.1f}")

# (The rest of the script remains identical, but is now fully automated)
st.header("💡 Strategy Recommendation Engine")
strategy_decision = determine_hedging_strategy(iv, rv, rsi, daily_funding_rate, LTV, thresholds)
if strategy_decision['hedge_with_perp']: st.success(f"**Recommendation: Sell {option_type.split(' ')[0]} + Hedge with 1x Short Perpetual**")
else: st.info(f"**Recommendation: Sell {option_type.split(' ')[0]} (No Perp Hedge Needed)**")
st.caption(f"Reasoning: {strategy_decision['reason']}")

st.header("💸 Payoff Analysis")
strike_price_put = ETH_PRICE_INITIAL * (1 - DCDS_COVERAGE); strike_price_call = ETH_PRICE_INITIAL * 1.2; strike = strike_price_put if option_type.startswith('Put') else strike_price_call
premium = simulate_black_scholes_premium(live_eth_price, strike, TARGET_DTE/365, 0.02, iv, option_type.split(' ')[0].lower())
current_strategy = {'option_type': option_type.split(' ')[0].lower(), 'strike': strike, 'premium': premium, 'hedge_with_perp': strategy_decision['hedge_with_perp'], 'daily_funding_rate': daily_funding_rate,}
params = {'eth_deposited': ETH_DEPOSITED, 'eth_price_initial': ETH_PRICE_INITIAL, 'aave_apy': AAVE_APY, 'dcds_coverage': DCDS_COVERAGE, 'dcds_premium': DCDS_PREMIUM,}
pcol1, pcol2 = st.columns([1,2])
with pcol1:
    st.markdown(f"**Strategy Details:**"); st.write(f"🔹 **Selling 1x {option_type.split(' ')[0].capitalize()}**"); st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Strike: `${strike:,.0f}`"); st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Premium collected: `${premium:,.2f}`")
    if current_strategy['hedge_with_perp']: st.write(f"🔹 **Hedging with 1x Short Perpetual**")
    eth_price_final = st.slider("Set Target ETH Price ($) for PnL Breakdown", int(ETH_PRICE_INITIAL * 0.5), int(ETH_PRICE_INITIAL * 1.5), int(ETH_PRICE_INITIAL))
    total_pnl, pnl_aave, pnl_dcds, pnl_option, pnl_perp = calculate_final_pnl(eth_price_final, params, current_strategy)
    st.metric("Total Projected PnL at Target Price", f"${total_pnl:,.2f}")
    with st.expander("Show PnL Contribution Breakdown"):
        st.metric("PnL from Underlying ETH", f"${(eth_price_final - ETH_PRICE_INITIAL) * ETH_DEPOSITED:,.2f}", delta_color="off"); st.metric("PnL from AAVE Yield", f"${pnl_aave:,.2f}"); st.metric("PnL from dCDS (Net)", f"${pnl_dcds:,.2f}"); st.metric("PnL from Sold Option", f"${pnl_option:,.2f}"); st.metric("PnL from Perpetual Hedge", f"${pnl_perp:,.2f}")
with pcol2:
    price_range = np.linspace(ETH_PRICE_INITIAL * 0.6, ETH_PRICE_INITIAL * 1.4, 200)
    pnl_values = [calculate_final_pnl(p, params, current_strategy)[0] for p in price_range]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3))); fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even"); fig.add_vline(x=eth_price_final, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${total_pnl:,.0f}", annotation_position="top right"); fig.add_vline(x=ETH_PRICE_INITIAL, line_width=1, line_dash="dot", line_color="grey", annotation_text="Initial Price", annotation_position="bottom right")
    fig.update_layout(title=f"Payoff Diagram: {current_strategy['option_type'].capitalize()} Sale {'+ Perp Hedge' if current_strategy['hedge_with_perp'] else ''}", xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
