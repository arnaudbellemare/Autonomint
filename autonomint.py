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
from scipy.stats import norm

# =====================================================================================
# ==                            IMPORTS & CONFIGURATION                            ==
# =====================================================================================

st.set_page_config(page_title="Autonomint Quant Strategy Optimizer", page_icon="💡", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_URL_THALEX = "https://thalex.com/api/v2/public"

# =====================================================================================
# ==                           DATA FETCHING & CACHING                           ==
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
    URL_TICKER = f"{BASE_URL_THALEX}/ticker"; API_DELAY_TICKER = 0.2; time.sleep(API_DELAY_TICKER)
    response = requests.get(URL_TICKER, params={"instrument_name": instrument_name}, timeout=15)
    response.raise_for_status(); return response.json().get("result", {})

@st.cache_data(ttl=300)
def get_thalex_actual_daily_funding_rate(coin_symbol: str) -> float:
    instrument_name = f"{coin_symbol.upper()}-PERPETUAL"; ticker_data = get_instrument_ticker(instrument_name)
    if ticker_data:
        if pd.notna(ticker_data.get('average_funding_rate_24h')): return float(ticker_data['average_funding_rate_24h'])
        elif pd.notna(ticker_data.get('funding_rate')): return float(ticker_data['funding_rate']) * 3
    return 0.0

@st.cache_data(ttl=600)
@with_retries()
def get_all_options_data():
    URL_INSTRUMENTS = f"{BASE_URL_THALEX}/instruments"; response = requests.get(URL_INSTRUMENTS, timeout=15); response.raise_for_status()
    instruments = response.json().get("result", []); now_utc, parsed_options = datetime.now(timezone.utc), []
    date_pattern = re.compile(r'ETH-(\d{2}[A-Z]{3}\d{2})-(\d+)-([CP])')
    for instr in instruments:
        if instr.get('type') != 'option' or not instr.get('instrument_name'): continue
        match = date_pattern.match(instr['instrument_name'])
        if match:
            expiry_str, strike_str, type_char = match.groups()
            try:
                expiry_dt = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8, minute=0, tzinfo=timezone.utc)
                if expiry_dt > now_utc:
                    dte = (expiry_dt - now_utc).total_seconds() / (24 * 3600)
                    parsed_options.append({'instrument_name': instr['instrument_name'], 'strike': float(strike_str), 'option_type': 'call' if type_char == 'C' else 'put', 'dte': dte})
            except (ValueError, TypeError): continue
    return pd.DataFrame(parsed_options)

@st.cache_data(ttl=300)
def fetch_atm_iv(options_df: pd.DataFrame, target_dte: int, live_price: float) -> float:
    if options_df is None or options_df.empty: return 0.0
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs()
    closest_expiry_row = options_df.loc[options_df['dte_diff'].idxmin()]
    expiry_df = options_df[np.isclose(options_df['dte'], closest_expiry_row['dte'])].copy()
    calls_df = expiry_df[expiry_df['option_type'] == 'call']
    if calls_df.empty: return 0.0
    calls_df['strike_diff'] = (calls_df['strike'] - live_price).abs()
    atm_call_instrument = calls_df.loc[calls_df['strike_diff'].idxmin()]['instrument_name']
    ticker_data = get_instrument_ticker(atm_call_instrument)
    if ticker_data and pd.notna(ticker_data.get('iv')): return float(ticker_data['iv']) / 100.0
    return 0.0

@st.cache_data(ttl=900)
@with_retries()
def fetch_historical_prices(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 50, timeframe='1d'):
    exchange = getattr(ccxt, exchange_id)(); limit = days_lookback if timeframe == '1d' else days_lookback * 24 + 5
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv: return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume']); df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True); return df

# =====================================================================================
# ==                          CORE CALCULATIONS & METRICS                          ==
# =====================================================================================
def calculate_rv(prices: pd.Series, window: int = 30) -> float:
    log_returns = np.log(prices / prices.shift(1)); return log_returns.rolling(window=window).std().iloc[-1] * np.sqrt(365)
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0); avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_loss.iloc[-1] == 0: return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]; return 100 - (100 / (1 + rs))

def calculate_final_pnl(eth_price_final, params, sold_put, sold_call, hedge_with_perp):
    underlying_pnl = (eth_price_final - params['eth_price_initial']) * params['eth_deposited']; aave_yield = params['eth_price_initial'] * params['eth_deposited'] * (params['aave_apy'] / 12)
    dcds_hedge_strike = params['eth_price_initial'] * (1 - params['dcds_coverage_percent']); hedged_value = params['eth_price_initial'] * params['eth_deposited'] * params['dcds_coverage_percent']; dcds_fee = hedged_value * params['dcds_fee_percent']; dcds_upside_cost = max(0, underlying_pnl) * params['dcds_upside_sharing_percent']; dcds_payout = max(0, dcds_hedge_strike - eth_price_final) * params['eth_deposited']; net_dcds_pnl = dcds_payout - dcds_fee - dcds_upside_cost
    option_pnl = 0
    if sold_put: option_pnl += sold_put['price'] - max(0, sold_put['strike'] - eth_price_final)
    if sold_call: option_pnl += sold_call['price'] - max(0, eth_price_final - sold_call['strike'])
    perp_pnl = 0
    if hedge_with_perp: funding_pnl = params['eth_price_initial'] * params['daily_funding_rate'] * 30; price_change_pnl = (params['eth_price_initial'] - eth_price_final); perp_pnl = funding_pnl + price_change_pnl
    total_pnl = underlying_pnl + aave_yield + net_dcds_pnl + option_pnl + perp_pnl
    return total_pnl, underlying_pnl, aave_yield, net_dcds_pnl, option_pnl, perp_pnl

# =====================================================================================
# ==                      STRATEGY ENGINE & DECISION LOGIC                       ==
# =====================================================================================

def generate_optimal_strategy(iv, rv, rsi_daily, rsi_hourly, thresholds):
    """The Primary Engine: Determines the best strategy using a dual-RSI trend filter."""
    if rv > 0 and iv < rv * (1 + thresholds['min_iv_rv_premium']):
        return {'action': 'HOLD', 'reason': f"IV ({iv:.1%}) is not sufficiently above RV ({rv:.1%}). No statistical edge."}
    
    if rsi_daily > thresholds['rsi_overbought'] and rsi_hourly > 50:
        return {'action': 'SELL_PUT', 'reason': f"Confirmed Bullish Trend (Daily RSI: {rsi_daily:.1f}, Hourly > 50). Selling puts captures premium while retaining upside."}
    elif rsi_daily < thresholds['rsi_oversold'] and rsi_hourly < 50:
        return {'action': 'SELL_CALL', 'reason': f"Confirmed Bearish Trend (Daily RSI: {rsi_daily:.1f}, Hourly < 50). Selling calls generates income to hedge falling collateral value."}
    else:
        return {'action': 'SELL_STRANGLE', 'reason': f"Neutral or Conflicted Trend (Daily RSI: {rsi_daily:.1f}). Selling a strangle maximizes premium collection in a range-bound market."}

def determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, ltv, thresholds):
    reasons = [];
    if iv > thresholds['iv_high']: reasons.append(f"High IV ({iv:.1%})")
    if rv > 0 and iv > rv * (1 + thresholds['iv_rv_spread']): reasons.append(f"IV/RV spread > {thresholds['iv_rv_spread']:.0%}")
    if rsi_daily > 70 or rsi_daily < 30: reasons.append(f"Extreme trend (Daily RSI:{rsi_daily:.1f})")
    if abs(daily_funding_rate) > thresholds['funding_rate_high']: reasons.append(f"High daily funding ({daily_funding_rate:.4%})")
    if ltv > thresholds['ltv_high']: reasons.append(f"High LTV ({ltv:.1%})")
    if reasons: return {'hedge_with_perp': True, 'reason': "Tactical hedge is a MUST. Trigger(s): " + " | ".join(reasons)}
    return {'hedge_with_perp': False, 'reason': "Market conditions neutral. Tactical perp hedge not required."}

def find_best_option_to_sell(options_df, option_type, target_delta, min_premium_ratio, live_price, target_dte):
    if options_df.empty: return None
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs(); closest_dte = options_df.loc[options_df['dte_diff'].idxmin()]['dte']; target_expiry_options = options_df[np.isclose(options_df['dte'], closest_dte)].copy()
    with st.spinner(f"Scanning {len(target_expiry_options)} options..."):
        tickers = {row['instrument_name']: get_instrument_ticker(row['instrument_name']) for _, row in target_expiry_options.iterrows()}
        target_expiry_options['ticker_data'] = target_expiry_options['instrument_name'].map(tickers)
    target_expiry_options.dropna(subset=['ticker_data'], inplace=True); target_expiry_options['delta'] = target_expiry_options['ticker_data'].apply(lambda x: x.get('delta')); target_expiry_options['price'] = target_expiry_options['ticker_data'].apply(lambda x: x.get('mark_price')); target_expiry_options.dropna(subset=['delta', 'price'], inplace=True)
    candidates = target_expiry_options[target_expiry_options['option_type'] == option_type].copy()
    if live_price > 0: candidates = candidates[candidates['price'] / live_price >= min_premium_ratio]
    if candidates.empty: return None
    candidates['delta_diff'] = (candidates['delta'].abs() - target_delta).abs()
    return candidates.loc[candidates['delta_diff'].idxmin()].to_dict()

# =====================================================================================
# ==                              UI & APP LAYOUT                                  ==
# =====================================================================================

st.title("Autonomint Quant Strategy Optimizer")
with st.sidebar:
    st.header("1. Core Position"); ETH_DEPOSITED = st.number_input("ETH Deposited", 1.0, 10.0, 2.0, 0.5); ETH_PRICE_INITIAL = st.number_input("Initial ETH Price ($)", 1000.0, 10000.0, 2000.0, 100.0); AAVE_APY = st.slider("AAVE Supply APY (%)", 0.1, 10.0, 3.0, 0.1) / 100.0; LTV = st.slider("Loan-to-Value (LTV) (%)", 50.0, 95.0, 80.0, 1.0) / 100.0
    st.header("2. dCDS Hedge Parameters"); DCDS_COVERAGE_PERCENT = st.slider("Downside Coverage (%)", 10., 50., 20., 1.)/ 100.0; DCDS_FEE_PERCENT = st.slider("Upfront Fee (% of hedged value)", 1., 20., 12., 0.5) / 100.0; DCDS_UPSIDE_SHARING_PERCENT = st.slider("Upside Sharing Cost (%)", 0., 10., 3., 0.5) / 100.0
    st.header("3. Option Execution Criteria"); TARGET_DTE = st.slider("Target Days to Expiry (DTE)", 7, 60, 30, 1); TARGET_DELTA = st.slider("Target Delta", 0.10, 0.45, 0.35, 0.01); MIN_PREMIUM_RATIO = st.slider("Min Premium-to-Spot Ratio (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
    st.header("4. Strategy Engine Thresholds"); col1, col2 = st.columns(2)
    with col1: st.markdown("**Regime Definition**"); MIN_IV_RV_PREMIUM = col1.slider("Min IV > RV Edge (%)", 0, 20, 5, 1) / 100.0; RSI_OVERBOUGHT = col1.slider("Daily RSI Overbought", 60, 75, 65); RSI_OVERSOLD = col1.slider("Daily RSI Oversold", 25, 40, 35)
    with col2: st.markdown("**Perp Hedge Triggers**"); IV_HIGH = col2.slider("High IV Threshold (%)", 50., 100., 80., 1.) / 100.0; FUNDING_HIGH = col2.slider("High Daily Funding Rate (%)", 0.05, 0.3, 0.15) / 100.0; IV_RV_SPREAD = col2.slider("IV > RV by (%)", 5., 50., 25., 1.) / 100.0
    thresholds = {'min_iv_rv_premium': MIN_IV_RV_PREMIUM, 'rsi_overbought': RSI_OVERBOUGHT, 'rsi_oversold': RSI_OVERSOLD, 'iv_high': IV_HIGH, 'iv_rv_spread': IV_RV_SPREAD, 'ltv_high': 85, 'funding_rate_high': FUNDING_HIGH}

with st.spinner("Fetching all live market data..."):
    live_eth_price = fetch_live_eth_price(); daily_funding_rate = get_thalex_actual_daily_funding_rate('ETH'); all_options = get_all_options_data()
    daily_historical_df = fetch_historical_prices(days_lookback=50, timeframe='1d'); hourly_historical_df = fetch_historical_prices(days_lookback=7, timeframe='1h')
if not live_eth_price or daily_funding_rate is None or daily_historical_df.empty or hourly_historical_df.empty or all_options.empty: st.error("Failed to fetch critical market data."); st.stop()

rv = calculate_rv(daily_historical_df['mark_price_close']); rsi_daily = calculate_rsi(daily_historical_df['mark_price_close']); rsi_hourly = calculate_rsi(hourly_historical_df['mark_price_close']); iv = fetch_atm_iv(all_options, TARGET_DTE, live_eth_price)

st.header("Live Market Dashboard"); mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5); mcol1.metric("Live ETH Price", f"${live_eth_price:,.2f}"); mcol2.metric("30D Realized Volatility (RV)", f"{rv:.2%}"); mcol3.metric(f"~{TARGET_DTE}-Day ATM IV", f"{iv:.2%}"); mcol4.metric("Daily RSI", f"{rsi_daily:.1f}"); mcol5.metric("Hourly RSI", f"{rsi_hourly:.1f}")

st.header("Optimal Strategy Recommendation")
optimal_strategy = generate_optimal_strategy(iv, rv, rsi_daily, rsi_hourly, thresholds)
st.info(f"**Optimal Strategy: {optimal_strategy['action'].replace('_', ' ').title()}**\n\n*Reasoning: {optimal_strategy['reason']}*", icon="💡")

sold_put, sold_call, hedge_with_perp = None, None, False
if optimal_strategy['action'] != 'HOLD':
    with st.spinner("Searching for best options to execute strategy..."):
        if optimal_strategy['action'] in ['SELL_PUT', 'SELL_STRANGLE']: sold_put = find_best_option_to_sell(all_options, 'put', TARGET_DELTA, MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)
        if optimal_strategy['action'] in ['SELL_CALL', 'SELL_STRANGLE']: sold_call = find_best_option_to_sell(all_options, 'call', TARGET_DELTA, MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)
    perp_decision = determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, LTV, thresholds); hedge_with_perp = perp_decision['hedge_with_perp']

st.subheader("Actionable Trade(s)")
if optimal_strategy['action'] == 'HOLD': st.success("No compelling trade setup found. The optimal action is to hold the base position and wait.")
elif not sold_put and not sold_call: st.warning(f"Engine recommended to **{optimal_strategy['action'].replace('_', ' ')}**, but no option was found that meets your specific Delta and Premium criteria.")
else:
    if hedge_with_perp: st.success(f"**Tactical Action: Add a 1x Short Perpetual Hedge.**\n\n*Reasoning: {perp_decision['reason']}*")
    else: st.info(f"**Tactical Action: No Perpetual Hedge Needed.**\n\n*Reasoning: {perp_decision['reason']}*")
    put_col, call_col = st.columns(2)
    with put_col:
        if sold_put: st.metric("Sell Put Strike", f"${sold_put['strike']:.0f}", f"Premium: ${sold_put['price']:.2f}")
    with call_col:
        if sold_call: st.metric("Sell Call Strike", f"${sold_call['strike']:.0f}", f"Premium: ${sold_call['price']:.2f}")

st.header("Position Payoff Analysis")
params = {'eth_deposited': ETH_DEPOSITED, 'eth_price_initial': ETH_PRICE_INITIAL, 'aave_apy': AAVE_APY, 'daily_funding_rate': daily_funding_rate, 'dcds_coverage_percent': DCDS_COVERAGE_PERCENT, 'dcds_fee_percent': DCDS_FEE_PERCENT, 'dcds_upside_sharing_percent': DCDS_UPSIDE_SHARING_PERCENT}
pcol1, pcol2 = st.columns([1,2])
with pcol1:
    eth_price_final = st.slider("Set Target ETH Price ($) for PnL Breakdown", int(ETH_PRICE_INITIAL * 0.5), int(ETH_PRICE_INITIAL * 1.5), int(ETH_PRICE_INITIAL))
    total_pnl, pnl_underlying, pnl_aave, pnl_dcds, pnl_option, pnl_perp = calculate_final_pnl(eth_price_final, params, sold_put, sold_call, hedge_with_perp)
    st.metric("Total Projected PnL at Target Price", f"${total_pnl:,.2f}")
    with st.expander("Show PnL Contribution Breakdown"):
        st.metric("PnL from Underlying ETH", f"${pnl_underlying:,.2f}", delta_color="off"); st.metric("PnL from AAVE Yield", f"${pnl_aave::.2f}"); st.metric("PnL from dCDS (Net)", f"${pnl_dcds:,.2f}"); st.metric("PnL from Sold Options", f"${pnl_option:,.2f}"); st.metric("PnL from Perpetual Hedge", f"${pnl_perp:,.2f}")
with pcol2:
    price_range = np.linspace(ETH_PRICE_INITIAL * 0.6, ETH_PRICE_INITIAL * 1.4, 200)
    pnl_values = [calculate_final_pnl(p, params, sold_put, sold_call, hedge_with_perp)[0] for p in price_range]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3))); fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even"); fig.add_vline(x=eth_price_final, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${total_pnl:,.2f}", annotation_position="top right"); fig.add_vline(x=ETH_PRICE_INITIAL, line_width=1, line_dash="dot", line_color="grey", annotation_text="Initial Price", annotation_position="bottom right")
    title = f"Payoff: dCDS + {optimal_strategy['action'].replace('_', ' ').title()}{' + Perp Hedge' if hedge_with_perp else ''}"
    fig.update_layout(title=title, xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
