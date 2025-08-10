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

st.set_page_config(page_title="Autonomint Quant Strategy", page_icon="📈", layout="wide")
st.title("📈 Autonomint: Delta-Neutral Yield Strategy Engine")
st.markdown("An implementation of the delta-neutral strategy combining dCDS protection, option selling, and dynamic perpetual future hedging based on live market conditions.")

# =====================================================================================
# ==                      HELPER & DATA FETCHING FUNCTIONS                         ==
# =====================================================================================

# (with_retries decorator remains the same)
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

@st.cache_data(ttl=300)
@with_retries()
def fetch_funding_rate(exchange_id: str = 'binance', symbol: str = 'ETH/USDT'):
    exchange = getattr(ccxt, exchange_id)(); markets = exchange.load_markets()
    market = exchange.market(f"{symbol}:USDT") # For Binance perpetuals
    funding_rate_data = exchange.fetch_funding_rate(market['id'])
    return funding_rate_data.get('fundingRate')

@st.cache_data(ttl=900)
@with_retries()
def fetch_historical_prices(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 40, timeframe='1d'):
    exchange = getattr(ccxt, exchange_id)(); limit = days_lookback + 5
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv: return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume'])
    df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True)
    return df

def calculate_rv(prices: pd.Series, window: int = 30) -> float:
    log_returns = np.log(prices / prices.shift(1)); return log_returns.rolling(window=window).std().iloc[-1] * np.sqrt(365)

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss; return 100 - (100 / (1 + rs.iloc[-1]))

# =====================================================================================
# ==                          STRATEGY SIMULATION & LOGIC                          ==
# =====================================================================================

def simulate_black_scholes_premium(S, K, T, r, sigma, option_type='call'):
    """A simplified B-S model to estimate premium for display purposes."""
    from scipy.stats import norm
    if T <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else: # put
        price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return price

def determine_hedging_strategy(iv, rv, rsi, funding_rate, ltv, thresholds):
    """Codifies the logic from the research paper to decide on using perps."""
    reasons = []
    # --- "MUST HEDGE" Conditions ---
    if iv > thresholds['iv_high']:
        reasons.append(f"High IV ({iv:.1%}) exceeds threshold ({thresholds['iv_high']:.1%}).")
    if iv > rv * (1 + thresholds['iv_rv_spread']):
        reasons.append(f"IV/RV spread is > {thresholds['iv_rv_spread']:.0%} (IV: {iv:.1%}, RV: {rv:.1%}).")
    if rsi > thresholds['rsi_high'] or rsi < thresholds['rsi_low']:
        reasons.append(f"Strong trend detected (RSI: {rsi:.1f}).")
    if abs(funding_rate) > thresholds['funding_rate_high']:
        reasons.append(f"High funding rate ({funding_rate:.4%}) offers hedging yield.")
    if ltv > thresholds['ltv_high']:
        reasons.append(f"High LTV ({ltv:.1%}) increases liquidation risk.")
    
    if reasons:
        return {'hedge_with_perp': True, 'reason': "Perp hedge is a MUST. Trigger(s): " + " | ".join(reasons)}

    # --- "NOT NECESSARY" Conditions (for context) ---
    reason_skip = "Perp hedge not necessary. "
    if rv < thresholds['rv_low'] and iv < thresholds['iv_low']:
        reason_skip += f"Low volatility regime (RV: {rv:.1%}, IV: {iv:.1%})."
    elif thresholds['rsi_low'] <= rsi <= thresholds['rsi_high']:
        reason_skip += "Market is range-bound (RSI is neutral)."
    else:
        reason_skip += "No strong signal met."
        
    return {'hedge_with_perp': False, 'reason': reason_skip}

def calculate_final_pnl(eth_price_final, params, strategy):
    """Calculates the PnL of the entire combined strategy."""
    # 1. AAVE Yield
    aave_yield = params['eth_deposited'] * params['eth_price_initial'] * (params['aave_apy'] / 12)
    # 2. dCDS Payout
    dcds_payout = max(0, params['eth_deposited'] * params['eth_price_initial'] * params['dcds_coverage'] - (params['eth_deposited'] * (params['eth_price_initial'] - eth_price_final)))
    dcds_cost = params['eth_deposited'] * params['eth_price_initial'] * params['dcds_coverage'] * params['dcds_premium']
    net_dcds_pnl = dcds_payout - dcds_cost
    # 3. Option PnL
    if strategy['option_type'] == 'put':
        option_pnl = strategy['premium'] - max(0, strategy['strike'] - eth_price_final)
    else: # call
        option_pnl = strategy['premium'] - max(0, eth_price_final - strategy['strike'])
    # 4. Perpetual Future PnL
    perp_pnl = 0
    if strategy['hedge_with_perp']:
        funding_pnl = params['eth_price_initial'] * strategy['funding_rate'] * 30 # Simplified 30-day estimate
        price_change_pnl = (params['eth_price_initial'] - eth_price_final) # PnL from 1x short ETH
        perp_pnl = funding_pnl + price_change_pnl
    # 5. Underlying ETH PnL
    underlying_pnl = (eth_price_final - params['eth_price_initial']) * params['eth_deposited']
    
    total_pnl = underlying_pnl + aave_yield + net_dcds_pnl + option_pnl + perp_pnl
    return total_pnl, aave_yield, net_dcds_pnl, option_pnl, perp_pnl

# =====================================================================================
# ==                          SIDEBAR FOR USER INPUTS                              ==
# =====================================================================================

with st.sidebar:
    st.header("⚙️ Core Configuration")
    ETH_DEPOSITED = st.number_input("ETH Deposited", 1.0, 10.0, 2.0, 0.5)
    ETH_PRICE_INITIAL = st.number_input("Initial ETH Price ($)", 1000.0, 10000.0, 2000.0, 100.0)

    st.header("📜 Protocol Parameters (Simulated)")
    AAVE_APY = st.slider("AAVE Supply APY (%)", 0.1, 10.0, 3.0, 0.1) / 100.0
    DCDS_COVERAGE = st.slider("dCDS Downside Coverage (%)", 10.0, 50.0, 20.0, 1.0) / 100.0
    DCDS_PREMIUM = st.slider("dCDS Premium (% of covered amount)", 1.0, 15.0, 6.0, 0.5) / 100.0
    LTV = st.slider("Loan-to-Value (LTV) (%)", 50.0, 95.0, 80.0, 1.0) / 100.0

    st.header("🧠 Strategy Decision Thresholds")
    option_type = st.radio("Option to Sell", ["Put (Upside Retention)", "Call (Capped Upside)"], horizontal=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Volatility**")
        IV_HIGH = col1.slider("High IV Threshold (%)", 50., 100., 70., 1.) / 100.0
        IV_LOW = col1.slider("Low IV Threshold (%)", 20., 60., 50., 1.) / 100.0
        RV_LOW = col1.slider("Low RV Threshold (%)", 10., 50., 40., 1.) / 100.0
        IV_RV_SPREAD = col1.slider("IV/RV Spread Trigger (%)", 5., 50., 20., 1.) / 100.0
    with col2:
        st.markdown("**Trend & Funding**")
        RSI_HIGH = col2.slider("High RSI (Overbought)", 60, 80, 70)
        RSI_LOW = col2.slider("Low RSI (Oversold)", 20, 40, 30)
        FUNDING_HIGH = col2.slider("High Funding Rate Trigger (%)", 0.05, 0.3, 0.15) / 100.0
        LTV_HIGH = col2.slider("High LTV Trigger (%)", 75, 95, 80) / 100.0

    thresholds = {'iv_high': IV_HIGH, 'iv_low': IV_LOW, 'rv_low': RV_LOW, 'iv_rv_spread': IV_RV_SPREAD, 
                  'rsi_high': RSI_HIGH, 'rsi_low': RSI_LOW, 'funding_rate_high': FUNDING_HIGH, 'ltv_high': LTV_HIGH}

# =====================================================================================
# ==                      MAIN APP LOGIC & DISPLAY                                 ==
# =====================================================================================

# --- Step 1: Fetch all live market data ---
with st.spinner("Fetching live market data (Price, Funding, Volatility, RSI)..."):
    live_eth_price = fetch_live_eth_price()
    funding_rate = fetch_funding_rate()
    historical_df = fetch_historical_prices(days_lookback=40)

if not live_eth_price or funding_rate is None or historical_df.empty:
    st.error("Failed to fetch critical market data. Please try again later."); st.stop()

# --- Step 2: Calculate metrics from data ---
rv = calculate_rv(historical_df['mark_price_close'])
rsi = calculate_rsi(historical_df['mark_price_close'])
# For demonstration, we'll use a mock IV. A real app would get this from an options API.
iv = st.number_input("Implied Volatility (IV) (%) - (Set for Simulation)", 20.0, 150.0, 67.0, 1.0) / 100.0

# --- Step 3: Display Market State ---
st.header("📊 Live Market Conditions")
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Live ETH Price", f"${live_eth_price:,.2f}")
mcol2.metric("30D Realized Volatility (RV)", f"{rv:.2%}")
mcol3.metric("Funding Rate (Annualized)", f"{funding_rate*3*365:.2%}", help=f"Daily: {funding_rate:.4%}")
mcol4.metric("14D RSI", f"{rsi:.1f}")

# --- Step 4: Run the Decision Engine ---
st.header("💡 Strategy Recommendation Engine")
strategy_decision = determine_hedging_strategy(iv, rv, rsi, funding_rate, LTV, thresholds)

if strategy_decision['hedge_with_perp']:
    st.success(f"**Recommendation: Sell {option_type.split(' ')[0]} + Hedge with 1x Short Perpetual**")
else:
    st.info(f"**Recommendation: Sell {option_type.split(' ')[0]} (No Perp Hedge Needed)**")
st.caption(f"Reasoning: {strategy_decision['reason']}")

# --- Step 5: Setup and Visualize PnL ---
st.header("💸 Payoff Analysis")
strike_price_put = ETH_PRICE_INITIAL * (1 - DCDS_COVERAGE)
strike_price_call = ETH_PRICE_INITIAL * 1.2 # Example: 20% OTM call
strike = strike_price_put if option_type == 'put' else strike_price_call

# Simulate a realistic premium for the chosen option
premium = simulate_black_scholes_premium(live_eth_price, strike, 30/365, 0.02, iv, option_type.split(' ')[0].lower())

current_strategy = {
    'option_type': option_type.split(' ')[0].lower(),
    'strike': strike,
    'premium': premium,
    'hedge_with_perp': strategy_decision['hedge_with_perp'],
    'funding_rate': funding_rate,
}
params = {
    'eth_deposited': ETH_DEPOSITED, 'eth_price_initial': ETH_PRICE_INITIAL, 'aave_apy': AAVE_APY,
    'dcds_coverage': DCDS_COVERAGE, 'dcds_premium': DCDS_PREMIUM,
}

pcol1, pcol2 = st.columns([1,2])
with pcol1:
    st.markdown(f"**Strategy Details:**")
    st.write(f"🔹 **Selling 1x {option_type.split(' ')[0].capitalize()}**")
    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Strike: `${strike:,.0f}`")
    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;Premium collected: `${premium:,.2f}`")
    if current_strategy['hedge_with_perp']:
        st.write(f"🔹 **Hedging with 1x Short Perpetual**")
    
    eth_price_final = st.slider("Set Target ETH Price ($) for PnL Breakdown", 
                                int(ETH_PRICE_INITIAL * 0.5), int(ETH_PRICE_INITIAL * 1.5), int(ETH_PRICE_INITIAL))
    
    # Calculate PnL for the slider value
    total_pnl, pnl_aave, pnl_dcds, pnl_option, pnl_perp = calculate_final_pnl(eth_price_final, params, current_strategy)
    st.metric("Total Projected PnL at Target Price", f"${total_pnl:,.2f}")
    with st.expander("Show PnL Contribution Breakdown"):
        st.metric("PnL from Underlying ETH", f"${(eth_price_final - ETH_PRICE_INITIAL) * ETH_DEPOSITED:,.2f}", delta_color="off")
        st.metric("PnL from AAVE Yield", f"${pnl_aave:,.2f}")
        st.metric("PnL from dCDS (Net)", f"${pnl_dcds:,.2f}")
        st.metric("PnL from Sold Option", f"${pnl_option:,.2f}")
        st.metric("PnL from Perpetual Hedge", f"${pnl_perp:,.2f}")

with pcol2:
    price_range = np.linspace(ETH_PRICE_INITIAL * 0.6, ETH_PRICE_INITIAL * 1.4, 200)
    pnl_values = [calculate_final_pnl(p, params, current_strategy)[0] for p in price_range]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3)))
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even")
    fig.add_vline(x=eth_price_final, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${total_pnl:,.0f}", annotation_position="top right")
    fig.add_vline(x=ETH_PRICE_INITIAL, line_width=1, line_dash="dot", line_color="grey", annotation_text="Initial Price", annotation_position="bottom right")

    fig.update_layout(
        title=f"Payoff Diagram: {current_strategy['option_type'].capitalize()} Sale {'+ Perp Hedge' if current_strategy['hedge_with_perp'] else ''}",
        xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)
