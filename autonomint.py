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

st.set_page_config(
    page_title="Autonomint Strategy Analyzer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Autonomint: Interactive Strategy Analysis")
st.markdown("An interactive dashboard to analyze and monitor yield strategies based on market conditions.")

# --- API Configuration ---
BASE_URL = "https://thalex.com/api/v2/public"
INSTRUMENTS_ENDPOINT = "instruments"
URL_INSTRUMENTS = f"{BASE_URL}/{INSTRUMENTS_ENDPOINT}"
TICKER_ENDPOINT = "ticker"
URL_TICKER = f"{BASE_URL}/{TICKER_ENDPOINT}"
REQUEST_TIMEOUT = 15
API_DELAY_TICKER = 0.4

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
            logging.error(f"API call to {func.__name__} failed after {max_retries} retries.")
            return None
        return wrapper
    return decorator

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

@st.cache_data(ttl=3600)
@with_retries()
def fetch_historical_data(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 30, timeframe='1d'):
    try:
        exchange = getattr(ccxt, exchange_id)();
        if not exchange.has['fetchOHLCV']: return pd.DataFrame()
        since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
        ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=days_lookback + 5)
        if not ohlcv: return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume'])
        df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True)
        return df
    except Exception as e:
        logging.error(f"CCXT fetch failed: {e}"); return pd.DataFrame()

def calculate_realized_volatility(prices: pd.Series) -> float:
    if len(prices) < 2: return 0.0
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.std() * np.sqrt(365)

def calculate_health(eth_price_current: float, debt: float, eth_deposited: float) -> float:
    if debt == 0: return float('inf')
    return (eth_deposited * eth_price_current) / debt

# =====================================================================================
# ==               NEW, ROBUST DATA LOADING AND PARSING LOGIC                      ==
# =====================================================================================

def _calculate_dte(expiry_str: str, current_date_utc: datetime) -> float | None:
    try:
        expiry_dt_obj = datetime.strptime(expiry_str, "%d%b%y").replace(hour=8, minute=0, tzinfo=timezone.utc)
        if expiry_dt_obj <= current_date_utc: return None
        time_to_expiry = expiry_dt_obj - current_date_utc
        return time_to_expiry.days + (time_to_expiry.seconds / (24 * 3600))
    except (ValueError, TypeError):
        return None

@st.cache_data(ttl=300)
def get_clean_options_df():
    instruments = fetch_instruments()
    if not instruments:
        st.error("Failed to fetch market instruments.")
        return None

    now_utc = datetime.now(timezone.utc)
    parsed_options = []
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
                parsed_options.append({
                    'instrument_name': name,
                    'strike': float(strike_str),
                    'option_type': 'call' if type_char == 'C' else 'put',
                    'dte': dte
                })
    
    if not parsed_options:
        st.error("No valid future options could be parsed from the market data.")
        return None

    return pd.DataFrame(parsed_options)

# =====================================================================================
# ==                            PAYOFF VISUALIZATION                             ==
# =====================================================================================

def create_payoff_diagram(
    strategy: str,
    price_range: np.ndarray,
    eth_deposited: float,
    eth_price_initial: float,
    aave_yield: float,
    call_strike: float | None,
    call_price: float | None,
    put_strike: float | None,
    put_price: float | None,
    dcds_premium_rate: float | None,
    predicted_price: float
):
    """Generates an interactive Plotly payoff diagram for the given strategy."""
    collateral_value_initial = eth_deposited * eth_price_initial
    # PnL from holding the underlying ETH
    underlying_pnl = (price_range * eth_deposited) - collateral_value_initial
    
    # Calculate PnL based on the selected strategy
    if strategy == "Bullish 🐂":
        # PnL from the long call option
        option_pnl = np.maximum(0, price_range - call_strike) - call_price
        total_pnl = underlying_pnl + aave_yield + option_pnl
        title_text = "Payoff: ETH Holdings + AAVE Yield + Long Call"
    
    elif strategy == "Bearish / Neutral 🐻":
        # PnL from the long put option
        option_pnl = np.maximum(0, put_strike - price_range) - put_price
        # Payout from dCDS (using the app's formula)
        dcds_payout = np.maximum(0, (eth_price_initial - price_range) * eth_deposited * (1 - dcds_premium_rate))
        total_pnl = underlying_pnl + aave_yield + option_pnl + dcds_payout
        title_text = "Payoff: ETH Holdings + AAVE Yield + Long Put + dCDS"
    else:
        return None

    # Create the Plotly figure
    fig = go.Figure()

    # Add the main payoff trace
    fig.add_trace(go.Scatter(
        x=price_range, y=total_pnl, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3)
    ))

    # Add a zero line for break-even reference
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even")

    # Add a vertical line for the user's predicted price
    predicted_pnl_index = np.argmin(np.abs(price_range - predicted_price))
    predicted_pnl = total_pnl[predicted_pnl_index]
    fig.add_vline(x=predicted_price, line_width=2, line_dash="dot", line_color="orange",
                  annotation_text=f"Predicted PnL: ${predicted_pnl:,.0f}",
                  annotation_position="top right")

    # Style the chart
    fig.update_layout(
        title=title_text,
        xaxis_title="ETH Price at Expiry ($)",
        yaxis_title="Overall Profit / Loss ($)",
        yaxis_tickprefix='$',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

# =====================================================================================
# ==                          SIDEBAR FOR USER INPUTS                              ==
# =====================================================================================

with st.sidebar:
    st.header("⚙️ Strategy Configuration")
    ETH_DEPOSITED = st.number_input("ETH Deposited", min_value=0.1, value=2.0, step=0.1, format="%.1f")
    ETH_PRICE_INITIAL = st.number_input("Initial ETH Price (at deposit)", min_value=1.0, value=3600.0, step=50.0)

    st.header("📈 Market & Hedging")
    expiry_choice = st.selectbox(
        "Select Expiry Horizon", ['30 Days', '60 Days', '90 Days'],
        help="The app will find the closest available market expiry to your target."
    )
    TARGET_DTE = int(expiry_choice.split(' ')[0])
    
    AAVE_APY_PERCENT = st.slider("AAVE Supply APY", 0.0, 10.0, 3.0, format="%.2f%%")
    DCDS_PREMIUM_PERCENT = st.slider("dCDS Premium Rate", 0.0, 15.0, 5.0, format="%.2f%%")
    AAVE_APY = AAVE_APY_PERCENT / 100.0
    DCDS_PREMIUM_RATE = DCDS_PREMIUM_PERCENT / 100.0

    st.header("🧠 Strategy Parameters")
    vol_premium_percent = st.slider(
        "IV Premium Threshold (%)", 1, 50, 10,
        help="How much higher (in %) must IV be than RV to consider options 'expensive'?"
    )
    VOLATILITY_PREMIUM_THRESHOLD = 1 + (vol_premium_percent / 100.0)
    
# =====================================================================================
# ==                      MAIN APP LOGIC & DISPLAY                                 ==
# =====================================================================================

@st.cache_data(ttl=120)
def fetch_live_eth_price(exchange_id: str = 'kraken', symbol: str = 'ETH/USD'):
    try:
        exchange = getattr(ccxt, exchange_id)(); ticker = exchange.fetch_ticker(symbol)
        return ticker.get('last')
    except Exception as e:
        logging.error(f"Could not fetch live ETH price: {e}"); return None

# --- Step 1: Fetch all necessary data ---
live_eth_price = fetch_live_eth_price()
if not live_eth_price:
    st.error("Could not fetch live ETH price. App cannot continue."); st.stop()
    
options_df = get_clean_options_df()
if options_df is None: st.stop()
    
historical_df = fetch_historical_data()
realized_vol = calculate_realized_volatility(historical_df['mark_price_close']) if not historical_df.empty else 0.0

# --- Step 2: Select best options based on LIVE price and fetch their tickers ---
with st.spinner(f"Finding options for ~{TARGET_DTE} day expiry..."):
    best_expiry_dte = options_df.iloc[(options_df['dte'] - TARGET_DTE).abs().idxmin()]['dte']
    
    if abs(best_expiry_dte - TARGET_DTE) > 20:
        st.error(f"No suitable options found near {TARGET_DTE} days. Closest is ~{best_expiry_dte:.0f} days away.")
        st.stop()

    expiry_df = options_df[np.isclose(options_df['dte'], best_expiry_dte)].copy()
    st.info(f"Using options chain for expiry ~{best_expiry_dte:.0f} days from now.", icon="🗓️")
    
    puts_df, calls_df = expiry_df[expiry_df['option_type'] == 'put'], expiry_df[expiry_df['option_type'] == 'call']
    
    best_put_row, best_call_row = None, None
    if not puts_df.empty: best_put_row = puts_df.loc[(puts_df['strike'] - live_eth_price).abs().idxmin()]
    if not calls_df.empty: best_call_row = calls_df.loc[(calls_df['strike'] - live_eth_price).abs().idxmin()]

    put_price, put_iv, STRIKE_PRICE_PUT = None, None, None
    if best_put_row is not None:
        ticker = fetch_ticker(best_put_row['instrument_name'])
        if ticker:
            put_price, put_iv = ticker.get('mark_price'), ticker.get('iv')
            STRIKE_PRICE_PUT = best_put_row['strike']
            st.info(f"Selected closest PUT to live price: Strike ${STRIKE_PRICE_PUT:,.0f}", icon="🎯")

    call_price, call_iv, STRIKE_PRICE_CALL = None, None, None
    if best_call_row is not None:
        ticker = fetch_ticker(best_call_row['instrument_name'])
        if ticker:
            call_price, call_iv = ticker.get('mark_price'), ticker.get('iv')
            STRIKE_PRICE_CALL = best_call_row['strike']
            st.info(f"Selected closest CALL to live price: Strike ${STRIKE_PRICE_CALL:,.0f}", icon="🎯")

if put_price is None or call_price is None:
    st.error("Could not fetch valid option prices. The market may be illiquid. Please try again."); st.stop()

# --- Step 3: Consolidate final data ---
all_ivs = [iv for iv in [put_iv, call_iv] if iv]
for i, iv in enumerate(all_ivs):
    if iv > 1.5: all_ivs[i] = iv / 100.0
implied_vol = np.mean(all_ivs) if all_ivs else 0.0
initial_slider_value = live_eth_price

# --- Step 4: Display UI and run calculations ---
st.divider()
st.metric("Live ETH Price (Kraken)", f"${live_eth_price:,.2f}")

st.header("Interactive Analysis")
st.markdown("_Adjust the sliders below to run what-if scenarios based on your market outlook._")
slider_col1, slider_col2 = st.columns(2)
eth_price_current = slider_col1.slider("ETH Price ($) for Health Check", float(initial_slider_value * 0.5), float(initial_slider_value * 2.0), float(initial_slider_value), 10.0)
eth_price_predicted = slider_col2.slider("Your Predicted ETH Price ($) at Expiry", float(initial_slider_value * 0.5), float(initial_slider_value * 2.0), float(initial_slider_value * 1.1), 10.0)

collateral_value_initial = ETH_DEPOSITED * ETH_PRICE_INITIAL
usda_minted = collateral_value_initial * 0.8
health = calculate_health(eth_price_current, usda_minted, ETH_DEPOSITED)
aave_yield_monthly = (AAVE_APY / 12) * collateral_value_initial

st.divider()
st.subheader("📊 Live Position Monitoring")
health_col, collateral_col, yield_col = st.columns(3)
health_col.metric("🛡️ Health Factor", f"{health:.2f}", help="Ratio of collateral value to debt. Liquidation at ≤ 1.0.")
if health <= 1.2: st.error("🔥 DANGER: Health is critically low. LIQUIDATION IMMINENT.")
elif health <= 1.5: st.warning("⚠️ WARNING: Health is low. Monitor your position.")
collateral_col.metric("Initial Collateral Value", f"${collateral_value_initial:,.2f}")
yield_col.metric("💰 AAVE Yield (Monthly)", f"${aave_yield_monthly:.2f}")

st.divider()
st.subheader("🧠 Recommended Strategy & PnL Projection")

options_are_expensive = implied_vol > realized_vol * VOLATILITY_PREMIUM_THRESHOLD if realized_vol > 0 else False
user_is_bullish = eth_price_predicted > eth_price_current
is_bullish_strategy = user_is_bullish and not options_are_expensive
strategy_name = "Bullish 🐂" if is_bullish_strategy else "Bearish / Neutral 🐻"

st.info(f"**Recommended Strategy:** {strategy_name}", icon="💡")

if strategy_name == "Bullish 🐂":
    st.markdown("**Action:** Re-leverage assets. Consider buying a Call option for upside.")
    abond_redeemed_eth = ETH_DEPOSITED + (aave_yield_monthly / eth_price_current if eth_price_current > 0 else 0)
    call_pnl = max(0, eth_price_predicted - STRIKE_PRICE_CALL) - call_price
    final_asset_value = abond_redeemed_eth * eth_price_predicted * 0.97
    profit = (final_asset_value - collateral_value_initial) + aave_yield_monthly + call_pnl
    apy = (profit / collateral_value_initial) * 12 * 100 if collateral_value_initial > 0 else 0
    pnl_col1, pnl_col2 = st.columns(2)
    pnl_col1.metric("Projected Profit", f"${profit:,.2f}", f"{apy:.2f}%% APY")
    pnl_col2.metric("Call Option PnL", f"${call_pnl:,.2f}")
else: # Bearish / Neutral Strategy
    st.markdown("**Action:** De-leverage, protect with dCDS, and buy a Put option.")
    dcds_payout = max(0, (ETH_PRICE_INITIAL - eth_price_predicted) * ETH_DEPOSITED * (1 - DCDS_PREMIUM_RATE))
    put_pnl = max(0, STRIKE_PRICE_PUT - eth_price_predicted) - put_price
    final_asset_value = eth_price_predicted * ETH_DEPOSITED
    profit = (final_asset_value - collateral_value_initial) + aave_yield_monthly + dcds_payout + put_pnl
    apy = (profit / collateral_value_initial) * 12 * 100 if collateral_value_initial > 0 else 0
    pnl_col1, pnl_col2, pnl_col3 = st.columns(3)
    pnl_col1.metric("Projected Profit", f"${profit:,.2f}", f"{apy:.2f}%% APY")
    pnl_col2.metric("dCDS Payout (net)", f"${dcds_payout:,.2f}")
    pnl_col3.metric("Put Option PnL", f"${put_pnl:,.2f}")

# --- NEW: Payoff Diagram Visualization ---
st.subheader("📈 Payoff Diagram at Expiry")
price_range = np.linspace(eth_price_current * 0.4, eth_price_current * 1.6, num=200)

payoff_fig = create_payoff_diagram(
    strategy=strategy_name,
    price_range=price_range,
    eth_deposited=ETH_DEPOSITED,
    eth_price_initial=ETH_PRICE_INITIAL,
    aave_yield=aave_yield_monthly,
    call_strike=STRIKE_PRICE_CALL,
    call_price=call_price,
    put_strike=STRIKE_PRICE_PUT,
    put_price=put_price,
    dcds_premium_rate=DCDS_PREMIUM_RATE,
    predicted_price=eth_price_predicted
)

if payoff_fig:
    st.plotly_chart(payoff_fig, use_container_width=True)
else:
    st.warning("Could not generate a payoff diagram for the selected strategy.")

with st.expander("Show Underlying Volatility & Options Data"):
    st.write({
        "Consolidated Implied Volatility (IV)": f"{implied_vol:.2%}" if implied_vol > 0 else "N/A",
        "30-Day Realized Volatility (RV)": f"{realized_vol:.2%}" if realized_vol > 0 else "N/A",
        f"Fetched Put Price (Strike ${STRIKE_PRICE_PUT:,.0f})": f"${put_price:.2f}",
        f"Fetched Call Price (Strike ${STRIKE_PRICE_CALL:,.0f})": f"${call_price:.2f}",
    })
