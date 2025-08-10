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
    if options_df is None or options_df.empty or live_price is None or live_price <= 0: return 0.0
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs(); closest_expiry_row = options_df.loc[options_df['dte_diff'].idxmin()]
    expiry_df = options_df[np.isclose(options_df['dte'], closest_expiry_row['dte'])].copy(); calls_df = expiry_df[expiry_df['option_type'] == 'call']
    if calls_df.empty: return 0.0
    calls_df['strike_diff'] = (calls_df['strike'] - live_price).abs(); atm_call_instrument = calls_df.loc[calls_df['strike_diff'].idxmin()]['instrument_name']
    ticker_data = get_instrument_ticker(atm_call_instrument)
    if ticker_data:
        iv_value = ticker_data.get('iv')
        if iv_value is not None and pd.notna(iv_value):
            try:
                iv_float = float(iv_value)
                normalized_iv = iv_float / 100.0 if iv_float > 1.0 else iv_float
                if 0.05 < normalized_iv < 3.0: return normalized_iv
            except (ValueError, TypeError): return 0.0
    return 0.0

@st.cache_data(ttl=3600)
@with_retries()
def fetch_historical_prices(symbol_pair: str = "ETH/USD", exchange_id: str = 'kraken', days_lookback: int = 365, timeframe='1d'):
    exchange = getattr(ccxt, exchange_id)(); limit = days_lookback + 50
    since = exchange.parse8601((datetime.now(timezone.utc) - timedelta(days=days_lookback)).isoformat())
    ohlcv = exchange.fetch_ohlcv(symbol_pair, timeframe=timeframe, since=since, limit=limit)
    if not ohlcv: return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=['date_time', 'open', 'high', 'low', 'mark_price_close', 'volume']); df['date_time'] = pd.to_datetime(df['date_time'], unit='ms', utc=True); return df

# =====================================================================================
# ==                          CORE CALCULATIONS & METRICS                          ==
# =====================================================================================
def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1: return 50.0
    delta = prices.diff(); gain = delta.where(delta > 0, 0); loss = -delta.where(delta < 0, 0); avg_gain = gain.ewm(com=period - 1, min_periods=period).mean(); avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    if avg_loss.empty or avg_loss.iloc[-1] == 0: return 100.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]; return 100 - (100 / (1 + rs))

def calculate_volatility_rank(historical_prices: pd.DataFrame, window: int = 30) -> dict:
    if historical_prices.empty or len(historical_prices) < 200:
        return {'rank': 50, 'current': 0, 'min': 0, 'max': 0}
    log_returns = np.log(historical_prices['mark_price_close'] / historical_prices['mark_price_close'].shift(1))
    historical_rv = log_returns.rolling(window=window).std() * np.sqrt(365)
    historical_rv.dropna(inplace=True)
    if historical_rv.empty: return {'rank': 50, 'current': 0, 'min': 0, 'max': 0}
    current_vol = historical_rv.iloc[-1]; min_vol_1y = historical_rv.min(); max_vol_1y = historical_rv.max()
    if (max_vol_1y - min_vol_1y) == 0: return {'rank': 50, 'current': current_vol, 'min': min_vol_1y, 'max': max_vol_1y}
    rank = (current_vol - min_vol_1y) / (max_vol_1y - min_vol_1y) * 100
    return {'rank': rank, 'current': current_vol, 'min': min_vol_1y, 'max': max_vol_1y}

def calculate_final_pnl(eth_price_final, params, sold_put, sold_call, hedge_with_perp):
    underlying_pnl = (eth_price_final - params['eth_price_initial']) * params['eth_deposited']; aave_yield = params['eth_price_initial'] * params['eth_deposited'] * (params['aave_apy'] / 12)
    dcds_hedge_strike = params['eth_price_initial'] * (1 - params['dcds_coverage_percent']); hedged_value = params['eth_price_initial'] * params['eth_deposited'] * params['dcds_coverage_percent']; dcds_fee = hedged_value * params['dcds_fee_percent']; dcds_upside_cost = max(0, underlying_pnl) * params['dcds_upside_sharing_percent']; dcds_payout = max(0, dcds_hedge_strike - eth_price_final) * params['eth_deposited']; net_dcds_pnl = dcds_payout - dcds_fee - dcds_upside_cost
    option_pnl = 0
    if sold_put: option_pnl += sold_put['price'] - max(0, sold_put['strike'] - eth_price_final)
    if sold_call: option_pnl += sold_call['price'] - max(0, eth_price_final - sold_call['strike'])
    perp_pnl = 0
    if hedge_with_perp:
        funding_pnl = params['eth_price_initial'] * params['daily_funding_rate'] * 30; price_change_pnl = (params['eth_price_initial'] - eth_price_final); perp_pnl = (funding_pnl + price_change_pnl) * params['eth_deposited']
    total_pnl = underlying_pnl + aave_yield + net_dcds_pnl + option_pnl + perp_pnl
    return total_pnl, underlying_pnl, aave_yield, net_dcds_pnl, option_pnl, perp_pnl

# =====================================================================================
# ==                      STRATEGY ENGINE & DECISION LOGIC                       ==
# =====================================================================================
def determine_market_sentiment(iv, vol_rank_data, rsi_daily, rsi_hourly, thresholds):
    vol_rank = vol_rank_data['rank']; rv = vol_rank_data['current']
    is_vol_high_rank = vol_rank >= thresholds['min_vol_rank']; has_vol_premium = iv > 0 and rv > 0 and iv >= rv * (1 + thresholds['min_iv_rv_premium'])
    is_daily_bullish = rsi_daily > thresholds['rsi_overbought']; is_daily_bearish = rsi_daily < thresholds['rsi_oversold']
    is_hourly_confirming_bullish = rsi_hourly > 50; is_hourly_confirming_bearish = rsi_hourly < 50

    if not is_vol_high_rank:
        sentiment, reason, can_sell_options = 'NEUTRAL', f"Volatility Rank is {vol_rank:.0f}%, below the {thresholds['min_vol_rank']:.0f}% threshold. Premium selling is not attractive as volatility is historically low.", False
    elif not has_vol_premium:
        sentiment, reason, can_sell_options = 'NEUTRAL', f"Implied Volatility ({iv:.1%}) is not sufficiently above Realized Volatility ({rv:.1%}). No statistical edge exists to justify selling options.", False
    else:
        can_sell_options = True
        if is_daily_bullish and is_hourly_confirming_bullish:
            sentiment = 'BULLISH'
            reason = f"Confirmed Uptrend: Daily RSI ({rsi_daily:.1f}) is strong and confirmed by Hourly RSI (>50). With high Volatility Rank ({vol_rank:.0f}%), selling Puts is optimal to collect premium without capping upside."
        elif is_daily_bearish and is_hourly_confirming_bearish:
            sentiment = 'BEARISH'
            reason = f"Confirmed Downtrend: Daily RSI ({rsi_daily:.1f}) is weak and confirmed by Hourly RSI (<50). With high Volatility Rank ({vol_rank:.0f}%), selling Calls is optimal to generate hedging income."
        else:
            sentiment = 'NEUTRAL'
            reason = f"Conflicted or Neutral Trend: Daily RSI ({rsi_daily:.1f}) and Hourly RSI ({rsi_hourly:.1f}) do not show a clear, confirmed trend. With high Volatility Rank ({vol_rank:.0f}%), a Strangle is optimal to harvest premium from range-bound action."
            
    return {'sentiment': sentiment, 'reason': reason, 'can_sell_options': can_sell_options}

def generate_optimal_strategy(market_sentiment_result):
    sentiment = market_sentiment_result['sentiment']; can_sell = market_sentiment_result['can_sell_options']; action = 'HOLD'
    if can_sell:
        if sentiment == 'BULLISH': action = 'SELL_PUT'
        elif sentiment == 'BEARISH': action = 'SELL_CALL'
        elif sentiment == 'NEUTRAL': action = 'SELL_STRANGLE'
    return {'action': action, 'sentiment': sentiment, 'reason': market_sentiment_result['reason']}

def determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, ltv, thresholds):
    reasons = [];
    if iv > thresholds['iv_high']: reasons.append(f"High IV ({iv:.1%})")
    if rv > 0 and iv > rv * (1 + thresholds['iv_rv_spread']): reasons.append(f"IV/RV spread > {thresholds['iv_rv_spread']:.0%}")
    if rsi_daily > 75 or rsi_daily < 25: reasons.append(f"Extreme trend (Daily RSI:{rsi_daily:.1f})")
    if abs(daily_funding_rate) > thresholds['funding_rate_high']: reasons.append(f"High daily funding ({daily_funding_rate:.4%})")
    if ltv > thresholds['ltv_high']: reasons.append(f"High LTV ({ltv:.1%})")
    if reasons: return {'hedge_with_perp': True, 'reason': "Tactical hedge is a MUST. Trigger(s): " + " | ".join(reasons)}
    return {'hedge_with_perp': False, 'reason': "Market conditions neutral. Tactical perp hedge not required."}

def find_best_option_to_sell(options_df, option_type, target_delta, min_premium_ratio, live_price, target_dte):
    if options_df.empty: return None
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs(); closest_dte = options_df.loc[options_df['dte_diff'].idxmin()]['dte']; target_expiry_options = options_df[np.isclose(options_df['dte'], closest_dte)].copy()
    ticker_data_map = {};
    with st.spinner(f"Scanning matching {option_type} options..."):
        for _, row in target_expiry_options.iterrows():
            if row['option_type'] == option_type: ticker_data_map[row['instrument_name']] = get_instrument_ticker(row['instrument_name'])
    candidates_data = []
    for name, data in ticker_data_map.items():
        if data and data.get('delta') is not None and data.get('mark_price') is not None: candidates_data.append({'instrument_name': name, 'delta': data['delta'], 'price': data['mark_price']})
    if not candidates_data: return None
    candidates_df = pd.DataFrame(candidates_data); original_info = options_df[['instrument_name', 'strike', 'option_type', 'dte']].set_index('instrument_name'); candidates_df = candidates_df.join(original_info, on='instrument_name')
    if live_price > 0: candidates_df = candidates_df[candidates_df['price'] / live_price >= min_premium_ratio]
    if candidates_df.empty: return None
    candidates_df['delta_diff'] = (candidates_df['delta'].abs() - target_delta).abs(); return candidates_df.loc[candidates_df['delta_diff'].idxmin()].to_dict()

# =====================================================================================
# ==                              UI & APP LAYOUT                                  ==
# =====================================================================================
st.title("Autonomint Quant Strategy Optimizer")
with st.sidebar:
    st.header("1. Core Position"); ETH_DEPOSITED = st.number_input("ETH Deposited", 1.0, 10.0, 2.0, 0.5); ETH_PRICE_INITIAL = st.number_input("Initial ETH Price ($)", 1000.0, 10000.0, 4254.0, 100.0); AAVE_APY = st.slider("AAVE Supply APY (%)", 0.1, 10.0, 3.0, 0.1) / 100.0; LTV = st.slider("Loan-to-Value (LTV) (%)", 50.0, 95.0, 80.0, 1.0) / 100.0
    st.header("2. dCDS Hedge Parameters"); DCDS_COVERAGE_PERCENT = st.slider("Downside Coverage (%)", 10., 50., 20., 1.)/ 100.0; DCDS_FEE_PERCENT = st.slider("Upfront Fee (% of hedged value)", 1., 20., 12., 0.5) / 100.0; DCDS_UPSIDE_SHARING_PERCENT = st.slider("Upside Sharing Cost (%)", 0., 10., 3., 0.5) / 100.0
    st.header("3. Option Execution Criteria"); TARGET_DTE = st.slider("Target Days to Expiry (DTE)", 7, 60, 30, 1); TARGET_DELTA = st.slider("Target Delta", 0.10, 0.45, 0.35, 0.01); MIN_PREMIUM_RATIO = st.slider("Min Premium-to-Spot Ratio (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
    st.header("4. Strategy Engine Thresholds");
    MIN_VOL_RANK = st.slider("Min Volatility Rank to Sell Premium (%)", 0, 100, 50, help="Only sell options if the current volatility is above this percentile of its 1-year range.")
    col1, col2 = st.columns(2)
    with col1: st.markdown("**Regime Definition**"); MIN_IV_RV_PREMIUM = col1.slider("Min IV > RV Edge (%)", 0, 20, 5, 1) / 100.0; RSI_OVERBOUGHT = col1.slider("Daily RSI Overbought", 60, 80, 70); RSI_OVERSOLD = col1.slider("Daily RSI Oversold", 20, 40, 30)
    with col2: st.markdown("**Perp Hedge Triggers**"); IV_HIGH = col2.slider("High IV Threshold (%)", 50., 120., 80., 1.) / 100.0; FUNDING_HIGH = col2.slider("High Daily Funding Rate (%)", 0.05, 0.3, 0.15) / 100.0; IV_RV_SPREAD = col2.slider("IV > RV by (%)", 5., 50., 25., 1.) / 100.0
    thresholds = {'min_iv_rv_premium': MIN_IV_RV_PREMIUM, 'rsi_overbought': RSI_OVERBOUGHT, 'rsi_oversold': RSI_OVERSOLD, 'iv_high': IV_HIGH, 'iv_rv_spread': IV_RV_SPREAD, 'ltv_high': 0.85, 'funding_rate_high': FUNDING_HIGH, 'min_vol_rank': MIN_VOL_RANK}
    st.header("5. Manual Overrides"); perp_hedge_override = st.selectbox("Perpetual Hedge Strategy", ["Automatic (Recommended)", "Force Short Hedge", "Force No Hedge"], help="Manually override the tactical perpetual hedge recommendation.")

with st.spinner("Fetching all live market data..."):
    live_eth_price = fetch_live_eth_price()
    if live_eth_price is None: st.error("Could not fetch live ETH price. Please refresh."); st.stop()
    daily_funding_rate = get_thalex_actual_daily_funding_rate('ETH'); all_options = get_all_options_data(); 
    yearly_historical_df = fetch_historical_prices(days_lookback=365)
    hourly_historical_df = fetch_historical_prices(days_lookback=7, timeframe='1h')
    if daily_funding_rate is None or yearly_historical_df.empty or hourly_historical_df.empty or all_options.empty: st.error("Failed to fetch critical market data. Please refresh."); st.stop()

vol_rank_data = calculate_volatility_rank(yearly_historical_df)
rv = vol_rank_data['current']
rsi_daily = calculate_rsi(yearly_historical_df['mark_price_close'])
rsi_hourly = calculate_rsi(hourly_historical_df['mark_price_close'])
iv = fetch_atm_iv(all_options, TARGET_DTE, live_eth_price)

st.header("Live Market Dashboard")
row1_col1, row1_col2, row1_col3 = st.columns(3)
row1_col1.metric("Live ETH Price", f"${live_eth_price:,.2f}")
row1_col2.metric("30D Realized Volatility (RV)", f"{rv:.2%}")
iv_display_text = f"{iv:.2%}" if iv > 0 else "N/A"
row1_col3.metric(f"~{TARGET_DTE}-Day ATM IV", iv_display_text)

row2_col1, row2_col2, row2_col3 = st.columns(3)
row2_col1.metric("1Y Volatility Rank", f"{vol_rank_data['rank']:.0f}%", help=f"Current 30D RV is at the {vol_rank_data['rank']:.0f} percentile of its 1-year range ({vol_rank_data['min']:.1%} - {vol_rank_data['max']:.1%})")
row2_col2.metric("Daily RSI", f"{rsi_daily:.1f}")
row2_col3.metric("Hourly RSI", f"{rsi_hourly:.1f}")

st.header("Optimal Strategy Recommendation")
market_sentiment_result = determine_market_sentiment(iv, vol_rank_data, rsi_daily, rsi_hourly, thresholds)
optimal_strategy = generate_optimal_strategy(market_sentiment_result)

sentiment_color_map = {'BULLISH': 'green', 'BEARISH': 'red', 'NEUTRAL': 'orange'}; sentiment_icon_map = {'BULLISH': '🐂', 'BEARISH': '🐻', 'NEUTRAL': '⚖️'}
color = sentiment_color_map[optimal_strategy['sentiment']]; icon = sentiment_icon_map[optimal_strategy['sentiment']]

st.markdown(f"""
<div style="border: 2px solid {color}; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
    <h3 style="color:{color}; margin-top:0;">{icon} Market Sentiment: {optimal_strategy['sentiment']}</h3>
    <p><b>Optimal Strategy:</b> {optimal_strategy['action'].replace('_', ' ').title()}</p>
    <p><b>Reasoning:</b> {optimal_strategy['reason']}</p>
</div>
""", unsafe_allow_html=True)

sold_put, sold_call, hedge_with_perp, hedge_reason = None, None, False, ""
if optimal_strategy['action'] != 'HOLD':
    if optimal_strategy['action'] in ['SELL_PUT', 'SELL_STRANGLE']: sold_put = find_best_option_to_sell(all_options, 'put', TARGET_DELTA, MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)
    if optimal_strategy['action'] in ['SELL_CALL', 'SELL_STRANGLE']: sold_call = find_best_option_to_sell(all_options, 'call', TARGET_DELTA, MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)

if perp_hedge_override == "Automatic (Recommended)": perp_decision = determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, LTV, thresholds); hedge_with_perp = perp_decision['hedge_with_perp']; hedge_reason = perp_decision['reason']
elif perp_hedge_override == "Force Short Hedge": hedge_with_perp = True; hedge_reason = "Manual override: User forced a short perpetual hedge."
else: hedge_with_perp = False; hedge_reason = "Manual override: User forced no perpetual hedge."

st.subheader("Actionable Trade(s)")
if optimal_strategy['action'] == 'HOLD': st.success("No compelling trade setup found. The optimal action is to hold the base position and wait.")
elif optimal_strategy['action'] != 'HOLD' and not sold_put and not sold_call: st.warning(f"Engine recommended to **{optimal_strategy['action'].replace('_', ' ')}**, but no option was found that meets your specific Delta and Premium criteria.")
else:
    if hedge_with_perp: st.success(f"**Tactical Action: Add a 1x Short Perpetual Hedge.**\n\n*Reasoning: {hedge_reason}*")
    else: st.info(f"**Tactical Action: No Perpetual Hedge Needed.**\n\n*Reasoning: {hedge_reason}*")
    put_col, call_col = st.columns(2)
    with put_col:
        if sold_put: st.metric("Sell Put Strike", f"${sold_put['strike']:.0f}", f"Premium: ${sold_put['price']:.2f}")
    with call_col:
        if sold_call: st.metric("Sell Call Strike", f"${sold_call['strike']:.0f}", f"Premium: ${sold_call['price']:.2f}")

st.header("Position Payoff Analysis")
params = {'eth_deposited': ETH_DEPOSITED, 'eth_price_initial': live_eth_price, 'aave_apy': AAVE_APY, 'daily_funding_rate': daily_funding_rate, 'dcds_coverage_percent': DCDS_COVERAGE_PERCENT, 'dcds_fee_percent': DCDS_FEE_PERCENT, 'dcds_upside_sharing_percent': DCDS_UPSIDE_SHARING_PERCENT}
pcol1, pcol2 = st.columns([1, 2])
with pcol1:
    price_slider_start = int(live_eth_price * 0.5); price_slider_end = int(live_eth_price * 1.5); eth_price_final = st.slider("Set Target ETH Price ($) for PnL Breakdown", price_slider_start, price_slider_end, int(live_eth_price))
    total_pnl, pnl_underlying, pnl_aave, pnl_dcds, pnl_option, pnl_perp = calculate_final_pnl(eth_price_final, params, sold_put, sold_call, hedge_with_perp)
    st.metric("Total Projected PnL at Target Price", f"${total_pnl:,.2f}")
    with st.expander("Show PnL Contribution Breakdown"): st.metric("PnL from Underlying ETH", f"${pnl_underlying:,.2f}", delta_color="off"); st.metric("PnL from AAVE Yield", f"${pnl_aave:.2f}"); st.metric("PnL from dCDS (Net)", f"${pnl_dcds:,.2f}"); st.metric("PnL from Sold Options", f"${pnl_option:,.2f}"); st.metric("PnL from Perpetual Hedge", f"${pnl_perp:,.2f}")
with pcol2:
    price_range = np.linspace(live_eth_price * 0.6, live_eth_price * 1.4, 200); pnl_values = [calculate_final_pnl(p, params, sold_put, sold_call, hedge_with_perp)[0] for p in price_range]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3))); fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even"); fig.add_vline(x=eth_price_final, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${total_pnl:,.2f}", annotation_position="top right"); fig.add_vline(x=live_eth_price, line_width=1, line_dash="dot", line_color="grey", annotation_text="Initial Price", annotation_position="bottom right")
    title_strategy = "Hold" if (optimal_strategy['action'] == 'HOLD' or (optimal_strategy['action'] != 'HOLD' and not sold_put and not sold_call)) else optimal_strategy['action'].replace('_', ' ').title()
    title_hedge = ' + Perp Hedge' if hedge_with_perp else ''
    title = f"Payoff: dCDS + {title_strategy}{title_hedge}"; fig.update_layout(title=title, xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
