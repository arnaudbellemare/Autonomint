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
# ==                        BLACK-SCHOLES GREEK CALCULATOR                         ==
# =====================================================================================
class BlackScholes:
    def __init__(self, T, K, S, sigma, r):
        self.T = max(T, 1e-9); self.K = float(K); self.S = float(S); self.sigma = float(sigma); self.r = float(r)
        self.sigma_sqrt_T = self.sigma * np.sqrt(self.T)
        if self.sigma_sqrt_T < 1e-9:
            self.d1 = np.inf if self.S > self.K else -np.inf; self.d2 = self.d1
        else:
            self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / self.sigma_sqrt_T
            self.d2 = self.d1 - self.sigma_sqrt_T

    def calculate_gamma(self):
        if self.sigma_sqrt_T < 1e-9 or self.S < 1e-9: return 0.0
        return norm.pdf(self.d1) / (self.S * self.sigma_sqrt_T)

    def calculate_theta(self):
        if self.sigma_sqrt_T < 1e-9:
            call_theta_val = -self.r * self.K * np.exp(-self.r*self.T) if self.S > self.K else 0.0
            put_theta_val = self.r * self.K * np.exp(-self.r*self.T) if self.S < self.K else 0.0
            return call_theta_val / 365.25, put_theta_val / 365.25
        term1_annual = - (self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
        call_theta_annual = term1_annual - self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(self.d2)
        put_theta_annual = term1_annual + self.r * self.K * np.exp(-self.r*self.T) * norm.cdf(-self.d2)
        return call_theta_annual / 365.25, put_theta_annual / 365.25

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
    URL_TICKER = f"{BASE_URL_THALEX}/ticker"; API_DELAY_TICKER = 0.1; time.sleep(API_DELAY_TICKER)
    response = requests.get(URL_TICKER, params={"instrument_name": instrument_name}, timeout=15)
    response.raise_for_status(); return response.json().get("result", {})

@st.cache_data(ttl=300)
def get_thalex_actual_daily_funding_rate(coin_symbol: str) -> float:
    instrument_name = f"{coin_symbol.upper()}-PERPETUAL"; ticker_data = get_instrument_ticker(instrument_name)
    if ticker_data and pd.notna(ticker_data.get('funding_rate')): return float(ticker_data['funding_rate']) * 3
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
                    parsed_options.append({'instrument_name': instr['instrument_name'], 'expiry_str': expiry_str, 'strike': float(strike_str), 'option_type': 'call' if type_char == 'C' else 'put', 'dte': dte})
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
    if ticker_data and pd.notna(ticker_data.get('iv')):
        iv_float = float(ticker_data['iv'])
        return iv_float / 100.0 if iv_float > 1.0 else iv_float
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
    if historical_prices.empty or len(historical_prices) < 200: return {'rank': 50, 'current': 0, 'min': 0, 'max': 0}
    log_returns = np.log(historical_prices['mark_price_close'] / historical_prices['mark_price_close'].shift(1))
    historical_rv = log_returns.rolling(window=window).std() * np.sqrt(365)
    historical_rv.dropna(inplace=True)
    if historical_rv.empty: return {'rank': 50, 'current': 0, 'min': 0, 'max': 0}
    current_vol = historical_rv.iloc[-1]; min_vol_1y = historical_rv.min(); max_vol_1y = historical_rv.max()
    if (max_vol_1y - min_vol_1y) == 0: return {'rank': 50, 'current': current_vol, 'min': min_vol_1y, 'max': max_vol_1y}
    rank = (current_vol - min_vol_1y) / (max_vol_1y - min_vol_1y) * 100
    return {'rank': rank, 'current': current_vol, 'min': min_vol_1y, 'max': max_vol_1y}

def calculate_price_z_score(historical_prices: pd.DataFrame, window: int) -> float:
    if historical_prices.empty or len(historical_prices) < window: return 0.0
    df = historical_prices.copy()
    df['log_price'] = np.log(df['mark_price_close'])
    mean_log_price = df['log_price'].rolling(window=window).mean(); std_log_price = df['log_price'].rolling(window=window).std()
    current_log_price = df['log_price'].iloc[-1]; current_mean = mean_log_price.iloc[-1]; current_std = std_log_price.iloc[-1]
    if pd.isna(current_std) or current_std == 0: return 0.0
    return (current_log_price - current_mean) / current_std

@st.cache_data(ttl=600)
def create_global_option_screener(options_df, live_price, risk_free_rate):
    if options_df.empty or live_price <= 0: return pd.DataFrame()
    
    filtered_df = options_df[(options_df['dte'] >= 1) & (options_df['dte'] <= 90)].copy()
    strike_min, strike_max = live_price * 0.5, live_price * 1.5
    filtered_df = filtered_df[(filtered_df['strike'] > strike_min) & (filtered_df['strike'] < strike_max)]

    if filtered_df.empty: return pd.DataFrame()
    
    enriched_options = []
    with st.spinner(f"Calculating greeks for {len(filtered_df)} relevant options..."):
        for _, row in filtered_df.iterrows():
            ticker_data = get_instrument_ticker(row['instrument_name'])
            if ticker_data and all(pd.notna(ticker_data.get(k)) for k in ['mark_price', 'iv', 'delta']) and ticker_data.get('mark_price', 0) > 0:
                iv_raw = ticker_data['iv']
                iv_decimal = iv_raw / 100.0 if iv_raw > 1.0 else iv_raw
                
                ttm = row['dte'] / 365.25
                bs_model = BlackScholes(T=ttm, K=row['strike'], S=live_price, sigma=iv_decimal, r=risk_free_rate)
                gamma_val = bs_model.calculate_gamma()
                theta_call, theta_put = bs_model.calculate_theta()
                theta_val = theta_call if row['option_type'] == 'call' else theta_put

                enriched_options.append({
                    'instrument': row['instrument_name'], 'expiry': row['expiry_str'], 'DTE': row['dte'], 'strike': row['strike'], 'type': row['option_type'],
                    'premium': ticker_data['mark_price'], 'iv': iv_decimal, 'delta': ticker_data['delta'],
                    'theta': theta_val, 'gamma': gamma_val,
                })

    if not enriched_options: return pd.DataFrame()

    df = pd.DataFrame(enriched_options)
    df['annualized_yield'] = (df['premium'] / df['strike']) / (df['DTE'] / 365.25)
    df['risk_adjusted_yield'] = df['annualized_yield'] * (1 - abs(df['delta']))
    df['theta_gamma_ratio'] = df['theta'].abs() / (df['gamma'] + 1e-9)
    df['cushion_%'] = abs(df['strike'] - live_price) / live_price * 100
    
    return df

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
def determine_market_sentiment(iv, rv, vrp, vol_rank, rsi_daily, rsi_hourly, thresholds):
    is_vol_high_rank = vol_rank >= thresholds['min_vol_rank']; has_vrp = vrp >= thresholds['min_vrp']
    is_daily_bullish = rsi_daily > thresholds['rsi_overbought']; is_daily_bearish = rsi_daily < thresholds['rsi_oversold']
    is_hourly_confirming_bullish = rsi_hourly > 50; is_hourly_confirming_bearish = rsi_hourly < 50

    if not is_vol_high_rank:
        sentiment, reason, can_sell = 'NEUTRAL', f"Volatility Rank is {vol_rank:.0f}%, below the {thresholds['min_vol_rank']:.0f}% threshold. Environment is not favorable for selling premium.", False
    elif not has_vrp:
        sentiment, reason, can_sell = 'NEUTRAL', f"The Volatility Risk Premium (VRP) is only {vrp:.1%}, below the {thresholds['min_vrp']:.1f}% threshold. There is not enough edge to sell options.", False
    else:
        can_sell = True
        if is_daily_bullish and is_hourly_confirming_bullish:
            sentiment = 'BULLISH'; reason = f"Confirmed Uptrend (Daily RSI: {rsi_daily:.1f}) + High Vol Rank ({vol_rank:.0f}%) + Positive VRP ({vrp:.1%}). Ideal conditions to sell Puts."
        elif is_daily_bearish and is_hourly_confirming_bearish:
            sentiment = 'BEARISH'; reason = f"Confirmed Downtrend (Daily RSI: {rsi_daily:.1f}) + High Vol Rank ({vol_rank:.0f}%) + Positive VRP ({vrp:.1%}). Ideal conditions to sell Calls for hedging."
        else:
            sentiment = 'NEUTRAL'; reason = f"Conflicted/Neutral Trend (RSI Daily/Hourly: {rsi_daily:.1f}/{rsi_hourly:.1f}) + High Vol Rank ({vol_rank:.0f}%) + Positive VRP ({vrp:.1%}). Favorable for a Strangle."
    return {'sentiment': sentiment, 'reason': reason, 'can_sell_options': can_sell}

def generate_optimal_strategy(market_sentiment_result):
    sentiment, can_sell, action = market_sentiment_result['sentiment'], market_sentiment_result['can_sell_options'], 'HOLD'
    if can_sell:
        if sentiment == 'BULLISH': action = 'SELL_PUT'
        elif sentiment == 'BEARISH': action = 'SELL_CALL'
        elif sentiment == 'NEUTRAL': action = 'SELL_STRANGLE'
    return {'action': action, 'sentiment': sentiment, 'reason': market_sentiment_result['reason']}

def determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, ltv, thresholds):
    reasons = []; vrp = iv - rv
    if iv > thresholds['iv_high']: reasons.append(f"High IV ({iv:.1%})")
    if vrp > thresholds['min_vrp'] + 0.1: reasons.append(f"High VRP ({vrp:.1%})")
    if rsi_daily > 75 or rsi_daily < 25: reasons.append(f"Extreme trend (Daily RSI:{rsi_daily:.1f})")
    if abs(daily_funding_rate) > thresholds['funding_rate_high']: reasons.append(f"High daily funding ({daily_funding_rate:.4%})")
    if ltv > thresholds['ltv_high']: reasons.append(f"High LTV ({ltv:.1%})")
    if reasons: return {'hedge_with_perp': True, 'reason': "Tactical hedge is a MUST. Trigger(s): " + " | ".join(reasons)}
    return {'hedge_with_perp': False, 'reason': "Market conditions neutral. Tactical perp hedge not required."}

def find_best_option_to_sell(options_df, option_type, target_delta, min_premium_ratio, live_price, target_dte):
    if options_df.empty: return None
    options_df['dte_diff'] = (options_df['dte'] - target_dte).abs(); closest_dte = options_df.loc[options_df['dte_diff'].idxmin()]['dte']; target_expiry_options = options_df[np.isclose(options_df['dte'], closest_dte)].copy()
    ticker_data_map = {}
    with st.spinner(f"Scanning matching {option_type} options..."):
        for _, row in target_expiry_options.iterrows():
            if row['option_type'] == option_type: ticker_data_map[row['instrument_name']] = get_instrument_ticker(row['instrument_name'])
    candidates_data = []
    for name, data in ticker_data_map.items():
        if data and data.get('delta') is not None and data.get('mark_price') is not None: candidates_data.append({'instrument_name': name, 'delta': data['delta'], 'price': data['mark_price'], 'strike': next((item['strike'] for item in options_df.to_dict('records') if item['instrument_name'] == name), 0)})
    if not candidates_data: return None
    candidates_df = pd.DataFrame(candidates_data)
    if live_price > 0: candidates_df = candidates_df[candidates_df['price'] / live_price >= min_premium_ratio]
    if candidates_df.empty: return None
    candidates_df['delta_diff'] = (candidates_df['delta'].abs() - target_delta).abs(); return candidates_df.loc[candidates_df['delta_diff'].idxmin()].to_dict()

# =====================================================================================
# ==                              UI & APP LAYOUT                                  ==
# =====================================================================================
st.title("Autonomint Quant Strategy Optimizer")
with st.sidebar:
    st.header("1. Core Position"); ETH_DEPOSITED = st.number_input("ETH Deposited", 1.0, 10.0, 2.0, 0.5); AAVE_APY = st.slider("AAVE Supply APY (%)", 0.1, 10.0, 3.0, 0.1) / 100.0; LTV = st.slider("Loan-to-Value (LTV) (%)", 50.0, 95.0, 80.0, 1.0) / 100.0
    st.header("2. dCDS Hedge Parameters"); DCDS_COVERAGE_PERCENT = st.slider("Downside Coverage (%)", 10., 50., 20., 1.)/ 100.0; DCDS_FEE_PERCENT = st.slider("Upfront Fee (% of hedged value)", 1., 20., 12., 0.5) / 100.0; DCDS_UPSIDE_SHARING_PERCENT = st.slider("Upside Sharing Cost (%)", 0., 10., 3., 0.5) / 100.0
    st.header("3. Option Execution Criteria"); TARGET_DTE = st.slider("Target Days to Expiry (DTE)", 7, 60, 30, 1);
    st.header("4. Strategy Engine Thresholds")
    MIN_VOL_RANK = st.slider("Min Volatility Rank to Sell Premium (%)", 0, 100, 50, help="Only sell options if current volatility is above this percentile of its 1-year range.")
    MIN_VRP = st.slider("Minimum VRP to Trade (IV - RV)", 0.0, 15.0, 5.0, 0.5, format="%.1f%%", help="Minimum Volatility Risk Premium required to sell options.") / 100.0
    RISK_FREE_RATE = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 4.0, 0.1, help="Proxy for risk-free rate, used in Black-Scholes calculations (e.g., Aave USDC rate).") / 100.0
    Z_SCORE_WINDOW = st.slider("Z-Score Lookback Period (Days)", 30, 180, 90, help="The lookback window for the mean-reversion Price Z-Score.")
    RSI_OVERBOUGHT = st.slider("Daily RSI Overbought", 60, 80, 70); RSI_OVERSOLD = st.slider("Daily RSI Oversold", 20, 40, 30)
    st.header("5. Global Screener Filters")
    SCREENER_MAX_DELTA = st.slider("Max Abs. Delta for Screener", 0.10, 0.50, 0.35, 0.01)
    SCREENER_MIN_PREMIUM_RATIO = st.slider("Min Premium/Spot Ratio for Screener (%)", 0.1, 5.0, 1.0, 0.1) / 100.0
    st.header("6. Tactical Hedge Triggers")
    IV_HIGH = st.slider("High IV Threshold (%)", 50., 120., 80., 1.) / 100.0; FUNDING_HIGH = st.slider("High Daily Funding Rate (%)", 0.05, 0.3, 0.15) / 100.0;
    thresholds = {'min_vol_rank': MIN_VOL_RANK, 'min_vrp': MIN_VRP, 'rsi_overbought': RSI_OVERBOUGHT, 'rsi_oversold': RSI_OVERSOLD, 'iv_high': IV_HIGH, 'funding_rate_high': FUNDING_HIGH, 'ltv_high': 0.85}
    st.header("7. Manual Overrides"); perp_hedge_override = st.selectbox("Perpetual Hedge Strategy", ["Automatic (Recommended)", "Force Short Hedge", "Force No Hedge"])

with st.spinner("Fetching all live market data..."):
    live_eth_price = fetch_live_eth_price()
    if live_eth_price is None: st.error("Could not fetch live ETH price. Please refresh."); st.stop()
    daily_funding_rate = get_thalex_actual_daily_funding_rate('ETH'); all_options = get_all_options_data()
    yearly_historical_df = fetch_historical_prices(days_lookback=365); hourly_historical_df = fetch_historical_prices(days_lookback=7, timeframe='1h')
    if daily_funding_rate is None or yearly_historical_df.empty or hourly_historical_df.empty or all_options.empty: st.error("Failed to fetch critical market data."); st.stop()

vol_rank_data = calculate_volatility_rank(yearly_historical_df); rv = vol_rank_data['current']
rsi_daily = calculate_rsi(yearly_historical_df['mark_price_close']); rsi_hourly = calculate_rsi(hourly_historical_df['mark_price_close'])
iv = fetch_atm_iv(all_options, TARGET_DTE, live_eth_price)
price_z_score = calculate_price_z_score(yearly_historical_df, Z_SCORE_WINDOW); vrp = iv - rv

st.header("Live Market Dashboard")
col1, col2 = st.columns(2)
with col1:
    st.metric("Live ETH Price", f"${live_eth_price:,.2f}")
    st.metric("30D Realized Volatility (RV)", f"{rv:.2%}")
    st.metric("1Y Volatility Rank", f"{vol_rank_data['rank']:.0f}%", help=f"Current RV is at the {vol_rank_data['rank']:.0f} percentile of its 1-year range ({vol_rank_data['min']:.1%} - {vol_rank_data['max']:.1%})")
with col2:
    iv_display_text = f"{iv:.2%}" if iv > 0 else "N/A"
    st.metric(f"~{TARGET_DTE}-Day ATM IV", iv_display_text)
    st.metric("Volatility Risk Premium (VRP)", f"{vrp:.2%}", delta="Edge to Sell" if vrp > 0 else "No Edge", help="VRP = IV - RV.")
    z_score_delta = "Below Trend" if price_z_score < 0 else "Above Trend"
    st.metric(f"{Z_SCORE_WINDOW}-Day Price Z-Score", f"{price_z_score:.2f}", delta=z_score_delta, help="Std. deviations from the mean log-price.")

st.header("Optimal Strategy Recommendation")
market_sentiment_result = determine_market_sentiment(iv, rv, vrp, vol_rank_data['rank'], rsi_daily, rsi_hourly, thresholds)
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
    if optimal_strategy['action'] in ['SELL_PUT', 'SELL_STRANGLE']: sold_put = find_best_option_to_sell(all_options, 'put', SCREENER_MAX_DELTA, SCREENER_MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)
    if optimal_strategy['action'] in ['SELL_CALL', 'SELL_STRANGLE']: sold_call = find_best_option_to_sell(all_options, 'call', SCREENER_MAX_DELTA, SCREENER_MIN_PREMIUM_RATIO, live_eth_price, TARGET_DTE)
if perp_hedge_override == "Automatic (Recommended)": perp_decision = determine_perp_hedge_necessity(iv, rv, rsi_daily, daily_funding_rate, LTV, thresholds); hedge_with_perp = perp_decision['hedge_with_perp']; hedge_reason = perp_decision['reason']
elif perp_hedge_override == "Force Short Hedge": hedge_with_perp = True; hedge_reason = "Manual override: User forced a short perpetual hedge."
else: hedge_with_perp = False; hedge_reason = "Manual override: User forced no perpetual hedge."

st.subheader("Actionable Trade(s)")
if optimal_strategy['action'] == 'HOLD': st.success("No compelling trade setup found. The optimal action is to hold the base position and wait.")
elif optimal_strategy['action'] != 'HOLD' and not sold_put and not sold_call: st.warning(f"Engine recommended to **{optimal_strategy['action'].replace('_', ' ')}**, but no option was found that meets your specific criteria for the target DTE.")
else:
    if hedge_with_perp: st.success(f"**Tactical Action: Add a 1x Short Perpetual Hedge.**\n\n*Reasoning: {hedge_reason}*")
    else: st.info(f"**Tactical Action: No Perpetual Hedge Needed.**\n\n*Reasoning: {hedge_reason}*")
    put_col, call_col = st.columns(2)
    with put_col:
        if sold_put: st.metric("Sell Put Strike", f"${sold_put['strike']:.0f}", f"Premium: ${sold_put['price']:.2f}")
    with call_col:
        if sold_call: st.metric("Sell Call Strike", f"${sold_call['strike']:.0f}", f"Premium: ${sold_call['price']:.2f}")

# =====================================================================================
# ==                      CORRECTED UI SECTION WITH HIGHLIGHTING                     ==
# =====================================================================================

with st.expander("🌍 Global Actionable Option Chain"):
    st.markdown("This chain scans all relevant expiries and is filtered by the criteria in the sidebar. The single best candidate in each table is highlighted.")
    df_enriched = create_global_option_screener(all_options, live_eth_price, RISK_FREE_RATE)
    
    if not df_enriched.empty:
        df_filtered = df_enriched[(df_enriched['delta'].abs() <= SCREENER_MAX_DELTA) & (df_enriched['premium'] >= (live_eth_price * SCREENER_MIN_PREMIUM_RATIO))].copy()

        if df_filtered.empty:
            st.warning("No options across ANY expiry met the filtering criteria. This could indicate a very low volatility environment or tight criteria.")
        else:
            calls_global = df_filtered[df_filtered['type'] == 'call'].sort_values(by='risk_adjusted_yield', ascending=False)
            puts_global = df_filtered[df_filtered['type'] == 'put'].sort_values(by='risk_adjusted_yield', ascending=False)

            cols_to_display = ['instrument', 'expiry', 'DTE', 'strike', 'premium', 'iv', 'delta', 'risk_adjusted_yield', 'theta_gamma_ratio', 'cushion_%']
            
            # --- NEW FORMATTING LOGIC ---
            calls_global['theta_gamma_ratio'] /= 1000
            puts_global['theta_gamma_ratio'] /= 1000
            style_formats = {'DTE': '{:.1f}', 'strike': '{:,.0f}', 'premium': '${:,.4f}', 'iv': '{:.2%}', 'delta': '{:.3f}', 'risk_adjusted_yield': '{:.2%}', 'theta_gamma_ratio': '{:,.0f}K', 'cushion_%': '{:.1f}%'}

            # --- NEW HIGHLIGHTING FUNCTION ---
            def highlight_top_row(df):
                # Create a blank DataFrame with the same shape to hold our styles
                style_df = pd.DataFrame('', index=df.index, columns=df.columns)
                # Get the index of the first row (the best one)
                top_row_index = df.index[0]
                # Apply the highlight style to that row
                style_df.loc[top_row_index, :] = 'background-color: #004d00; color: white; border-bottom: 2px solid #FFFFFF;'
                return style_df

            st.subheader("Best Call Candidates (Sorted by Best Yield)")
            if not calls_global.empty:
                styled_df = (calls_global[cols_to_display].head(20).style
                             .format(style_formats)
                             .background_gradient(subset=['risk_adjusted_yield'], cmap='Greens')
                             .background_gradient(subset=['theta_gamma_ratio'], cmap='YlOrRd')
                             .apply(highlight_top_row, axis=None)) # axis=None applies the style DataFrame
                st.dataframe(styled_df, use_container_width=True)
            else: 
                st.info("No Call options met the criteria.")

            st.subheader("Best Put Candidates (Sorted by Best Yield)")
            if not puts_global.empty:
                styled_df = (puts_global[cols_to_display].head(20).style
                             .format(style_formats)
                             .background_gradient(subset=['risk_adjusted_yield'], cmap='Greens')
                             .background_gradient(subset=['theta_gamma_ratio'], cmap='YlOrRd')
                             .apply(highlight_top_row, axis=None)) # axis=None applies the style DataFrame
                st.dataframe(styled_df, use_container_width=True)
            else: 
                st.info("No Put options met the criteria.")
    else:
        st.warning("Could not retrieve data for Global Actionable Chain analysis.")

st.header("Position Payoff Analysis")
params = {'eth_deposited': ETH_DEPOSITED, 'eth_price_initial': live_eth_price, 'aave_apy': AAVE_APY, 'daily_funding_rate': daily_funding_rate, 'dcds_coverage_percent': DCDS_COVERAGE_PERCENT, 'dcds_fee_percent': DCDS_FEE_PERCENT, 'dcds_upside_sharing_percent': DCDS_UPSIDE_SHARING_PERCENT}
pcol1, pcol2 = st.columns([1, 2])
with pcol1:
    price_slider_start = int(live_eth_price * 0.5); price_slider_end = int(live_eth_price * 1.5); eth_price_final = st.slider("Set Target ETH Price ($) for PnL Breakdown", price_slider_start, price_slider_end, int(live_eth_price))
    total_pnl, pnl_underlying, pnl_aave, pnl_dcds, pnl_option, pnl_perp = calculate_final_pnl(eth_price_final, params, sold_put, sold_call, hedge_with_perp)
    st.metric("Total Projected PnL at Target Price", f"${total_pnl:,.2f}")
    with st.expander("Show PnL Contribution Breakdown"): st.metric("PnL from Underlying ETH", f"${pnl_underlying:,.2f}", delta_color="off"); st.metric("PnL from AAVE Yield", f"${pnl_aave:.2f}"); st.metric("PnL from dCDS (Net)", f"${pnl_dcds:,.2f}"); st.metric("PnL from Sold Options", f"${pnl_option:.2f}"); st.metric("PnL from Perpetual Hedge", f"${pnl_perp:.2f}")
with pcol2:
    price_range = np.linspace(live_eth_price * 0.6, live_eth_price * 1.4, 200); pnl_values = [calculate_final_pnl(p, params, sold_put, sold_call, hedge_with_perp)[0] for p in price_range]
    fig = go.Figure(); fig.add_trace(go.Scatter(x=price_range, y=pnl_values, mode='lines', name='Total PnL', line=dict(color='royalblue', width=3))); fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", annotation_text="Break-Even"); fig.add_vline(x=eth_price_final, line_width=2, line_dash="dot", line_color="orange", annotation_text=f"Target PnL: ${total_pnl:,.2f}", annotation_position="top right"); fig.add_vline(x=live_eth_price, line_width=1, line_dash="dot", line_color="grey", annotation_text="Initial Price", annotation_position="bottom right")
    title_strategy = "Hold" if (optimal_strategy['action'] == 'HOLD' or (optimal_strategy['action'] != 'HOLD' and not sold_put and not sold_call)) else optimal_strategy['action'].replace('_', ' ').title()
    title_hedge = ' + Perp Hedge' if hedge_with_perp else ''
    title = f"Payoff: dCDS + {title_strategy}{title_hedge}"; fig.update_layout(title=title, xaxis_title="ETH Price at Expiry ($)", yaxis_title="Overall Profit / Loss ($)", yaxis_tickprefix='$', margin=dict(l=40, r=40, t=50, b=40))
    st.plotly_chart(fig, use_container_width=True)
