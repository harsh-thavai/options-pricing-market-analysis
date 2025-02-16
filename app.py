import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from numpy import log, sqrt, exp
import yfinance as yf
from datetime import timedelta
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------
# Page Configuration & Custom CSS
# ------------------------------
st.set_page_config(
    page_title="Options Pricing & Market Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean and modern look
st.markdown("""
    <style>
    body { font-family: 'Segoe UI', sans-serif; }
    .title { text-align: center; font-size: 2.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .subtitle { text-align: center; font-size: 1.25rem; margin-bottom: 2rem; color: #555; }
    .metric-box {
         border-radius: 10px;
         padding: 15px;
         text-align: center;
         color: #fff;
         font-size: 1.5rem;
         font-weight: bold;
         margin: 10px;
    }
    .call-box { background-color: #2ecc71; }
    .put-box { background-color: #e74c3c; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Options Pricing & Market Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A unified tool for Black-Scholes pricing and market data visualization</div>', unsafe_allow_html=True)

# ------------------------------
# Black-Scholes Model Definition
# ------------------------------
class BlackScholes:
    def __init__(self, time_to_maturity: float, strike: float, current_price: float, volatility: float, interest_rate: float):
        self.t = time_to_maturity
        self.K = strike
        self.S = current_price
        self.sigma = volatility
        self.r = interest_rate

    def calculate_prices(self):
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.t) / (self.sigma * sqrt(self.t))
        d2 = d1 - self.sigma * sqrt(self.t)
        call_price = self.S * norm.cdf(d1) - self.K * exp(-self.r * self.t) * norm.cdf(d2)
        put_price = self.K * exp(-self.r * self.t) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
        return call_price, put_price

# ------------------------------
# Helper Functions for Visuals
# ------------------------------
def generate_heatmap_data(bs_model, spot_range, vol_range, strike):
    call_prices = np.zeros((len(vol_range), len(spot_range)))
    put_prices = np.zeros((len(vol_range), len(spot_range)))
    
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            temp_model = BlackScholes(bs_model.t, strike, spot, vol, bs_model.r)
            cp, pp = temp_model.calculate_prices()
            call_prices[i, j] = cp
            put_prices[i, j] = pp
    return call_prices, put_prices

def create_plotly_heatmap(data, x, y, title):
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=np.round(x, 2),
        y=np.round(y, 2),
        colorscale='Viridis',
        colorbar=dict(title="Price")
    ))
    fig.update_layout(title=title, xaxis_title="Spot Price", yaxis_title="Volatility")
    return fig

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan
    def objective(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price
    try:
        vol = brentq(objective, 1e-6, 5)
    except Exception:
        vol = np.nan
    return vol

# ------------------------------
# App Tabs: Option Pricing & Market Analysis
# ------------------------------
tabs = st.tabs(["Option Pricing", "Market Analysis"])

# ==============================
# Tab 1: Option Pricing
# ==============================
with tabs[0]:
    st.header("Black-Scholes Option Pricing")
    st.write("Adjust the parameters to compute theoretical option prices and view how they change with different spot prices and volatilities.")
    
    # Input parameters for Black-Scholes
    col_params = st.columns(5)
    with col_params[0]:
        S = st.number_input("Asset Price (S)", value=100.0, min_value=1.0)
    with col_params[1]:
        K = st.number_input("Strike Price (K)", value=100.0, min_value=1.0)
    with col_params[2]:
        t = st.number_input("Time to Maturity (T, years)", value=1.0, min_value=0.1)
    with col_params[3]:
        sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
    with col_params[4]:
        r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, step=0.01)
    
    bs = BlackScholes(t, K, S, sigma, r)
    call_val, put_val = bs.calculate_prices()
    
    # Display computed option prices as attractive metric boxes
    col_prices = st.columns(2)
    with col_prices[0]:
        st.markdown(f'<div class="metric-box call-box">CALL: ${call_val:.2f}</div>', unsafe_allow_html=True)
    with col_prices[1]:
        st.markdown(f'<div class="metric-box put-box">PUT: ${put_val:.2f}</div>', unsafe_allow_html=True)
    
    # Heatmap parameter controls (using minimal inputs)
    st.subheader("Interactive Heatmaps")
    col_heat = st.columns(2)
    with col_heat[0]:
        spot_min = st.number_input("Min Spot Price", value=S*0.8, min_value=0.1, step=0.1, key="min_spot")
    with col_heat[1]:
        spot_max = st.number_input("Max Spot Price", value=S*1.2, min_value=0.1, step=0.1, key="max_spot")
    
    vol_min = st.slider("Min Volatility", min_value=0.01, max_value=1.0, value=sigma*0.5, step=0.01)
    vol_max = st.slider("Max Volatility", min_value=0.01, max_value=1.0, value=sigma*1.5, step=0.01)
    
    spot_range = np.linspace(spot_min, spot_max, 20)
    vol_range = np.linspace(vol_min, vol_max, 20)
    
    call_prices, put_prices = generate_heatmap_data(bs, spot_range, vol_range, K)
    fig_call = create_plotly_heatmap(call_prices, spot_range, vol_range, "Call Price Heatmap")
    fig_put = create_plotly_heatmap(put_prices, spot_range, vol_range, "Put Price Heatmap")
    
    # Display heatmaps side-by-side
    col_maps = st.columns(2)
    with col_maps[0]:
        st.plotly_chart(fig_call, use_container_width=True)
    with col_maps[1]:
        st.plotly_chart(fig_put, use_container_width=True)
    
    # Download option pricing data
    prices_df = pd.DataFrame({
        "Call Price": [call_val],
        "Put Price": [put_val]
    })
    st.download_button("Download Option Prices", prices_df.to_csv(index=False), "option_prices.csv", "text/csv")

# ==============================
# Tab 2: Market Analysis
# ==============================
with tabs[1]:
    st.header("Market Analysis: Implied Volatility & Historical Data")
    st.write("This section uses live market data to compute the implied volatility surface and display historical price trends.")
    
    # Sidebar-like inputs for market analysis (placed at the top of the tab)
    col_market = st.columns(3)
    with col_market[0]:
        ticker_sym = st.text_input("Ticker Symbol", value="SPY", max_chars=10).upper()
    with col_market[1]:
        market_r = st.number_input("Risk-Free Rate", value=0.015, step=0.005, format="%.4f")
    with col_market[2]:
        div_yield = st.number_input("Dividend Yield", value=0.013, step=0.005, format="%.4f")
    
    ticker_obj = yf.Ticker(ticker_sym)
    today = pd.Timestamp("today").normalize()
    try:
        expirations = ticker_obj.options
    except Exception as e:
        st.error(f"Error fetching options for {ticker_sym}: {e}")
        st.stop()
    
    # Filter expiration dates (only those > 7 days away)
    exp_dates = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > today + timedelta(days=7)]
    if not exp_dates:
        st.error(f"No valid expiration dates found for {ticker_sym}.")
    else:
        options_list = []
        for exp in exp_dates:
            try:
                chain = ticker_obj.option_chain(exp.strftime('%Y-%m-%d'))
                calls = chain.calls
            except Exception:
                continue
            calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
            for _, row in calls.iterrows():
                mid_price = (row['bid'] + row['ask']) / 2
                options_list.append({
                    "expiration": exp,
                    "strike": row["strike"],
                    "mid": mid_price
                })
        if not options_list:
            st.error("No option data available after filtering.")
        else:
            options_df = pd.DataFrame(options_list)
            # Get the current spot price
            try:
                hist = ticker_obj.history(period="5d")
                if hist.empty:
                    st.error("No historical data available.")
                    st.stop()
                else:
                    spot_price = hist["Close"].iloc[-1]
            except Exception as e:
                st.error(f"Error fetching historical data: {e}")
                st.stop()
            
            options_df["daysToExp"] = (options_df["expiration"] - today).dt.days
            options_df["T"] = options_df["daysToExp"] / 365
            
            # Compute implied volatility for each option
            options_df["ImplVol"] = options_df.apply(
                lambda row: implied_volatility(row["mid"], spot_price, row["strike"], row["T"], market_r, div_yield),
                axis=1
            )
            options_df = options_df.dropna(subset=["ImplVol"])
            options_df["ImplVol"] *= 100  # convert to percentage
            
            # Create the 3D implied volatility surface
            X = options_df["T"].values
            Y = options_df["strike"].values
            Z = options_df["ImplVol"].values
            
            ti = np.linspace(X.min(), X.max(), 50)
            yi = np.linspace(Y.min(), Y.max(), 50)
            T_mesh, Y_mesh = np.meshgrid(ti, yi)
            Zi = griddata((X, Y), Z, (T_mesh, Y_mesh), method="linear")
            Zi = np.ma.array(Zi, mask=np.isnan(Zi))
            
            fig_surface = go.Figure(data=[go.Surface(
                x=T_mesh, y=Y_mesh, z=Zi,
                colorscale='Viridis',
                colorbar=dict(title="Impl. Vol (%)")
            )])
            fig_surface.update_layout(
                title=f"Implied Volatility Surface for {ticker_sym}",
                scene=dict(
                    xaxis_title="Time to Expiration (years)",
                    yaxis_title="Strike Price ($)",
                    zaxis_title="Implied Volatility (%)"
                ),
                autosize=True,
                margin=dict(l=65, r=50, b=65, t=90)
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
            
            # Historical price chart for the underlying asset
            st.subheader(f"Historical Prices for {ticker_sym}")
            fig_hist = px.line(hist, x=hist.index, y="Close", title=f"{ticker_sym} Historical Close Prices",
                               labels={"Close": "Price ($)", "Date": "Date"})
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Download options data
            st.download_button("Download Options Data", options_df.to_csv(index=False), "market_options.csv", "text/csv")
