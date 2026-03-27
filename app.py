import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.optimize as sco
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="Live NSE Quant Dashboard | IIMA Demo", page_icon="📈", layout="wide")
st.title("📈 Portfolio Restructuring Engine | IIMA Demo")
st.caption("Current Holdings vs. Markowitz Optimal | #QuantFinance")

# --- 2. SIDEBAR (HOLDINGS + DIVERSIFICATION) ---
with st.sidebar:
    st.header("⚙️ 1. Define Assets")
    raw_tickers = st.text_input("Portfolio Tickers", "RELIANCE.NS, IRCON.NS, JWL.NS, ABFRL.NS")
    tickers = [t.strip().upper() for t in raw_tickers.split(',') if t.strip()]
    
    if len(tickers) < 2:
        st.warning("Enter at least TWO tickers.")
        st.stop()

    st.markdown("---")
    st.header("💰 2. Current Holdings")
    st.caption("Enter the exact ₹ amount you currently have in each stock:")
    
    current_investments = []
    for t in tickers:
        amount = st.number_input(f"{t} (₹)", min_value=0.0, value=25000.0, step=5000.0, format="%f")
        current_investments.append(amount)
        
    current_investments = np.array(current_investments)
    total_capital = np.sum(current_investments)
    st.markdown(f"**Total Capital: ₹{total_capital:,.2f}**")

    st.markdown("---")
    st.header("⚖️ 3. Diversification Rules")
    st.caption("Prevent the '100% in one stock' problem.")
    min_weight = st.slider("Min Weight per Asset (%)", 0, 30, 5) / 100.0
    max_weight = st.slider("Max Weight per Asset (%)", 10, 100, 40) / 100.0
    st.markdown("---")

    period = st.selectbox("Historical Period", ["6mo", "1y", "2y", "5y"])
    rf_rate = 0.07  # 7% Risk-Free Rate
    
    if st.button("🔄 Refresh Live Data"):
        st.cache_data.clear()

# --- SAFETY CHECKS ---
if total_capital <= 0:
    st.sidebar.error("Total capital must be greater than zero.")
    st.stop()
if min_weight * len(tickers) > 1.0:
    st.sidebar.error(f"Error: Minimum weight is too high. {min_weight*100}% × {len(tickers)} stocks > 100%.")
    st.stop()
if max_weight * len(tickers) < 1.0:
    st.sidebar.error(f"Error: Maximum weight is too low. {max_weight*100}% × {len(tickers)} stocks < 100%.")
    st.stop()

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=3600)
def fetch_data(t_list, p):
    df = yf.download(t_list + ['^NSEI'], period=p, progress=False)
    prices = df['Adj Close'] if 'Adj Close' in df.columns else df['Close']
    return prices.dropna()

try:
    data = fetch_data(tickers, period)
    ret = data[tickers].pct_change().dropna()
    mu, cov = ret.mean() * 252, ret.cov() * 252
except Exception as e:
    st.error("Data Fetch Error: Please check if all ticker symbols are correct.")
    st.stop()

# --- 4. CORE MATH: CURRENT VS OPTIMAL ---
# A. Current Portfolio Stats
current_weights = current_investments / total_capital
curr_ret = np.dot(current_weights, mu)
curr_vol = np.sqrt(current_weights.T @ cov @ current_weights)
curr_sharpe = (curr_ret - rf_rate) / curr_vol

# B. Optimal Portfolio Stats (SciPy) WITH USER BOUNDS
def neg_sharpe(w): 
    return -(np.dot(w, mu) - rf_rate) / np.sqrt(w.T @ cov @ w)

# Re-introduced the strict diversification boundaries!
bounds = tuple((min_weight, max_weight) for _ in range(len(tickers)))
constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
res = sco.minimize(neg_sharpe, [1./len(tickers)] * len(tickers), method='SLSQP', bounds=bounds, constraints=constraints)

opt_w = res.x
opt_ret, opt_vol = np.dot(opt_w, mu), np.sqrt(opt_w.T @ cov @ opt_w)
opt_sharpe = (opt_ret - rf_rate) / opt_vol

# Monte Carlo (2000 Sims) bounded by user limits
sim_ret, sim_vol, sim_sharpe = [], [], []
for _ in range(2000):
    w = np.random.uniform(min_weight, max_weight, len(tickers))
    w /= sum(w)
    r, v = np.dot(w, mu), np.sqrt(w.T @ cov @ w)
    sim_ret.append(r); sim_vol.append(v); sim_sharpe.append((r - rf_rate) / v)

# --- 5. HERO KPIs ---
st.markdown("### 📊 Portfolio Analysis: Current vs. Target")
c1, c2, c3 = st.columns(3)
c1.metric("Sharpe Ratio", f"{opt_sharpe:.2f}", f"{(opt_sharpe - curr_sharpe):.2f} Improvement from Current")
c2.metric("Expected Annual Return", f"{opt_ret*100:.2f}%", f"{(opt_ret - curr_ret)*100:.2f}% vs Current")
c3.metric("Annual Volatility (Risk)", f"{opt_vol*100:.2f}%", f"{(opt_vol - curr_vol)*100:.2f}% vs Current", delta_color="inverse")

# --- 6. REBALANCING TABLE ---
st.markdown("### ⚖️ Recommended Rebalancing Actions")
stats_df = pd.DataFrame({
    'Stock': tickers, 
    'Current ₹': current_investments,
    'Target ₹': opt_w * total_capital,
})
stats_df['Action Required'] = stats_df['Target ₹'] - stats_df['Current ₹']

def color_action(val):
    color = '#2e8b57' if val > 0 else '#d62728' if val < 0 else 'grey'
    return f'color: {color}; font-weight: bold'

styled_df = stats_df.style.format({
    'Current ₹': '₹{:,.2f}', 
    'Target ₹': '₹{:,.2f}', 
    'Action Required': '₹{:,.2f}'
}).map(color_action, subset=['Action Required'])

st.dataframe(styled_df, use_container_width=True)

# --- 7. PLOTLY DASHBOARD ---
fig = make_subplots(rows=2, cols=2, row_heights=[0.5, 0.5], 
                    specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "heatmap"}, {"type": "xy"}]],
                    subplot_titles=("Efficient Frontier: You vs. Math", "Allocation Shift (₹)", 
                                    "Asset Correlation Heatmap", "Cumulative Returns (Last 30 Days)"))

# R1C1
fig.add_trace(go.Scatter(x=sim_vol, y=sim_ret, mode='markers', 
                         marker=dict(color=sim_sharpe, colorscale='Viridis', showscale=True, size=5, 
                                     colorbar=dict(title='Sharpe', x=0.45, len=0.45, y=0.8)), 
                         name='Simulations', hoverinfo='none'), row=1, col=1)
fig.add_trace(go.Scatter(x=[curr_vol], y=[curr_ret], mode='markers', 
                         marker=dict(color='cyan', symbol='square', size=14, line=dict(color='white', width=1)),
                         name='Current Holdings', hovertext=f"<b>CURRENT</b><br>Sharpe: {curr_sharpe:.2f}<br>Ret: {curr_ret*100:.2f}%<br>Vol: {curr_vol*100:.2f}%"), row=1, col=1)
fig.add_trace(go.Scatter(x=[opt_vol], y=[opt_ret], mode='markers', 
                         marker=dict(color='gold', symbol='diamond', size=16, line=dict(color='black', width=1)),
                         name='Math Optimal', hovertext=f"<b>TARGET</b><br>Sharpe: {opt_sharpe:.2f}<br>Ret: {opt_ret*100:.2f}%<br>Vol: {opt_vol*100:.2f}%"), row=1, col=1)

# R1C2
fig.add_trace(go.Bar(name='Current Allocation ₹', x=tickers, y=current_investments, marker_color='cyan'), row=1, col=2)
fig.add_trace(go.Bar(name='Target Allocation ₹', x=tickers, y=opt_w * total_capital, marker_color='gold'), row=1, col=2)

# R2C1 & R2C2
corr = ret.corr()
fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1, showscale=False), row=2, col=1)
cum_ret = (1 + ret.tail(30)).cumprod() - 1
for t in tickers:
    fig.add_trace(go.Scatter(x=cum_ret.index, y=cum_ret[t], mode='lines', name=t), row=2, col=2)

fig.update_layout(height=700, template="plotly_dark", barmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_xaxes(title_text="Volatility", row=1, col=1)
fig.update_yaxes(title_text="Return", row=1, col=1)
st.plotly_chart(fig, use_container_width=True)
