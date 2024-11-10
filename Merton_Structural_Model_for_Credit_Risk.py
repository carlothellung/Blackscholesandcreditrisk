import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.title('Merton Structural Model for Credit Risk')

st.write("""
The Structural Merton Model is used to assess credit risk by modeling a firm's capital structure and asset value dynamics. 
It treats a firm's equity as a call option on its assets, with the firm's debt as the strike price: thus, the model follows the rules of Black-Scholes. The model calculates various important quantities such as the value of equity, the value of debt, the distance to default (DD), the expected default frequency (EDF), and the credit spread.
""")

# Value of Equity
st.write("**1. Value of Equity (E):**")
st.latex(r"E = \max(0, V_T - D)")
st.write("""
The value of equity is the difference between the firm's asset value (V_T) and its debt (D). If the asset value falls below the debt level, the equity value becomes zero.
""")

# Value of Debt
st.write("**2. Value of Debt (D):**")
st.latex(r"D = V_T \cdot N(d_2) - D \cdot e^{-rT} \cdot N(d_1)")
st.write("""
The value of debt is calculated using the firm's asset value (V_T), the risk-free rate (r), and time to maturity (T). Here, N(d_1) and N(d_2) are the cumulative standard normal distribution functions. The formula accounts for the probability of default and the expected payoff to debt holders.
""")

# Distance to Default
st.write("**3. Distance to Default (DD):**")
st.latex(r"DD = \frac{\ln\left(\frac{V_T}{D}\right) + \left( r - \frac{\sigma^2}{2} \right) T}{\sigma \sqrt{T}}")
st.write("""
The Distance to Default measures how far the firm's asset value is from the default point (debt level). A larger value of DD suggests a lower probability of default. It depends on the asset volatility (σ), the risk-free rate (r), and time to maturity (T).
""")

# Expected Default Frequency (EDF)
st.write("**4. Expected Default Frequency (EDF):**")
st.latex(r"EDF = N(-DD)")
st.write("""
The Expected Default Frequency (EDF) is the probability that the firm will default. It is the cumulative probability that the firm's asset value will fall below the debt level, which is given by the normal distribution function evaluated at the negative Distance to Default.
""")

# Credit Spread
st.write("**5. Credit Spread (CS):**")
st.latex(r"CS = r - \text{Implied Yield on Debt}")
st.write("""
The Credit Spread represents the difference between the risk-free rate (r) and the implied yield on the firm's debt. It compensates investors for the credit risk associated with the firm’s potential default.

By plugging in the inputs below, the graph will display the variation of credit spread for maturities that go from the one selected to the following 10 maturities.
""")

V = st.number_input('Asset Value (V)', value = 100.00, key='V_key')
D = st.number_input('Debt Face Value (D)', value = 60.00, key='D_key')
rfr = st.number_input('Risk-Free Rate (rfr)', value = 0.10, key='rfr_key')
maturity = st.number_input('Time to Maturity', value = 1.00, key='maturity_key')
sigmaV = st.number_input('Asset Volatility (σv)', value=0.3, key='sigmaV_key')

def MertonEquity(V,D,rfr,maturity,sigmaV):
    d1merton = (np.log(V / D) + (rfr + 0.5 * sigmaV**2) * maturity) / (sigmaV * np.sqrt(maturity))
    d2merton = d1merton - sigmaV * np.sqrt(maturity)
    MertonEquity = V*norm.cdf(d1merton) - D*np.exp(-rfr*maturity)*norm.cdf(d2merton)
    return (MertonEquity)

def MertonDebt(V,D,rfr,maturity,sigmaV):
    MertonEquity_value = MertonEquity(V, D, rfr, maturity, sigmaV)
    MertonDebt = V - MertonEquity_value
    return (MertonDebt)

#KMV model

def DistanceToDefault(V,D,rfr,maturity,sigmaV):
    d1merton = (np.log(V / D) + (rfr + 0.5 * sigmaV**2) * maturity) / (sigmaV * np.sqrt(maturity))
    d2merton = d1merton - sigmaV * np.sqrt(maturity)
    DistanceToDefault = d2merton
    return (DistanceToDefault)

def ExpectedDefaultFrequency(V,D,rfr,maturity,sigmaV):
    distance_to_default_value = DistanceToDefault(V, D, rfr, maturity, sigmaV)
    ExpectedDefaultFrequency = norm.cdf(- distance_to_default_value)
    return (ExpectedDefaultFrequency)

def CreditSpread(V,D,rfr,maturity,sigmaV):
    d1merton = (np.log(V / D) + (rfr + 0.5 * sigmaV**2) * maturity) / (sigmaV * np.sqrt(maturity))
    d2merton = d1merton - sigmaV * np.sqrt(maturity)
    DiscDebt = D * np.exp(- rfr * maturity)
    L = DiscDebt / V
    YieldToMaturity = rfr - np.log(norm.cdf(d2merton) + norm.cdf(- d1merton) / L) / maturity 
    CreditSpread = YieldToMaturity - rfr
    return (CreditSpread)                 



MertonEquity_value = MertonEquity(V,D,rfr,maturity,sigmaV)
MertonDebt_value = MertonDebt(V,D,rfr,maturity,sigmaV)
DistanceToDefault_value = DistanceToDefault(V,D,rfr,maturity,sigmaV)
ExpectedDefaultFrequency_value = ExpectedDefaultFrequency(V,D,rfr,maturity,sigmaV)
CreditSpread_value = CreditSpread(V,D,rfr,maturity,sigmaV)

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Value of Equity", f"${MertonEquity_value:.2f}")
with col2:
    st.metric("Value of Debt", f"${MertonDebt_value:.2f}")
with col3:
    st.metric("Distance to Default", f"{DistanceToDefault_value:.4f}")
with col4:
    st.metric("Expected Default Frequency", f"{ExpectedDefaultFrequency_value:.4f}")
with col5:
    st.metric("Credit Spread", f"{CreditSpread_value:.4f}")

# Generate maturity values from 1 to 10 years
maturities = np.linspace(maturity, maturity + 10, 50)

# Calculate credit spread for each maturity
credit_spreads = [CreditSpread(V, D, rfr, maturity, sigmaV) * 1e4 for maturity in maturities]  # Convert to basis points

# Plotting the credit spread as a function of maturity
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(maturities, credit_spreads, color='#FFD700', linestyle='-', linewidth=1.5)
ax.set_xlabel('Maturity (years)', fontsize=8, color='white')
ax.set_ylabel('Credit Spread (bps)', fontsize=8, color='white')
ax.set_title('Credit Spread vs Maturity', color='white')
ax.tick_params(colors='white')
ax.plot([], [], color='#FFD700', linewidth=1.5, label='Credit Spread')  # Invisible line for legend
ax.legend(loc='upper right', fontsize=8, frameon=False, labelcolor='lightgray')
for spine in ax.spines.values():
    spine.set_color('grey')

ax.grid(False)
ax.set_facecolor("none")  # Set background color to match ggplot2's look
fig.patch.set_facecolor('none')


# Display the plot in Streamlit
st.pyplot(fig)