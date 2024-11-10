import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import norm
import plotly.graph_objs as go
import matplotlib.pyplot as plt

st.title('Black-Scholes Option Pricing')
st.write("""
The Black-Scholes model, developed by Fischer Black and Myron Scholes in 1973, is a mathematical model for pricing European-style options. 
The model provides a formula that gives a theoretical estimate of the price of European call and put options. It revolutionized the financial industry by allowing traders to price options consistently, 
which led to the growth of options trading worldwide.
""")

# Black-Scholes formula explanation
st.subheader("Black-Scholes Formula")
st.latex(r"""
\text{Call Option Price} = S e^{-δ T} N(d_1) - K e^{-r T} N(d_2) 
""")
st.write("""
The Black-Scholes formula calculates the price of a call option as shown above. Let's break down each component:
- \( S \): Current stock price
- \( K \): Strike price of the option
- \( T \): Time to expiration (in years)
- \( r \): Risk-free interest rate
- \( σ \): Volatility of the stock's returns (standard deviation)
- \( δ \): Dividend yield, or the rate at which the stock pays dividends
- \( N(d_1) \) and \( N(d_2) \): The cumulative distribution function of the standard normal distribution evaluated at \( d_1 \) and \( d_2 \)

Where:
""")
st.latex(r"""
d_1 = \frac{\ln(S / K) + (r - δ + 0.5 \sigma^2) T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
""")
def black_scholes(S, K, T, r, d, sigma, option_type):
    d1 = (np.log(S / K) + (r - d + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp(-d * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-d * T) * norm.cdf(-d1)
    else:
        raise ValueError("Option type must be 'call' or 'put'")
    
    return price


S = st.number_input('Stock Price (S)', value=100.0)
K = st.number_input('Strike Price (K)', value=100.0)
T = st.number_input('Time to Expiration (T) in years', value=1.0)
r = st.number_input('Risk-Free Rate (r)', value=0.05)
d = st.number_input('Dividend Yield (δ)', value=0.00)
sigma = st.number_input('Volatility (σ)', value=0.2)
option_type = st.selectbox('Option Type', ('call', 'put'))
if st.button('Calculate Option Price', key = 'calculate_button_1'):
    price = black_scholes(S, K, T, r, d, sigma, option_type)
    container = st.container(border=True)
    container.write(f'The {option_type} option price is: {price}')

st.subheader("The Greeks")
st.write("""
The Greeks are measures of an option's sensitivity to different variables. They help traders understand the risk of an option's price changing in response to market conditions. Key Greeks include:
- **Delta (Δ)**: Measures the rate of change of the option's price with respect to the underlying asset's price.
- **Gamma (Γ)**: Measures the rate of change of delta with respect to changes in the underlying asset price.
- **Vega (ν)**: Measures sensitivity to volatility. It shows how much the option price will change as the stock's volatility changes.
- **Theta (Θ)**: Measures time decay, or how much the option price decreases as expiration approaches.
- **Rho (ρ)**: Measures sensitivity to the risk-free interest rate.
""")
    
def greeks(S, K, T, r, d, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r - d + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == "call":
        delta = np.exp(-d * T) * norm.cdf(d1)
        theta = ((-S * np.exp(-d * T) * sigma * norm.pdf(d1) / (2 * np.sqrt(T)))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + d * S * np.exp(-d * T) * norm.cdf(d1)) 
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        delta = -np.exp(-d * T) * norm.cdf(-d1)
        theta = ((-S * np.exp(-d * T) * sigma * norm.pdf(d1) / (2 * np.sqrt(T)))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - d * S * np.exp(-d * T) * norm.cdf(-d1)) 
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    gamma = np.exp(-d * T) * (norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    vega = S * np.exp(-d * T) * norm.pdf(d1) * np.sqrt(T)

    return {
        'delta': round(delta, 3),
        'gamma': round(gamma, 3),
        'theta': round(theta / 365, 4),   
        'vega': round(vega * 0.01, 3),    
        'rho': round(rho * 0.01, 3)      
}


def greek_summary(S, K, T, r, d, sigma):
    greeks_list = ['delta', 'gamma', 'vega', 'theta', 'rho']
    Call = [greeks(S, K, T, r, d, sigma, 'call')[greek] for greek in greeks_list]
    Put = [greeks(S, K, T, r, d, sigma, 'put')[greek] for greek in greeks_list]
    summary = {'Call': Call, 'Put': Put}
    df = pd.DataFrame(summary, index=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
    return df

def color_negatives(val):
    color = '#EC0808' if (val < 0) else '#008000'
    return f'color: {color}'

if st.button('Show Greeks Table'):
    summary_combined = greek_summary(S, K, T, r, d, sigma)
    st.subheader('Greeks Table')
    st.dataframe(summary_combined.style.applymap(color_negatives, subset=['Call', 'Put']))


def greek_visualisation(S, K, T, r, d, sigma, option_type, greek):
    min_s = S * 0.8  # Minimum stock price for visualization
    max_s = S * 1.2  # Maximum stock price for visualization
    S_values = np.linspace(min_s, max_s, 200)  # Stock price range

    # Calculate the specified Greek for each stock price in S_values
    greek_values = [
        greeks(S_val, K, T, r, d, sigma, option_type)[greek] for S_val in S_values
    ]

    line_color = '#FFAF33' if option_type == 'call' else '#D623BE'
    
    fig, ax = plt.subplots()
    ax.plot(S_values, greek_values, label=f"{option_type.capitalize()} {greek.capitalize()}", color=line_color)
    ax.set_xlabel("Stock Price (S)", color="white")
    ax.set_ylabel(greek.capitalize(), color="white")
    ax.legend(labelcolor="white", frameon=True)
    
    return fig

if st.button('Show Greeks Plots'):
    greeks_list = ['delta', 'gamma', 'theta', 'vega', 'rho']
    greek_vis = st.container(border = True)
    greek_vis.subheader('Greeks vs Stock Price')
    call_col, put_col = greek_vis.columns(2)
    for greek in greeks_list:
        fig_greeks_call = greek_visualisation(S, K, T, r, d, sigma, 'call', greek)
        fig_greeks_put = greek_visualisation(S, K, T, r, d, sigma, 'put', greek)
        call_col.plotly_chart(fig_greeks_call)
        put_col.plotly_chart(fig_greeks_put)


st.subheader('Implied Volatility')
st.write("""
Implied Volatility is a metric derived from the price of an option and represents the market's forecast of the likely movement or volatility of the underlying asset over the option's remaining life. Unlike historical volatility, which measures past price fluctuations, implied volatility reflects the market's expectations of future volatility. It is essentially "implied" by the observed option price, as it’s the volatility value that would make the Black-Scholes model (or another pricing model) output a theoretical price equal to the market price of the option.

Higher implied volatility typically indicates greater uncertainty or risk about future price movements of the underlying asset, which can lead to higher option premiums. Traders often track changes in implied volatility to assess market sentiment and anticipate potential price swings.

Finding implied volatility requires computing the σ which solves the Black-Scholes formula for a given option price. This step is made possible by the implementation of a popular numerical procedure called "Newton-Raphson method".
""")

N_prime = norm.pdf


def vega(S, K, T, r, d, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r -d + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    
    vega = S * np.sqrt(T) * np.exp(-d * T) * N_prime(d1)
    return vega

def implied_volatility_call(price, S, K, T, r, d, option_type, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    vol_guess = 0.3

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes(S, K, T, r, d, vol_guess, option_type) - price

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            
            break

        ### use newton rapshon to update the estimate
        vol_guess = vol_guess - diff / vega(S, K, T, r, d, vol_guess)

    return vol_guess

observed_price = st.number_input('Observed market price', value=50.00)

if st.button('Calculate implied volatility'):
    imp_vol = implied_volatility_call(observed_price, S, K, T, r, d, option_type)
    container = st.container(border=True)
    container.write(f'Implied volatility (with Newton-Raphson) is: {imp_vol}')




