import streamlit as st

st.set_page_config(
    page_title="Multipage App"
)
st.title("Black-Scholes and Credit Risk")
st.sidebar.success("Select a page above.")
st.write("""
The **Black-Scholes Model** is a foundational mathematical model used to price options by calculating the fair value of a call or put option based on factors like the stock price, strike price, time to expiration, risk-free rate, and volatility. 
The Black-Scholes model has the following assumptions:
- **Continuously compounded returns** on the stocks are normally distributed and independent over time.
- The **volatility** of continuously compounded returns is known and constant.
- **Future dividends** are known, either as dollar amounts or as a fixed dividend yield.
- The **risk-free rate** is known and constant over time.
- There are **no transaction costs or taxes**, allowing for seamless trading.
- It is possible to **short-sell without costs** and to borrow at the risk-free rate.

These assumptions are critical to deriving the Black-Scholes formula, which is the basis for calculating option prices. While these conditions may not hold exactly in real markets, they provide a framework for understanding how option values depend on factors such as stock price volatility, time to expiration, and interest rates.

**Merton's Structural Model** for credit risk builds on the Black-Scholes option-pricing framework, applying it to the valuation of corporate debt. In this model, a firm's equity is viewed as a call option on its assets, with the debt acting as the strike price. If the firm's asset value falls below the debt level at maturity, the firm defaults, allowing for an estimation of the probability of default, credit spread, and other key credit risk metrics. This structural approach connects the firm's asset volatility, debt level, and other market conditions to assess the firm's credit risk.

This app allows a quick and straightforward interaction with the models by simply plugging the desired inputs in. Despite the models being quite simplistic, they do provide a good proxy for the interested quantities, thus making the app an accessible tool in the determination of significant financial metrics.
""")

st.markdown("Made by: Carlo Thellung [🔗](https://www.linkedin.com/in/carlo-thellung-056b11248/)")