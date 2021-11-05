import streamlit
import streamlit as st
from datetime import date

import yfinance
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction")

stocks = ("AAPL", "GOOG", "MSFT", "GME", "PAGEIND.NS", "RELIANCE.NS", "HDFC.NS")

selected_stock = st.selectbox("Select dataset for predictions", stocks)

n_years = st.slider("Years of Predictions", 1, 4)
periods = n_years * 365


@st.cache
def load_data(ticker):
    data = yfinance.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text("loading...")
data = load_data(selected_stock)
data_load_state.text("loading data...done")

st.subheader("Raw Data")
st.write(data.tail(10))


def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="Open"))
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(fig)


plot_raw_data()

# Forecasting
df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": 'ds', "Close": 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=periods)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail(10))

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast Component")
fig2 = m.plot_components(forecast)
st.write(fig2)
