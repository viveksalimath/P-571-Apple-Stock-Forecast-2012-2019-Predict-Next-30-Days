
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.title("Apple Stock Forecast — 30 Days")

uploaded = st.file_uploader("Upload CSV with Date & Close columns", type=["csv"])
lookback = st.number_input("Lookback window", min_value=10, max_value=200, value=60, step=5)
epochs = st.number_input("Epochs", min_value=1, max_value=200, value=10, step=1)

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip().title() for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    st.line_chart(df.set_index('Date')['Close'])

    # Train simple LSTM
    scaler = MinMaxScaler((0,1))
    data = scaler.fit_transform(df[['Close']].values)

    def make_seq(data, n):
        X,y = [],[]
        for i in range(n, len(data)):
            X.append(data[i-n:i,0])
            y.append(data[i,0])
        X = np.array(X).reshape(-1,n,1)
        y = np.array(y)
        return X,y

    X,y = make_seq(data, lookback)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X,y,epochs=epochs,batch_size=32,verbose=0)

    # Forecast next 30
    seq = data[-lookback:].copy()
    preds = []
    for _ in range(30):
        p = model.predict(seq.reshape(1,lookback,1), verbose=0)[0,0]
        preds.append(p)
        seq = np.vstack([seq[1:], [p]])

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()

    future_dates = pd.date_range(df['Date'].max()+pd.Timedelta(days=1), periods=30)
    out = pd.DataFrame({"date": future_dates, "forecast_close": preds})

    st.line_chart(out.set_index('date'))
    st.download_button("Download Forecast CSV", out.to_csv(index=False), file_name="forecast_30d.csv")
