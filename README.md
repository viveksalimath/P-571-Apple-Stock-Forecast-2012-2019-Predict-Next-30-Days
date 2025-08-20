# 📈 P-571: Apple Stock Forecasting (2012–2019 → Next 30 Days)

## 📌 Business Objective  
The goal of this project is to **predict Apple’s stock price for the next 30 days** using historical OHLC data (2012–2019).  
We explore both **short-term and long-term trends**, evaluate performance, and deploy a working prototype.

---

## 📂 Project Workflow  
1. **Kickoff & Objective Discussion**  
2. **Dataset Preparation** (Apple OHLC data 2012–2019)  
3. **Exploratory Data Analysis (EDA)**  
   - Trend analysis (Close price, moving averages)  
   - Volatility (High–Low daily range)  
   - Rolling statistics  
4. **Model Building**  
   - Train/Test Split → 2019 as test set  
   - Models: **LSTM (Deep Learning)** for sequential forecasting  
5. **Model Evaluation**  
   - Metrics: MAE, RMSE, MAPE  
   - Visualization of predicted vs actual values  
6. **30-Day Forecast**  
   - Recursive prediction for future 30 days  
   - Save forecast as CSV  
7. **Deployment**  
   - **Streamlit Web App** to upload stock data and generate forecasts  

---

## 🛠️ Tech Stack  
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow/Keras, Streamlit  
- **Deployment:** Streamlit  

---

## 📊 Results  
- Built an LSTM model to capture sequential dependencies.  
- Evaluated on **2019 test set** with MAE, RMSE, and MAPE.  
- Generated **30-day forward forecast** (saved as `apple_30day_forecast.csv`).  

---

## 🚀 How to Run  

### 🔹 Clone Repository
```bash
git clone https://github.com/<your-username>/P571-Apple-Stock-Forecast.git
cd P571-Apple-Stock-Forecast
```

### 🔹 Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔹 Run Jupyter Notebook
```bash
jupyter notebook P571_Apple_Stock_Forecast.ipynb
```

### 🔹 Run Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 📂 Repository Structure
```
├── P571_Apple_Stock_Forecast.ipynb   # Main notebook (EDA + Model + Forecast)
├── streamlit_app.py                  # Deployment app
├── apple_30day_forecast.csv          # Sample forecast output
├── AAPL.csv                          # Input dataset
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## 📌 Future Work  
- Add external event features (earnings, product launches, macroeconomic events).  
- Try other models (ARIMA, Prophet, GRU).  
- Deploy as API for real-time forecasting.  

---

## 👨‍💻 Author  
**Vivek V Salimath**  
AI/ML Engineer | Data Scientist | Generative AI Enthusiast  
📧 [salimath823@gmail.com](mailto:salimath823@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/vivek-v-salimath)  
