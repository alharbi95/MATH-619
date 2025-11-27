# MATH 619

Paper title: Hybrid CNN-LSTM Stock Price Forecasting Using Technical and Sentiment Indicators

This repository contains the full code, documentation, and supporting files for my MATH-619 graduate project at King Fahd University of Petroleum and Minerals (KFUPM).  
The project explores deep learning–based stock movement prediction using a hybrid **CNN–LSTM architecture**, combining **technical indicators**, **market volatility (VIX)**, and **Twitter sentiment**.

---

##  Project Overview

Financial markets are highly dynamic and sentiment-driven. Traditional models often struggle to integrate both numerical (technical) and psychological (sentiment) factors.  
This project aims to bridge this gap by:

- Developing a **hybrid CNN-LSTM model** to capture spatial (CNN) and temporal (LSTM) patterns  
- Integrating **TradingView technical indicators** (2015–2025)  
- Incorporating **VIX volatility index**  
- Enhancing predictions with **Twitter sentiment** (pre-scored Kaggle dataset, 2021–2022)  
- Comparing:
  - **Model A:** Technical-only  
  - **Model B:** Technical + Sentiment  
- Using **Explainable AI** to interpret model behavior

---

##  Dataset Summary

### **Technical Data (2015–2025)**
Source: TradingView  
Companies: AAPL, AMZN, GOOG, MSFT, TSLA  
Indicators:
- EMA  
- RSI  
- MACD & MACD Histogram  
- Bollinger Bands  
- Volume  
- OHLC  

### **Market Fear Index (VIX)**
Source: TradingView  
Years: 2015–2025  

### **Twitter Sentiment (2021–2022)**
Source: Kaggle (*Stock Tweets for Sentiment Analysis and Prediction*)  
Notes:
- Sentiment already scored → no NLP needed  
- Aggregated into daily average sentiment + daily tweet volume  

>  Full datasets are **not included** in this repository due to size and licensing.  
> Only **sample data** and **data-processing scripts** are provided.

---

##  Models

### **Model A – Technical-Only**
- Inputs: Technical indicators + VIX  
- Training period: 2015–2025  
- Evaluation period: 2021–2022  

### **Model B – Sentiment-Enhanced**
- Inputs: Technical indicators + VIX + Twitter sentiment + Tweet volume  
- Training & evaluation period: 2021–2022  

Both models are evaluated on the **same final 12-month period** to ensure fair comparison.

---

##  Repository Structure

MATH-619/
│
├── data/ # sample datasets (no full raw data)
│ ├── sample_technical.csv
│ ├── sample_sentiment.csv
│
├── notebooks/
│ ├── 01_data_preparation.ipynb
│ ├── 02_sentiment_integration.ipynb
│ ├── 03_modelA_cnn_lstm.ipynb
│ ├── 04_modelB_sentiment.ipynb
│ ├── 05_shap_explainability.ipynb
│
├── src/ # Python scripts for modular code
│ ├── data_loader.py
│ ├── technical_features.py
│ ├── model_cnn_lstm.py
│ ├── evaluation.py
│
├── results/
│ ├── performance/
│ ├── shap/
│
├── README.md # this file
└── requirements.txt # pip dependencies

