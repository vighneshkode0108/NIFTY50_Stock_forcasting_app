# 📊 AI-Based Stock Price Forecasting App (NIFTY 50)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Framework](https://img.shields.io/badge/framework-Streamlit-green)

A visually enhanced and interactive AI-powered web app to forecast **NIFTY50 stock prices** using **LSTM neural networks**, with dynamic graphs, data download, and custom UI. **( Follow the below  instructions to successfully run the system.)**

---

## 🚀 Demo

[![Watch the video](https://img.youtube.com/vi/MRwdyijC6jQ/0.jpg)](https://www.youtube.com/watch?v=MRwdyijC6jQ)


---

## 🌟 Features

| Feature             | Description                                                             |
|---------------------|-------------------------------------------------------------------------|
| 📊 LSTM Forecast     | Predicts future stock prices using LSTM models                         | 
| 📈 Plotly Charts     | Dynamic, interactive price visualization with comparison graphs        | 
| 🎨 Custom UI         | Blue-themed background with modern animations                          | 
| 📁 Data Download     | Download original and forecast data as CSV                             | 
| 📂 Pre-trained Models| Loads Keras `.keras` models based on selected NIFTY50 company          |
| 🧠 Scaled Predictions| Uses MinMaxScaler to normalize input for LSTM                          | 
| ⏱️ Fast Execution    | Optimized for low-latency prediction and UI response                   |

---

## 📺 Screenshots

### Home Page
![Home Page](Screenshot2025-04-17013957.png)

### Dashboard
![Dashboard](assets/screenshot2.png)




---

## 🧠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Visualization**: Plotly
- **Backend / AI**: TensorFlow/Keras (LSTM), Scikit-learn
- **Data**: Historical stock data (.csv)
- **Model I/O**: `.keras` format
- **Styling**: Custom CSS in Streamlit markdown

---

## 📁 Project Structure
stock-forecasting-app/
│

├── data/                      # Historical CSV files for each company

├── models/                    # Trained LSTM model files (.keras)

├── utils.py                   # Utility script with NIFTY50 company mapping

├── fetch_data.py              # (Optional) Script for data retrieval

├── train_models.py            # Model training script

├── frontend.py                # Streamlit web app code

├──config.ini                  # Configuration file


---

## 📦 Installation

### 🔧 Requirements

- Python 3.8+
- pip

---

## 🐍 Create Environment

```bash
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate


```
---

## 🔧 Installation of required dependencies & librabries

### Install or update dependencies in  requirements.txt by following bash commands.
### ("Replace the dependencies in requirements.txt with their updated versions if there are any version conflicts.")
```bash
pip install -r requirements.txt

```


(Update the packages in  requirements.txt )

---

### ▶️ Run the App
```bash
#--------(Run the following Command one at a time line by line.)--------
python data_fetch.py # Use to fetch .csv files of stocks(Run Single Command at a time)

python train_models.py   # Use to train the models(Run Single Command at a time)
 
streamlit run frontend.py  # use to Run the program(Run Single Command at a time)
```

---


### 🧠 How It Works
1.Loads historical stock data (must have 100+ records) from CSV.

2.Applies a pre-trained LSTM model to forecast future prices.

3.Visualizes both actual and forecasted stock prices using Plotly charts.

4.Allows CSV download of historical and forecasted data.

---

### 🔐 Disclaimer
This project is intended for educational and research purposes only.
The forecasts are based on AI models and are not guaranteed to be accurate.
Please do not use this application for live financial decisions without consulting professionals.

---
### 🙌 Contributions
Feel free to fork, star ⭐, and contribute! If you'd like to improve the UI, integrate real-time data, or enhance accuracy, pull requests are welcome.


