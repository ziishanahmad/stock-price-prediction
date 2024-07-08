

# Stock Price Prediction

## Overview
This project aims to develop a machine learning model to predict stock prices using historical data from Yahoo Finance. The dataset contains the daily stock prices of a specific company over a certain period.

## Key Features
- **Data Preprocessing:** Fetching, cleaning, and scaling the stock price data.
- **Model Building:** Training a Linear Regression model to predict future stock prices.
- **Model Evaluation:** Assessing model performance using Mean Squared Error (MSE) and R-squared (R2) score.
- **Model Visualization:** Visualizing actual vs predicted stock prices.

## Installation

### Clone the Repository
To get started, clone this repository to your local machine using the following command:
```sh
git clone https://github.com/ziishanahmad/stock-price-prediction.git
cd stock-price-prediction
```

### Set Up a Virtual Environment
It is recommended to use a virtual environment to manage your dependencies. You can set up a virtual environment using `venv`:
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Required Libraries
Install the necessary libraries using `pip`:
```sh
pip install -r requirements.txt
```

## Usage

### Run the Jupyter Notebook
Open the Jupyter notebook to run the project step-by-step:
1. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Open the `stock_price_prediction.ipynb` notebook.
3. Run the cells step-by-step to preprocess the data, train the model, evaluate its performance, and visualize the results.

## Detailed Explanation of the Code

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Fetch the Data
```python
ticker = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-01-01'

stock_data = yf.download(ticker, start=start_date, end=end_date)
print(stock_data.head())
print(stock_data.describe())
```

### Exploratory Data Analysis (EDA)
```python
plt.figure(figsize=(12, 6))
plt.plot(stock_data['Close'])
plt.title('Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.show()

plt.figure(figsize=(12, 10))
sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

### Data Preprocessing
```python
data = stock_data[['Close']].copy()
data['Prediction'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Close']].values
y = data['Prediction'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Build the Model
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Evaluate the Model
```python
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Prices')
plt.plot(y_pred, label='Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Close Price USD ($)')
plt.legend()
plt.show()
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- The stock price data is fetched from Yahoo Finance using the `yfinance` library.
- The developers of TensorFlow for their deep learning framework.

## Contact
For any questions or feedback, please contact:
- **Name:** Zeeshan Ahmad
- **Email:** ziishanahmad@gmail.com
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)
