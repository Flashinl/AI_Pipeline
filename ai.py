import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from transformers import pipeline, logging as hf_logging
import warnings
from datetime import datetime, timedelta
import pandas_datareader.data as web
from newsapi import NewsApiClient # pip install newsapi-python
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
# Suppress warnings for a cleaner experience
warnings.filterwarnings('ignore')
hf_logging.set_verbosity_error() # Suppress verbose transformers logs

# IMPORTANT: You need a free API key from https://newsapi.org/
# Replace 'YOUR_API_KEY' with your actual key to enable real-time news sentiment.
NEWS_API_KEY = 'YOUR_API_KEY' 

# --- Section 1: Advanced Data Collection ---

def get_user_input():
    """Gets ticker and prediction horizon from the user interactively."""
    ticker = input("Enter the stock ticker symbol (e.g., NVDA, TSLA): ").upper()
    while True:
        try:
            horizon = int(input("Enter the prediction horizon in days (e.g., 7, 30, 90): "))
            if horizon > 0:
                return ticker, horizon
            else:
                print("Horizon must be a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a number for the horizon.")

def get_stock_data(ticker, period="10y"):
    """Fetches and cleans historical stock data, including SPY for relative strength."""
    print(f"Fetching historical data for {ticker} and SPY...")
    # Fetch ticker data
    stock_df = yf.Ticker(ticker).history(period=period)[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_df.index = stock_df.index.tz_localize(None) # Remove timezone for consistency
    
    # Fetch SPY data for market context
    spy_df = yf.Ticker('SPY').history(period=period)[['Close']]
    spy_df.index = spy_df.index.tz_localize(None)
    spy_df.rename(columns={'Close': 'SPY_Close'}, inplace=True)
    
    # Merge and calculate relative performance
    merged_df = stock_df.merge(spy_df, left_index=True, right_index=True, how='left')
    return merged_df

def get_fundamental_data(ticker):
    """Fetches key fundamental data points for the company."""
    print(f"Fetching fundamental data for {ticker}...")
    try:
        stock_info = yf.Ticker(ticker).info
        fundamentals = {
            'market_cap': stock_info.get('marketCap'),
            'forward_pe': stock_info.get('forwardPE'),
            'trailing_pe': stock_info.get('trailingPE'),
            'price_to_sales': stock_info.get('priceToSalesTrailing12Months'),
            'forward_eps': stock_info.get('forwardEps'),
            'enterprise_to_revenue': stock_info.get('enterpriseToRevenue'),
            'beta': stock_info.get('beta'),
            'profit_margins': stock_info.get('profitMargins')
        }
        return fundamentals
    except Exception as e:
        print(f"Could not fetch fundamental data for {ticker}: {e}")
        return {}

def get_economic_indicators(start_date, end_date):
    """Fetches real macroeconomic data from the Federal Reserve (FRED)."""
    print("Fetching economic indicators from FRED...")
    # T10YIE: Market's inflation expectation
    # T10Y2Y: Yield curve slope (recession indicator)
    # VIXCLS: Volatility Index
    economic_symbols = ['T10YIE', 'T10Y2Y', 'VIXCLS']
    try:
        eco_data = web.DataReader(economic_symbols, 'fred', start_date, end_date)
        eco_data.rename(columns={'T10YIE': 'inflation_expectation', 'T10Y2Y': 'yield_curve', 'VIXCLS': 'vix'}, inplace=True)
        return eco_data
    except Exception as e:
        print(f"Could not fetch FRED data: {e}")
        return pd.DataFrame()

def get_real_news_sentiment(ticker, api_key):
    """Fetches and analyzes real news headlines from NewsAPI."""
    if api_key == 'YOUR_API_KEY' or not api_key:
        print("Warning: NewsAPI key not provided. Skipping real news sentiment analysis.")
        return pd.DataFrame(columns=['date', 'sentiment_score'])
    
    print(f"Fetching and analyzing real news for {ticker}...")
    newsapi = NewsApiClient(api_key=api_key)
    from_date = (datetime.now() - timedelta(days=29)).strftime('%Y-%m-%d')
    try:
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', from_param=from_date)['articles']
        if not articles:
            return pd.DataFrame(columns=['date', 'sentiment_score'])
            
        sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        news_data = []
        for article in articles:
            try:
                # Truncate headline to avoid model length limits
                headline = article['title'][:512]
                sentiment = sentiment_pipeline(headline)[0]
                score = sentiment['score'] if sentiment['label'] == 'positive' else -sentiment['score']
                news_data.append({
                    "date": pd.to_datetime(article['publishedAt']).normalize(),
                    "sentiment_score": score
                })
            except Exception:
                continue # Skip articles that cause errors
                
        sentiment_df = pd.DataFrame(news_data)
        # Average the sentiment for each day
        daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
        return daily_sentiment

    except Exception as e:
        print(f"Error fetching or analyzing news: {e}. Check your API key and usage limits.")
        return pd.DataFrame(columns=['date', 'sentiment_score'])

# --- Section 2: Feature Engineering ---

def create_features(df):
    """Calculates a robust set of technical, volatility, and momentum indicators."""
    print("Engineering features...")
    # Simple and Exponential Moving Averages
    for w in [10, 20, 50, 200]:
        df[f'SMA_{w}'] = df['Close'].rolling(window=w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Average True Range (ATR) - for Volatility
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR_14'] = np.max(ranges, axis=1).rolling(14).mean()

    # Relative Strength vs. SPY
    df['Rel_Strength_SPY'] = (df['Close'] / df['SMA_200']) / (df['SPY_Close'] / df['SPY_Close'].rolling(window=200).mean())
    
    # Price Rate of Change
    df['ROC_20'] = df['Close'].pct_change(20)
    
    return df

# --- Section 3: Model Training and Evaluation ---

def plot_feature_importance(model, features):
    """Visualizes the most important features from the trained model."""
    importances = model.get_booster().get_score(importance_type='weight')
    if not importances:
        print("Could not get feature importances.")
        return
        
    importance_df = pd.DataFrame({'Feature': list(importances.keys()), 'Importance': list(importances.values())})
    importance_df = importance_df.sort_values('Importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Top 15 Most Important Features for Prediction', fontsize=16)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

def run_prediction_pipeline(ticker, horizon):
    """Executes the entire data-to-prediction pipeline."""
    # Step 1: Collect all data
    price_data = get_stock_data(ticker)
    fundamental_data = get_fundamental_data(ticker)
    eco_data = get_economic_indicators(price_data.index.min(), price_data.index.max())
    news_sentiment = get_real_news_sentiment(ticker, NEWS_API_KEY)
    
    # Step 2: Engineer features
    featured_df = create_features(price_data)

    # Step 3: Merge all data sources into a single dataframe
    print("Merging all data sources...")
    master_df = featured_df.merge(eco_data, left_index=True, right_index=True, how='left')
    master_df.ffill(inplace=True) # Forward-fill for days without new economic data
    
    if not news_sentiment.empty:
        master_df = master_df.merge(news_sentiment, left_index=True, right_on='date', how='left')
        master_df.set_index('date', inplace=True, drop=True)
        master_df['sentiment_score'] = master_df['sentiment_score'].fillna(0)
    else:
        master_df['sentiment_score'] = 0.0

    for key, value in fundamental_data.items():
        master_df[key] = value

    # Step 4: Prepare data for machine learning
    master_df['target'] = master_df['Close'].shift(-horizon)
    final_df = master_df.dropna()
    
    features = [f for f in final_df.columns if f not in ['target', 'SPY_Close']]
    # Filter out non-numeric features just in case
    features = [f for f in features if pd.api.types.is_numeric_dtype(final_df[f])]

    X = final_df[features]
    y = final_df['target']
    
    if len(X) < 100:
        print("Not enough data to train the model after processing. Exiting.")
        return

    # Step 5: Train the XGBoost model with robust parameters
    print(f"Training XGBoost model on {len(X)} data points...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use a chronological split for time-series data
    train_size = int(len(X_scaled) * 0.9)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    xg_reg = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=2000, learning_rate=0.01,
        max_depth=6, subsample=0.7, colsample_bytree=0.7, random_state=42, n_jobs=-1,
        early_stopping_rounds=50 # Add early stopping
    )
    xg_reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Step 6: Evaluate the model
    preds = xg_reg.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"\nModel Evaluation RMSE on Test Set: {rmse:.4f}")

    # Step 7: Make a future prediction
    print("\n" + "="*35)
    print("--- PREDICTION FOR THE FUTURE ---")
    last_data_point = final_df[features].iloc[[-1]]
    last_data_point_scaled = scaler.transform(last_data_point)
    future_prediction = xg_reg.predict(last_data_point_scaled)
    
    current_price = last_data_point['Close'].values[0]
    predicted_price = future_prediction[0]
    predicted_growth_pct = ((predicted_price - current_price) / current_price) * 100

    print(f"Ticker: {ticker}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Predicted Price in {horizon} days: ${predicted_price:,.2f}")
    print(f"Predicted Growth Potential: {predicted_growth_pct:.2f}%")
    print("="*35 + "\n")

    # Step 8: Explain the prediction
    plot_feature_importance(xg_reg, features)

if __name__ == '__main__':
    TICKER, PREDICTION_HORIZON = get_user_input()
    run_prediction_pipeline(TICKER, PREDICTION_HORIZON)
