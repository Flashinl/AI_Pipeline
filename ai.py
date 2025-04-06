# =============================================================================
# Enhanced Omega Pipeline – Ultra-Advanced Stock Prediction System
# User inputs a stock ticker and the system predicts its growth potential
# =============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import time
import requests
import json
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Sentiment Analysis Libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Technical Indicators
from ta import add_all_ta_features
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

# Macroeconomic Data
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

# Google Trends
try:
    from pytrends.request import TrendReq
    TRENDS_AVAILABLE = True
except ImportError:
    TRENDS_AVAILABLE = False

# Social Data: Twitter & Reddit
try:
    import tweepy
    TWITTER_AVAILABLE = True
except ImportError:
    TWITTER_AVAILABLE = False

try:
    import praw
    REDDIT_AVAILABLE = True
except ImportError:
    REDDIT_AVAILABLE = False

# Machine Learning & Ensembling
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Deep Learning 
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, GRU, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Experiment Logging
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# For advanced backtest and trading simulation
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False

# For order book and market microstructure (would use real API in production)
try:
    import ccxt  # For order book data simulation
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# For options data
try:
    import mibian  # For options pricing models
    MIBIAN_AVAILABLE = True
except ImportError:
    MIBIAN_AVAILABLE = False

# -----------------------------------------------------------------------------
# 1. Enhanced Data Sourcing & Alternative Data
# -----------------------------------------------------------------------------

def get_stock_data(ticker, start_date, end_date, include_dividends=True):
    """
    Get historical stock data with additional metrics.
    """
    try:
        # Download base data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(df) == 0:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Include dividends and stock splits if requested
        if include_dividends:
            div_df = yf.Ticker(ticker).dividends
            if not div_df.empty:
                div_df = div_df.loc[start_date:end_date]
                df['Dividends'] = div_df
                df['Dividends'].fillna(0, inplace=True)
        
        # Get stock info for additional metadata
        stock_info = yf.Ticker(ticker).info
        sector = stock_info.get('sector', 'Unknown')
        industry = stock_info.get('industry', 'Unknown')
                
        # Format and return data
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"Downloaded {len(df)} days of data for {ticker} ({sector})")
        return df, sector, industry
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
        return pd.DataFrame(), "Unknown", "Unknown"

def get_spy_data(start_date, end_date):
    """Get S&P 500 data for market comparison"""
    try:
        spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False)
        spy_df.reset_index(inplace=True)
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        spy_df.rename(columns={
            'Close': 'SPY_Close',
            'Volume': 'SPY_Volume',
            'Open': 'SPY_Open',
            'High': 'SPY_High',
            'Low': 'SPY_Low'
        }, inplace=True)
        return spy_df[['Date', 'SPY_Close', 'SPY_Volume', 'SPY_Open', 'SPY_High', 'SPY_Low']]
    except Exception as e:
        print(f"Error downloading SPY data: {e}")
        return pd.DataFrame()

def get_sector_etf_data(sector, start_date, end_date):
    """Get sector ETF data based on stock's sector"""
    sector_etfs = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Healthcare': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Industrials': 'XLI',
        'Communication Services': 'XLC',
        'Consumer Defensive': 'XLP',
        'Energy': 'XLE',
        'Basic Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
    }
    
    etf = sector_etfs.get(sector, 'SPY')  # Default to SPY if sector not found
    try:
        etf_df = yf.download(etf, start=start_date, end=end_date, progress=False)
        etf_df.reset_index(inplace=True)
        etf_df['Date'] = pd.to_datetime(etf_df['Date'])
        etf_df.rename(columns={
            'Close': f'{etf}_Close',
            'Volume': f'{etf}_Volume'
        }, inplace=True)
        return etf_df[['Date', f'{etf}_Close', f'{etf}_Volume']]
    except Exception as e:
        print(f"Error downloading sector ETF data: {e}")
        return pd.DataFrame()

def get_vix_data(start_date, end_date):
    """Get VIX volatility index data"""
    try:
        vix_df = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        vix_df.reset_index(inplace=True)
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix_df = vix_df[['Date', 'Close']]
        vix_df.rename(columns={'Close': 'VIX'}, inplace=True)
        return vix_df
    except Exception as e:
        print(f"Error downloading VIX data: {e}")
        return pd.DataFrame()

def get_yield_curve_data(start_date, end_date):
    """Get yield curve data (10Y-2Y spread)"""
    if not FRED_AVAILABLE:
        return pd.DataFrame()
    
    try:
        fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with actual key
        dgs2 = fred.get_series('DGS2', start_date, end_date)  # 2-year Treasury
        dgs10 = fred.get_series('DGS10', start_date, end_date)  # 10-year Treasury
        
        # Calculate spread
        spread = dgs10 - dgs2
        df = pd.DataFrame({
            'Date': spread.index,
            'Yield_Curve': spread.values
        })
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        print(f"Error downloading yield curve data: {e}")
        return pd.DataFrame()

def get_macro_indicators(start_date, end_date):
    """Get key macroeconomic indicators"""
    if not FRED_AVAILABLE:
        return pd.DataFrame()
    
    try:
        fred = Fred(api_key='YOUR_FRED_API_KEY')  # Replace with actual key
        indicators = {
            'UNRATE': 'Unemployment_Rate',
            'CPIAUCSL': 'CPI',
            'FEDFUNDS': 'Fed_Funds_Rate',
            'M2': 'M2_Money_Supply',
            'INDPRO': 'Industrial_Production'
        }
        
        macro_df = pd.DataFrame()
        
        for code, name in indicators.items():
            series = fred.get_series(code, start_date, end_date)
            if macro_df.empty:
                macro_df = pd.DataFrame({
                    'Date': series.index,
                    name: series.values
                })
            else:
                macro_df[name] = pd.Series(index=series.index, data=series.values)
        
        # Fill missing values using forward fill (appropriate for monthly data)
        macro_df = macro_df.ffill()
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        return macro_df
    except Exception as e:
        print(f"Error downloading macroeconomic data: {e}")
        return pd.DataFrame()

def get_fundamental_data(ticker):
    """Get fundamental data for a stock"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get financial statements
        balance_sheet = stock.balance_sheet
        income_stmt = stock.income_stmt
        cash_flow = stock.cashflow
        
        # Extract key metrics
        fundamentals = {}
        
        # Latest values from income statement
        if not income_stmt.empty and income_stmt.shape[1] > 0:
            fundamentals['Revenue'] = income_stmt.loc['Total Revenue'].iloc[0]
            fundamentals['NetIncome'] = income_stmt.loc['Net Income'].iloc[0]
            if 'EBITDA' in income_stmt.index:
                fundamentals['EBITDA'] = income_stmt.loc['EBITDA'].iloc[0]
            
        # Latest values from balance sheet
        if not balance_sheet.empty and balance_sheet.shape[1] > 0:
            fundamentals['TotalAssets'] = balance_sheet.loc['Total Assets'].iloc[0]
            fundamentals['TotalDebt'] = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            fundamentals['TotalEquity'] = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
        
        # Calculate ratios
        info = stock.info
        fundamentals['MarketCap'] = info.get('marketCap', 0)
        fundamentals['PE_Ratio'] = info.get('trailingPE', 0)
        fundamentals['PB_Ratio'] = info.get('priceToBook', 0)
        fundamentals['Dividend_Yield'] = info.get('dividendYield', 0)
        fundamentals['Profit_Margin'] = info.get('profitMargins', 0)
        
        # Return single row DataFrame with fundamental metrics
        df = pd.DataFrame([fundamentals])
        return df
    except Exception as e:
        print(f"Error retrieving fundamental data for {ticker}: {e}")
        return pd.DataFrame()

def get_analyst_recommendations(ticker):
    """Get analyst recommendations and price targets"""
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is not None and not recommendations.empty:
            # Get the most recent recommendations
            recent_recs = recommendations.sort_index(ascending=False).head(10)
            
            # Calculate consensus metrics
            buy_count = sum(recent_recs['To Grade'].isin(['Buy', 'Outperform', 'Strong Buy']))
            sell_count = sum(recent_recs['To Grade'].isin(['Sell', 'Underperform', 'Strong Sell']))
            hold_count = sum(recent_recs['To Grade'].isin(['Hold', 'Neutral', 'Equal-Weight']))
            
            total_recs = buy_count + sell_count + hold_count
            if total_recs > 0:
                buy_ratio = buy_count / total_recs
                sell_ratio = sell_count / total_recs
                hold_ratio = hold_count / total_recs
            else:
                buy_ratio = sell_ratio = hold_ratio = 0
                
            # Get price target information
            target_info = stock.info
            current_price = target_info.get('currentPrice', 0)
            target_price = target_info.get('targetMeanPrice', 0)
            target_upside = (target_price / current_price - 1) * 100 if current_price > 0 else 0
            
            analyst_data = {
                'Buy_Ratio': buy_ratio,
                'Sell_Ratio': sell_ratio,
                'Hold_Ratio': hold_ratio,
                'Target_Price': target_price,
                'Target_Upside': target_upside
            }
            
            return pd.DataFrame([analyst_data])
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving analyst recommendations for {ticker}: {e}")
        return pd.DataFrame()

def get_google_trends(keyword, start_date, end_date):
    """Get Google Trends data for a keyword"""
    if not TRENDS_AVAILABLE:
        return pd.DataFrame()
    
    try:
        pytrend = TrendReq()
        timeframe = f"{start_date} {end_date}"
        pytrend.build_payload(kw_list=[keyword], timeframe=timeframe)
        trends_df = pytrend.interest_over_time()
        
        if trends_df.empty:
            return pd.DataFrame()
            
        if 'isPartial' in trends_df.columns:
            trends_df = trends_df.drop(columns=['isPartial'])
            
        trends_df.reset_index(inplace=True)
        trends_df.rename(columns={keyword: 'Trend'}, inplace=True)
        return trends_df
    except Exception as e:
        print(f"Error retrieving Google Trends data: {e}")
        return pd.DataFrame()

def get_options_data(ticker):
    """Get options data for implied volatility and put/call ratio"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get options expiration dates
        expirations = stock.options
        
        if not expirations:
            return pd.DataFrame()
            
        # Get nearest expiration date
        next_exp = expirations[0]
        
        # Get options chain
        opt = stock.option_chain(next_exp)
        
        calls = opt.calls
        puts = opt.puts
        
        # Calculate total volume and open interest
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        call_oi = calls['openInterest'].sum()
        put_oi = puts['openInterest'].sum()
        
        # Calculate put/call ratios
        vol_put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
        oi_put_call_ratio = put_oi / call_oi if call_oi > 0 else 0
        
        # Get ATM options for implied volatility
        stock_price = stock.info.get('currentPrice', 0)
        if stock_price > 0:
            calls['strike_diff'] = abs(calls['strike'] - stock_price)
            puts['strike_diff'] = abs(puts['strike'] - stock_price)
            
            atm_call = calls.loc[calls['strike_diff'].idxmin()]
            atm_put = puts.loc[puts['strike_diff'].idxmin()]
            
            # Average the implied volatility
            avg_iv = (atm_call['impliedVolatility'] + atm_put['impliedVolatility']) / 2
        else:
            avg_iv = 0
            
        options_data = {
            'Put_Call_Volume_Ratio': vol_put_call_ratio,
            'Put_Call_OI_Ratio': oi_put_call_ratio,
            'Implied_Volatility': avg_iv
        }
        
        return pd.DataFrame([options_data])
    except Exception as e:
        print(f"Error retrieving options data for {ticker}: {e}")
        return pd.DataFrame()

def get_institutional_ownership(ticker):
    """Get institutional ownership data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get institutional holders
        inst_holders = stock.institutional_holders
        
        if inst_holders is not None and not inst_holders.empty:
            total_shares = stock.info.get('sharesOutstanding', 0)
            
            if total_shares > 0:
                # Calculate total institutional ownership
                total_inst_shares = inst_holders['Shares'].sum()
                inst_ownership_pct = total_inst_shares / total_shares
                
                # Get top 5 holders percentage
                top5_pct = inst_holders.nlargest(5, 'Shares')['Shares'].sum() / total_shares
                
                inst_data = {
                    'Institutional_Ownership': inst_ownership_pct,
                    'Top5_Ownership': top5_pct,
                    'Institutional_Holder_Count': len(inst_holders)
                }
                
                return pd.DataFrame([inst_data])
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving institutional ownership data for {ticker}: {e}")
        return pd.DataFrame()

def get_insider_trading(ticker):
    """Get insider trading data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get insider transactions
        insiders = stock.insiderTransactions
        
        if insiders is not None and not insiders.empty:
            # Filter to recent transactions (last 3 months)
            recent = insiders[insiders['startDate'] >= (datetime.datetime.now() - datetime.timedelta(days=90))]
            
            if not recent.empty:
                # Calculate buy/sell ratio
                buys = recent[recent['transactionDescription'].str.contains('Buy', case=False)]
                sells = recent[recent['transactionDescription'].str.contains('Sell', case=False)]
                
                buy_vol = buys['shares'].sum() if not buys.empty else 0
                sell_vol = sells['shares'].sum() if not sells.empty else 0
                
                # Calculate net insider activity
                net_activity = buy_vol - sell_vol
                buy_sell_ratio = buy_vol / sell_vol if sell_vol > 0 else float('inf')
                
                insider_data = {
                    'Insider_Net_Activity': net_activity,
                    'Insider_Buy_Sell_Ratio': buy_sell_ratio if buy_sell_ratio != float('inf') else 10,
                    'Insider_Transaction_Count': len(recent)
                }
                
                return pd.DataFrame([insider_data])
        
        return pd.DataFrame()
    except Exception as e:
        print(f"Error retrieving insider trading data for {ticker}: {e}")
        return pd.DataFrame()

def get_sentiment_data(ticker, start_date, end_date):
    """
    Placeholder for comprehensive sentiment analysis.
    In a production system, this would connect to Twitter, Reddit, news APIs, etc.
    """
    # Initialize sentiment data
    date_range = pd.date_range(start=start_date, end=end_date)
    sentiment_df = pd.DataFrame({'Date': date_range})
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    
    # Simulate sentiment data with some randomness but following a pattern
    np.random.seed(42)  # For reproducibility
    
    # Base sentiment 
    base_sentiment = np.random.normal(0.1, 0.3, len(date_range))
    
    # Add some trends and patterns
    for i in range(len(date_range)):
        # Weekend effect - lower sentiment volatility on weekends
        if date_range[i].weekday() >= 5:  # 5=Saturday, 6=Sunday
            base_sentiment[i] *= 0.5
        
        # End of month effect - slightly more positive
        if date_range[i].day >= 25:
            base_sentiment[i] += 0.1
            
        # Clip to reasonable range [-1, 1]
        base_sentiment[i] = max(min(base_sentiment[i], 1.0), -1.0)
    
    sentiment_df['News_Sentiment'] = base_sentiment
    
    # Add some variation for different sources
    sentiment_df['Social_Sentiment'] = sentiment_df['News_Sentiment'] * 0.7 + np.random.normal(0, 0.2, len(date_range))
    sentiment_df['Social_Sentiment'] = sentiment_df['Social_Sentiment'].clip(-1, 1)
    
    sentiment_df['Analyst_Sentiment'] = sentiment_df['News_Sentiment'] * 0.5 + np.random.normal(0.2, 0.15, len(date_range))
    sentiment_df['Analyst_Sentiment'] = sentiment_df['Analyst_Sentiment'].clip(-1, 1)
    
    # Calculate weighted average sentiment
    sentiment_df['Combined_Sentiment'] = (
        0.4 * sentiment_df['News_Sentiment'] + 
        0.4 * sentiment_df['Social_Sentiment'] + 
        0.2 * sentiment_df['Analyst_Sentiment']
    )
    
    return sentiment_df

def get_satellite_data(ticker, start_date, end_date):
    """Simulate satellite imagery data (parking lot counts, etc.)"""
    # In a production system, this would connect to actual satellite data APIs
    # For simulation, we'll create synthetic data
    
    if ticker not in ['WMT', 'TGT', 'HD', 'LOW', 'COST']:  # Only retail companies have relevant parking lot data
        return pd.DataFrame()
    
    date_range = pd.date_range(start=start_date, end=end_date)
    satellite_df = pd.DataFrame({'Date': date_range})
    satellite_df['Date'] = pd.to_datetime(satellite_df['Date'])
    
    # Generate synthetic store traffic data with weekly seasonality
    np.random.seed(43)
    base_traffic = np.random.normal(100, 10, len(date_range))
    
    # Add weekly seasonality (weekends have higher traffic)
    for i in range(len(date_range)):
        # Weekend effect
        if date_range[i].weekday() >= 5:  # Weekend
            base_traffic[i] *= 1.4
        elif date_range[i].weekday() == 4:  # Friday
            base_traffic[i] *= 1.2
        
        # Holiday effects (approximate)
        if date_range[i].month == 12 and date_range[i].day > 15:  # Christmas season
            base_traffic[i] *= 1.5
        elif date_range[i].month == 11 and date_range[i].day > 22:  # Thanksgiving/Black Friday
            base_traffic[i] *= 1.6
    
    satellite_df['Store_Traffic'] = base_traffic
    satellite_df['YoY_Traffic_Change'] = np.random.normal(0.03, 0.05, len(date_range))  # YoY change
    
    return satellite_df

def get_earnings_call_sentiment(ticker):
    """Extract sentiment from earnings call transcripts"""
    # In a production system, this would use NLP on actual transcripts
    try:
        # Simulate earnings call metrics
        sentiment_metrics = {
            'CEO_Sentiment': np.random.normal(0.2, 0.5),  # Range from -1 to 1
            'Guidance_Sentiment': np.random.normal(0.1, 0.6),
            'QA_Sentiment': np.random.normal(0, 0.4),
            'Forward_Looking_Ratio': np.random.uniform(0.2, 0.4),  # % of statements about future
            'Uncertainty_Words': np.random.randint(5, 30)  # Count of uncertainty words
        }
        
        # Clip sentiment values to -1 to 1 range
        for key in ['CEO_Sentiment', 'Guidance_Sentiment', 'QA_Sentiment']:
            sentiment_metrics[key] = max(min(sentiment_metrics[key], 1.0), -1.0)
            
        return pd.DataFrame([sentiment_metrics])
    except Exception as e:
        print(f"Error simulating earnings call sentiment: {e}")
        return pd.DataFrame()

def get_orderbook_data(ticker):
    """Simulate order book data and liquidity metrics"""
    if not CCXT_AVAILABLE:
        return pd.DataFrame()
        
    try:
        # This is a simulation - in reality, would connect to exchange API
        liquidity_metrics = {
            'Bid_Ask_Spread': np.random.uniform(0.01, 0.5),  # Percentage spread
            'Depth_5pct': np.random.uniform(100000, 5000000),  # $ depth within 5% of mid
            'Order_Imbalance': np.random.normal(0, 0.3),  # Positive means more bids than asks
            'Large_Order_Count': np.random.randint(5, 50),  # Count of large orders (>$100k)
            'Market_Impact_100k': np.random.uniform(0.05, 0.5)  # Est. price impact for $100k order
        }
        
        # Clip imbalance to reasonable range
        liquidity_metrics['Order_Imbalance'] = max(min(liquidity_metrics['Order_Imbalance'], 1.0), -1.0)
        
        return pd.DataFrame([liquidity_metrics])
    except Exception as e:
        print(f"Error simulating order book data: {e}")
        return pd.DataFrame()

def get_credit_card_data(ticker):
    """Simulate credit card transaction data for consumer companies"""
    # In production, this would connect to alternative data providers
    consumer_tickers = ['AMZN', 'AAPL', 'WMT', 'TGT', 'SBUX', 'MCD', 'NKE', 'COST', 'HD']
    
    if ticker not in consumer_tickers:
        return pd.DataFrame()
        
    try:
        # Simulate transaction metrics
        transaction_metrics = {
            'MoM_Sales_Growth': np.random.normal(0.02, 0.03),  # Month-over-month growth
            'YoY_Sales_Growth': np.random.normal(0.08, 0.07),  # Year-over-year growth
            'Avg_Transaction_Change': np.random.normal(0.01, 0.05),  # Change in average transaction value
            'Customer_Retention': np.random.uniform(0.7, 0.95),  # Retention rate
            'New_Customer_Rate': np.random.uniform(0.05, 0.25)  # Rate of new customers
        }
        
        return pd.DataFrame([transaction_metrics])
    except Exception as e:
        print(f"Error simulating credit card data: {e}")
        return pd.DataFrame()

def get_market_breadth_data(start_date, end_date):
    """Get market breadth indicators like advance/decline, new highs/lows"""
    try:
        # In production, would get from market data provider
        # For now, simulate data based on S&P 500
        spy_df = yf.download("SPY", start=start_date, end=end_date, progress=False)
        spy_df.reset_index(inplace=True)
        spy_df['Date'] = pd.to_datetime(spy_df['Date'])
        
        # Create synthetic breadth data
        breadth_df = pd.DataFrame({'Date': spy_df['Date']})
        
        # Advance-Decline is somewhat correlated with market movement but has unique info
        breadth_df['Adv_Dec_Ratio'] = spy_df['Close'].pct_change().apply(
            lambda x: np.random.normal(2.0 if x > 0 else 0.5, 0.5))
        
        # New Highs / New Lows
        spy_returns = spy_df['Close'].pct_change(21)  # Look at monthly returns
        breadth_df['New_Highs'] = spy_returns.apply(
            lambda x: np.random.randint(50, 200) if x > 0.02 else np.random.randint(10, 60))
        breadth_df['New_Lows'] = spy_returns.apply(
            lambda x: np.random.randint(50, 200) if x < -0.02 else np.random.randint(10, 60))
        
        # McClellan Oscillator and Summation Index (synthetic)
        breadth_df['McClellan_Osc'] = spy_df['Close'].rolling(window=19).mean().diff() / 10
        breadth_df['McClellan_Sum'] = breadth_df['McClellan_Osc'].cumsum()
        
        return breadth_df
    except Exception as e:
        print(f"Error getting market breadth data: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# 2. Enhanced Technical Indicators & Feature Engineering
# -----------------------------------------------------------------------------
def compute_technical_indicators(df):
    """Compute comprehensive technical indicators"""
    try:
        # Basic indicators using TA library
        df = add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
        )
        
        # Calculate returns at different timeframes
        df['Daily_Return'] = df['Close'].pct_change()
        df['Weekly_Return'] = df['Close'].pct_change(5)
        df['Monthly_Return'] = df['Close'].pct_change(21)
        df['Quarterly_Return'] = df['Close'].pct_change(63)
        
        # Additional momentum indicators
        rsi = RSIIndicator(df['Close'])
        df['RSI'] = rsi.rsi()
        
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Volatility indicators
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'])
        df['ATR'] = atr.average_true_range()
        df['ATR_Pct'] = df['ATR'] / df['Close']  # ATR as percentage of price
        
        bb = BollingerBands(df['Close'])
        df['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        df['BB_Position'] = (df['Close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Trend indicators
        macd = MACD(df['Close'])
        df['MACD_Line'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Moving averages at different timeframes
        for window in [10, 20, 50, 100, 200]:
            sma = SMAIndicator(df['Close'], window=window)
            ema = EMAIndicator(df['Close'], window=window)
            df[f'SMA_{window}'] = sma.sma_indicator()
            df[f'EMA_{window}'] = ema.ema_indicator()
            
            # Price relative to moving average
            df[f'Price_to_SMA_{window}'] = df['Close'] / df[f'SMA_{window}'] - 1
            df[f'Price_to_EMA_{window}'] = df['Close'] / df[f'EMA_{window}'] - 1
        
        # Volume indicators
        df['Dollar_Volume'] = df['Close'] * df['Volume']
        df['Rel_Volume'] = df['Volume'] / df['Volume'].rolling(window=30).mean()
        
        # Volatility over different time periods
        for window in [5, 10, 21, 63]:
            df[f'Volatility_{window}d'] = df['Daily_Return'].rolling(window=window).std()
        
        # Cross-sectional momentum (relative to SPY)
        if 'SPY_Close' in df.columns:
            df['Rel_Strength_SPY'] = df['Close'] / df['Close'].shift(21) / (df['SPY_Close'] / df['SPY_Close'].shift(21))
            df['Excess_Return'] = df['Daily_Return'] - df['SPY_Close'].pct_change()
        
        # Advanced pattern detection
        # Detect price gaps
        df['Gap_Up'] = ((df['Low'] > df['High'].shift(1)) * 1)
        df['Gap_Down'] = ((df['High'] < df['Low'].shift(1)) * 1)
        
        # Calculate volume profile metrics
        df['Close_Quartile'] = pd.qcut(df['Close'], 4, labels=False)
        
        # Market regime (high vol vs low vol)
        vix_median = df['VIX'].median() if 'VIX' in df.columns else 20
        df['High_Vol_Regime'] = (df['VIX'] > vix_median) * 1 if 'VIX' in df.columns else 0
        
        return df
    
    except Exception as e:
        print(f"Error computing technical indicators: {e}")
        return df

def add_calendar_features(df):
    """Add calendar-based features"""
    try:
        # Extract basic date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['Day'] = df['Date'].dt.day
        df['Weekday'] = df['Date'].dt.weekday
        df['Is_Month_End'] = df['Date'].dt.is_month_end * 1
        df['Is_Month_Start'] = df['Date'].dt.is_month_start * 1
        df['Is_Quarter_End'] = df['Date'].dt.is_quarter_end * 1
        df['Is_Quarter_Start'] = df['Date'].dt.is_quarter_start * 1
        df['Is_Year_End'] = df['Date'].dt.is_year_end * 1
        df['Is_Year_Start'] = df['Date'].dt.is_year_start * 1
        
        # Weekend effect
        df['Is_Weekend'] = (df['Weekday'] >= 5) * 1
        
        # Day of week one-hot encoding
        for i in range(5):  # Monday to Friday
            df[f'Weekday_{i}'] = (df['Weekday'] == i) * 1
            
        # Month one-hot encoding
        for i in range(1, 13):
            df[f'Month_{i}'] = (df['Month'] == i) * 1
            
        # Market seasonality
        # January effect
        df['January_Effect'] = (df['Month'] == 1) * 1
        
        # Turn of the month effect (last trading day and first three of month)
        df['Month_Turn'] = ((df['Is_Month_End'] == 1) | 
                           (df['Day'] <= 3)) * 1
                           
        # End of quarter effect
        df['Quarter_End_Effect'] = ((df['Date'].dt.month.isin([3, 6, 9, 12])) & 
                                   (df['Date'].dt.day >= 28)) * 1
        
        # Add holiday proximity flags (would be more comprehensive in production)
        # For simplicity, just flagging December holiday season
        df['Holiday_Season'] = ((df['Month'] == 12) & (df['Day'] >= 15)) * 1
        
        return df
    except Exception as e:
        print(f"Error adding calendar features: {e}")
        return df

def create_interaction_features(df):
    """Create interaction features between different data sources"""
    try:
        # Interact technical and sentiment
        if 'Combined_Sentiment' in df.columns:
            df['RSI_x_Sentiment'] = df['RSI'] * df['Combined_Sentiment']
            df['Return_x_Sentiment'] = df['Daily_Return'] * df['Combined_Sentiment']
            
        # Interact technical and volume
        df['RSI_x_RelVolume'] = df['RSI'] * df['Rel_Volume']
        
        # Interact volatility and trend
        df['Trend_x_Vol'] = df['Price_to_SMA_50'] * df['Volatility_21d']
        
        # Interact market regime and stock metrics
        if 'VIX' in df.columns:
            df['High_Vol_x_Beta'] = df['High_Vol_Regime'] * df['Beta'] if 'Beta' in df.columns else 0
            df['VIX_x_Return'] = df['VIX'] * df['Daily_Return']
        
        # Fundamental x Technical interactions
        if 'PE_Ratio' in df.columns:
            df['PE_x_Momentum'] = df['PE_Ratio'] * df['Weekly_Return']
        
        # Analyst sentiment x Technical interactions
        if 'Target_Upside' in df.columns:
            df['Target_x_RSI'] = df['Target_Upside'] * df['RSI'] / 100
        
        return df
    except Exception as e:
        print(f"Error creating interaction features: {e}")
        return df

def create_target_variables(df, horizon=[1, 5, 21, 63], threshold=0.0):
    """Create target variables for different prediction horizons"""
    try:
        for days in horizon:
            # Forward returns
            df[f'Forward_Return_{days}d'] = df['Close'].pct_change(days).shift(-days)
            
            # Binary classification targets
            df[f'Target_{days}d'] = (df[f'Forward_Return_{days}d'] > threshold) * 1
            
            # Multi-class targets 
            # -1: significant loss, 0: flat, 1: moderate gain, 2: significant gain
            conditions = [
                (df[f'Forward_Return_{days}d'] < -0.05),
                (df[f'Forward_Return_{days}d'] < 0.02),
                (df[f'Forward_Return_{days}d'] < 0.05),
                (df[f'Forward_Return_{days}d'] >= 0.05)
            ]
            values = [-1, 0, 1, 2]
            df[f'Target_Class_{days}d'] = np.select(conditions, values)
            
            # Risk-adjusted targets (normalized by volatility)
            df[f'Forward_Return_Vol_Adj_{days}d'] = df[f'Forward_Return_{days}d'] / df['Volatility_21d']
            
            # Calculate excess returns compared to SPY
            if 'SPY_Close' in df.columns:
                spy_fwd_return = df['SPY_Close'].pct_change(days).shift(-days)
                df[f'Forward_Excess_Return_{days}d'] = df[f'Forward_Return_{days}d'] - spy_fwd_return
                df[f'Target_Excess_{days}d'] = (df[f'Forward_Excess_Return_{days}d'] > 0) * 1
            
        return df
    except Exception as e:
        print(f"Error creating target variables: {e}")
        return df
        
# -----------------------------------------------------------------------------
# 3. Enhanced Machine Learning Pipeline
# -----------------------------------------------------------------------------
def prepare_ml_data(df, target_days=5, classification=True, test_size=0.2):
    """Prepare data for machine learning"""
    try:
        # Select target variable based on horizon and task type
        if classification:
            target_col = f'Target_{target_days}d'
        else:
            target_col = f'Forward_Return_{target_days}d'
            
        print(f"Preparing ML data for {target_col}")
        
        # Remove rows with NaN in target
        df = df.dropna(subset=[target_col])
        
        # Exclude non-feature columns
        exclude_cols = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',  # Price data
            'SPY_Close', 'SPY_Volume', 'SPY_Open', 'SPY_High', 'SPY_Low',  # Market data
            'Dividends', 'Stock Splits'  # Corporate actions
        ]
        
        # Also exclude target columns and forward-looking data
        forward_cols = [col for col in df.columns if 'Forward_' in col or 'Target_' in col]
        exclude_cols = exclude_cols + forward_cols
        
        # Get feature columns by excluding non-features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Filter out features with too many NaNs
        valid_features = []
        for col in feature_cols:
            nan_ratio = df[col].isna().mean()
            if nan_ratio < 0.3:  # Allow up to 30% missing values
                valid_features.append(col)
            else:
                print(f"Excluding {col} due to {nan_ratio:.1%} missing values")
        
        # Fill remaining NaNs with column medians
        for col in valid_features:
            if df[col].isna().any():
                med = df[col].median()
                df[col] = df[col].fillna(med)
        
        # Convert to numpy arrays
        X = df[valid_features].values
        y = df[target_col].values
        
        # Split into train and test sets (time-aware split)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Prepared {X_train.shape[1]} features, {len(X_train)} training samples, {len(X_test)} testing samples")
        
        return X_train, X_test, y_train, y_test, valid_features, scaler
    
    except Exception as e:
        print(f"Error preparing ML data: {e}")
        return None, None, None, None, None, None

def train_baseline_models(X_train, y_train, X_test, y_test, classification=True):
    """Train baseline machine learning models"""
    models = {}
    results = {}
    
    try:
        # Define models based on task type
        if classification:
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
                'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
            }
            
            # Add models if available
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
            
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            # Regression models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
            
            # Add models if available
            if XGBOOST_AVAILABLE:
                models['XGBoost'] = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
            
            if LIGHTGBM_AVAILABLE:
                models['LightGBM'] = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Make predictions
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            
            # Evaluate
            if classification:
                train_acc = accuracy_score(y_train, train_preds)
                test_acc = accuracy_score(y_test, test_preds)
                
                results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'class_report': classification_report(y_test, test_preds, output_dict=True)
                }
                
                print(f"{name} - Train accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}")
            else:
                train_mse = mean_squared_error(y_train, train_preds)
                test_mse = mean_squared_error(y_test, test_preds)
                
                results[name] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
                
                print(f"{name} - Train RMSE: {np.sqrt(train_mse):.4f}, Test RMSE: {np.sqrt(test_mse):.4f}")
        
        return models, results
    
    except Exception as e:
        print(f"Error training baseline models: {e}")
        return {}, {}

def train_ensemble_model(models, X_train, y_train, classification=True):
    """Create an ensemble of the best models"""
    try:
        if len(models) < 2:
            print("Not enough models for ensemble")
            return None
            
        print("Training ensemble model...")
        
        # Configure ensemble based on task type
        if classification:
            ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in models.items()],
                voting='soft'  # Use probability estimates for classification
            )
        else:
            ensemble = VotingRegressor(
                estimators=[(name, model) for name, model in models.items()]
            )
            
        # Train the ensemble
        ensemble.fit(X_train, y_train)
        
        return ensemble
    
    except Exception as e:
        print(f"Error training ensemble model: {e}")
        return None

def hypertune_best_model(best_model_name, X_train, y_train, X_test, y_test, classification=True):
    """Hypertune the best performing model using Optuna"""
    if not OPTUNA_AVAILABLE:
        print("Optuna not available for hyperparameter tuning")
        return None
        
    try:
        print(f"Hypertuning {best_model_name} model...")
        
        # Define the objective function based on model type
        def objective(trial):
            if best_model_name == 'RandomForest':
                if classification:
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 5, 30),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=42
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 5, 30),
                        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                        random_state=42
                    )
                    
            elif best_model_name == 'XGBoost' and XGBOOST_AVAILABLE:
                if classification:
                    model = XGBClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=42
                    )
                else:
                    model = XGBRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=42
                    )
                    
            elif best_model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
                if classification:
                    model = LGBMClassifier(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=42
                    )
                else:
                    model = LGBMRegressor(
                        n_estimators=trial.suggest_int('n_estimators', 50, 300),
                        max_depth=trial.suggest_int('max_depth', 3, 10),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        random_state=42
                    )
            else:
                # Fallback to logistic regression
                if classification:
                    model = LogisticRegression(
                        C=trial.suggest_float('C', 0.01, 10.0, log=True),
                        max_iter=1000,
                        random_state=42
                    )
                else:
                    # Nothing to tune for basic linear regression
                    return 0
                
            # Train the model    
            model.fit(X_train, y_train)
            
            # Evaluate based on task type
            if classification:
                preds = model.predict(X_test)
                score = accuracy_score(y_test, preds)
            else:
                preds = model.predict(X_test)
                score = -mean_squared_error(y_test, preds)  # Negative because Optuna minimizes
                
            return score
            
        # Create and run the study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print(f"Best parameters: {study.best_params}")
        print(f"Best score: {study.best_value}")
        
        # Create and train the best model
        if best_model_name == 'RandomForest':
            if classification:
                best_model = RandomForestClassifier(
                    **study.best_params,
                    random_state=42
                )
            else:
                best_model = RandomForestClassifier(
                    **study.best_params,
                    random_state=42
                )
                
        elif best_model_name == 'XGBoost' and XGBOOST_AVAILABLE:
            if classification:
                best_model = XGBClassifier(
                    **study.best_params,
                    random_state=42
                )
            else:
                best_model = XGBRegressor(
                    **study.best_params,
                    random_state=42
                )
                
        elif best_model_name == 'LightGBM' and LIGHTGBM_AVAILABLE:
            if classification:
                best_model = LGBMClassifier(
                    **study.best_params,
                    random_state=42
                )
            else:
                best_model = LGBMRegressor(
                    **study.best_params,
                    random_state=42
                )
        else:
            if classification:
                best_model = LogisticRegression(
                    **study.best_params,
                    max_iter=1000,
                    random_state=42
                )
            else:
                return None
                
        # Train the best model
        best_model.fit(X_train, y_train)
        
        return best_model
        
    except Exception as e:
        print(f"Error hypertuning model: {e}")
        return None

def train_lstm_model(X_train, y_train, X_test, y_test, sequence_length=10, classification=True):
    """Train deep learning LSTM model for time series prediction"""
    if not TF_AVAILABLE:
        print("TensorFlow not available for deep learning")
        return None, None
        
    try:
        print("Training LSTM model...")
        
        # Transform data into sequences
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(len(X) - seq_length):
                X_seq.append(X[i:i + seq_length])
                y_seq.append(y[i + seq_length])
            return np.array(X_seq), np.array(y_seq)
            
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
        
        print(f"LSTM sequence data shapes: {X_train_seq.shape}, {y_train_seq.shape}")
        
        # Define model architecture
        n_features = X_train.shape[1]
        
        if classification:
            # Binary classification model
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, n_features)),
                Dropout(0.2),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            # Regression model
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=True), input_shape=(sequence_length, n_features)),
                Dropout(0.2),
                Bidirectional(LSTM(32)),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1)
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True)
        ]
        
        # Train the model
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_seq, y_test_seq),
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history
        
    except Exception as e:
        print(f"Error training LSTM model: {e}")
        return None, None

def evaluate_model_performance(model, X_test, y_test, classification=True):
    """Comprehensive evaluation of model performance"""
    try:
        # Get predictions
        if hasattr(model, 'predict_proba') and classification:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] >= 2:  # Binary classification with 2+ classes
                y_prob = y_prob[:, 1]  # Probability of positive class
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test)
            y_prob = None
            
        # Evaluation metrics
        results = {}
        
        if classification:
            # Classification metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            
            # If probabilities are available, calculate ROC AUC
            if y_prob is not None:
                from sklearn.metrics import roc_auc_score, roc_curve
                results['roc_auc'] = roc_auc_score(y_test, y_prob)
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                results['roc_curve'] = (fpr, tpr)
                
                # Calculate precision-recall curve
                from sklearn.metrics import precision_recall_curve, average_precision_score
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                results['pr_curve'] = (precision, recall)
                results['avg_precision'] = average_precision_score(y_test, y_prob)
        else:
            # Regression metrics
            results['mse'] = mean_squared_error(y_test, y_pred)
            results['rmse'] = np.sqrt(results['mse'])
            results['mae'] = np.mean(np.abs(y_test - y_pred))
            
            # Calculate R-squared
            from sklearn.metrics import r2_score
            results['r2'] = r2_score(y_test, y_pred)
            
        return results
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}

def feature_importance_analysis(model, feature_names):
    """Analyze and visualize feature importance"""
    try:
        # Check if model has feature importance attribute
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return None
            
        # Create sorted feature importance DataFrame
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_imp = feature_imp.sort_values('Importance', ascending=False)
        
        return feature_imp
        
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return None
