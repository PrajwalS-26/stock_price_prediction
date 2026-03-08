import yfinance as yf
import pandas as pd


def load_stock_data(ticker, period="10y"):

    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    df.reset_index(inplace=True)

    # Check if Adj Close exists
    if "Adj Close" in df.columns:
        df["Price"] = df["Adj Close"]
    else:
        df["Price"] = df["Close"]

    # Keep useful columns
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Price"]]
    df.dropna(inplace=True)

    return df


if __name__ == "__main__":

    data = load_stock_data("AAPL")

    print(data.head())