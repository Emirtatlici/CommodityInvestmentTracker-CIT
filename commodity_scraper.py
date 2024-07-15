import requests
import matplotlib.pyplot as plt
import json
import pandas as pd
import logging
from datetime import datetime
import seaborn as sns

def get_commodity_data():
    start_date = input("Enter the starting date (dd-mm-yy): ")
    commodity_type = input("Please enter the commodity type (gold/silver): ")

    start_date_unix = int(datetime.strptime(start_date, '%d-%m-%Y').timestamp())

    if commodity_type.lower() == 'gold':
        instrument = "XAU/USD"
        price_column_name = "Gold_USD_Price"
    elif commodity_type.lower() == 'silver':
        instrument = "XAG/USD"
        price_column_name = "Silver_USD_Price"
    else:
        raise ValueError("Invalid commodity type. Please use 'gold' or 'silver'.")

    url = "https://www.fxempire.com/api/v1/en/commodities/chart/candles"
    querystring = {
        "instrument": instrument,
        "granularity": "D",
        "from": str(start_date_unix),
        "price": "M",
        "count": "5000"
    }

    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,tr-TR;q=0.8,tr;q=0.7",
        "api_version": "$GITHUB_SHA",
        "priority": "u=1, i",
        "referer": "https://www.fxempire.com/commodities/silver",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "Windows",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "token": "null",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    data = json.loads(response.text)

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%d-%m-%Y')
    df = df[["Date","Close"]]
    df = df.rename(columns={'Close': price_column_name})

    return df


def analyze_investment(df):
    start_date = input("Please enter the start date in format of dd-mm-yyyy: ")

    data_frame = df.copy()
    data_frame['Date'] = pd.to_datetime(data_frame['Date'], format='%d-%m-%Y')
    data_frame.set_index('Date', inplace=True)
    
    start_value = data_frame.loc[start_date].iloc[0]
    data_frame['Investment_Value'] = (100 / start_value) * data_frame.iloc[:, 0]
    data_frame_after_start = data_frame[data_frame.index >= start_date]

    min_value_after_start = data_frame_after_start['Investment_Value'].min()
    max_value = data_frame['Investment_Value'].max()
    final_value = data_frame.iloc[-1]['Investment_Value']
    total_return = final_value - 100
    annualized_return = (final_value / 100) ** (1 / ((data_frame.index[-1] - data_frame.index[0]).days / 365.25)) - 1

    metrics_text = (f"Minimum Value of Investment (After {start_date}): {min_value_after_start:.4f}\n"
                    f"Maximum Value of Investment: {max_value:.4f}\n"
                    f"Final Investment Value: ${final_value:.2f}\n"
                    f"Total Return: ${total_return:.2f}\n"
                    f"Annualized Return: {annualized_return:.2%}")

    plt.figure(figsize=(18, 14))
    plt.suptitle("Financial Metrics\n" + metrics_text, fontsize=12, fontweight='bold')

    plt.subplot(2, 2, 1)
    plt.plot(data_frame.index, data_frame.iloc[:, 0], label='Price')
    plt.title('Commodity Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(data_frame_after_start.index, data_frame_after_start['Investment_Value'], label='Investment Value', color='green')
    plt.title(f'Investment Value Over Time (100 USD Initial Investment on {start_date})')
    plt.xlabel('Date')
    plt.ylabel('Value (USD)')
    plt.legend()

    rolling_window = 30
    plt.subplot(2, 2, 3)
    data_frame['Rolling_Mean'] = data_frame.iloc[:, 0].rolling(window=rolling_window).mean()
    data_frame['Rolling_STD'] = data_frame.iloc[:, 0].rolling(window=rolling_window).std()
    plt.plot(data_frame.index, data_frame['Rolling_Mean'], label='Rolling Mean', color='orange')
    plt.plot(data_frame.index, data_frame['Rolling_STD'], label='Rolling Std Dev', color='red')
    plt.title(f'Rolling Mean and Std Dev ({rolling_window}-day)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(2, 2, 4)
    data_frame['Daily_Returns'] = data_frame.iloc[:, 0].pct_change() * 100
    data_frame['Month'] = data_frame.index.month
    data_frame['Year'] = data_frame.index.year
    heatmap_data = data_frame.pivot_table(values='Daily_Returns', index='Year', columns='Month', aggfunc='mean')
    sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f")
    plt.title('Heatmap of Mean Daily Returns by Month')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the suptitle
    plt.show()

    # stats = {
    #     "min_after_start": min_value_after_start,
    #     "max": max_value,
    #     "final_value": final_value,
    #     "total_return": total_return,
    #     "annualized_return": annualized_return,
    # }

    return "----------------------------------------"

def save_to_excel(df, file_path):
    try:
        df.to_excel(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while saving data to Excel: {e}")

