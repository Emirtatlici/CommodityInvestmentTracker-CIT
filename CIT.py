import requests
import matplotlib.pyplot as plt
import pandas as pd
import logging
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, Any
from colorama import Fore, Style, init
import argparse
import matplotlib.dates as mdates
pd.options.mode.chained_assignment = None 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CommodityInvestmentTracker:
    def __init__(self):
        self.gold_df = None
        self.silver_df = None

    def get_commodity_data(self, start_date: str, commodity_type: str) -> pd.DataFrame:
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

        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logging.error(f"Error fetching data from API: {e}")
            raise

        df = pd.DataFrame(data)
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime('%d-%m-%Y')
        df = df[["Date", "Close"]]
        df = df.rename(columns={'Close': price_column_name})
        df[price_column_name] = pd.to_numeric(df[price_column_name], errors='coerce')
        df.set_index('Date', inplace=True)

        return df


    def analyze_investment(self, df: pd.DataFrame, start_date: str, initial_investment: float = 100, end_date: str = None) -> Dict[str, Any]:
        start_date = pd.to_datetime(start_date, format='%d-%m-%Y')
        
        if end_date is None:
            end_date = pd.to_datetime('today')
        else:
            end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
        
        df.index = pd.to_datetime(df.index, format='%d-%m-%Y')
        
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df_filtered.empty:
            raise ValueError("No data available in the specified date range.")
        
        start_price = df_filtered.iloc[0, 0]
        units_bought = initial_investment / start_price
        
        df_filtered['Investment_Value'] = df_filtered.iloc[:, 0] * units_bought
        
        min_value_after_start = df_filtered['Investment_Value'].min()
        max_value = df_filtered['Investment_Value'].max()
        final_value = df_filtered['Investment_Value'].iloc[-1]
        total_return = final_value - initial_investment
        years = (end_date - start_date).days / 365.25
        annualized_return = (final_value / initial_investment) ** (1 / years) - 1 if years > 0 else 0

        plt.figure(figsize=(15, 10))

        metrics_text = (f"Initial Investment: ${initial_investment:.2f}\n"
                        f"Final Value: ${final_value:.2f}\n"
                        f"Total Return: ${total_return:.2f}\n"
                        f"Annualized Return: {annualized_return:.2%}")
        plt.suptitle(f"Investment Analysis from {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')}\n{metrics_text}", fontsize=12, fontweight='bold')

        plt.subplot(2, 2, 1)
        plt.plot(df_filtered.index, df_filtered.iloc[:, 0], label='Price')
        plt.title('Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(df_filtered.index, df_filtered['Investment_Value'], label='Investment Value', color='green')
        plt.title('Investment Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value (USD)')
        plt.legend()

        plt.subplot(2, 2, 3)
        rolling_window = 30
        df_filtered['Rolling_Mean'] = df_filtered.iloc[:, 0].rolling(window=rolling_window).mean()
        df_filtered['Rolling_STD'] = df_filtered.iloc[:, 0].rolling(window=rolling_window).std()
        plt.plot(df_filtered.index, df_filtered['Rolling_Mean'], label='Rolling Mean', color='orange')
        plt.plot(df_filtered.index, df_filtered['Rolling_STD'], label='Rolling Standard Deviation', color='purple')
        plt.title(f'Rolling Mean and Std Dev ({rolling_window}-Days)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(2, 2, 4)
        df_filtered['Daily_Returns'] = df_filtered.iloc[:, 0].pct_change() * 100
        df_filtered['Month'] = df_filtered.index.month
        df_filtered['Year'] = df_filtered.index.year
        heatmap_data = df_filtered.pivot_table(values='Daily_Returns', index='Year', columns='Month', aggfunc='mean')
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Heatmap of Monthly Mean Daily Returns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return "Analysis Completed"    

    def analyze_and_plot_periodic_investment(self, df: pd.DataFrame, start_date: str, end_date: str, 
                                             interval_days: int, investment_amount: float, 
                                             commodity_type: str) -> Dict[str, Any]:
        start_date = pd.to_datetime(start_date, format='%d-%m-%Y')
        end_date = pd.to_datetime(end_date, format='%d-%m-%Y')
        df.index = pd.to_datetime(df.index, format='%d-%m-%Y')
        
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            raise ValueError("Belirtilen tarih aralığı için veri bulunamadı.")

        investment_dates = pd.date_range(start=start_date, end=end_date, freq=f'{interval_days}D')
        total_invested = 0
        units_bought = 0
        investment_growth = []

        for date in investment_dates:
            if date in df.index:
                price = df.loc[date].iloc[0]
                units_bought += investment_amount / price
                total_invested += investment_amount
                current_value = units_bought * price
                investment_growth.append({'Date': date, 'Value': current_value, 'Total_Invested': total_invested})

        growth_df = pd.DataFrame(investment_growth)
        growth_df.set_index('Date', inplace=True)

        final_value = units_bought * df.iloc[-1].iloc[0]
        total_return = final_value - total_invested
        years = (end_date - start_date).days / 365.25
        annualized_return = (final_value / total_invested) ** (1 / years) - 1 if years > 0 else 0

        plt.figure(figsize=(15, 10))
        
        metrics_text = (f"Periodical Investment Analysis:\n"
                        f"Total invested: ${total_invested:.2f}\n"
                        f"Final Value: ${final_value:.2f}\n"
                        f"Total Return: ${total_return:.2f}\n"
                        f"Annualized Return: {annualized_return:.2%}")

        plt.suptitle(f"{commodity_type.capitalize()} Investment Analysis\n{metrics_text}", fontsize=12, fontweight='bold')

        plt.subplot(2, 2, 1)
        plt.plot(df.index, df.iloc[:, 0], label='Price')
        plt.title(f'{commodity_type.capitalize()} Price in Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(growth_df.index, growth_df['Value'], label='Investment Value', color='green')
        plt.plot(growth_df.index, growth_df['Total_Invested'], label='Total Invested', color='red', linestyle='--')
        plt.title('Periodical Investment Growth In time')
        plt.xlabel('Date')
        plt.ylabel('Value (USD)')
        plt.legend()

        plt.subplot(2, 2, 3)
        rolling_window = 30
        df['Rolling_Mean'] = df.iloc[:, 0].rolling(window=rolling_window).mean()
        df['Rolling_STD'] = df.iloc[:, 0].rolling(window=rolling_window).std()
        plt.plot(df.index, df['Rolling_Mean'], label='Rolling Mean', color='orange')
        plt.plot(df.index, df['Rolling_STD'], label='Rolling Standard Deviation', color='purple')
        plt.title(f'Rolling Mean and Rolling Std. ({rolling_window}-Days)')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()

        plt.subplot(2, 2, 4)
        df['Daily_Returns'] = df.iloc[:, 0].pct_change() * 100
        df['Month'] = df.index.month
        df['Year'] = df.index.year
        heatmap_data = df.pivot_table(values='Daily_Returns', index='Year', columns='Month', aggfunc='mean')
        sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f")
        plt.title('Heatmap of Monthly mean of the daily returns')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return "Analysis Completed"
        
   

    def save_to_excel(self, df: pd.DataFrame, file_path: str):
        try:
            df.to_excel(file_path, index=True)
            print(f"{Fore.GREEN}Data successfully saved to {file_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error saving data to Excel: {e}{Style.RESET_ALL}")

    def compare_commodities(self, start_date: str, end_date: str):
        try:
            self.gold_df = self.get_commodity_data(start_date, commodity_type='gold')
            self.silver_df = self.get_commodity_data(start_date, commodity_type='silver')
            
            self.gold_df.index = pd.to_datetime(self.gold_df.index, format='%d-%m-%Y')
            self.silver_df.index = pd.to_datetime(self.silver_df.index, format='%d-%m-%Y')
            
            common_dates = self.gold_df.index.intersection(self.silver_df.index)
            self.gold_df = self.gold_df.loc[common_dates]
            self.silver_df = self.silver_df.loc[common_dates]
            
            start = pd.to_datetime(start_date, format='%d-%m-%Y')
            end = pd.to_datetime(end_date, format='%d-%m-%Y')
            mask = (self.gold_df.index >= start) & (self.gold_df.index <= end)
            self.gold_df = self.gold_df.loc[mask]
            self.silver_df = self.silver_df.loc[mask]
            
            gold_normalized = self.gold_df['Gold_USD_Price'] / self.gold_df['Gold_USD_Price'].iloc[0] * 100
            silver_normalized = self.silver_df['Silver_USD_Price'] / self.silver_df['Silver_USD_Price'].iloc[0] * 100
            
            gold_return = (self.gold_df['Gold_USD_Price'].iloc[-1] - self.gold_df['Gold_USD_Price'].iloc[0]) / self.gold_df['Gold_USD_Price'].iloc[0] * 100
            silver_return = (self.silver_df['Silver_USD_Price'].iloc[-1] - self.silver_df['Silver_USD_Price'].iloc[0]) / self.silver_df['Silver_USD_Price'].iloc[0] * 100
            
            plt.figure(figsize=(12, 6))
            plt.plot(self.gold_df.index, gold_normalized, label='Gold')
            plt.plot(self.silver_df.index, silver_normalized, label='Silver')
            plt.title('Gold vs Silver Price Comparison (Normalized)')
            plt.xlabel('Date')
            plt.ylabel('Normalized Price (Base = 100)')
            plt.legend()
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.gcf().autofmt_xdate()  
            
            plt.tight_layout()
            plt.show()
            
            print(f"{Fore.YELLOW}Performance Comparison:{Style.RESET_ALL}")
            print(f"Gold return: {gold_return:.2f}%")
            print(f"Silver return: {silver_return:.2f}%")
        except Exception as e:
            logging.error(f"Error in compare_commodities: {e}")
            print(f"{Fore.RED}Error comparing commodities: {e}{Style.RESET_ALL}")
def main():
    parser = argparse.ArgumentParser(description="Commodity Investment Analyzer")
    parser.add_argument("--commodity", choices=['gold', 'silver'], help="Commodity type (gold or silver)")
    parser.add_argument("--start_date", help="Start date in dd-mm-yyyy format")
    parser.add_argument("--end_date", help="End date in dd-mm-yyyy format")
    parser.add_argument("--interval", type=int, help="Investment interval in days")
    parser.add_argument("--amount", type=float, help="Investment amount for each period")
    args = parser.parse_args()

    print(f"{Fore.CYAN}******Commodity Investment Analysis******{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Welcome to the Commodity Investment Analyzer{Style.RESET_ALL}\n")

    analyzer = CommodityInvestmentTracker()

    if args.commodity and args.start_date:
        commodity_type = args.commodity
        start_date = args.start_date
    else:
        commodity_type = input(f"{Fore.YELLOW}Please enter the commodity type (gold/silver): {Style.RESET_ALL}").lower()
        start_date = input(f"{Fore.YELLOW}Enter the starting date (dd-mm-yyyy): {Style.RESET_ALL}")
    
    try:
        df = analyzer.get_commodity_data(start_date, commodity_type)
        print(f"{Fore.GREEN}Data fetched successfully.{Style.RESET_ALL}")
    except Exception as e:
        logging.error(f"Error getting commodity data: {e}")
        print(f"{Fore.RED}Error getting commodity data: {e}{Style.RESET_ALL}")
        return

    while True:
        print(f"\n{Fore.CYAN}What would you like to do?{Style.RESET_ALL}")
        print("1. Analyze periodic investments")