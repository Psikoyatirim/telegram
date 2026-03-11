
import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from tradingview_screener import get_all_symbols
import warnings
import requests # Import the requests library for sending HTTP requests
import schedule
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def ehlers_fisher_transform(data, length=10, repaint=False):
    """
    Calculates the Ehlers Fisher Transform and generates entry/exit signals.

    Args:
        data (pd.DataFrame): DataFrame with 'close' prices.
        length (int): Lookback period for highest/lowest values.
        repaint (bool): If True, current bar's Fisher value is used for signals.
                        If False, previous bar's Fisher value is used.

    Returns:
        pd.DataFrame: Original DataFrame with 'maxSrc', 'minSrc', 'sto', 'v1', 'fish',
                      'Entry', and 'Exit' columns added.
    """
    df = data.copy()

    # Calculate the highest and lowest values for the given length
    df['maxSrc'] = df['close'].rolling(window=length).max()
    df['minSrc'] = df['close'].rolling(window=length).min()

    # Calculate the stochastic oscillator
    # Avoid division by zero by checking df['maxSrc'] - df['minSrc'] != 0
    df['sto'] = np.where(df['maxSrc'] - df['minSrc'] != 0,
                         (df['close'] - df['minSrc']) / (df['maxSrc'] - df['minSrc']), 0)

    # Initialize v1 and fish columns
    df['v1'] = 0.0
    df['fish'] = 0.0

    # Calculate v1 and fish values iteratively
    # The loop starts from 'length' to ensure enough historical data is available
    for i in range(length, len(df)):
        v1_prev = df.at[i-1, 'v1']
        # Apply the Fisher Transform formula for v1, clamping values between -0.999 and 0.999
        v1_new = max(min((0.33 * 2 * (df.at[i, 'sto'] - 0.5)) + (0.67 * v1_prev), 0.999), -0.999)
        df.at[i, 'v1'] = v1_new

        fish_prev = df.at[i-1, 'fish']
        # Apply the Fisher Transform formula for fish
        fish_new = (0.5 * np.log((1 + v1_new) / (1 - v1_new))) + (0.5 * fish_prev)
        df.at[i, 'fish'] = fish_new

    # If repaint is False, shift Fisher values to prevent look-ahead bias
    if repaint == False:
        df['fish'] = df['fish'].shift(1)

    # Generate entry and exit signals based on Fisher Transform crossovers
    df['Entry'] = df['fish'] > df['fish'].shift(1)
    df['Exit'] = df['fish'] < df['fish'].shift(1)
    return df

def bollinger_bands(data, window=20, num_std_dev=2):
    """
    Calculates Bollinger Bands (Middle, Upper, Lower).

    Args:
        data (pd.DataFrame): DataFrame with 'close' prices.
        window (int): Rolling window for SMA and standard deviation.
        num_std_dev (int): Number of standard deviations for upper and lower bands.
    Returns:
        pd.DataFrame: Original DataFrame with 'MiddleBand', 'UpperBand', and 'LowerBand' columns added.
    """
    df = data.copy()
    # Calculate Middle Band (Simple Moving Average)
    df['MiddleBand'] = df['close'].rolling(window=window).mean()
    # Calculate Standard Deviation
    df['StdDev'] = df['close'].rolling(window=window).std()
    # Calculate Upper Band
    df['UpperBand'] = df['MiddleBand'] + (df['StdDev'] * num_std_dev)
    # Calculate Lower Band
    df['LowerBand'] = df['MiddleBand'] - (df['StdDev'] * num_std_dev)
    return df

def send_telegram_message(bot_token, chat_id, message):
    """
    Sends a message to a Telegram chat.

    Args:
        bot_token (str): Your Telegram bot token.
        chat_id (str): The chat ID where the message will be sent.
        message (str): The text message to send.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        'parse_mode': 'HTML' # Use HTML for basic formatting
    }
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status() # Raise an exception for HTTP errors
        print("Telegram mesajı başarıyla gönderildi.")
    except requests.exceptions.RequestException as e:
        print(f"Telegram mesajı gönderilirken hata oluştu: {e}")

def stock_scanner():
    """
    Ana tarama fonksiyonu - tüm hisse senetlerini tarar ve sinyalleri kontrol eder
    """
    print(f"\n{'='*50}")
    print(f"Tarama başlatılıyor: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    # Telegram bot ayarları
    bot_token = '8035211094:AAEqHt4ZosBJsuT1FxdCcTR9p9uJ1O073zY'
    bot_chatID = '-1002715468798'

    # Initialize TvDatafeed for accessing TradingView data
    tv = TvDatafeed()

    # Get all symbols for the Turkish market (BIST) and clean them
    Hisseler = get_all_symbols(market='turkey')
    Hisseler = [symbol.replace('BIST:', '') for symbol in Hisseler]
    Hisseler = sorted(Hisseler) # Sort the list of symbols

    # Define titles for the signal report DataFrame
    Titles = ['Hisse Adı', 'Son Fiyat', 'Giriş Sinyali (Fisher & BB)', 'Çıkış Sinyali (Fisher)']
    df_signals = pd.DataFrame(columns=Titles)

    # Iterate through each stock to fetch data and generate signals
    for hisse in Hisseler:
        try:
            # Fetch 4-hour historical data for the last 500 bars
            data = tv.get_hist(symbol=hisse, exchange='BIST', interval=Interval.in_4_hour, n_bars=500)
            if data is None or data.empty:
                print(f"No data found for {hisse}. Skipping.")
                continue

            data = data.reset_index() # Reset index to make 'datetime' a column

            # Calculate Fisher Transform signals
            FisherSignal = ehlers_fisher_transform(data, length=9, repaint=True)

            # Calculate Bollinger Bands (default window=20, num_std_dev=2)
            FisherSignal = bollinger_bands(FisherSignal, window=20, num_std_dev=2)

            # Rename columns for consistency and set 'datetime' as index
            FisherSignal.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
            FisherSignal.set_index('datetime', inplace=True)

            Buy = False
            Sell = False

            # Get the last two rows to check for signal crossovers
            Signals = FisherSignal.tail(2)
            Signals = Signals.reset_index() # Reset index for easy access with .loc

            # Fisher Transform Entry Condition
            fisher_entry_condition = (Signals.loc[0, 'Entry'] == False and Signals.loc[1, 'Entry'] == True)

            # Bollinger Band Lower Band Crossover Condition
            # Check if current close crossed above the lower band from below or equal
            bb_lower_band_crossover = (Signals.loc[0, 'Close'] <= Signals.loc[0, 'LowerBand'] and
                                       Signals.loc[1, 'Close'] > Signals.loc[1, 'LowerBand'])

            # Combined Buy Signal: Fisher Entry AND BB Lower Band Crossover
            Buy = fisher_entry_condition and bb_lower_band_crossover

            # Sell Signal: Fisher Transform Exit (unchanged)
            Sell = (Signals.loc[0, 'Exit'] == False and Signals.loc[1, 'Exit'] == True)

            Last_Price = Signals.loc[1, 'Close'] # Get the latest closing price

            # Prepare the list for the DataFrame row
            L1 = [hisse, Last_Price, str(Buy), str(Sell)]
            df_signals.loc[len(df_signals)] = L1 # Add the row to the DataFrame
            print(L1) # Print the signal for the current stock

        except Exception as e:
            # Catch any errors during data fetching or signal calculation for a specific stock
            print(f"Error processing {hisse}: {e}")
            pass # Continue to the next stock even if an error occurs

    # Filter and print stocks with a 'True' Buy signal
    df_True = df_signals[(df_signals['Giriş Sinyali (Fisher & BB)'] == 'True')]
    print("\n--- Stocks with Buy Signals (Fisher Transform & Bollinger Band Crossover) ---")
    print(df_True)

    # Prepare message for Telegram
    telegram_output_messages = []
    if not df_True.empty:
        for index, row in df_True.iterrows():
            hisse_adi = row['Hisse Adı']
            son_fiyat = row['Son Fiyat']
            # Format the message for each stock similar to the image
            message_part = (
                f"🔥 <b>GÜÇLÜ SİNYAL</b>\n"
                f"🚀 <b>{hisse_adi}</b> - {son_fiyat:.2f}₺\n"
                f"📊 Fisher & BB Sinyal: Dip Bulucu\n" # Updated line
            )
            telegram_output_messages.append(message_part)

        # Join all individual stock messages into one
        full_telegram_message = "\n".join(telegram_output_messages)
        send_telegram_message(bot_token, bot_chatID, full_telegram_message)
    else:
        send_telegram_message(bot_token, bot_chatID, "Bugün yeni alım sinyali bulunmamaktadır.")

    print(f"\nTarama tamamlandı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

def main():
    """
    Ana program - otomatik taramayı başlatır
    """
    print("🚀 Otomatik Hisse Tarama Sistemi Başlatılıyor...")
    print("📅 Her 30 dakikada bir tarama yapılacak")
    print("⏰ Durdurmak için Ctrl+C tuşlayın\n")

    # İlk taramayı hemen yap
    print("İlk tarama başlatılıyor...")
    stock_scanner()

    # Her 30 dakikada bir tarama yapmak için zamanla
    schedule.every(30).minutes.do(stock_scanner)

    print(f"\n⏰ Sonraki tarama: {datetime.now().strftime('%H:%M:%S')} + 30 dakika")

    # Sürekli çalışma döngüsü
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Her dakika kontrol et
    except KeyboardInterrupt:
        print("\n\n🛑 Program kullanıcı tarafından durduruldu.")
        print("👋 Görüşmek üzere!")

if __name__ == "__main__":
    main()
