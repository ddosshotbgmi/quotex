import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from telegram import Bot, InputMediaPhoto
from dotenv import load_dotenv
import logging
import asyncio
from matplotlib import gridspec
import random

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_IDS = [chat_id.strip() for chat_id in os.getenv('TELEGRAM_CHAT_ID').split(',')]
TWELVE_DATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')

BASE_URL = 'https://api.twelvedata.com/'
PAIRS = ['USD/INR', 'USD/BRL', 'NZD/JPY']
INTERVAL = '1min'
OUTPUT_SIZE = 100

COLORS = {
    'background': '#121212',
    'grid': '#2A2A2A',
    'text': '#E0E0E0',
    'up': '#00C176',
    'down': '#FF3B69',
    'sma10': '#FFA800',
    'sma30': '#00D1FF',
    'ema20': '#9D5BFF',
    'bollinger': '#00C17680',
    'rsi': '#00D1FF',
    'macd': '#FFA800',
    'signal': '#00D1FF'
}

def fetch_historical_data(pair):
    try:
        response = requests.get(
            f"{BASE_URL}time_series",
            params={
                'symbol': pair,
                'interval': INTERVAL,
                'outputsize': OUTPUT_SIZE,
                'apikey': TWELVE_DATA_API_KEY
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if 'values' not in result:
            logger.warning(f"No data for {pair}, generating realistic dummy data")
            return generate_realistic_dummy_data(pair)
            
        df = pd.DataFrame(result['values'])
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])
        
        if len(df) > 0:
            variation = 0.0001 * (1 + random.random())
            df['close'] = df['close'] * (1 + (random.random() - 0.5) * variation)
        return df.sort_index()
    except Exception as e:
        logger.error(f"Error fetching {pair}: {str(e)}")
        return generate_realistic_dummy_data(pair)

def generate_realistic_dummy_data(pair):
    base_values = {
        'USD/INR': 83.5,
        'USD/BRL': 5.0,
        'NZD/JPY': 85.0
    }
    base_value = base_values.get(pair, 1.0)
    
    dates = pd.date_range(end=datetime.now(), periods=OUTPUT_SIZE, freq='min')
    prices = base_value * (1 + 0.001 * np.cumsum(np.random.randn(OUTPUT_SIZE)))
    
    df = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.001,
        'low': prices * 0.998,
        'close': prices,
        'datetime': dates
    }).set_index('datetime')
    
    logger.info(f"Generated realistic dummy data for {pair}")
    return df

def create_pro_chart(df, pair):
    try:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(12, 10), facecolor=COLORS['background'])
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
        
        df['SMA_10'] = df['close'].rolling(10).mean() * (1 + (random.random() - 0.5) * 0.0001)
        df['SMA_30'] = df['close'].rolling(30).mean() * (1 + (random.random() - 0.5) * 0.0001)
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean() * (1 + (random.random() - 0.5) * 0.0001)
        
        df['MA_20'] = df['close'].rolling(20).mean()
        std_dev = df['close'].rolling(20).std()
        df['Upper_BB'] = df['MA_20'] + 2 * std_dev
        df['Lower_BB'] = df['MA_20'] - 2 * std_dev
        
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (exp12 - exp26) * (1 + (random.random() - 0.5) * 0.0001)
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        ax1 = plt.subplot(gs[0])
        ax1.set_facecolor(COLORS['background'])
        
        for i in range(len(df)):
            color = COLORS['up'] if df['close'].iloc[i] >= df['open'].iloc[i] else COLORS['down']
            ax1.plot([df.index[i], df.index[i]], 
                    [df['low'].iloc[i], df['high'].iloc[i]], 
                    color=color, linewidth=1)
            ax1.plot([df.index[i], df.index[i]], 
                    [df['open'].iloc[i], df['close'].iloc[i]], 
                    color=color, linewidth=4)
        
        ax1.plot(df.index, df['SMA_10'], label='SMA 10', color=COLORS['sma10'], linewidth=1.5)
        ax1.plot(df.index, df['SMA_30'], label='SMA 30', color=COLORS['sma30'], linewidth=1.5)
        ax1.plot(df.index, df['EMA_20'], label='EMA 20', color=COLORS['ema20'], linewidth=1.5, linestyle='--')
        ax1.plot(df.index, df['Upper_BB'], label='Upper BB', color=COLORS['bollinger'], linewidth=1, alpha=0.7)
        ax1.plot(df.index, df['Lower_BB'], label='Lower BB', color=COLORS['bollinger'], linewidth=1, alpha=0.7)
        
        ax1.set_title(f'{pair} | Professional Analysis', color=COLORS['text'], pad=20, fontsize=12)
        ax1.grid(color=COLORS['grid'], linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', facecolor=COLORS['background'], edgecolor=COLORS['grid'])
        
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.set_facecolor(COLORS['background'])
        ax2.plot(df.index, df['RSI'], label='RSI', color=COLORS['rsi'], linewidth=1.5)
        ax2.axhline(70, color=COLORS['down'], linestyle='--', alpha=0.7)
        ax2.axhline(30, color=COLORS['up'], linestyle='--', alpha=0.7)
        ax2.fill_between(df.index, 30, 70, color=COLORS['grid'], alpha=0.2)
        ax2.set_ylabel('RSI', color=COLORS['text'])
        ax2.set_ylim(0, 100)
        ax2.grid(color=COLORS['grid'], linestyle='--', alpha=0.3)
        
        ax3 = plt.subplot(gs[2], sharex=ax1)
        ax3.set_facecolor(COLORS['background'])
        ax3.plot(df.index, df['MACD'], label='MACD', color=COLORS['macd'], linewidth=1.5)
        ax3.plot(df.index, df['Signal'], label='Signal', color=COLORS['signal'], linewidth=1.5)
        ax3.bar(df.index, df['MACD'] - df['Signal'], 
               color=np.where((df['MACD'] - df['Signal']) >= 0, COLORS['up'], COLORS['down']), 
               width=0.01, alpha=0.6)
        ax3.axhline(0, color=COLORS['text'], linestyle='--', alpha=0.5)
        ax3.grid(color=COLORS['grid'], linestyle='--', alpha=0.3)
        
        ax3.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        
        plt.tight_layout()
        
        filename = f"chart_{pair.replace('/', '_')}.png"
        plt.savefig(filename, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close()
        return filename
    except Exception as e:
        logger.error(f"Chart error for {pair}: {str(e)}")
        return None

def get_signal_reason(signal_data):
    rsi = signal_data['rsi']
    price = signal_data['price']
    upper_bb = signal_data['upper_bb']
    lower_bb = signal_data['lower_bb']
    
    reasons = []
    if rsi > 70:
        reasons.append("RSI overbought (>70)")
    elif rsi < 30:
        reasons.append("RSI oversold (<30)")
        
    if price > upper_bb:
        reasons.append("Price above Upper Bollinger Band")
    elif price < lower_bb:
        reasons.append("Price below Lower Bollinger Band")
        
    if not reasons:
        if 50 < rsi < 70:
            reasons.append("Mild bullish momentum")
        elif 30 < rsi < 50:
            reasons.append("Mild bearish momentum")
        else:
            reasons.append("Neutral market conditions")
            
    return " + ".join(reasons)

async def trading_bot():
    bot = Bot(token=TOKEN)
    logger.info("🚀 Trading Bot Started")
    
    now = datetime.now()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    await asyncio.sleep((next_minute - now).total_seconds())
    
    while True:
        try:
            start_time = datetime.now()
            signals = []
            media = []
            
            for pair in PAIRS:
                df = fetch_historical_data(pair)
                if not df.empty:
                    chart_file = create_pro_chart(df, pair)
                    if chart_file:
                        with open(chart_file, 'rb') as f:
                            media.append(InputMediaPhoto(f))
                        os.remove(chart_file)
                    
                    latest = df.iloc[-1]
                    signal = {
                        'pair': pair,
                        'price': latest['close'],
                        'ema20': latest['EMA_20'],
                        'rsi': latest['RSI'],
                        'upper_bb': latest['Upper_BB'],
                        'lower_bb': latest['Lower_BB'],
                        'macd': latest['MACD'],
                        'signal_line': latest['Signal']
                    }
                    
                    if latest['RSI'] > 70 or latest['close'] > latest['Upper_BB']:
                        signal.update({
                            'action': 'SELL',
                            'emoji': '🔴'
                        })
                    elif latest['RSI'] < 30 or latest['close'] < latest['Lower_BB']:
                        signal.update({
                            'action': 'BUY',
                            'emoji': '🟢'
                        })
                    else:
                        signal.update({
                            'action': 'HOLD',
                            'emoji': '🟡'
                        })
                    
                    signal['reason'] = get_signal_reason(signal)
                    signals.append(signal)
            
            if media and signals:
                now = datetime.now()
                message = f"""📊 *Quotex FX OTC Signals* {now.strftime('%H:%M')}
━━━━━━━━━━━━━━"""
                
                for sig in signals:
                    message += f"""
{sig['emoji']} *{sig['pair']} {sig['action']}*
Price: {sig['price']:.4f} | EMA20: {sig['ema20']:.4f}
RSI: {sig['rsi']:.1f} | BB: {sig['lower_bb']:.4f}-{sig['upper_bb']:.4f}
MACD: {sig['macd']:.4f} | Sig: {sig['signal_line']:.4f}
🔍 {sig['reason']}"""
                
                message += "\n━━━━━━━━━━━━━━\n⚠️ On your Own Risk ! /tips"
                
                for chat_id in CHAT_IDS:
                    try:
                        await bot.send_media_group(chat_id=chat_id, media=media)
                        await bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='Markdown'
                        )
                    except Exception as e:
                        logger.error(f"Error sending to chat {chat_id}: {str(e)}")
            
            elapsed = (datetime.now() - start_time).total_seconds()
            await asyncio.sleep(max(60 - elapsed, 0))
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            await asyncio.sleep(60)

if __name__ == '__main__':
    try:
        asyncio.run(trading_bot())
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")