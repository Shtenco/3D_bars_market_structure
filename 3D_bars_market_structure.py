# Author: Evgeniy Koshtenko
# Copyright (c) 2024. All rights reserved.
# 
# This code is released under CC BY-NC-ND license:
# - Commercial use is prohibited
# - Modifications are prohibited
# - Free distribution of the unmodified code is allowed with attribution
# - For personal use only
#
# If you find this code useful, you can support future development
# BTC donations: 1KrvpVrtW4WvWfq9zHRspRMb3T8ikDur81
# License: MIT

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mplfinance as mpf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_true_3d_renko(symbol, timeframe, start_date, end_date, min_spread_multiplier=45, volume_brick=500):
    """
    Creates 4D stationary features with same interface as 3D Renko
    """
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"Error getting data for {symbol}")
        return None, None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    if df.isnull().any().any():
        print("Missing values detected, cleaning...")
        df = df.dropna()
        if len(df) == 0:
            print("No data for analysis after cleaning")
            return None, None
    
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Failed to get symbol info for {symbol}")
        return None, None
    
    try:
        min_price_brick = symbol_info.spread * min_spread_multiplier * symbol_info.point
        if min_price_brick <= 0:
            print("Invalid block size")
            return None, None
    except AttributeError as e:
        print(f"Error getting symbol parameters: {e}")
        return None, None
    
    scaler = MinMaxScaler(feature_range=(3, 9))
    df_blocks = []
    
    try:
        # Time dimension
        df['time_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        df['time_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        df['time_numeric'] = (df['time'] - df['time'].min()).dt.total_seconds()
        
        # Price dimension
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_return'] = df['typical_price'].pct_change()
        df['price_acceleration'] = df['price_return'].diff()
        
        # Volume dimension
        df['volume_change'] = df['tick_volume'].pct_change()
        df['volume_acceleration'] = df['volume_change'].diff()
        
        # Volatility dimension
        df['volatility'] = df['price_return'].rolling(20).std()
        df['volatility_change'] = df['volatility'].pct_change()
        
        for idx in range(20, len(df)):
            window = df.iloc[idx-20:idx+1]
            
            block = {
                'time': df.iloc[idx]['time'],
                'time_numeric': scaler.fit_transform([[float(df.iloc[idx]['time_numeric'])]]).item(),
                'open': float(window['price_return'].iloc[-1]),
                'high': float(window['price_acceleration'].iloc[-1]),
                'low': float(window['volume_change'].iloc[-1]),
                'close': float(window['volatility_change'].iloc[-1]),
                'tick_volume': float(window['volume_acceleration'].iloc[-1]),
                'direction': np.sign(window['price_return'].iloc[-1]),
                'spread': float(df.iloc[idx]['time_sin']),
                'type': float(df.iloc[idx]['time_cos']),
                'trend_count': len(window),
                'price_change': float(window['price_return'].mean()),
                'volume_intensity': float(window['volume_change'].mean()),
                'price_velocity': float(window['price_acceleration'].mean())
            }
            df_blocks.append(block)
                
    except Exception as e:
        print(f"Error processing data: {e}")
        if len(df_blocks) == 0:
            return None, None
    
    if len(df_blocks) == 0:
        print("Failed to create any blocks")
        return None, None
        
    result_df = pd.DataFrame(df_blocks)
    
    # Scale all features
    features_to_scale = [col for col in result_df.columns if col != 'time' and col != 'direction']
    result_df[features_to_scale] = scaler.fit_transform(result_df[features_to_scale])
    
    # Add same analytical metrics as in original function
    result_df['ma_5'] = result_df['close'].rolling(5).mean()
    result_df['ma_20'] = result_df['close'].rolling(20).mean()
    result_df['volume_ma_5'] = result_df['tick_volume'].rolling(5).mean()
    result_df['price_volatility'] = result_df['price_change'].rolling(10).std()
    result_df['volume_volatility'] = result_df['tick_volume'].rolling(10).std()
    result_df['trend_strength'] = result_df['trend_count'] * result_df['direction']
    
    # Scale moving averages and volatility
    ma_columns = ['ma_5', 'ma_20', 'volume_ma_5', 'price_volatility', 'volume_volatility', 'trend_strength']
    result_df[ma_columns] = scaler.fit_transform(result_df[ma_columns])
    
    # Add statistical metrics and scale them
    result_df['zscore_price'] = stats.zscore(result_df['close'], nan_policy='omit')
    result_df['zscore_volume'] = stats.zscore(result_df['tick_volume'], nan_policy='omit')
    zscore_columns = ['zscore_price', 'zscore_volume']
    result_df[zscore_columns] = scaler.fit_transform(result_df[zscore_columns])
    
    return result_df, min_price_brick


def create_interactive_3d(df, symbol, save_dir):
    """
    Creates interactive 3D visualization with smoothed data and original price chart
    """
    try:
        save_dir = Path(save_dir)
        
        # Smooth all series with MA(100)
        df_smooth = df.copy()
        smooth_columns = ['close', 'tick_volume', 'price_volatility', 'volume_volatility']
        
        for col in smooth_columns:
            df_smooth[f'{col}_smooth'] = df_smooth[col].rolling(window=100, min_periods=1).mean()
        
        # Create subplots: 3D view and original chart side by side
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=(f'{symbol} 3D View (MA100)', f'{symbol} Original Price'),
            horizontal_spacing=0.05
        )
        
        # Add 3D scatter plot
        fig.add_trace(
            go.Scatter3d(
                x=np.arange(len(df_smooth)),
                y=df_smooth['tick_volume_smooth'],
                z=df_smooth['close_smooth'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=df_smooth['price_volatility_smooth'],
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True,
                    colorbar=dict(x=0.45)
                ),
                hovertemplate=
                "Time: %{x}<br>" +
                "Volume: %{y:.2f}<br>" +
                "Price: %{z:.5f}<br>" +
                "Volatility: %{marker.color:.5f}",
                name='3D View'
            ),
            row=1, col=1
        )
        
        # Add original price chart
        fig.add_trace(
            go.Candlestick(
                x=np.arange(len(df)),
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=2
        )
        
        # Add smoothed price line
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(df_smooth)),
                y=df_smooth['close_smooth'],
                line=dict(color='blue', width=1),
                name='MA100'
            ),
            row=1, col=2
        )
        
        # Update 3D layout
        fig.update_scenes(
            xaxis_title='Time',
            yaxis_title='Volume',
            zaxis_title='Price',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
        
        # Update 2D layout
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Price", row=1, col=2)
        
        # Update overall layout
        fig.update_layout(
            width=1500,  # Double width to accommodate both plots
            height=750,
            showlegend=True,
            title_text=f"{symbol} Combined Analysis"
        )
        
        # Save interactive HTML
        fig.write_html(save_dir / f'{symbol}_combined_view.html')
        
        # Create additional plots with smoothed data (unchanged)
        fig2 = make_subplots(rows=2, cols=2, 
                            subplot_titles=('Smoothed Price', 'Smoothed Volume',
                                          'Smoothed Price Volatility', 'Smoothed Volume Volatility'))
        
        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['close_smooth'],
                      name='Price MA100'),
            row=1, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['tick_volume_smooth'],
                      name='Volume MA100'),
            row=1, col=2
        )
        
        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['price_volatility_smooth'],
                      name='Price Vol MA100'),
            row=2, col=1
        )
        
        fig2.add_trace(
            go.Scatter(x=np.arange(len(df_smooth)), y=df_smooth['volume_volatility_smooth'],
                      name='Volume Vol MA100'),
            row=2, col=2
        )
        
        fig2.update_layout(
            height=750,
            width=750,
            showlegend=True,
            title_text=f"{symbol} Smoothed Data Analysis"
        )
        
        fig2.write_html(save_dir / f'{symbol}_smoothed_analysis.html')
        
        print(f"Interactive visualizations saved in {save_dir}")
        
    except Exception as e:
        print(f"Error creating interactive visualization: {e}")
        raise
    
def create_visualizations(df, symbol, save_dir):
    """
    Creates extended set of visualizations for 3D analysis
    """
    if df is None or len(df) == 0:
        print("No data for visualization")
        return
        
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Base settings for all plots
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = [7.5, 5]  # 750px at 100dpi
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'

        # 1. Main 3D visualization
        fig = plt.figure(figsize=(7.5, 5))
        ax = fig.add_subplot(111, projection='3d')
        
        # Data normalization
        scaler = MinMaxScaler()
        time_norm = np.arange(len(df))
        volume_norm = scaler.fit_transform(df[['tick_volume']])
        price_norm = scaler.fit_transform(df[['close']])
        
        # Create 3D bars
        for i in range(1, len(df)):
            dx = 0.3
            dy = volume_norm[i][0]
            dz = abs(price_norm[i][0] - price_norm[i-1][0])
            
            color = 'g' if df.iloc[i]['direction'] > 0 else 'r'
            alpha = min(0.7 + df.iloc[i]['trend_count'] * 0.05, 0.95)
            
            x = time_norm[i]
            y = volume_norm[i][0] / 2
            z = min(price_norm[i][0], price_norm[i-1][0])
            
            ax.bar3d(x, y, z, dx, dy, dz, 
                    color=color, alpha=alpha,
                    shade=True)
        
        ax.plot(time_norm, volume_norm.flatten(), price_norm.flatten(), 
                'k--', alpha=0.3, label='Price path')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Volume')
        ax.set_zlabel('Price')
        ax.set_title(f'{symbol} 3D Analysis')
        ax.view_init(elev=20, azim=45)
        ax.grid(True)
        
        plt.savefig(save_dir / f'{symbol}_3d_main.png', bbox_inches='tight')
        plt.close()

        # 2. Trend analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5))
        
        # Price and trends plot
        ax1.plot(df.index, df['close'], 'b-', label='Price', linewidth=1)
        for i in range(1, len(df)):
            if df.iloc[i]['direction'] > 0:
                ax1.fill_between([i-1, i], 
                               [df.iloc[i-1]['close'], df.iloc[i]['close']], 
                               df.iloc[i-1]['close'],
                               color='g', alpha=0.2)
            else:
                ax1.fill_between([i-1, i], 
                               [df.iloc[i-1]['close'], df.iloc[i]['close']], 
                               df.iloc[i-1]['close'],
                               color='r', alpha=0.2)
        ax1.set_title('Price Trends')
        ax1.grid(True)
        
        # Volume plot
        ax2.bar(df.index, df['tick_volume'], 
                color=['g' if d > 0 else 'r' for d in df['direction']], 
                alpha=0.6)
        ax2.set_title('Volume by Direction')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{symbol}_trends.png')
        plt.close()

        # 3. Statistical analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5, 7.5))
        
        # Price distribution
        ax1.hist(df['close'], bins=50, alpha=0.6, color='b')
        ax1.axvline(df['close'].mean(), color='r', linestyle='--', label='Mean')
        ax1.axvline(df['close'].median(), color='g', linestyle='--', label='Median')
        ax1.set_title('Price Distribution')
        ax1.grid(True)
        ax1.legend()
        
        # Price QQ-plot
        stats.probplot(df['close'], dist="norm", plot=ax2)
        ax2.set_title('Price Q-Q Plot')
        ax2.grid(True)
        
        # Volatility
        ax3.plot(df.index, df['price_volatility'], 'b-', label='Price Volatility')
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df.index, df['volume_volatility'], 'r-', label='Volume Volatility')
        ax3.set_title('Price and Volume Volatility')
        ax3.grid(True)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        # Volume distribution
        ax4.hist(df['tick_volume'], bins=50, alpha=0.6, color='g')
        ax4.axvline(df['tick_volume'].mean(), color='r', linestyle='--', label='Mean Volume')
        ax4.set_title('Volume Distribution')
        ax4.grid(True)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / f'{symbol}_stats.png')
        plt.close()

        # Add interactive 3D visualization
        create_interactive_3d(df, symbol, save_dir)
        print(f"Visualizations saved in directory {save_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        raise


def main():
    try:
        # Initialize MT5
        if not mt5.initialize():
            print("MetaTrader5 initialization error")
            return

        # Analysis parameters
        symbols = ["EURUSD", "GBPUSD"]
        timeframes = {
            "M15": mt5.TIMEFRAME_M15
        }
        
        # 7D analysis parameters
        params = {
            "min_spread_multiplier": 45,
            "volume_brick": 500
        }

        # Date range for data fetching
        start_date = datetime(2023, 4, 21)
        end_date = datetime(2023, 8, 10)

        # Analysis for each symbol and timeframe
        for symbol in symbols:
            print(f"\nAnalyzing symbol {symbol}")
            
            # Create symbol directory
            symbol_dir = Path('charts') / symbol
            symbol_dir.mkdir(parents=True, exist_ok=True)
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"Failed to get symbol info for {symbol}")
                continue

            print(f"Spread: {symbol_info.spread} points")
            print(f"Tick: {symbol_info.point}")
            
            # Analysis for each timeframe
            for tf_name, tf in timeframes.items():
                print(f"\nAnalyzing timeframe {tf_name}")
                
                # Create timeframe directory
                tf_dir = symbol_dir / tf_name
                tf_dir.mkdir(exist_ok=True)
                
                # Get and analyze data
                print("Getting data...")
                df, brick_size = create_true_3d_renko(
                    symbol=symbol,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    min_spread_multiplier=params["min_spread_multiplier"],
                    volume_brick=params["volume_brick"]
                )
                
                if df is not None and brick_size is not None:
                    print(f"Created {len(df)} 7D bars")
                    print(f"Block size: {brick_size}")
                    
                    # Basic statistics
                    print("\nBasic statistics:")
                    print(f"Average volume: {df['tick_volume'].mean():.2f}")
                    print(f"Average trend length: {df['trend_count'].mean():.2f}")
                    print(f"Max uptrend length: {df[df['direction'] > 0]['trend_count'].max()}")
                    print(f"Max downtrend length: {df[df['direction'] < 0]['trend_count'].max()}")
                    
                    # Create visualizations
                    print("\nCreating visualizations...")
                    create_visualizations(df, symbol, tf_dir)
                    
                    # Save data
                    csv_file = tf_dir / f"{symbol}_{tf_name}_7d_data.csv"
                    df.to_csv(csv_file)
                    print(f"Data saved to {csv_file}")
                    
                    # Results analysis
                    trend_ratio = len(df[df['direction'] > 0]) / len(df[df['direction'] < 0])
                    print(f"\nUp/Down bars ratio: {trend_ratio:.2f}")
                    
                    volume_corr = df['tick_volume'].corr(df['price_change'].abs())
                    print(f"Volume-Price change correlation: {volume_corr:.2f}")
                    
                    # Print warnings if anomalies detected
                    if df['price_volatility'].max() > df['price_volatility'].mean() * 3:
                        print("\nWARNING: High volatility periods detected!")
                        
                    if df['volume_volatility'].max() > df['volume_volatility'].mean() * 3:
                        print("WARNING: Abnormal volume spikes detected!")
                else:
                    print(f"Failed to create 3D bars for {symbol} on {tf_name}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    main()
