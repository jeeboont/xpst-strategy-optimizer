"""
XPST Trading Strategy Optimizer v3.2.0
Streamlit Application with Pepperstone Position Sizing Integration

Features:
- Accurate Pepperstone broker simulation
- Fixed zero trades bug
- Real broker position sizing formulas
- Multi-asset support (Crypto, Forex, Gold)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import zipfile
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import json

# Import the Pepperstone position sizing module
from pepperstone_position_sizing import PepperstonePositionSizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XPSTOptimizer:
    """XPST Strategy Optimizer with Pepperstone-accurate position sizing"""
    
    def __init__(self):
        self.position_sizer = PepperstonePositionSizer()
        
    def calculate_supertrend(self, data: pd.DataFrame, atr_length: int = 10, atr_factor: float = 3.0) -> pd.DataFrame:
        """Calculate SuperTrend indicator with given parameters"""
        df = data.copy()
        
        # Calculate ATR
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=atr_length).mean()
        
        # Calculate basic upper and lower bands
        df['basic_ub'] = (df['High'] + df['Low']) / 2 + (atr_factor * df['ATR'])
        df['basic_lb'] = (df['High'] + df['Low']) / 2 - (atr_factor * df['ATR'])
        
        # Initialize arrays
        df['final_ub'] = 0.0
        df['final_lb'] = 0.0
        df['supertrend'] = 0.0
        df['trend'] = 0  # 1 for uptrend, -1 for downtrend
        
        for i in range(len(df)):
            if i == 0:
                df.loc[i, 'final_ub'] = df.loc[i, 'basic_ub']
                df.loc[i, 'final_lb'] = df.loc[i, 'basic_lb']
            else:
                # Final upper band
                if df.loc[i, 'basic_ub'] < df.loc[i-1, 'final_ub'] or df.loc[i-1, 'Close'] > df.loc[i-1, 'final_ub']:
                    df.loc[i, 'final_ub'] = df.loc[i, 'basic_ub']
                else:
                    df.loc[i, 'final_ub'] = df.loc[i-1, 'final_ub']
                
                # Final lower band  
                if df.loc[i, 'basic_lb'] > df.loc[i-1, 'final_lb'] or df.loc[i-1, 'Close'] < df.loc[i-1, 'final_lb']:
                    df.loc[i, 'final_lb'] = df.loc[i, 'basic_lb']
                else:
                    df.loc[i, 'final_lb'] = df.loc[i-1, 'final_lb']
        
        # Determine SuperTrend and trend
        for i in range(len(df)):
            if i == 0:
                df.loc[i, 'supertrend'] = df.loc[i, 'final_ub']
                df.loc[i, 'trend'] = -1
            else:
                if df.loc[i-1, 'supertrend'] == df.loc[i-1, 'final_ub']:
                    if df.loc[i, 'Close'] <= df.loc[i, 'final_ub']:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_ub']
                        df.loc[i, 'trend'] = -1
                    else:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_lb']
                        df.loc[i, 'trend'] = 1
                else:
                    if df.loc[i, 'Close'] >= df.loc[i, 'final_lb']:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_lb']
                        df.loc[i, 'trend'] = 1
                    else:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_ub']
                        df.loc[i, 'trend'] = -1
        
        return df
    
    def backtest_strategy(self, 
                         data: pd.DataFrame, 
                         symbol: str,
                         atr_length: int = 10, 
                         atr_factor: float = 3.0, 
                         risk_percent: float = 2.0,
                         account_balance: float = 10000) -> Dict:
        """
        Backtest XPST strategy with Pepperstone-accurate position sizing
        
        Returns:
            Dict with backtest results including trade details
        """
        
        # Calculate SuperTrend
        df = self.calculate_supertrend(data, atr_length, atr_factor)
        
        # Initialize tracking variables
        trades = []
        current_position = None
        total_return = 0.0
        winning_trades = 0
        losing_trades = 0
        total_trades = 0
        equity_curve = [account_balance]
        current_equity = account_balance
        
        # Track for debugging
        signals_generated = 0
        position_sizing_failures = 0
        margin_failures = 0
        
        for i in range(1, len(df)):
            current_row = df.iloc[i]
            previous_row = df.iloc[i-1]
            
            # Detect trend changes (signals)
            trend_changed = current_row['trend'] != previous_row['trend']
            
            if trend_changed:
                signals_generated += 1
                
                # Close existing position if any
                if current_position is not None:
                    # Calculate profit/loss
                    if current_position['type'] == 'long':
                        pnl_points = current_row['Close'] - current_position['entry_price']
                    else:  # short
                        pnl_points = current_position['entry_price'] - current_row['Close']
                    
                    pnl_dollars = pnl_points * current_position['position_size']
                    current_equity += pnl_dollars
                    
                    # Record trade
                    trade_result = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_row.name,
                        'type': current_position['type'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': current_row['Close'],
                        'position_size': current_position['position_size'],
                        'pnl_points': pnl_points,
                        'pnl_dollars': pnl_dollars,
                        'stop_loss': current_position['stop_loss'],
                        'trade_value': current_position['trade_value'],
                        'required_margin': current_position['required_margin']
                    }
                    trades.append(trade_result)
                    
                    # Update statistics
                    total_trades += 1
                    if pnl_dollars > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
                    
                    current_position = None
                
                # Open new position based on new trend
                if current_row['trend'] == 1:  # Bullish trend - Long
                    trade_type = 'long'
                    entry_price = current_row['Close']
                    stop_loss_price = current_row['supertrend']  # Use SuperTrend as stop loss
                else:  # Bearish trend - Short
                    trade_type = 'short' 
                    entry_price = current_row['Close']
                    stop_loss_price = current_row['supertrend']  # Use SuperTrend as stop loss
                
                # Calculate position size using Pepperstone rules
                risk_amount = current_equity * (risk_percent / 100)
                
                try:
                    position_calc = self.position_sizer.calculate_position_size(
                        symbol=symbol,
                        entry_price=entry_price,
                        stop_loss_price=stop_loss_price,
                        risk_amount=risk_amount,
                        account_balance=current_equity
                    )
                    
                    if position_calc['valid'] and position_calc['position_size'] > 0:
                        # Create position
                        current_position = {
                            'type': trade_type,
                            'entry_time': current_row.name,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss_price,
                            'position_size': position_calc['position_size'],
                            'trade_value': position_calc['trade_value'],
                            'required_margin': position_calc['required_margin'],
                            'actual_risk': position_calc['actual_risk']
                        }
                    else:
                        # Position sizing failed
                        if position_calc['required_margin'] > current_equity * 0.8:
                            margin_failures += 1
                        else:
                            position_sizing_failures += 1
                            
                except Exception as e:
                    position_sizing_failures += 1
                    logger.error(f"Position sizing error: {e}")
            
            # Update equity curve
            if current_position is not None:
                # Mark-to-market current position
                if current_position['type'] == 'long':
                    unrealized_pnl = (current_row['Close'] - current_position['entry_price']) * current_position['position_size']
                else:
                    unrealized_pnl = (current_position['entry_price'] - current_row['Close']) * current_position['position_size']
                current_mark_to_market = current_equity + unrealized_pnl
            else:
                current_mark_to_market = current_equity
            
            equity_curve.append(current_mark_to_market)
        
        # Close final position if exists
        if current_position is not None:
            final_row = df.iloc[-1]
            if current_position['type'] == 'long':
                pnl_points = final_row['Close'] - current_position['entry_price']
            else:
                pnl_points = current_position['entry_price'] - final_row['Close']
            
            pnl_dollars = pnl_points * current_position['position_size']
            current_equity += pnl_dollars
            
            trade_result = {
                'entry_time': current_position['entry_time'],
                'exit_time': final_row.name,
                'type': current_position['type'],
                'entry_price': current_position['entry_price'],
                'exit_price': final_row['Close'],
                'position_size': current_position['position_size'],
                'pnl_points': pnl_points,
                'pnl_dollars': pnl_dollars,
                'stop_loss': current_position['stop_loss'],
                'trade_value': current_position['trade_value'],
                'required_margin': current_position['required_margin']
            }
            trades.append(trade_result)
            
            total_trades += 1
            if pnl_dollars > 0:
                winning_trades += 1
            else:
                losing_trades += 1
        
        # Calculate performance metrics
        final_equity = current_equity
        total_return = ((final_equity - account_balance) / account_balance) * 100
        
        # Calculate additional metrics
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        profit_factor = sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] > 0) / abs(sum(t['pnl_dollars'] for t in trades if t['pnl_dollars'] < 0)) if any(t['pnl_dollars'] < 0 for t in trades) else float('inf')
        
        avg_trade = sum(t['pnl_dollars'] for t in trades) / len(trades) if trades else 0
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_equity': final_equity,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'signals_generated': signals_generated,
            'position_sizing_failures': position_sizing_failures,
            'margin_failures': margin_failures,
            'parameters': {
                'atr_length': atr_length,
                'atr_factor': atr_factor,
                'risk_percent': risk_percent,
                'symbol': symbol,
                'account_balance': account_balance
            }
        }
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown percentage"""
        if len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak * 100
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def optimize_parameters(self, 
                           data: pd.DataFrame,
                           symbol: str,
                           atr_length_range: Tuple[int, int] = (5, 20),
                           atr_factor_range: Tuple[float, float] = (1.0, 5.0),
                           risk_percent: float = 2.0,
                           account_balance: float = 10000) -> pd.DataFrame:
        """Optimize XPST parameters with Pepperstone position sizing"""
        
        results = []
        total_combinations = 0
        
        # Generate parameter combinations
        atr_lengths = range(atr_length_range[0], atr_length_range[1] + 1, 1)
        atr_factors = np.arange(atr_factor_range[0], atr_factor_range[1] + 0.1, 0.1)
        
        total_combinations = len(atr_lengths) * len(atr_factors)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        combination = 0
        for atr_length in atr_lengths:
            for atr_factor in atr_factors:
                combination += 1
                
                status_text.text(f'Testing combination {combination}/{total_combinations}: ATR Length={atr_length}, ATR Factor={atr_factor:.1f}')
                
                # Backtest with current parameters
                result = self.backtest_strategy(
                    data=data,
                    symbol=symbol,
                    atr_length=atr_length,
                    atr_factor=atr_factor,
                    risk_percent=risk_percent,
                    account_balance=account_balance
                )
                
                # Store results
                results.append({
                    'atr_length': atr_length,
                    'atr_factor': round(atr_factor, 1),
                    'total_trades': result['total_trades'],
                    'win_rate': result['win_rate'],
                    'total_return': result['total_return'],
                    'profit_factor': result['profit_factor'],
                    'max_drawdown': result['max_drawdown'],
                    'avg_trade': result['avg_trade'],
                    'signals_generated': result['signals_generated'],
                    'position_sizing_failures': result['position_sizing_failures'],
                    'margin_failures': result['margin_failures']
                })
                
                progress_bar.progress(combination / total_combinations)
        
        status_text.text('Optimization complete!')
        
        return pd.DataFrame(results)


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="XPST Strategy Optimizer v3.2.0 - Pepperstone Edition",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 XPST Trading Strategy Optimizer")
    st.subheader("v3.2.0 - Pepperstone Position Sizing Edition")
    
    st.markdown("""
    **✅ NEW FEATURES:**
    - **Fixed Zero Trades Bug** - Now uses accurate Pepperstone position sizing
    - **Multi-Asset Support** - BTC, ETH, EUR/USD, Gold with correct formulas
    - **Real Broker Simulation** - Matches actual Pepperstone trading conditions
    - **Enhanced Debugging** - Shows signal generation vs actual trades
    
    **📊 Upload your trading data CSV files to get started!**
    """)
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload Trading Data (CSV)",
        accept_multiple_files=True,
        type=['csv']
    )
    
    if uploaded_files:
        # File selection
        file_options = {f.name: f for f in uploaded_files}
        selected_file = st.selectbox("Select data file for optimization:", list(file_options.keys()))
        
        if selected_file:
            # Load and display data
            data = pd.read_csv(file_options[selected_file])
            
            # Parse datetime
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            
            st.success(f"✅ Loaded {len(data)} bars of data from {selected_file}")
            
            # Data preview
            with st.expander("📊 Data Preview"):
                st.dataframe(data.head(10))
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Bars", len(data))
                with col2:
                    st.metric("Date Range", f"{data.index[0].date()} to {data.index[-1].date()}")
                with col3:
                    st.metric("Current Price", f"${data['Close'].iloc[-1]:,.2f}")
                with col4:
                    st.metric("Price Range", f"${data['Low'].min():,.2f} - ${data['High'].max():,.2f}")
            
            # Extract symbol from filename
            symbol = selected_file.split('_')[0].upper()
            st.info(f"🔍 Detected symbol: **{symbol}**")
            
            # Configuration section
            st.header("⚙️ Configuration")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📈 Strategy Parameters")
                atr_length_min = st.number_input("ATR Length Min", min_value=5, max_value=50, value=10)
                atr_length_max = st.number_input("ATR Length Max", min_value=5, max_value=50, value=20)
                
                atr_factor_min = st.number_input("ATR Factor Min", min_value=0.5, max_value=10.0, value=1.0, step=0.1)
                atr_factor_max = st.number_input("ATR Factor Max", min_value=0.5, max_value=10.0, value=5.0, step=0.1)
            
            with col2:
                st.subheader("💰 Risk Management")
                risk_percent = st.number_input("Risk per Trade (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
                account_balance = st.number_input("Account Balance ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
                
                # Show position sizing preview
                with st.expander("🔍 Position Sizing Preview"):
                    sizer = PepperstonePositionSizer()
                    test_entry = data['Close'].iloc[-1]
                    test_stop = test_entry * 0.97  # 3% stop loss for preview
                    test_risk = account_balance * (risk_percent / 100)
                    
                    preview_calc = sizer.calculate_position_size(
                        symbol=symbol,
                        entry_price=test_entry,
                        stop_loss_price=test_stop,
                        risk_amount=test_risk,
                        account_balance=account_balance
                    )
                    
                    st.write(f"**Sample Trade Preview for {symbol}:**")
                    st.write(f"- Entry Price: ${test_entry:,.2f}")
                    st.write(f"- Stop Loss: ${test_stop:,.2f}")
                    st.write(f"- Position Size: {preview_calc['position_size']:.6f} units")
                    st.write(f"- Trade Value: ${preview_calc['trade_value']:,.2f}")
                    st.write(f"- Required Margin: ${preview_calc['required_margin']:,.2f}")
                    st.write(f"- Asset Type: {preview_calc['asset_type']}")
                    st.write(f"- Leverage: {preview_calc['leverage']}:1")
            
            with col3:
                st.subheader("🚀 Execution")
                
                if st.button("▶️ Start Optimization", type="primary"):
                    optimizer = XPSTOptimizer()
                    
                    st.info("🔄 Running optimization with Pepperstone position sizing...")
                    
                    # Run optimization
                    results = optimizer.optimize_parameters(
                        data=data,
                        symbol=symbol,
                        atr_length_range=(atr_length_min, atr_length_max),
                        atr_factor_range=(atr_factor_min, atr_factor_max),
                        risk_percent=risk_percent,
                        account_balance=account_balance
                    )
                    
                    # Display results
                    st.success("✅ Optimization Complete!")
                    
                    # Sort by total return
                    results = results.sort_values('total_return', ascending=False)
                    
                    # Best result summary
                    best_result = results.iloc[0]
                    
                    st.header("🏆 Best Parameters Found")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("ATR Length", int(best_result['atr_length']))
                    with col2:
                        st.metric("ATR Factor", f"{best_result['atr_factor']:.1f}")
                    with col3:
                        st.metric("Total Return", f"{best_result['total_return']:.2f}%")
                    with col4:
                        st.metric("Total Trades", int(best_result['total_trades']))
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Win Rate", f"{best_result['win_rate']:.1f}%")
                    with col2:
                        st.metric("Profit Factor", f"{best_result['profit_factor']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{best_result['max_drawdown']:.2f}%")
                    with col4:
                        st.metric("Avg Trade", f"${best_result['avg_trade']:.2f}")
                    
                    # Debug information
                    st.header("🔍 Debug Information")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Signals Generated", int(best_result['signals_generated']))
                    with col2:
                        st.metric("Position Sizing Failures", int(best_result['position_sizing_failures']))
                    with col3:
                        st.metric("Margin Failures", int(best_result['margin_failures']))
                    
                    if best_result['total_trades'] == 0:
                        st.error("⚠️ Zero trades generated! Check your position sizing settings.")
                        st.info("💡 Try: Lower risk percentage, increase account balance, or check symbol format.")
                    
                    # Results table
                    st.header("📊 All Results")
                    st.dataframe(
                        results.head(20),
                        use_container_width=True
                    )
                    
                    # Heatmap visualization
                    st.header("🗺️ Performance Heatmap")
                    
                    # Create pivot table for heatmap
                    pivot_data = results.pivot(
                        index='atr_length',
                        columns='atr_factor', 
                        values='total_return'
                    )
                    
                    fig = px.imshow(
                        pivot_data.values,
                        x=pivot_data.columns,
                        y=pivot_data.index,
                        color_continuous_scale='RdYlGn',
                        title='Total Return (%) Heatmap'
                    )
                    
                    fig.update_layout(
                        xaxis_title="ATR Factor",
                        yaxis_title="ATR Length"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    st.header("💾 Download Results")
                    
                    # Create download data
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    # CSV download
                    csv_buffer = io.StringIO()
                    results.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="📄 Download Results CSV",
                        data=csv_buffer.getvalue(),
                        file_name=f"xpst_optimization_results_{symbol}_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    # Best parameters for cBot
                    cbot_config = {
                        'ATR_Length': int(best_result['atr_length']),
                        'ATR_Factor': float(best_result['atr_factor']),
                        'Risk_Percent': risk_percent,
                        'Symbol': symbol,
                        'Optimization_Date': timestamp,
                        'Expected_Return': float(best_result['total_return']),
                        'Expected_Trades': int(best_result['total_trades']),
                        'Expected_Win_Rate': float(best_result['win_rate'])
                    }
                    
                    st.download_button(
                        label="⚙️ Download cBot Settings JSON",
                        data=json.dumps(cbot_config, indent=2),
                        file_name=f"xpst_cbot_settings_{symbol}_{timestamp}.json",
                        mime="application/json"
                    )

    # Sidebar information
    with st.sidebar:
        st.header("ℹ️ About XPST v3.2.0")
        st.markdown("""
        **XPST** (Cross Price SuperTrend) is a trend-following strategy that:
        
        - Uses SuperTrend indicator for trend detection
        - Enters long when price crosses above SuperTrend
        - Enters short when price crosses below SuperTrend
        - Uses SuperTrend line as dynamic stop loss
        
        **v3.2.0 Improvements:**
        - ✅ Fixed zero trades bug
        - ✅ Accurate Pepperstone position sizing
        - ✅ Multi-asset support (Crypto, Forex, Gold)
        - ✅ Real broker margin calculations
        - ✅ Enhanced debugging tools
        
        **Pepperstone Position Sizing:**
        - Crypto: Risk$ / Stop_Loss_Points
        - Forex: (Risk$ × 10k) / Stop_Loss_Pips
        - Gold: Risk$ / Stop_Loss_Points
        - Respects minimum position sizes
        - Accounts for leverage limits
        """)
        
        st.header("📈 Supported Assets")
        st.markdown("""
        **Crypto:** BTC, ETH, LTC, XRP, ADA
        **Forex:** EUR/USD, GBP/USD, USD/JPY, etc.
        **Commodities:** Gold (GC, XAUUSD)
        """)
        
        st.header("🎯 Performance Metrics")
        st.markdown("""
        **Total Return:** Overall profit/loss percentage
        **Win Rate:** Percentage of profitable trades
        **Profit Factor:** Gross profit / Gross loss
        **Max Drawdown:** Largest peak-to-trough decline
        **Avg Trade:** Average profit/loss per trade
        """)


if __name__ == "__main__":
    main()
