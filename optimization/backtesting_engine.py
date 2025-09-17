import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class XPSTBacktester:
    """
    Backtesting engine for XPST (eXtended Pivot SuperTrend) strategy
    """
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.balance = 10000  # Starting balance
        
    def backtest_strategy(self, data: pd.DataFrame, parameters: Dict) -> Tuple[List[Dict], Dict]:
        """
        Main backtesting function that runs the XPST strategy
        """
        # Reset state
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        self.balance = 10000
        
        # Validate data
        if len(data) < 50:
            raise ValueError("Insufficient data for backtesting")
        
        # Calculate indicators
        df = self._calculate_indicators(data.copy(), parameters)
        
        # Generate signals and execute trades
        df = self._generate_signals(df, parameters)
        
        # Execute trading logic
        self._execute_trades(df, parameters)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        return self.trades, metrics
    
    def _calculate_indicators(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for XPST strategy
        """
        # Calculate ATR
        df = self._calculate_atr(df, params['atr_period'])
        
        # Calculate Pivot Points
        df = self._calculate_pivot_points(df, params['pivot_period'])
        
        # Calculate SuperTrend
        df = self._calculate_supertrend(df, params)
        
        # Calculate optional filters
        if params.get('use_adx', False):
            df = self._calculate_adx(df, 14)  # Standard ADX period
        
        if params.get('use_ema', False):
            df = self._calculate_ema(df, params.get('ema_period', 50))
        
        if params.get('use_xtrend', False):
            df = self._calculate_xtrend(df)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Average True Range
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        df['ATR'] = true_range.rolling(window=period).mean()
        
        return df
    
    def _calculate_pivot_points(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Pivot Points for highs and lows
        """
        df['PivotHigh'] = df['High'].rolling(window=period*2+1, center=True).max() == df['High']
        df['PivotLow'] = df['Low'].rolling(window=period*2+1, center=True).min() == df['Low']
        
        # Forward fill pivot levels
        df['LastPivotHigh'] = df.loc[df['PivotHigh'], 'High'].reindex(df.index).fillna(method='ffill')
        df['LastPivotLow'] = df.loc[df['PivotLow'], 'Low'].reindex(df.index).fillna(method='ffill')
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Calculate SuperTrend based on pivot points and ATR
        """
        atr_factor = params['atr_factor']
        
        # Use previous ATR if specified
        if params.get('use_prev_atr', False):
            atr_values = df['ATR'].shift(1)
        else:
            atr_values = df['ATR']
        
        # Calculate SuperTrend bands
        hl2 = (df['High'] + df['Low']) / 2
        
        upper_band = hl2 + (atr_factor * atr_values)
        lower_band = hl2 - (atr_factor * atr_values)
        
        # SuperTrend calculation
        supertrend = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        
        for i in range(1, len(df)):
            if pd.isna(upper_band.iloc[i-1]) or pd.isna(lower_band.iloc[i-1]):
                continue
                
            # Upper band
            if upper_band.iloc[i] < upper_band.iloc[i-1] or df['Close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Lower band
            if lower_band.iloc[i] > lower_band.iloc[i-1] or df['Close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # Trend direction
            if pd.isna(trend.iloc[i-1]):
                trend.iloc[i] = 1
            elif trend.iloc[i-1] == 1 and df['Close'].iloc[i] <= lower_band.iloc[i]:
                trend.iloc[i] = -1
            elif trend.iloc[i-1] == -1 and df['Close'].iloc[i] >= upper_band.iloc[i]:
                trend.iloc[i] = 1
            else:
                trend.iloc[i] = trend.iloc[i-1]
            
            # SuperTrend value
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
        
        df['SuperTrend'] = supertrend
        df['Trend'] = trend
        df['UpperBand'] = upper_band
        df['LowerBand'] = lower_band
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX)
        """
        # Calculate directional movement
        df['HighDiff'] = df['High'] - df['High'].shift(1)
        df['LowDiff'] = df['Low'].shift(1) - df['Low']
        
        df['PlusDM'] = np.where((df['HighDiff'] > df['LowDiff']) & (df['HighDiff'] > 0), df['HighDiff'], 0)
        df['MinusDM'] = np.where((df['LowDiff'] > df['HighDiff']) & (df['LowDiff'] > 0), df['LowDiff'], 0)
        
        # Calculate smoothed directional movement
        df['PlusDM_smooth'] = df['PlusDM'].rolling(window=period).mean()
        df['MinusDM_smooth'] = df['MinusDM'].rolling(window=period).mean()
        df['ATR_smooth'] = df['ATR'].rolling(window=period).mean()
        
        # Calculate directional indicators
        df['PlusDI'] = 100 * df['PlusDM_smooth'] / df['ATR_smooth']
        df['MinusDI'] = 100 * df['MinusDM_smooth'] / df['ATR_smooth']
        
        # Calculate ADX
        df['DX'] = 100 * np.abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """
        Calculate Exponential Moving Average
        """
        df['EMA'] = df['Close'].ewm(span=period).mean()
        return df
    
    def _calculate_xtrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate XTrend indicator (simplified implementation)
        """
        # This is a simplified XTrend - in real implementation you'd use the actual XTrend logic
        # For now, using EMA crossover as proxy
        ema_fast = df['Close'].ewm(span=10).mean()
        ema_slow = df['Close'].ewm(span=20).mean()
        
        df['XTrend'] = np.where(ema_fast > ema_slow, 1, -1)
        return df
    
    def _generate_signals(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """
        Generate buy/sell signals based on strategy rules
        """
        # Initialize signal columns
        df['Signal'] = 0
        df['SignalReason'] = ''
        
        for i in range(1, len(df)):
            signal = 0
            reason = ''
            
            # Check for pivot-based signals
            if (df['Trend'].iloc[i] == 1 and df['Trend'].iloc[i-1] == -1):
                signal = 1  # Buy signal
                reason = 'Pivot_Buy'
            elif (df['Trend'].iloc[i] == -1 and df['Trend'].iloc[i-1] == 1):
                signal = -1  # Sell signal
                reason = 'Pivot_Sell'
            
            # Apply filters
            if signal != 0:
                signal, reason = self._apply_filters(df, i, signal, reason, params)
            
            df.loc[df.index[i], 'Signal'] = signal
            df.loc[df.index[i], 'SignalReason'] = reason
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame, i: int, signal: int, reason: str, params: Dict) -> Tuple[int, str]:
        """
        Apply various filters to the signal
        """
        # ADX Filter
        if params.get('use_adx', False):
            adx_threshold = params.get('adx_threshold', 15)
            if df['ADX'].iloc[i] < adx_threshold:
                return 0, f'{reason}_Blocked_ADX'
        
        # EMA Filter
        if params.get('use_ema', False):
            ema_value = df['EMA'].iloc[i]
            close_price = df['Close'].iloc[i]
            
            if signal == 1 and close_price < ema_value:  # Buy signal but below EMA
                return 0, f'{reason}_Blocked_EMA'
            elif signal == -1 and close_price > ema_value:  # Sell signal but above EMA
                return 0, f'{reason}_Blocked_EMA'
        
        # XTrend Filter
        if params.get('use_xtrend', False):
            xtrend_value = df['XTrend'].iloc[i]
            
            if signal == 1 and xtrend_value != 1:  # Buy signal but XTrend bearish
                return 0, f'{reason}_Blocked_XTrend'
            elif signal == -1 and xtrend_value != -1:  # Sell signal but XTrend bullish
                return 0, f'{reason}_Blocked_XTrend'
        
        return signal, reason
    
    def _execute_trades(self, df: pd.DataFrame, params: Dict):
        """
        Execute trades based on signals
        """
        risk_percent = params.get('risk_percent', 2.0) / 100
        
        for i, row in df.iterrows():
            if row['Signal'] != 0:
                self._process_signal(row, i, risk_percent, params)
    
    def _process_signal(self, row, timestamp, risk_percent: float, params: Dict):
        """
        Process individual trading signals
        """
        signal = row['Signal']
        price = row['Close']
        atr = row['ATR']
        
        # Close existing position if opposite signal
        if self.current_position and self.current_position['direction'] != signal:
            self._close_position(price, timestamp, 'Opposite_Signal')
        
        # Open new position if no current position
        if not self.current_position:
            self._open_position(signal, price, timestamp, atr, risk_percent, params)
    
    def _open_position(self, direction: int, price: float, timestamp, atr: float, 
                      risk_percent: float, params: Dict):
        """
        Open a new trading position
        """
        # Calculate position size based on risk
        stop_distance = atr * params.get('stop_loss_atr_multiplier', 1.0)
        risk_amount = self.balance * risk_percent
        
        # Ensure minimum stop distance
        stop_distance = max(stop_distance, price * 0.001)  # Minimum 0.1%
        
        position_size = risk_amount / stop_distance
        
        # Calculate stop loss and take profit
        if direction == 1:  # Long position
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * params.get('take_profit_ratio', 2.0))
        else:  # Short position
            stop_loss = price + stop_distance
            take_profit = price - (stop_distance * params.get('take_profit_ratio', 2.0))
        
        self.current_position = {
            'direction': direction,
            'entry_price': price,
            'entry_time': timestamp,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_balance': self.balance
        }
    
    def _close_position(self, exit_price: float, exit_time, reason: str):
        """
        Close the current trading position
        """
        if not self.current_position:
            return
        
        position = self.current_position
        
        # Calculate P&L
        if position['direction'] == 1:  # Long position
            pnl = (exit_price - position['entry_price']) * position['position_size']
        else:  # Short position
            pnl = (position['entry_price'] - exit_price) * position['position_size']
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'direction': 'Long' if position['direction'] == 1 else 'Short',
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'position_size': position['position_size'],
            'pnl': pnl,
            'pnl_percent': (pnl / position['entry_balance']) * 100,
            'exit_reason': reason,
            'trade_duration': exit_time - position['entry_time']
        }
        
        self.trades.append(trade)
        self.current_position = None
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics
        """
        if not self.trades:
            return self._empty_metrics()
        
        trade_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trade_df)
        winning_trades = len(trade_df[trade_df['pnl'] > 0])
        losing_trades = len(trade_df[trade_df['pnl'] < 0])
        
        # Performance metrics
        total_return = ((self.balance - 10000) / 10000) * 100
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = trade_df[trade_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trade_df[trade_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf') if winning_trades > 0 else 0
        
        # Risk metrics
        returns = trade_df['pnl_percent'].values
        max_drawdown = self._calculate_max_drawdown(trade_df)
        
        # Sharpe ratio (assuming 252 trading days)
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        
        metrics = {
            'total_return': round(total_return, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'final_balance': round(self.balance, 2),
            'largest_win': round(trade_df['pnl'].max(), 2),
            'largest_loss': round(trade_df['pnl'].min(), 2),
            'avg_trade_duration': trade_df['trade_duration'].mean(),
            'expectancy': round((win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss), 2)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, trade_df: pd.DataFrame) -> float:
        """
        Calculate maximum drawdown from trade data
        """
        balance_curve = [10000]  # Starting balance
        
        for pnl in trade_df['pnl']:
            balance_curve.append(balance_curve[-1] + pnl)
        
        peak = balance_curve[0]
        max_dd = 0
        
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _empty_metrics(self) -> Dict:
        """
        Return empty metrics when no trades are generated
        """
        return {
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'final_balance': 10000,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_trade_duration': pd.Timedelta(0),
            'expectancy': 0
        }
