# XPST Strategy Optimization Engine - Google Colab
# High-Performance Optimization with Parallel Processing

# ===============================
# CELL 1: Setup and Installation
# ===============================

# Install required packages
!pip install yfinance pandas numpy plotly scipy scikit-learn requests

# Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import json
import requests
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

print("📦 Environment Setup Complete")
print(f"🔥 Available CPU cores: {mp.cpu_count()}")

# ===============================
# CELL 2: Optimization Parameters
# ===============================

# Configuration - Edit these parameters
OPTIMIZATION_CONFIG = {
    "asset": "BTC-USD",
    "timeframe": "2m", 
    "period": "1mo",
    "steps": ["step1", "step2", "step3"],
    "risk_percent": 2.0,
    "max_combinations": 1000,
    "use_parallel": True,
    "max_workers": mp.cpu_count(),
    "early_stopping_patience": 100
}

# API Configuration for Streamlit communication
API_CONFIG = {
    "enable_api": True,
    "results_url": None,  # Will be set dynamically
    "status_callback": None
}

print("⚙️ Configuration loaded:")
for key, value in OPTIMIZATION_CONFIG.items():
    print(f"  {key}: {value}")

# ===============================
# CELL 3: Core Optimization Classes
# ===============================

class HighPerformanceBacktester:
    """
    Optimized backtesting engine for Google Colab
    """
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        
    def backtest_strategy_vectorized(self, data: pd.DataFrame, parameters: Dict) -> Tuple[List[Dict], Dict]:
        """
        Vectorized backtesting for faster performance
        """
        # Reset state
        self.trades = []
        balance = 10000
        
        # Calculate indicators efficiently
        df = self._calculate_indicators_vectorized(data.copy(), parameters)
        
        # Generate signals
        signals = self._generate_signals_vectorized(df, parameters)
        
        # Execute trades
        trades = self._execute_trades_vectorized(df, signals, parameters, balance)
        
        # Calculate metrics
        metrics = self._calculate_metrics_fast(trades, balance)
        
        return trades, metrics
    
    def _calculate_indicators_vectorized(self, df: pd.DataFrame, params: Dict) -> pd.DataFrame:
        """Vectorized indicator calculations"""
        
        # ATR calculation
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift(1))
        low_close = np.abs(df['Low'] - df['Close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['ATR'] = true_range.rolling(window=params['atr_period']).mean()
        
        # SuperTrend calculation (vectorized)
        atr_factor = params['atr_factor']
        hl2 = (df['High'] + df['Low']) / 2
        
        if params.get('use_prev_atr', False):
            atr_values = df['ATR'].shift(1)
        else:
            atr_values = df['ATR']
            
        upper_band = hl2 + (atr_factor * atr_values)
        lower_band = hl2 - (atr_factor * atr_values)
        
        # Vectorized SuperTrend
        supertrend, trend = self._calculate_supertrend_vectorized(
            df['Close'], upper_band, lower_band
        )
        
        df['SuperTrend'] = supertrend
        df['Trend'] = trend
        
        # Add filters if enabled
        if params.get('use_adx', False):
            df = self._calculate_adx_vectorized(df)
        
        if params.get('use_ema', False):
            df['EMA'] = df['Close'].ewm(span=params.get('ema_period', 50)).mean()
        
        return df
    
    def _calculate_supertrend_vectorized(self, close: pd.Series, 
                                       upper_band: pd.Series, 
                                       lower_band: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Vectorized SuperTrend calculation"""
        
        # Initialize arrays
        supertrend = np.full(len(close), np.nan)
        trend = np.full(len(close), 1, dtype=int)
        
        # Adjust bands
        for i in range(1, len(close)):
            # Upper band adjustment
            if upper_band.iloc[i] < upper_band.iloc[i-1] or close.iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Lower band adjustment  
            if lower_band.iloc[i] > lower_band.iloc[i-1] or close.iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # Trend calculation
            if trend[i-1] == 1 and close.iloc[i] <= lower_band.iloc[i]:
                trend[i] = -1
            elif trend[i-1] == -1 and close.iloc[i] >= upper_band.iloc[i]:
                trend[i] = 1
            else:
                trend[i] = trend[i-1]
            
            # SuperTrend value
            if trend[i] == 1:
                supertrend[i] = lower_band.iloc[i]
            else:
                supertrend[i] = upper_band.iloc[i]
        
        return pd.Series(supertrend, index=close.index), pd.Series(trend, index=close.index)
    
    def _calculate_adx_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Vectorized ADX calculation"""
        
        # Directional movement
        df['HighDiff'] = df['High'].diff()
        df['LowDiff'] = -df['Low'].diff()
        
        df['PlusDM'] = np.where((df['HighDiff'] > df['LowDiff']) & (df['HighDiff'] > 0), 
                               df['HighDiff'], 0)
        df['MinusDM'] = np.where((df['LowDiff'] > df['HighDiff']) & (df['LowDiff'] > 0), 
                                df['LowDiff'], 0)
        
        # Smoothed values
        df['PlusDM_smooth'] = df['PlusDM'].rolling(window=period).mean()
        df['MinusDM_smooth'] = df['MinusDM'].rolling(window=period).mean()
        df['ATR_smooth'] = df['ATR'].rolling(window=period).mean()
        
        # Directional indicators
        df['PlusDI'] = 100 * df['PlusDM_smooth'] / df['ATR_smooth']
        df['MinusDI'] = 100 * df['MinusDM_smooth'] / df['ATR_smooth']
        
        # ADX
        df['DX'] = 100 * np.abs(df['PlusDI'] - df['MinusDI']) / (df['PlusDI'] + df['MinusDI'])
        df['ADX'] = df['DX'].rolling(window=period).mean()
        
        return df
    
    def _generate_signals_vectorized(self, df: pd.DataFrame, params: Dict) -> pd.Series:
        """Vectorized signal generation"""
        
        # Base signals from trend changes
        trend_change = df['Trend'].diff()
        buy_signals = (trend_change == 2)  # -1 to 1
        sell_signals = (trend_change == -2)  # 1 to -1
        
        signals = pd.Series(0, index=df.index)
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        # Apply filters
        if params.get('use_adx', False):
            adx_filter = df['ADX'] >= params.get('adx_threshold', 15)
            signals = signals * adx_filter
        
        if params.get('use_ema', False):
            ema_filter_long = df['Close'] > df['EMA']
            ema_filter_short = df['Close'] < df['EMA']
            
            signals = np.where((signals == 1) & ~ema_filter_long, 0, signals)
            signals = np.where((signals == -1) & ~ema_filter_short, 0, signals)
        
        return pd.Series(signals, index=df.index)
    
    def _execute_trades_vectorized(self, df: pd.DataFrame, signals: pd.Series, 
                                 params: Dict, initial_balance: float) -> List[Dict]:
        """Vectorized trade execution"""
        
        trades = []
        signal_indices = signals[signals != 0].index
        
        if len(signal_indices) == 0:
            return trades
        
        # Calculate position sizes and P&L vectorized
        risk_percent = params.get('risk_percent', 2.0) / 100
        
        for i, signal_time in enumerate(signal_indices[:-1]):
            signal = signals[signal_time]
            entry_price = df.loc[signal_time, 'Close']
            atr = df.loc[signal_time, 'ATR']
            
            # Find exit
            next_signal_time = signal_indices[i + 1]
            exit_price = df.loc[next_signal_time, 'Close']
            
            # Calculate trade
            stop_distance = atr * params.get('stop_loss_atr_multiplier', 1.0)
            risk_amount = initial_balance * risk_percent
            position_size = risk_amount / stop_distance
            
            if signal == 1:  # Long
                pnl = (exit_price - entry_price) * position_size
            else:  # Short
                pnl = (entry_price - exit_price) * position_size
            
            trade = {
                'entry_time': signal_time,
                'exit_time': next_signal_time,
                'direction': 'Long' if signal == 1 else 'Short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'pnl': pnl,
                'pnl_percent': (pnl / initial_balance) * 100
            }
            
            trades.append(trade)
        
        return trades
    
    def _calculate_metrics_fast(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Fast metrics calculation"""
        
        if not trades:
            return self._empty_metrics()
        
        trade_df = pd.DataFrame(trades)
        
        total_trades = len(trade_df)
        winning_trades = len(trade_df[trade_df['pnl'] > 0])
        
        total_pnl = trade_df['pnl'].sum()
        total_return = (total_pnl / initial_balance) * 100
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        avg_win = trade_df[trade_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trade_df[trade_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else float('inf') if winning_trades > 0 else 0
        
        # Simple drawdown calculation
        cumulative_pnl = trade_df['pnl'].cumsum()
        peak = cumulative_pnl.expanding().max()
        drawdown = ((peak - cumulative_pnl) / (initial_balance + peak)) * 100
        max_drawdown = drawdown.max()
        
        # Sharpe ratio approximation
        returns = trade_df['pnl_percent'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_return': round(total_return, 2),
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'final_balance': round(initial_balance + total_pnl, 2),
            'largest_win': round(trade_df['pnl'].max(), 2),
            'largest_loss': round(trade_df['pnl'].min(), 2),
            'expectancy': round((win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss), 2)
        }
    
    def _empty_metrics(self) -> Dict:
        """Empty metrics for failed optimizations"""
        return {
            'total_return': 0, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0,
            'max_drawdown': 0, 'sharpe_ratio': 0, 'final_balance': 10000,
            'largest_win': 0, 'largest_loss': 0, 'expectancy': 0
        }

print("🚀 High-Performance Backtester Ready")

# ===============================
# CELL 4: Parallel Optimization Engine
# ===============================

class ParallelOptimizationEngine:
    """
    Multi-core optimization engine for Google Colab
    """
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.backtester = HighPerformanceBacktester()
        
    def optimize_step1_parallel(self, data: pd.DataFrame, progress_callback=None) -> List[Dict]:
        """Step 1: Parallel core parameter optimization"""
        
        print("🎯 Starting Step 1: Core Parameter Optimization (Parallel)")
        
        # Parameter grid
        coarse_grid = {
            'pivot_period': [3, 5, 7, 9, 12, 15],
            'atr_factor': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
            'atr_period': [12, 16, 20, 25, 30, 35]
        }
        
        # Generate combinations
        combinations = self._generate_combinations(coarse_grid)
        print(f"📊 Testing {len(combinations)} parameter combinations")
        
        # Parallel processing
        results = self._process_combinations_parallel(data, combinations, progress_callback)
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        
        print(f"✅ Step 1 completed. Best fitness: {results[0]['fitness']:.4f}")
        return results[:10]  # Top 10
    
    def optimize_step2_parallel(self, data: pd.DataFrame, step1_results: List[Dict]) -> List[Dict]:
        """Step 2: Parallel filter optimization"""
        
        print("🔍 Starting Step 2: Filter Optimization (Parallel)")
        
        filter_combinations = []
        
        # For each top core result, test filter combinations
        for core_result in step1_results[:3]:  # Top 3 core results
            core_params = core_result['parameters']
            
            # Generate filter combinations
            filter_options = [
                {'use_adx': False, 'use_ema': False},
                {'use_adx': True, 'adx_threshold': 10, 'use_ema': False},
                {'use_adx': True, 'adx_threshold': 15, 'use_ema': False},
                {'use_adx': True, 'adx_threshold': 20, 'use_ema': False},
                {'use_adx': False, 'use_ema': True, 'ema_period': 50},
                {'use_adx': False, 'use_ema': True, 'ema_period': 100},
                {'use_adx': False, 'use_ema': True, 'ema_period': 200},
                {'use_adx': True, 'adx_threshold': 15, 'use_ema': True, 'ema_period': 50},
                {'use_adx': True, 'adx_threshold': 15, 'use_ema': True, 'ema_period': 100}
            ]
            
            for filter_option in filter_options:
                combined_params = {**core_params, **filter_option}
                filter_combinations.append(combined_params)
        
        print(f"📊 Testing {len(filter_combinations)} filter combinations")
        
        # Parallel processing
        results = self._process_combinations_parallel(data, filter_combinations)
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        
        print(f"✅ Step 2 completed. Best fitness: {results[0]['fitness']:.4f}")
        return results[:5]  # Top 5
    
    def optimize_step3_parallel(self, data: pd.DataFrame, step2_results: List[Dict]) -> List[Dict]:
        """Step 3: Parallel CB & Re-entry optimization"""
        
        print("⚡ Starting Step 3: CB & Re-entry Optimization (Parallel)")
        
        cb_combinations = []
        
        # For each top filter result, test CB combinations
        for filter_result in step2_results[:3]:  # Top 3 filter results
            base_params = filter_result['parameters']
            
            # CB options
            cb_options = [
                {'enable_circuit_breaker': False},
                {'enable_circuit_breaker': True, 'circuit_breaker_buffer': 0.05},
                {'enable_circuit_breaker': True, 'circuit_breaker_buffer': 0.08},
                {'enable_circuit_breaker': True, 'circuit_breaker_buffer': 0.10}
            ]
            
            for cb_option in cb_options:
                combined_params = {**base_params, **cb_option}
                cb_combinations.append(combined_params)
        
        print(f"📊 Testing {len(cb_combinations)} CB combinations")
        
        # Parallel processing
        results = self._process_combinations_parallel(data, cb_combinations)
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        
        print(f"✅ Step 3 completed. Best fitness: {results[0]['fitness']:.4f}")
        return results[:3]  # Top 3 final results
    
    def _generate_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate parameter combinations"""
        import itertools
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _process_combinations_parallel(self, data: pd.DataFrame, 
                                     combinations: List[Dict], 
                                     progress_callback=None) -> List[Dict]:
        """Process combinations in parallel"""
        
        # Limit combinations for faster processing
        max_combinations = OPTIMIZATION_CONFIG.get('max_combinations', 1000)
        if len(combinations) > max_combinations:
            combinations = combinations[:max_combinations]
        
        results = []
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            
            # Submit all jobs
            future_to_params = {
                executor.submit(test_single_combination, data, params): params
                for params in combinations
            }
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_params):
                completed += 1
                
                if progress_callback:
                    progress_callback(completed / len(combinations))
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"⚠️ Error in parallel processing: {e}")
                    continue
                
                # Progress update
                if completed % 50 == 0:
                    print(f"📈 Completed {completed}/{len(combinations)} combinations")
        
        return results
    
    def calculate_fitness(self, metrics: Dict) -> float:
        """Calculate fitness score"""
        
        weights = {
            'return_weight': 0.30,
            'winrate_weight': 0.25,
            'profit_factor_weight': 0.25,
            'drawdown_penalty': 0.20
        }
        
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        max_drawdown = metrics.get('max_drawdown', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Normalize metrics
        normalized_return = min(max(total_return / 50, 0), 1)
        normalized_winrate = win_rate / 100
        normalized_pf = min(max((profit_factor - 1) / 2, 0), 1)
        drawdown_penalty = min(max(max_drawdown / 30, 0), 1)
        
        # Calculate composite score
        fitness = (
            weights['return_weight'] * normalized_return +
            weights['winrate_weight'] * normalized_winrate +
            weights['profit_factor_weight'] * normalized_pf -
            weights['drawdown_penalty'] * drawdown_penalty
        )
        
        # Trade count penalties
        if total_trades < 5:
            fitness *= 0.2
        elif total_trades < 10:
            fitness *= 0.6
        elif total_trades < 15:
            fitness *= 0.9
        
        return max(fitness, 0)

# Global function for multiprocessing
def test_single_combination(data: pd.DataFrame, parameters: Dict) -> Optional[Dict]:
    """Test a single parameter combination (for multiprocessing)"""
    try:
        # Add default parameters
        full_params = {
            'pivot_period': parameters.get('pivot_period', 5),
            'atr_factor': parameters.get('atr_factor', 1.2),
            'atr_period': parameters.get('atr_period', 12),
            'use_prev_atr': parameters.get('use_prev_atr', False),
            'use_adx': parameters.get('use_adx', False),
            'adx_threshold': parameters.get('adx_threshold', 15),
            'use_ema': parameters.get('use_ema', False),
            'ema_period': parameters.get('ema_period', 50),
            'enable_circuit_breaker': parameters.get('enable_circuit_breaker', False),
            'circuit_breaker_buffer': parameters.get('circuit_breaker_buffer', 0.08),
            'risk_percent': 2.0,
            'stop_loss_atr_multiplier': 1.0
        }
        
        # Run backtest
        backtester = HighPerformanceBacktester()
        trades, metrics = backtester.backtest_strategy_vectorized(data, full_params)
        
        # Calculate fitness
        engine = ParallelOptimizationEngine()
        fitness = engine.calculate_fitness(metrics)
        
        return {
            'parameters': parameters.copy(),
            'full_parameters': full_params,
            'metrics': metrics,
            'fitness': fitness,
            'trades': trades
        }
        
    except Exception as e:
        return None

print(f"⚡ Parallel Optimization Engine Ready ({mp.cpu_count()} cores)")

# ===============================
# CELL 5: Data Download and Validation
# ===============================

def download_optimization_data(symbol: str, timeframe: str, period: str) -> pd.DataFrame:
    """Download and prepare data for optimization"""
    
    print(f"📊 Downloading {symbol} data ({timeframe}, {period})")
    
    # Timeframe mapping
    timeframe_map = {
        '1m': '1m', '2m': '2m', '5m': '5m', 
        '15m': '15m', '1h': '1h', '4h': '1h'
    }
    
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=period,
            interval=timeframe_map.get(timeframe, '5m'),
            auto_adjust=True,
            prepost=False
        )
        
        # Handle 4h resampling
        if timeframe == '4h' and not data.empty:
            data = data.resample('4h').agg({
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        # Validate data
        if len(data) < 100:
            raise ValueError(f"Insufficient data: only {len(data)} bars")
        
        # Clean data
        data = data.dropna()
        
        print(f"✅ Downloaded {len(data)} bars")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        
        return data
        
    except Exception as e:
        raise Exception(f"Failed to download {symbol} data: {str(e)}")

print("📊 Data download functions ready")
# ===============================
# CELL 6: Configuration File Generator
# ===============================

def generate_cbot_config(optimized_params: Dict, asset: str, timeframe: str) -> Dict:
    """Generate cBot configuration"""
    
    config = {
        "Chart": {
            "Symbol": asset,
            "Period": timeframe
        },
        "Parameters": {
            "PosSizing": 1,
            "RiskPercent": 2.0,
            "RiskDollars": 50.0,
            "FixedUnits": 1000.0,
            "OverrideAsset": 1,
            "FloorToStep": True,
            "ForceMinSize": True,
            "MinStopDistance": 1.0,
            "PivotPeriod": optimized_params.get('pivot_period', 5),
            "AtrFactor": optimized_params.get('atr_factor', 1.2),
            "AtrPeriod": optimized_params.get('atr_period', 12),
            "UsePrevAtr": optimized_params.get('use_prev_atr', False),
            "UseXTrend": optimized_params.get('use_xtrend', False),
            "UseXTrendMTF": optimized_params.get('use_xtrend_mtf', False),
            "XTrendMTFTimeframe": optimized_params.get('xtrend_mtf_timeframe', 'm5'),
            "UseAdx": optimized_params.get('use_adx', False),
            "AdxThreshold": optimized_params.get('adx_threshold', 15),
            "UseEma": optimized_params.get('use_ema', False),
            "EmaPeriod": optimized_params.get('ema_period', 50),
            "EnableCircuitBreaker": optimized_params.get('enable_circuit_breaker', False),
            "CircuitBreakerBuffer": optimized_params.get('circuit_breaker_buffer', 0.08),
            "AllowReentry": optimized_params.get('allow_reentry', False),
            "ReentryCooldownBars": optimized_params.get('reentry_cooldown_bars', 2),
            "ReentryWindowBars": optimized_params.get('reentry_window_bars', 15),
            "EnableDetailedLogging": True,
            "LogSignalDetails": True,
            "ShowStatsTable": True,
            "ShowEntryExitMarkers": True
        }
    }
    
    return config

def generate_indicator_config(optimized_params: Dict, asset: str, timeframe: str) -> Dict:
    """Generate Indicator configuration"""
    
    config = {
        "Lines": [
            {
                "IsEnabled": True,
                "LineName": "PivotSupertrend",
                "Color": "#FF0000FF",
                "LineType": "Solid",
                "LineWidth": 3.0
            }
        ],
        "Parameters": {
            "PivotPeriod": str(optimized_params.get('pivot_period', 5)),
            "AtrFactor": str(optimized_params.get('atr_factor', 1.2)),
            "AtrPeriod": str(optimized_params.get('atr_period', 12)),
            "UsePrevAtr": str(optimized_params.get('use_prev_atr', False)),
            "UseAdx": str(optimized_params.get('use_adx', False)),
            "AdxThreshold": str(optimized_params.get('adx_threshold', 15)),
            "UseEma": str(optimized_params.get('use_ema', False)),
            "EmaPeriod": str(optimized_params.get('ema_period', 50)),
            "EnableCircuitBreaker": str(optimized_params.get('enable_circuit_breaker', False)),
            "CircuitBreakerBuffer": str(optimized_params.get('circuit_breaker_buffer', 0.08))
        }
    }
    
    return config

print("🔧 Configuration generators ready")

# ===============================
# CELL 7: Main Optimization Execution
# ===============================

def run_full_optimization():
    """Execute the complete 3-step optimization process"""
    
    print("🚀 Starting XPST Strategy Optimization")
    print("=" * 50)
    
    # Extract configuration
    asset = OPTIMIZATION_CONFIG["asset"]
    timeframe = OPTIMIZATION_CONFIG["timeframe"]
    period = OPTIMIZATION_CONFIG["period"]
    steps = OPTIMIZATION_CONFIG["steps"]
    
    optimization_results = {
        'asset': asset,
        'timeframe': timeframe,
        'period': period,
        'start_time': datetime.now().isoformat(),
        'steps_completed': [],
        'final_results': [],
        'all_results': {},
        'configuration_files': {},
        'status': 'running'
    }
    
    try:
        # Download data
        print(f"📊 Step 0: Data Download")
        data = download_optimization_data(asset, timeframe, period)
        
        optimization_results['data_stats'] = {
            'total_bars': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'timespan_days': (data.index[-1] - data.index[0]).days
        }
        
        # Initialize optimization engine
        engine = ParallelOptimizationEngine(max_workers=OPTIMIZATION_CONFIG["max_workers"])
        
        # Step 1: Core Parameters
        if "step1" in steps:
            print(f"\n🎯 Step 1: Core Parameter Optimization")
            step1_results = engine.optimize_step1_parallel(data)
            optimization_results['all_results']['step1'] = step1_results
            optimization_results['steps_completed'].append('step1')
            print(f"✅ Step 1 completed: {len(step1_results)} results")
        
        # Step 2: Filters  
        if "step2" in steps and "step1" in optimization_results['steps_completed']:
            print(f"\n🔍 Step 2: Filter Optimization")
            step2_results = engine.optimize_step2_parallel(data, optimization_results['all_results']['step1'])
            optimization_results['all_results']['step2'] = step2_results
            optimization_results['steps_completed'].append('step2')
            print(f"✅ Step 2 completed: {len(step2_results)} results")
        
        # Step 3: Circuit Breaker & Re-entry
        if "step3" in steps and "step2" in optimization_results['steps_completed']:
            print(f"\n⚡ Step 3: Circuit Breaker & Re-entry Optimization")
            step3_results = engine.optimize_step3_parallel(data, optimization_results['all_results']['step2'])
            optimization_results['all_results']['step3'] = step3_results
            optimization_results['steps_completed'].append('step3')
            print(f"✅ Step 3 completed: {len(step3_results)} results")
        
        # Get final results
        final_step = optimization_results['steps_completed'][-1]
        final_results = optimization_results['all_results'][final_step][:3]  # Top 3
        optimization_results['final_results'] = final_results
        
        # Generate configuration files
        print(f"\n📁 Generating Configuration Files")
        for i, result in enumerate(final_results):
            rank = i + 1
            params = result['parameters']
            
            # Generate cBot config
            cbot_config = generate_cbot_config(params, asset, timeframe)
            cbot_filename = f"{asset}_{timeframe}_rank{rank}.cbotset"
            optimization_results['configuration_files'][cbot_filename] = cbot_config
            
            # Generate Indicator config
            indicator_config = generate_indicator_config(params, asset, timeframe)
            indicator_filename = f"{asset}_{timeframe}_rank{rank}.indiset"
            optimization_results['configuration_files'][indicator_filename] = indicator_config
        
        # Mark as completed
        optimization_results['status'] = 'completed'
        optimization_results['end_time'] = datetime.now().isoformat()
        
        # Calculate total time
        start_time = datetime.fromisoformat(optimization_results['start_time'])
        end_time = datetime.fromisoformat(optimization_results['end_time'])
        total_time = (end_time - start_time).total_seconds()
        optimization_results['total_time_seconds'] = total_time
        
        print("=" * 50)
        print("🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.1f} seconds")
        print(f"🏆 Top result fitness: {final_results[0]['fitness']:.4f}")
        print(f"📈 Top result return: {final_results[0]['metrics']['total_return']:.2f}%")
        print(f"🎯 Win rate: {final_results[0]['metrics']['win_rate']:.1f}%")
        print(f"📊 Total trades: {final_results[0]['metrics']['total_trades']}")
        
        return optimization_results
        
    except Exception as e:
        optimization_results['status'] = 'error'
        optimization_results['error'] = str(e)
        optimization_results['end_time'] = datetime.now().isoformat()
        
        print(f"❌ Optimization failed: {str(e)}")
        return optimization_results

# ===============================
# CELL 8: Results Display and Export
# ===============================

def display_optimization_results(results: Dict):
    """Display formatted optimization results"""
    
    if results['status'] != 'completed':
        print(f"❌ Optimization Status: {results['status']}")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return
    
    print("\n" + "=" * 60)
    print("🏆 XPST OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Summary
    print(f"Asset: {results['asset']}")
    print(f"Timeframe: {results['timeframe']}")
    print(f"Data Period: {results['period']}")
    print(f"Total Bars: {results['data_stats']['total_bars']}")
    print(f"Steps Completed: {', '.join(results['steps_completed'])}")
    print(f"Optimization Time: {results['total_time_seconds']:.1f} seconds")
    
    # Top 3 Results
    print("\n🥇 TOP 3 OPTIMIZATION RESULTS:")
    print("-" * 60)
    
    for i, result in enumerate(results['final_results']):
        rank = i + 1
        metrics = result['metrics']
        params = result['parameters']
        
        print(f"\n🏅 RANK #{rank} (Fitness: {result['fitness']:.4f})")
        print(f"   📈 Total Return: {metrics['total_return']:.2f}%")
        print(f"   🎯 Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   💰 Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   📉 Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"   📊 Total Trades: {metrics['total_trades']}")
        print(f"   📐 Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        
        print(f"   ⚙️ Parameters:")
        print(f"      • Pivot Period: {params.get('pivot_period', 'N/A')}")
        print(f"      • ATR Factor: {params.get('atr_factor', 'N/A')}")
        print(f"      • ATR Period: {params.get('atr_period', 'N/A')}")
        print(f"      • Use ADX: {params.get('use_adx', False)}")
        if params.get('use_adx', False):
            print(f"      • ADX Threshold: {params.get('adx_threshold', 'N/A')}")
        print(f"      • Use EMA: {params.get('use_ema', False)}")
        if params.get('use_ema', False):
            print(f"      • EMA Period: {params.get('ema_period', 'N/A')}")
        print(f"      • Circuit Breaker: {params.get('enable_circuit_breaker', False)}")
        if params.get('enable_circuit_breaker', False):
            print(f"      • CB Buffer: {params.get('circuit_breaker_buffer', 'N/A'):.2%}")

def export_results_to_files(results: Dict):
    """Export results to downloadable files"""
    
    if results['status'] != 'completed':
        print("❌ Cannot export: Optimization not completed")
        return
    
    # Create results summary
    summary = {
        'optimization_summary': {
            'asset': results['asset'],
            'timeframe': results['timeframe'],
            'period': results['period'],
            'completion_time': results['end_time'],
            'total_time_seconds': results['total_time_seconds'],
            'steps_completed': results['steps_completed']
        },
        'top_results': results['final_results'],
        'data_statistics': results['data_stats']
    }
    
    # Save summary JSON
    with open('xpst_optimization_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save configuration files
    for filename, config in results['configuration_files'].items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    print("\n📁 FILES EXPORTED:")
    print("   • xpst_optimization_results.json - Complete results summary")
    
    for filename in results['configuration_files'].keys():
        print(f"   • {filename} - cTrader configuration")
    
    print("\n💾 Files are ready for download!")

# ===============================
# CELL 9: API Communication (for Streamlit)
# ===============================

def create_shareable_results_link(results: Dict) -> str:
    """Create a shareable link for results"""
    
    # For now, we'll use a simple approach - in production, you'd use a proper service
    # like GitHub Gist, Pastebin, or a custom API
    
    try:
        # Example: Upload to a simple file sharing service
        # This is a placeholder - implement with your preferred service
        
        results_json = json.dumps(results, indent=2, default=str)
        
        # For demo purposes, we'll just provide instructions
        print("\n🔗 To share results with Streamlit:")
        print("1. Download the 'xpst_optimization_results.json' file")
        print("2. Upload it to your preferred file sharing service")
        print("3. Use the public URL in your Streamlit app")
        
        return "results_exported_locally"
        
    except Exception as e:
        print(f"❌ Error creating shareable link: {e}")
        return None

def send_results_to_streamlit(results: Dict, callback_url: str = None):
    """Send results back to Streamlit app (if callback URL provided)"""
    
    if not callback_url:
        print("ℹ️ No callback URL provided - results available locally")
        return
    
    try:
        # Send results to Streamlit callback
        response = requests.post(
            callback_url,
            json=results,
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Results sent to Streamlit successfully!")
        else:
            print(f"⚠️ Failed to send to Streamlit: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error sending to Streamlit: {e}")

# ===============================
# CELL 10: Execution Control
# ===============================

# Set your optimization parameters here or modify OPTIMIZATION_CONFIG above
OPTIMIZATION_CONFIG.update({
    "asset": "BTC-USD",        # Change this to your desired asset
    "timeframe": "2m",         # Change timeframe
    "period": "7d",            # Change data period  
    "steps": ["step1"],        # Choose steps: ["step1"], ["step1", "step2"], or ["step1", "step2", "step3"]
    "max_combinations": 324,   # Limit for faster testing
    "max_workers": mp.cpu_count()
})

print("🎯 READY TO START OPTIMIZATION")
print("=" * 50)
print("Current Configuration:")
for key, value in OPTIMIZATION_CONFIG.items():
    print(f"  {key}: {value}")

print("\n🚀 To start optimization, run the next cell!")

# ===============================
# CELL 11: RUN OPTIMIZATION
# ===============================

# EXECUTE THE OPTIMIZATION
print("🚀 STARTING XPST OPTIMIZATION...")
print(f"Using {mp.cpu_count()} CPU cores for parallel processing\n")

# Run the optimization
optimization_results = run_full_optimization()

# Display results
display_optimization_results(optimization_results)

# Export files
export_results_to_files(optimization_results)

# Create shareable link
results_link = create_shareable_results_link(optimization_results)

print("\n🎉 OPTIMIZATION COMPLETE!")
print("📁 Check the files panel on the left to download your results")
print("🔗 Results are ready for integration with Streamlit")
