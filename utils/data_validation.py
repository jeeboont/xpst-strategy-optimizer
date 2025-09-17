import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

class DataValidator:
    """
    Validates data quality for optimization purposes
    """
    
    def __init__(self):
        # Minimum data requirements per timeframe
        self.min_bars_required = {
            '1m': 500,   # ~8 hours
            '2m': 500,   # ~16 hours  
            '5m': 400,   # ~33 hours
            '15m': 300,  # ~3 days
            '1h': 200,   # ~8 days
            '4h': 100,   # ~17 days
            '1d': 100    # ~3 months
        }
        
        # Quality thresholds
        self.max_missing_data_percent = 5.0  # Maximum 5% missing data
        self.min_volume_threshold = 0  # Minimum average volume
        self.max_gap_hours = 24  # Maximum gap between bars (hours)
    
    def validate_data_for_optimization(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Comprehensive data validation for optimization
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'data_stats': {},
            'quality_score': 0.0
        }
        
        # Basic data checks
        if data.empty:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Data is empty")
            return validation_results
        
        # Check minimum data requirements
        min_required = self.min_bars_required.get(timeframe, 200)
        if len(data) < min_required:
            validation_results['is_valid'] = False
            validation_results['errors'].append(
                f"Insufficient data: {len(data)} bars (need {min_required})"
            )
        
        # Validate required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing columns: {missing_columns}")
            return validation_results
        
        # Data quality checks
        self._check_missing_data(data, validation_results)
        self._check_price_data_integrity(data, validation_results)
        self._check_volume_data(data, validation_results)
        self._check_time_gaps(data, timeframe, validation_results)
        self._check_price_anomalies(data, validation_results)
        
        # Calculate data statistics
        validation_results['data_stats'] = self._calculate_data_stats(data, timeframe)
        
        # Calculate overall quality score
        validation_results['quality_score'] = self._calculate_quality_score(data, validation_results)
        
        return validation_results
    
    def _check_missing_data(self, data: pd.DataFrame, results: Dict):
        """
        Check for missing data points
        """
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        missing_percent = (missing_cells / total_cells) * 100
        
        if missing_percent > self.max_missing_data_percent:
            results['warnings'].append(
                f"High missing data: {missing_percent:.1f}% (threshold: {self.max_missing_data_percent}%)"
            )
        elif missing_percent > 0:
            results['warnings'].append(f"Missing data: {missing_percent:.1f}%")
    
    def _check_price_data_integrity(self, data: pd.DataFrame, results: Dict):
        """
        Check price data integrity (OHLC relationships)
        """
        # Check for invalid OHLC relationships
        invalid_high = (data['High'] < data['Open']) | (data['High'] < data['Close']) | \
                      (data['High'] < data['Low'])
        invalid_low = (data['Low'] > data['Open']) | (data['Low'] > data['Close']) | \
                     (data['Low'] > data['High'])
        
        invalid_count = invalid_high.sum() + invalid_low.sum()
        
        if invalid_count > 0:
            results['warnings'].append(f"Invalid OHLC relationships: {invalid_count} bars")
        
        # Check for zero or negative prices
        zero_prices = (data[['Open', 'High', 'Low', 'Close']] <= 0).any(axis=1).sum()
        if zero_prices > 0:
            results['warnings'].append(f"Zero or negative prices: {zero_prices} bars")
    
    def _check_volume_data(self, data: pd.DataFrame, results: Dict):
        """
        Check volume data quality
        """
        if 'Volume' in data.columns:
            zero_volume = (data['Volume'] == 0).sum()
            total_bars = len(data)
            zero_volume_percent = (zero_volume / total_bars) * 100
            
            if zero_volume_percent > 20:
                results['warnings'].append(
                    f"High zero volume bars: {zero_volume_percent:.1f}%"
                )
            
            avg_volume = data['Volume'].mean()
            if avg_volume < self.min_volume_threshold:
                results['warnings'].append("Very low average volume")
    
    def _check_time_gaps(self, data: pd.DataFrame, timeframe: str, results: Dict):
        """
        Check for unusual time gaps in data
        """
        if len(data) < 2:
            return
        
        # Calculate expected interval in minutes
        timeframe_minutes = {
            '1m': 1, '2m': 2, '5m': 5, '15m': 15, 
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        expected_interval = timeframe_minutes.get(timeframe, 5)
        
        # Check for gaps
        time_diffs = data.index.to_series().diff().dt.total_seconds() / 60
        large_gaps = time_diffs > (expected_interval * 3)  # 3x expected interval
        
        gap_count = large_gaps.sum()
        if gap_count > 0:
            max_gap_hours = time_diffs.max() / 60
            results['warnings'].append(
                f"Time gaps detected: {gap_count} gaps, largest: {max_gap_hours:.1f} hours"
            )
    
    def _check_price_anomalies(self, data: pd.DataFrame, results: Dict):
        """
        Check for price anomalies and outliers
        """
        # Calculate price changes
        price_changes = data['Close'].pct_change().dropna()
        
        # Check for extreme price movements
        extreme_threshold = 0.10  # 10% price change
        extreme_moves = np.abs(price_changes) > extreme_threshold
        extreme_count = extreme_moves.sum()
        
        if extreme_count > len(data) * 0.05:  # More than 5% of bars
            results['warnings'].append(
                f"Many extreme price movements: {extreme_count} bars > {extreme_threshold*100}%"
            )
        
        # Check for price spikes
        price_std = price_changes.std()
        spikes = np.abs(price_changes) > (price_std * 5)  # 5 standard deviations
        spike_count = spikes.sum()
        
        if spike_count > 0:
            results['warnings'].append(f"Price spikes detected: {spike_count} bars")
    
    def _calculate_data_stats(self, data: pd.DataFrame, timeframe: str) -> Dict:
        """
        Calculate comprehensive data statistics
        """
        stats = {
            'total_bars': len(data),
            'date_range': f"{data.index[0]} to {data.index[-1]}",
            'timespan_days': (data.index[-1] - data.index[0]).days,
            'missing_data_points': data.isnull().sum().sum(),
            'missing_data_percent': (data.isnull().sum().sum() / data.size) * 100,
            'timeframe': timeframe
        }
        
        # Price statistics
        if 'Close' in data.columns:
            stats.update({
                'price_min': data['Close'].min(),
                'price_max': data['Close'].max(),
                'price_mean': data['Close'].mean(),
                'price_volatility': data['Close'].pct_change().std() * 100,
                'total_price_change_percent': ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            })
        
        # Volume statistics
        if 'Volume' in data.columns:
            stats.update({
                'avg_volume': data['Volume'].mean(),
                'zero_volume_bars': (data['Volume'] == 0).sum(),
                'zero_volume_percent': ((data['Volume'] == 0).sum() / len(data)) * 100
            })
        
        return stats
    
    def _calculate_quality_score(self, data: pd.DataFrame, results: Dict) -> float:
        """
        Calculate overall data quality score (0-100)
        """
        score = 100.0
        
        # Deduct points for issues
        score -= len(results['errors']) * 25  # Major issues
        score -= len(results['warnings']) * 5  # Minor issues
        
        # Deduct for missing data
        missing_percent = (data.isnull().sum().sum() / data.size) * 100
        score -= missing_percent * 2  # 2 points per percent missing
        
        # Deduct for insufficient data
        min_required = self.min_bars_required.get(results['data_stats'].get('timeframe', '5m'), 200)
        if len(data) < min_required:
            shortage_percent = (1 - len(data) / min_required) * 100
            score -= shortage_percent * 0.5
        
        return max(score, 0.0)
    
    def get_validation_summary(self, validation_results: Dict) -> str:
        """
        Get human-readable validation summary
        """
        summary = []
        
        # Overall status
        status = "VALID" if validation_results['is_valid'] else "INVALID"
        quality_score = validation_results['quality_score']
        summary.append(f"Data Status: {status} (Quality Score: {quality_score:.1f}/100)")
        
        # Data statistics
        stats = validation_results['data_stats']
        summary.append(f"Data Points: {stats['total_bars']} bars")
        summary.append(f"Date Range: {stats['date_range']}")
        summary.append(f"Timespan: {stats['timespan_days']} days")
        
        # Issues
        if validation_results['errors']:
            summary.append("\nERRORS:")
            for error in validation_results['errors']:
                summary.append(f"  • {error}")
        
        if validation_results['warnings']:
            summary.append("\nWARNINGS:")
            for warning in validation_results['warnings']:
                summary.append(f"  • {warning}")
        
        # Recommendations
        recommendations = self._get_recommendations(validation_results)
        if recommendations:
            summary.append("\nRECOMMENDATIONS:")
            for rec in recommendations:
                summary.append(f"  • {rec}")
        
        return "\n".join(summary)
    
    def _get_recommendations(self, validation_results: Dict) -> List[str]:
        """
        Get recommendations based on validation results
        """
        recommendations = []
        
        if not validation_results['is_valid']:
            recommendations.append("Fix data issues before proceeding with optimization")
        
        quality_score = validation_results['quality_score']
        if quality_score < 70:
            recommendations.append("Consider obtaining higher quality data")
        
        stats = validation_results['data_stats']
        if stats['total_bars'] < 1000:
            recommendations.append("More data points would improve optimization reliability")
        
        if stats['missing_data_percent'] > 2:
            recommendations.append("Clean missing data before optimization")
        
        if 'zero_volume_percent' in stats and stats['zero_volume_percent'] > 10:
            recommendations.append("High zero-volume periods may affect results")
        
        return recommendations
    
    def prepare_data_for_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and clean data for optimization
        """
        cleaned_data = data.copy()
        
        # Remove rows with missing critical data
        critical_columns = ['Open', 'High', 'Low', 'Close']
        cleaned_data = cleaned_data.dropna(subset=critical_columns)
        
        # Fill missing volume with 0
        if 'Volume' in cleaned_data.columns:
            cleaned_data['Volume'] = cleaned_data['Volume'].fillna(0)
        
        # Remove invalid OHLC relationships
        valid_ohlc = (
            (cleaned_data['High'] >= cleaned_data['Low']) &
            (cleaned_data['High'] >= cleaned_data['Open']) &
            (cleaned_data['High'] >= cleaned_data['Close']) &
            (cleaned_data['Low'] <= cleaned_data['Open']) &
            (cleaned_data['Low'] <= cleaned_data['Close'])
        )
        
        cleaned_data = cleaned_data[valid_ohlc]
        
        # Remove zero or negative prices
        positive_prices = (cleaned_data[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
        cleaned_data = cleaned_data[positive_prices]
        
        # Sort by index (time)
        cleaned_data = cleaned_data.sort_index()
        
        return cleaned_data
