import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import itertools

class FilterOptimizer:
    """
    Optimizes filter parameters (XTrend, ADX, EMA) for Step 2 of optimization
    """
    
    def __init__(self):
        # Filter parameter ranges
        self.filter_ranges = {
            'xtrend_options': ['off', 'on'],
            'xtrend_mtf_timeframes': ['m1', 'm2', 'm3', 'm5', 'm15', 'm30', 'h1'],
            'adx_options': ['off', 'on'], 
            'adx_thresholds': list(range(5, 21)),  # 5-20
            'ema_options': ['off', 'on'],
            'ema_periods': [50, 100, 150, 200, 250]  # Step 50 as specified
        }
    
    def optimize_filters_for_core(self, data: pd.DataFrame, backtester, 
                                 core_params: Dict) -> List[Dict]:
        """
        Optimize filters for a given core parameter set using independent testing approach
        """
        print(f"Optimizing filters for core: {core_params}")
        
        # Test each filter type independently first
        xtrend_results = self._test_xtrend_options(data, backtester, core_params)
        adx_results = self._test_adx_options(data, backtester, core_params)
        ema_results = self._test_ema_options(data, backtester, core_params)
        
        # Get best options from each filter type
        best_xtrend = self._get_best_filter_options(xtrend_results, n=2)
        best_adx = self._get_best_filter_options(adx_results, n=2)
        best_ema = self._get_best_filter_options(ema_results, n=2)
        
        # Combine best filters
        combined_results = self._combine_best_filters(
            data, backtester, core_params, best_xtrend, best_adx, best_ema
        )
        
        return combined_results
    
    def _test_xtrend_options(self, data: pd.DataFrame, backtester, 
                            core_params: Dict) -> List[Dict]:
        """
        Test XTrend filter options
        """
        results = []
        
        # Test XTrend off
        xtrend_off_params = self._create_filter_params(core_params, {
            'use_xtrend': False,
            'use_xtrend_mtf': False
        })
        
        result = self._test_filter_combination(data, backtester, xtrend_off_params, 'xtrend_off')
        if result:
            results.append(result)
        
        # Test XTrend on with different MTF timeframes
        for timeframe in self.filter_ranges['xtrend_mtf_timeframes']:
            xtrend_on_params = self._create_filter_params(core_params, {
                'use_xtrend': True,
                'use_xtrend_mtf': True,
                'xtrend_mtf_timeframe': timeframe
            })
            
            result = self._test_filter_combination(data, backtester, xtrend_on_params, f'xtrend_{timeframe}')
            if result:
                results.append(result)
        
        return results
    
    def _test_adx_options(self, data: pd.DataFrame, backtester, 
                         core_params: Dict) -> List[Dict]:
        """
        Test ADX filter options
        """
        results = []
        
        # Test ADX off
        adx_off_params = self._create_filter_params(core_params, {
            'use_adx': False
        })
        
        result = self._test_filter_combination(data, backtester, adx_off_params, 'adx_off')
        if result:
            results.append(result)
        
        # Test ADX on with different thresholds
        for threshold in self.filter_ranges['adx_thresholds']:
            adx_on_params = self._create_filter_params(core_params, {
                'use_adx': True,
                'adx_threshold': threshold
            })
            
            result = self._test_filter_combination(data, backtester, adx_on_params, f'adx_{threshold}')
            if result:
                results.append(result)
        
        return results
    
    def _test_ema_options(self, data: pd.DataFrame, backtester, 
                         core_params: Dict) -> List[Dict]:
        """
        Test EMA filter options
        """
        results = []
        
        # Test EMA off
        ema_off_params = self._create_filter_params(core_params, {
            'use_ema': False
        })
        
        result = self._test_filter_combination(data, backtester, ema_off_params, 'ema_off')
        if result:
            results.append(result)
        
        # Test EMA on with different periods
        for period in self.filter_ranges['ema_periods']:
            ema_on_params = self._create_filter_params(core_params, {
                'use_ema': True,
                'ema_period': period
            })
            
            result = self._test_filter_combination(data, backtester, ema_on_params, f'ema_{period}')
            if result:
                results.append(result)
        
        return results
    
    def _combine_best_filters(self, data: pd.DataFrame, backtester, core_params: Dict,
                             best_xtrend: List[Dict], best_adx: List[Dict], 
                             best_ema: List[Dict]) -> List[Dict]:
        """
        Combine best filter options from each category
        """
        results = []
        
        # Generate all combinations of best filters
        for xtrend_result in best_xtrend:
            for adx_result in best_adx:
                for ema_result in best_ema:
                    
                    # Create combined parameter set
                    combined_params = core_params.copy()
                    
                    # Add XTrend settings
                    xtrend_filters = xtrend_result['filter_params']
                    combined_params.update({
                        'use_xtrend': xtrend_filters.get('use_xtrend', False),
                        'use_xtrend_mtf': xtrend_filters.get('use_xtrend_mtf', False),
                        'xtrend_mtf_timeframe': xtrend_filters.get('xtrend_mtf_timeframe', 'm5')
                    })
                    
                    # Add ADX settings
                    adx_filters = adx_result['filter_params']
                    combined_params.update({
                        'use_adx': adx_filters.get('use_adx', False),
                        'adx_threshold': adx_filters.get('adx_threshold', 15)
                    })
                    
                    # Add EMA settings
                    ema_filters = ema_result['filter_params']
                    combined_params.update({
                        'use_ema': ema_filters.get('use_ema', False),
                        'ema_period': ema_filters.get('ema_period', 50)
                    })
                    
                    # Create full parameter set
                    full_params = self._create_full_filter_parameter_set(combined_params)
                    
                    # Test combination
                    try:
                        trades, metrics = backtester.backtest_strategy(data, full_params)
                        fitness = self._calculate_fitness(metrics)
                        
                        result = {
                            'parameters': combined_params.copy(),
                            'full_parameters': full_params,
                            'metrics': metrics,
                            'fitness': fitness,
                            'trades': trades,
                            'filter_combination': {
                                'xtrend': xtrend_result['description'],
                                'adx': adx_result['description'],
                                'ema': ema_result['description']
                            }
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Error testing combined filters: {e}")
                        continue
        
        # Sort by fitness and return best combinations
        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results
    
    def _create_filter_params(self, core_params: Dict, filter_updates: Dict) -> Dict:
        """
        Create parameter set with core params and specific filter settings
        """
        params = core_params.copy()
        params.update(filter_updates)
        return params
    
    def _test_filter_combination(self, data: pd.DataFrame, backtester, 
                                params: Dict, description: str) -> Optional[Dict]:
        """
        Test a specific filter combination
        """
        try:
            # Create full parameter set
            full_params = self._create_full_filter_parameter_set(params)
            
            # Run backtest
            trades, metrics = backtester.backtest_strategy(data, full_params)
            
            # Calculate fitness
            fitness = self._calculate_fitness(metrics)
            
            return {
                'filter_params': params.copy(),
                'full_parameters': full_params,
                'metrics': metrics,
                'fitness': fitness,
                'trades': trades,
                'description': description
            }
            
        except Exception as e:
            print(f"Error testing {description}: {e}")
            return None
    
    def _create_full_filter_parameter_set(self, params: Dict) -> Dict:
        """
        Create full parameter set with all required parameters
        """
        full_params = {
            # Core parameters (from Step 1)
            'pivot_period': params.get('pivot_period', 5),
            'atr_factor': params.get('atr_factor', 1.2),
            'atr_period': params.get('atr_period', 12),
            'use_prev_atr': params.get('use_prev_atr', False),
            
            # Filter parameters (being optimized in Step 2)
            'use_xtrend': params.get('use_xtrend', False),
            'use_xtrend_mtf': params.get('use_xtrend_mtf', False),
            'xtrend_mtf_timeframe': params.get('xtrend_mtf_timeframe', 'm5'),
            'use_adx': params.get('use_adx', False),
            'adx_threshold': params.get('adx_threshold', 15),
            'use_ema': params.get('use_ema', False),
            'ema_period': params.get('ema_period', 50),
            
            # Circuit breaker and re-entry (defaults for Step 2)
            'enable_circuit_breaker': False,
            'circuit_breaker_buffer': 0.08,
            'allow_reentry': False,
            'reentry_cooldown_bars': 2,
            'reentry_window_bars': 15,
            
            # Risk management
            'risk_percent': 2.0,
            'stop_loss_atr_multiplier': 1.0,
            'take_profit_ratio': 2.0
        }
        
        return full_params
    
    def _get_best_filter_options(self, results: List[Dict], n: int = 2) -> List[Dict]:
        """
        Get best N options from filter test results
        """
        if not results:
            return []
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results[:n]
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score for filter optimization
        """
        # Fitness function for filter optimization
        weights = {
            'return_weight': 0.30,
            'winrate_weight': 0.25,
            'profit_factor_weight': 0.25,
            'sharpe_weight': 0.10,
            'drawdown_penalty': 0.10
        }
        
        # Extract metrics
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Normalize metrics
        normalized_return = min(max(total_return / 50, 0), 1)
        normalized_winrate = win_rate / 100
        normalized_pf = min(max((profit_factor - 1) / 2, 0), 1)
        normalized_sharpe = min(max(sharpe_ratio / 2, 0), 1)
        drawdown_penalty = min(max(max_drawdown / 30, 0), 1)
        
        # Calculate composite score
        fitness = (
            weights['return_weight'] * normalized_return +
            weights['winrate_weight'] * normalized_winrate +
            weights['profit_factor_weight'] * normalized_pf +
            weights['sharpe_weight'] * normalized_sharpe -
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
    
    def get_filter_optimization_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary of filter optimization results
        """
        if not results:
            return {}
        
        best_result = results[0]
        
        summary = {
            'total_combinations_tested': len(results),
            'best_fitness': best_result['fitness'],
            'best_filter_combination': best_result.get('filter_combination', {}),
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics'],
            'filters_tested': {
                'xtrend_combinations': len([r for r in results if 'xtrend' in r.get('description', '')]),
                'adx_combinations': len([r for r in results if 'adx' in r.get('description', '')]),
                'ema_combinations': len([r for r in results if 'ema' in r.get('description', '')])
            }
        }
        
        return summary
