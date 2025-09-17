import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import itertools

class CBReentryOptimizer:
    """
    Optimizes Circuit Breaker and Re-Entry parameters for Step 3 of optimization
    """
    
    def __init__(self):
        # Parameter ranges for CB and Re-entry
        self.cb_ranges = {
            'enable_cb': [False, True],
            'cb_buffer_values': [round(0.01 + i*0.01, 2) for i in range(15)]  # 0.01-0.15 in 0.01 steps
        }
        
        self.reentry_ranges = {
            'allow_reentry': [False, True],
            'cooldown_bars': list(range(0, 5)),     # 0-4 bars
            'window_bars': list(range(0, 21, 5))    # 0-20 bars in steps of 5
        }
    
    def optimize_cb_reentry_for_base(self, data: pd.DataFrame, backtester, 
                                    base_params: Dict) -> List[Dict]:
        """
        Optimize CB and Re-entry settings for a given base parameter set
        """
        print(f"Optimizing CB & Re-entry for: {base_params}")
        
        results = []
        
        # First optimize Circuit Breaker
        cb_results = self._optimize_circuit_breaker(data, backtester, base_params)
        
        # For each good CB setting, optimize re-entry (if XTrend is enabled)
        for cb_result in cb_results[:3]:  # Top 3 CB settings
            
            if cb_result['parameters'].get('use_xtrend', False):
                # XTrend is enabled, can test re-entry
                reentry_results = self._optimize_reentry(data, backtester, cb_result['parameters'])
                results.extend(reentry_results)
            else:
                # No XTrend, re-entry not applicable
                results.append(cb_result)
        
        # Sort all results by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results
    
    def _optimize_circuit_breaker(self, data: pd.DataFrame, backtester, 
                                 base_params: Dict) -> List[Dict]:
        """
        Optimize Circuit Breaker settings
        """
        results = []
        
        # Test CB disabled
        cb_off_params = base_params.copy()
        cb_off_params['enable_circuit_breaker'] = False
        
        result = self._test_cb_combination(data, backtester, cb_off_params, 'cb_off')
        if result:
            results.append(result)
        
        # Test CB enabled with different buffer values
        for buffer_value in self.cb_ranges['cb_buffer_values']:
            cb_on_params = base_params.copy()
            cb_on_params.update({
                'enable_circuit_breaker': True,
                'circuit_breaker_buffer': buffer_value
            })
            
            result = self._test_cb_combination(data, backtester, cb_on_params, f'cb_{buffer_value}')
            if result:
                results.append(result)
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results
    
    def _optimize_reentry(self, data: pd.DataFrame, backtester, 
                         cb_params: Dict) -> List[Dict]:
        """
        Optimize re-entry settings for a given CB configuration
        """
        results = []
        
        # Test re-entry disabled
        reentry_off_params = cb_params.copy()
        reentry_off_params['allow_reentry'] = False
        
        result = self._test_reentry_combination(data, backtester, reentry_off_params, 'reentry_off')
        if result:
            results.append(result)
        
        # Test re-entry enabled with different cooldown and window combinations
        for cooldown in self.reentry_ranges['cooldown_bars']:
            for window in self.reentry_ranges['window_bars']:
                
                # Skip invalid combinations (window should be >= cooldown)
                if window < cooldown:
                    continue
                
                reentry_on_params = cb_params.copy()
                reentry_on_params.update({
                    'allow_reentry': True,
                    'reentry_cooldown_bars': cooldown,
                    'reentry_window_bars': window
                })
                
                result = self._test_reentry_combination(
                    data, backtester, reentry_on_params, 
                    f'reentry_c{cooldown}_w{window}'
                )
                if result:
                    results.append(result)
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        return results
    
    def _test_cb_combination(self, data: pd.DataFrame, backtester, 
                            params: Dict, description: str) -> Optional[Dict]:
        """
        Test a specific CB combination
        """
        try:
            # Create full parameter set
            full_params = self._create_full_cb_parameter_set(params)
            
            # Run backtest
            trades, metrics = backtester.backtest_strategy(data, full_params)
            
            # Calculate fitness
            fitness = self._calculate_fitness(metrics)
            
            return {
                'parameters': params.copy(),
                'full_parameters': full_params,
                'metrics': metrics,
                'fitness': fitness,
                'trades': trades,
                'description': description,
                'optimization_step': 'circuit_breaker'
            }
            
        except Exception as e:
            print(f"Error testing CB {description}: {e}")
            return None
    
    def _test_reentry_combination(self, data: pd.DataFrame, backtester, 
                                 params: Dict, description: str) -> Optional[Dict]:
        """
        Test a specific re-entry combination
        """
        try:
            # Create full parameter set
            full_params = self._create_full_cb_parameter_set(params)
            
            # Run backtest
            trades, metrics = backtester.backtest_strategy(data, full_params)
            
            # Calculate fitness
            fitness = self._calculate_fitness(metrics)
            
            return {
                'parameters': params.copy(),
                'full_parameters': full_params,
                'metrics': metrics,
                'fitness': fitness,
                'trades': trades,
                'description': description,
                'optimization_step': 'reentry'
            }
            
        except Exception as e:
            print(f"Error testing re-entry {description}: {e}")
            return None
    
    def _create_full_cb_parameter_set(self, params: Dict) -> Dict:
        """
        Create full parameter set for CB/re-entry optimization
        """
        full_params = {
            # Core parameters (from Step 1)
            'pivot_period': params.get('pivot_period', 5),
            'atr_factor': params.get('atr_factor', 1.2),
            'atr_period': params.get('atr_period', 12),
            'use_prev_atr': params.get('use_prev_atr', False),
            
            # Filter parameters (from Step 2)
            'use_xtrend': params.get('use_xtrend', False),
            'use_xtrend_mtf': params.get('use_xtrend_mtf', False),
            'xtrend_mtf_timeframe': params.get('xtrend_mtf_timeframe', 'm5'),
            'use_adx': params.get('use_adx', False),
            'adx_threshold': params.get('adx_threshold', 15),
            'use_ema': params.get('use_ema', False),
            'ema_period': params.get('ema_period', 50),
            
            # Circuit breaker and re-entry (being optimized in Step 3)
            'enable_circuit_breaker': params.get('enable_circuit_breaker', False),
            'circuit_breaker_buffer': params.get('circuit_breaker_buffer', 0.08),
            'allow_reentry': params.get('allow_reentry', False),
            'reentry_cooldown_bars': params.get('reentry_cooldown_bars', 2),
            'reentry_window_bars': params.get('reentry_window_bars', 15),
            
            # Risk management
            'risk_percent': params.get('risk_percent', 2.0),
            'stop_loss_atr_multiplier': 1.0,
            'take_profit_ratio': 2.0
        }
        
        return full_params
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score for CB/re-entry optimization
        Focus on risk-adjusted returns and drawdown control
        """
        weights = {
            'return_weight': 0.25,
            'winrate_weight': 0.20,
            'profit_factor_weight': 0.20,
            'sharpe_weight': 0.15,
            'calmar_weight': 0.10,  # Return/Max Drawdown ratio
            'drawdown_penalty': 0.10
        }
        
        # Extract metrics
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        max_drawdown = metrics.get('max_drawdown', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Calculate Calmar ratio (return/max drawdown)
        calmar_ratio = total_return / max(max_drawdown, 1) if max_drawdown > 0 else 0
        
        # Normalize metrics
        normalized_return = min(max(total_return / 50, 0), 1)
        normalized_winrate = win_rate / 100
        normalized_pf = min(max((profit_factor - 1) / 2, 0), 1)
        normalized_sharpe = min(max(sharpe_ratio / 2, 0), 1)
        normalized_calmar = min(max(calmar_ratio / 2, 0), 1)
        drawdown_penalty = min(max(max_drawdown / 25, 0), 1)  # Stricter penalty
        
        # Calculate composite score
        fitness = (
            weights['return_weight'] * normalized_return +
            weights['winrate_weight'] * normalized_winrate +
            weights['profit_factor_weight'] * normalized_pf +
            weights['sharpe_weight'] * normalized_sharpe +
            weights['calmar_weight'] * normalized_calmar -
            weights['drawdown_penalty'] * drawdown_penalty
        )
        
        # Trade count considerations
        if total_trades < 5:
            fitness *= 0.3
        elif total_trades < 10:
            fitness *= 0.7
        elif total_trades < 15:
            fitness *= 0.9
        
        # Bonus for very low drawdown with decent returns
        if max_drawdown < 5 and total_return > 10:
            fitness *= 1.1
        
        return max(fitness, 0)
    
    def get_cb_reentry_optimization_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary of CB and re-entry optimization results
        """
        if not results:
            return {}
        
        best_result = results[0]
        
        # Analyze CB vs no CB performance
        cb_results = [r for r in results if r['parameters'].get('enable_circuit_breaker', False)]
        no_cb_results = [r for r in results if not r['parameters'].get('enable_circuit_breaker', False)]
        
        # Analyze re-entry vs no re-entry (for XTrend strategies)
        reentry_results = [r for r in results if r['parameters'].get('allow_reentry', False)]
        no_reentry_results = [r for r in results if not r['parameters'].get('allow_reentry', False)]
        
        summary = {
            'total_combinations_tested': len(results),
            'best_fitness': best_result['fitness'],
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics'],
            'circuit_breaker_analysis': {
                'cb_enabled_results': len(cb_results),
                'cb_disabled_results': len(no_cb_results),
                'best_cb_fitness': max([r['fitness'] for r in cb_results]) if cb_results else 0,
                'best_no_cb_fitness': max([r['fitness'] for r in no_cb_results]) if no_cb_results else 0,
                'cb_recommended': len(cb_results) > 0 and cb_results[0]['fitness'] > (no_cb_results[0]['fitness'] if no_cb_results else 0)
            },
            'reentry_analysis': {
                'reentry_enabled_results': len(reentry_results),
                'reentry_disabled_results': len(no_reentry_results),
                'best_reentry_fitness': max([r['fitness'] for r in reentry_results]) if reentry_results else 0,
                'best_no_reentry_fitness': max([r['fitness'] for r in no_reentry_results]) if no_reentry_results else 0,
                'reentry_recommended': len(reentry_results) > 0 and reentry_results[0]['fitness'] > (no_reentry_results[0]['fitness'] if no_reentry_results else 0)
            }
        }
        
        return summary
    
    def analyze_risk_impact(self, results: List[Dict]) -> Dict:
        """
        Analyze the impact of CB and re-entry on risk metrics
        """
        if not results:
            return {}
        
        # Group results by CB and re-entry settings
        cb_on = [r for r in results if r['parameters'].get('enable_circuit_breaker', False)]
        cb_off = [r for r in results if not r['parameters'].get('enable_circuit_breaker', False)]
        
        reentry_on = [r for r in results if r['parameters'].get('allow_reentry', False)]
        reentry_off = [r for r in results if not r['parameters'].get('allow_reentry', False)]
        
        def get_avg_metric(result_list: List[Dict], metric: str) -> float:
            if not result_list:
                return 0
            return sum(r['metrics'].get(metric, 0) for r in result_list) / len(result_list)
        
        analysis = {
            'circuit_breaker_impact': {
                'avg_drawdown_with_cb': get_avg_metric(cb_on, 'max_drawdown'),
                'avg_drawdown_without_cb': get_avg_metric(cb_off, 'max_drawdown'),
                'avg_return_with_cb': get_avg_metric(cb_on, 'total_return'),
                'avg_return_without_cb': get_avg_metric(cb_off, 'total_return'),
                'drawdown_reduction': get_avg_metric(cb_off, 'max_drawdown') - get_avg_metric(cb_on, 'max_drawdown')
            },
            'reentry_impact': {
                'avg_trades_with_reentry': get_avg_metric(reentry_on, 'total_trades'),
                'avg_trades_without_reentry': get_avg_metric(reentry_off, 'total_trades'),
                'avg_return_with_reentry': get_avg_metric(reentry_on, 'total_return'),
                'avg_return_without_reentry': get_avg_metric(reentry_off, 'total_return'),
                'trade_increase': get_avg_metric(reentry_on, 'total_trades') - get_avg_metric(reentry_off, 'total_trades')
            }
        }
        
        return analysis
