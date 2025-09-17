import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import itertools

class CoreParameterOptimizer:
    """
    Optimizes core XPST parameters using the coarse-to-fine approach discussed
    """
    
    def __init__(self):
        # Coarse grid for initial exploration
        self.coarse_grid = {
            'pivot_period': [3, 5, 7, 9, 12, 15],                    # 6 values
            'atr_factor': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],  # 9 values
            'atr_period': [12, 16, 20, 25, 30, 35]                   # 6 values
        }
        # Total coarse combinations: 6 × 9 × 6 = 324 combinations
        
        # Full ranges for fine-tuning
        self.full_ranges = {
            'pivot_period': list(range(2, 16)),                      # 2-15
            'atr_factor': [round(0.8 + i*0.05, 2) for i in range(25)], # 0.8-2.0 in 0.05 steps
            'atr_period': list(range(10, 41))                        # 10-40
        }
    
    def run_coarse_optimization(self, data: pd.DataFrame, backtester, 
                               progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Run coarse grid search first
        """
        print("Running coarse grid search...")
        
        # Generate all coarse combinations
        coarse_combinations = self._generate_parameter_combinations(self.coarse_grid)
        
        print(f"Testing {len(coarse_combinations)} coarse combinations")
        
        results = []
        
        for i, params in enumerate(coarse_combinations):
            if progress_callback:
                progress_callback((i / len(coarse_combinations)) * 0.7)  # 70% of Step 1
            
            try:
                # Create full parameter set with defaults
                full_params = self._create_full_parameter_set(params)
                
                # Run backtest
                trades, metrics = backtester.backtest_strategy(data, full_params)
                
                # Calculate fitness
                fitness = self._calculate_fitness(metrics)
                
                result = {
                    'parameters': params.copy(),
                    'full_parameters': full_params,
                    'metrics': metrics,
                    'fitness': fitness,
                    'trades': trades
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error in coarse optimization for {params}: {e}")
                continue
        
        # Sort by fitness and return top performers
        results.sort(key=lambda x: x['fitness'], reverse=True)
        
        print(f"Coarse optimization completed. Best fitness: {results[0]['fitness']:.4f}")
        return results
    
    def run_fine_optimization(self, data: pd.DataFrame, backtester, 
                             best_coarse_params: Dict) -> List[Dict]:
        """
        Run fine-tuning around the best coarse parameters
        """
        print(f"Fine-tuning around: {best_coarse_params}")
        
        # Create fine-tuning grid around best parameters
        fine_grid = self._create_fine_tune_grid(best_coarse_params)
        
        # Generate combinations
        fine_combinations = self._generate_parameter_combinations(fine_grid)
        
        print(f"Testing {len(fine_combinations)} fine-tuning combinations")
        
        results = []
        
        for params in fine_combinations:
            try:
                # Create full parameter set
                full_params = self._create_full_parameter_set(params)
                
                # Run backtest
                trades, metrics = backtester.backtest_strategy(data, full_params)
                
                # Calculate fitness
                fitness = self._calculate_fitness(metrics)
                
                result = {
                    'parameters': params.copy(),
                    'full_parameters': full_params,
                    'metrics': metrics,
                    'fitness': fitness,
                    'trades': trades
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error in fine optimization for {params}: {e}")
                continue
        
        # Sort by fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)
        
        if results:
            print(f"Fine optimization completed. Best fitness: {results[0]['fitness']:.4f}")
        
        return results
    
    def _create_fine_tune_grid(self, best_params: Dict) -> Dict:
        """
        Create fine-tuning grid around best parameters using the manual approach
        """
        fine_grid = {}
        
        # Pivot period: ±1 around best
        pivot_neighbors = self._get_neighbors(
            best_params['pivot_period'], 
            self.full_ranges['pivot_period'], 
            step=1
        )
        fine_grid['pivot_period'] = pivot_neighbors
        
        # ATR factor: ±0.05 around best (your manual approach)
        atr_factor_neighbors = self._get_atr_factor_neighbors(best_params['atr_factor'])
        fine_grid['atr_factor'] = atr_factor_neighbors
        
        # ATR period: ±1 around best
        atr_period_neighbors = self._get_neighbors(
            best_params['atr_period'],
            self.full_ranges['atr_period'],
            step=1
        )
        fine_grid['atr_period'] = atr_period_neighbors
        
        return fine_grid
    
    def _get_atr_factor_neighbors(self, best_value: float) -> List[float]:
        """
        Get ATR factor neighbors using the ±0.05 approach you described
        """
        neighbors = [best_value]  # Always include the best value
        
        # Test 0.05 below
        if best_value - 0.05 >= 0.8:
            neighbors.append(round(best_value - 0.05, 2))
        
        # Test 0.05 above
        if best_value + 0.05 <= 2.0:
            neighbors.append(round(best_value + 0.05, 2))
        
        return sorted(list(set(neighbors)))  # Remove duplicates and sort
    
    def _get_neighbors(self, best_value: int, valid_range: List[int], step: int = 1) -> List[int]:
        """
        Get neighboring values for integer parameters
        """
        neighbors = [best_value]
        
        # Add neighbors within valid range
        if best_value - step in valid_range:
            neighbors.append(best_value - step)
        if best_value + step in valid_range:
            neighbors.append(best_value + step)
        
        return sorted(list(set(neighbors)))
    
    def _generate_parameter_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Generate all combinations from parameter grid
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        return combinations
    
    def _create_full_parameter_set(self, core_params: Dict) -> Dict:
        """
        Create full parameter set with core parameters and defaults for other settings
        """
        full_params = {
            # Core parameters (being optimized)
            'pivot_period': core_params['pivot_period'],
            'atr_factor': core_params['atr_factor'],
            'atr_period': core_params['atr_period'],
            
            # Default values for other parameters (not optimized in Step 1)
            'use_prev_atr': False,
            'use_xtrend': False,
            'use_xtrend_mtf': False,
            'xtrend_mtf_timeframe': 'm5',
            'use_adx': False,
            'adx_threshold': 15,
            'use_ema': False,
            'ema_period': 50,
            'enable_circuit_breaker': False,
            'circuit_breaker_buffer': 0.08,
            'allow_reentry': False,
            'reentry_cooldown_bars': 2,
            'reentry_window_bars': 15,
            
            # Risk management (fixed for optimization)
            'risk_percent': 2.0,
            'stop_loss_atr_multiplier': 1.0,
            'take_profit_ratio': 2.0
        }
        
        return full_params
    
    def _calculate_fitness(self, metrics: Dict) -> float:
        """
        Calculate fitness score for core parameter optimization
        """
        # Fitness function optimized for core parameter selection
        weights = {
            'return_weight': 0.30,
            'winrate_weight': 0.25,
            'profit_factor_weight': 0.25,
            'drawdown_penalty': 0.20
        }
        
        # Extract metrics with defaults
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        max_drawdown = metrics.get('max_drawdown', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Normalize metrics to [0, 1] scale
        normalized_return = min(max(total_return / 50, 0), 1)  # Scale to 50% max return
        normalized_winrate = win_rate / 100
        normalized_pf = min(max((profit_factor - 1) / 2, 0), 1)  # Scale profit factor
        drawdown_penalty = min(max(max_drawdown / 30, 0), 1)  # Scale to 30% max drawdown
        
        # Calculate composite score
        fitness = (
            weights['return_weight'] * normalized_return +
            weights['winrate_weight'] * normalized_winrate +
            weights['profit_factor_weight'] * normalized_pf -
            weights['drawdown_penalty'] * drawdown_penalty
        )
        
        # Apply trade count penalties
        if total_trades < 5:
            fitness *= 0.1  # Heavy penalty for too few trades
        elif total_trades < 10:
            fitness *= 0.5
        elif total_trades < 15:
            fitness *= 0.8
        
        # Ensure non-negative fitness
        return max(fitness, 0)
    
    def get_optimization_summary(self, results: List[Dict]) -> Dict:
        """
        Get summary of optimization results
        """
        if not results:
            return {}
        
        best_result = results[0]
        
        summary = {
            'total_combinations_tested': len(results),
            'best_fitness': best_result['fitness'],
            'best_parameters': best_result['parameters'],
            'best_metrics': best_result['metrics'],
            'parameter_ranges_tested': {
                'pivot_period': [min(r['parameters']['pivot_period'] for r in results),
                               max(r['parameters']['pivot_period'] for r in results)],
                'atr_factor': [min(r['parameters']['atr_factor'] for r in results),
                              max(r['parameters']['atr_factor'] for r in results)],
                'atr_period': [min(r['parameters']['atr_period'] for r in results),
                              max(r['parameters']['atr_period'] for r in results)]
            }
        }
        
        return summary
