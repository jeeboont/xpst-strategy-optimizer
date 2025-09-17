import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import itertools
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .backtesting_engine import XPSTBacktester
from .core_optimizer import CoreParameterOptimizer
from .filter_optimizer import FilterOptimizer
from .cb_reentry_optimizer import CBReentryOptimizer

class ProgressiveOptimizer:
    """
    Progressive optimization engine that implements the 3-step optimization strategy
    """
    
    def __init__(self, max_combinations: int = 500):
        self.max_combinations = max_combinations
        self.backtester = XPSTBacktester()
        self.core_optimizer = CoreParameterOptimizer()
        self.filter_optimizer = FilterOptimizer()
        self.cb_reentry_optimizer = CBReentryOptimizer()
        
        # Optimization settings
        self.early_stopping_patience = 100
        self.top_n_to_keep = 3
        self.max_workers = 4
    
    def step1_core_optimization(self, data: pd.DataFrame, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Step 1: Optimize core parameters (Pivot Period, ATR Factor, ATR Period)
        """
        print("Starting Step 1: Core Parameter Optimization")
        
        # Use coarse-to-fine approach as discussed
        coarse_results = self.core_optimizer.run_coarse_optimization(
            data, self.backtester, progress_callback
        )
        
        # Get top performers from coarse search
        top_coarse = self._get_top_performers(coarse_results, n=5)
        
        # Fine-tune around top performers
        fine_results = []
        for i, result in enumerate(top_coarse):
            if progress_callback:
                progress_callback(0.7 + (i / len(top_coarse)) * 0.3)
            
            fine_result = self.core_optimizer.run_fine_optimization(
                data, self.backtester, result['parameters']
            )
            fine_results.extend(fine_result)
        
        # Return top 3 from fine-tuning
        final_results = self._get_top_performers(fine_results, n=self.top_n_to_keep)
        
        print(f"Step 1 completed. Found {len(final_results)} optimized core parameter sets.")
        return final_results
    
    def step2_filter_optimization(self, data: pd.DataFrame, step1_results: List[Dict], 
                                 progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Step 2: Optimize filters (XTrend, ADX, EMA) for top core parameter sets
        """
        print("Starting Step 2: Filter Optimization")
        
        all_results = []
        total_cores = len(step1_results)
        
        for i, core_result in enumerate(step1_results):
            if progress_callback:
                progress_callback(i / total_cores)
            
            print(f"Optimizing filters for core set {i+1}/{total_cores}")
            
            # Test each filter independently first
            filter_results = self.filter_optimizer.optimize_filters_for_core(
                data, self.backtester, core_result['parameters']
            )
            
            all_results.extend(filter_results)
        
        # Return top performers across all core sets
        final_results = self._get_top_performers(all_results, n=self.top_n_to_keep)
        
        print(f"Step 2 completed. Found {len(final_results)} optimized filter combinations.")
        return final_results
    
    def step3_cb_reentry_optimization(self, data: pd.DataFrame, step2_results: List[Dict],
                                     progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Step 3: Optimize Circuit Breaker and Re-Entry settings
        """
        print("Starting Step 3: Circuit Breaker & Re-Entry Optimization")
        
        all_results = []
        total_settings = len(step2_results)
        
        for i, base_result in enumerate(step2_results):
            if progress_callback:
                progress_callback(i / total_settings)
            
            print(f"Optimizing CB & Re-entry for setting {i+1}/{total_settings}")
            
            # Optimize CB and re-entry for this base setting
            cb_results = self.cb_reentry_optimizer.optimize_cb_reentry_for_base(
                data, self.backtester, base_result['parameters']
            )
            
            all_results.extend(cb_results)
        
        # Return top performers
        final_results = self._get_top_performers(all_results, n=self.top_n_to_keep)
        
        print(f"Step 3 completed. Found {len(final_results)} final optimized settings.")
        return final_results
    
    def _get_top_performers(self, results: List[Dict], n: int = 3) -> List[Dict]:
        """Get top N performers based on fitness score"""
        if not results:
            return []
        
        # Sort by fitness score (assuming higher is better)
        sorted_results = sorted(results, key=lambda x: x.get('fitness', 0), reverse=True)
        return sorted_results[:n]
    
    def _calculate_fitness_score(self, metrics: Dict) -> float:
        """
        Calculate fitness score for optimization results
        """
        # Weighted fitness function
        weights = {
            'return_weight': 0.35,
            'winrate_weight': 0.25,
            'profit_factor_weight': 0.20,
            'drawdown_penalty': 0.20
        }
        
        # Normalize metrics
        total_return = metrics.get('total_return', 0)
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 1)
        max_drawdown = metrics.get('max_drawdown', 100)
        total_trades = metrics.get('total_trades', 0)
        
        # Normalize to [0, 1] scale
        normalized_return = min(max(total_return / 100, 0), 1)
        normalized_winrate = win_rate / 100
        normalized_pf = min(max((profit_factor - 1) / 2, 0), 1)
        drawdown_penalty = min(max(max_drawdown / 50, 0), 1)
        
        # Calculate composite score
        fitness = (
            weights['return_weight'] * normalized_return +
            weights['winrate_weight'] * normalized_winrate +
            weights['profit_factor_weight'] * normalized_pf -
            weights['drawdown_penalty'] * drawdown_penalty
        )
        
        # Apply penalties for insufficient trades
        if total_trades < 10:
            fitness *= 0.5
        elif total_trades < 5:
            fitness *= 0.25
        
        return max(fitness, 0)  # Ensure non-negative
    
    def run_parallel_optimization(self, data: pd.DataFrame, parameter_combinations: List[Dict],
                                 progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Run optimization in parallel for faster processing
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all combinations
            future_to_params = {
                executor.submit(self._test_single_combination, data, params): params
                for params in parameter_combinations
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_params)):
                if progress_callback:
                    progress_callback(i / len(parameter_combinations))
                
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"Error in parallel optimization: {e}")
                    continue
        
        return results
    
    def _test_single_combination(self, data: pd.DataFrame, parameters: Dict) -> Optional[Dict]:
        """
        Test a single parameter combination
        """
        try:
            # Run backtest
            trades, metrics = self.backtester.backtest_strategy(data, parameters)
            
            # Calculate fitness
            fitness = self._calculate_fitness_score(metrics)
            
            return {
                'parameters': parameters.copy(),
                'metrics': metrics,
                'fitness': fitness,
                'trades': trades
            }
            
        except Exception as e:
            print(f"Error testing combination {parameters}: {e}")
            return None
    
    def optimize_with_early_stopping(self, data: pd.DataFrame, parameter_combinations: List[Dict],
                                    progress_callback: Optional[Callable] = None) -> List[Dict]:
        """
        Optimize with early stopping to prevent over-optimization
        """
        results = []
        best_fitness = 0
        no_improvement_count = 0
        
        # Shuffle combinations to avoid bias
        shuffled_combinations = parameter_combinations.copy()
        random.shuffle(shuffled_combinations)
        
        for i, params in enumerate(shuffled_combinations):
            if i >= self.max_combinations:
                break
            
            if progress_callback:
                progress_callback(i / min(len(shuffled_combinations), self.max_combinations))
            
            result = self._test_single_combination(data, params)
            if result:
                results.append(result)
                
                if result['fitness'] > best_fitness:
                    best_fitness = result['fitness']
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                # Early stopping check
                if (no_improvement_count >= self.early_stopping_patience and 
                    i >= 200):  # Minimum 200 tests
                    print(f"Early stopping at {i+1} combinations (no improvement)")
                    break
        
        return self._get_top_performers(results, n=10)  # Return top 10 for further processing
