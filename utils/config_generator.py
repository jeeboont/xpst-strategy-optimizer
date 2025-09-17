import json
from typing import Dict, List, Optional
from datetime import datetime

class ConfigurationGenerator:
    """
    Generates cBot and Indicator configuration files from optimization results
    """
    
    def __init__(self):
        self.cbot_template = self._load_cbot_template()
        self.indicator_template = self._load_indicator_template()
        
        # Parameter mapping between optimization and cTrader
        self.parameter_mapping = {
            # Core parameters (Step 1)
            'pivot_period': {'cbot': 'PivotPeriod', 'indicator': 'PivotPeriod'},
            'atr_factor': {'cbot': 'AtrFactor', 'indicator': 'AtrFactor'},
            'atr_period': {'cbot': 'AtrPeriod', 'indicator': 'AtrPeriod'},
            'use_prev_atr': {'cbot': 'UsePrevAtr', 'indicator': 'UsePrevAtr'},
            
            # Filter parameters (Step 2)
            'use_xtrend': {'cbot': 'UseXTrend', 'indicator': 'UseXTrend'},
            'use_xtrend_mtf': {'cbot': 'UseXTrendMTF', 'indicator': 'UseXTrendMTF'},
            'xtrend_mtf_timeframe': {'cbot': 'XTrendMTFTimeframe', 'indicator': 'XTrendMTFTimeframe'},
            'use_adx': {'cbot': 'UseAdx', 'indicator': 'UseAdx'},
            'adx_threshold': {'cbot': 'AdxThreshold', 'indicator': 'AdxThreshold'},
            'use_ema': {'cbot': 'UseEma', 'indicator': 'UseEma'},
            'ema_period': {'cbot': 'EmaPeriod', 'indicator': 'EmaPeriod'},
            
            # Circuit Breaker & Re-entry (Step 3)
            'enable_circuit_breaker': {'cbot': 'EnableCircuitBreaker', 'indicator': 'EnableCircuitBreaker'},
            'circuit_breaker_buffer': {'cbot': 'CircuitBreakerBuffer', 'indicator': 'CircuitBreakerBuffer'},
            'allow_reentry': {'cbot': 'AllowReentry', 'indicator': None},  # Only for cBot
            'reentry_cooldown_bars': {'cbot': 'ReentryCooldownBars', 'indicator': None},
            'reentry_window_bars': {'cbot': 'ReentryWindowBars', 'indicator': None}
        }
    
    def generate_cbot_config(self, optimized_params: Dict, asset: str, timeframe: str) -> Dict:
        """
        Generate cBot configuration with optimized parameters
        """
        config = self.cbot_template.copy()
        
        # Update chart info
        config["Chart"]["Symbol"] = asset
        config["Chart"]["Period"] = timeframe
        
        # Map optimization results to cBot parameters
        params = config["Parameters"]
        
        for opt_param, value in optimized_params.items():
            if opt_param in self.parameter_mapping:
                cbot_param = self.parameter_mapping[opt_param]['cbot']
                if cbot_param:  # Some parameters are cBot-only
                    params[cbot_param] = value
        
        return config
    
    def generate_indicator_config(self, optimized_params: Dict, asset: str, timeframe: str) -> Dict:
        """
        Generate Indicator configuration with optimized parameters
        """
        config = self.indicator_template.copy()
        
        # Map optimization results to indicator parameters (note: strings not numbers)
        params = config["Parameters"]
        
        for opt_param, value in optimized_params.items():
            if opt_param in self.parameter_mapping:
                indicator_param = self.parameter_mapping[opt_param]['indicator']
                if indicator_param:  # Some parameters don't apply to indicator
                    # Convert value to string for indicator
                    if isinstance(value, bool):
                        params[indicator_param] = "True" if value else "False"
                    else:
                        params[indicator_param] = str(value)
        
        return config
    
    def generate_optimized_configs(self, optimization_results: List[Dict], asset: str, timeframe: str) -> Dict:
        """
        Generate configuration files for all optimization results
        """
        configs = {}
        
        for i, result in enumerate(optimization_results[:3]):  # Top 3 results
            rank = i + 1
            params = result.get('parameters', {})
            
            # Generate cBot configuration
            cbot_config = self.generate_cbot_config(params, asset, timeframe)
            cbot_filename = f"{asset}_{timeframe}_rank{rank}.cbotset"
            configs[cbot_filename] = cbot_config
            
            # Generate Indicator configuration
            indicator_config = self.generate_indicator_config(params, asset, timeframe)
            indicator_filename = f"{asset}_{timeframe}_rank{rank}.indiset"
            configs[indicator_filename] = indicator_config
        
        return configs
    
    def create_readme_content(self, asset: str, timeframe: str, results: List[Dict]) -> str:
        """
        Create README content for configuration package
        """
        readme_content = f"""
XPST Strategy Optimization Results
==================================

Asset: {asset}
Timeframe: {timeframe}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Optimizer Version: XPST v3.0.0

OPTIMIZATION SUMMARY
===================

Total Results: {len(results)}
Configuration Files: {len(results) * 2} (cBot + Indicator pairs)

TOP RESULTS:
"""
        
        for i, result in enumerate(results[:3]):
            rank = i + 1
            metrics = result.get('metrics', {})
            params = result.get('parameters', {})
            
            readme_content += f"""
Rank #{rank}:
-----------
Performance Metrics:
  • Total Return: {metrics.get('total_return', 0):.2f}%
  • Win Rate: {metrics.get('win_rate', 0):.1f}%
  • Profit Factor: {metrics.get('profit_factor', 0):.2f}
  • Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
  • Total Trades: {metrics.get('total_trades', 0)}
  • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}

Optimized Parameters:
  • Pivot Period: {params.get('pivot_period', 'N/A')}
  • ATR Factor: {params.get('atr_factor', 'N/A')}
  • ATR Period: {params.get('atr_period', 'N/A')}
  • Use ADX: {params.get('use_adx', False)}
  • ADX Threshold: {params.get('adx_threshold', 'N/A')}
  • Use EMA: {params.get('use_ema', False)}
  • EMA Period: {params.get('ema_period', 'N/A')}
  • Circuit Breaker: {params.get('enable_circuit_breaker', False)}
  • CB Buffer: {params.get('circuit_breaker_buffer', 'N/A')}%

Files:
  • {asset}_{timeframe}_rank{rank}.cbotset
  • {asset}_{timeframe}_rank{rank}.indiset

"""
        
        readme_content += """
INSTALLATION INSTRUCTIONS
=========================

For cBot (.cbotset files):
1. Open cTrader platform
2. Go to cBot section
3. Click "Import" and select the .cbotset file
4. The cBot will be added to your available cBots
5. Drag to chart to start trading

For Indicator (.indiset files):
1. Open cTrader platform  
2. Go to Indicators section
3. Find "XPST" indicator
4. Right-click and select "Import Settings"
5. Select the .indiset file
6. Apply indicator to chart

IMPORTANT DISCLAIMERS
====================

⚠️  RISK WARNING:
• These parameters are optimized on HISTORICAL data
• Past performance does NOT guarantee future results
• ALWAYS test on demo account before live trading
• Use proper risk management and position sizing
• Markets can change - parameters may need re-optimization

⚠️  TESTING REQUIREMENTS:
• Test thoroughly on demo account for at least 2 weeks
• Monitor performance closely in different market conditions
• Consider re-optimizing periodically (monthly/quarterly)
• Adjust risk settings according to your risk tolerance

⚠️  TECHNICAL NOTES:
• Optimization period: Limited historical data
• Market regime changes may affect performance
• Consider fundamental analysis alongside technical signals
• Monitor correlations with other trading strategies

For support or questions about XPST strategy optimization,
please refer to the documentation or community forums.

Generated by XPST Strategy Optimizer v3.0.0
Copyright © 2025 - For educational and research purposes
"""
        
        return readme_content
    
    def _load_cbot_template(self) -> Dict:
        """
        Load cBot template with default values
        """
        return {
            "Chart": {
                "Symbol": "BTCUSD",
                "Period": "m2"
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
                "PivotPeriod": 5,
                "AtrFactor": 1.2,
                "AtrPeriod": 12,
                "UsePrevAtr": False,
                "UseXTrend": False,
                "UseXTrendMTF": False,
                "XTrendMTFTimeframe": "m5",
                "UseAdx": True,
                "AdxThreshold": 15,
                "UseEma": False,
                "EmaPeriod": 50,
                "EnableCircuitBreaker": True,
                "CircuitBreakerBuffer": 0.08,
                "AllowReentry": True,
                "ReentryCooldownBars": 2,
                "ReentryWindowBars": 15,
                "EnableDetailedLogging": True,
                "LogSignalDetails": True,
                "LogFilterStates": False,
                "LogEveryTick": False,
                "SummaryInterval": 50,
                "ShowStatsTable": True,
                "TablePosition": 1,
                "TextColorName": "White",
                "BackgroundOpacity": 0.8,
                "ShowEntryExitMarkers": True,
                "ShowTradeLineConnections": True,
                "EntryMarkerColor": "Yellow",
                "ProfitExitColor": "Lime",
                "LossExitColor": "Red",
                "ReentryMarkerColor": "Cyan",
                "MarkerSize": 3
            }
        }
    
    def _load_indicator_template(self) -> Dict:
        """
        Load Indicator template with default values
        """
        return {
            "Lines": [
                {
                    "IsEnabled": True,
                    "LineName": "PivotSupertrend",
                    "Color": "#FF0000FF",
                    "LineType": "Solid",
                    "LineWidth": 3.0
                },
                {
                    "IsEnabled": True,
                    "LineName": "XTrendLine",
                    "Color": "#FF808080",
                    "LineType": "Solid",
                    "LineWidth": 3.0
                },
                {
                    "IsEnabled": True,
                    "LineName": "EmaLine",
                    "Color": "#FF800080",
                    "LineType": "Solid",
                    "LineWidth": 1.0
                },
                {
                    "IsEnabled": True,
                    "LineName": "SupportLevel",
                    "Color": "#FF008000",
                    "LineType": "Dots",
                    "LineWidth": 1.0
                },
                {
                    "IsEnabled": True,
                    "LineName": "ResistanceLevel",
                    "Color": "#FFFF0000",
                    "LineType": "Dots",
                    "LineWidth": 1.0
                }
            ],
            "Levels": [],
            "Parameters": {
                "PivotPeriod": "5",
                "AtrFactor": "1.2",
                "AtrPeriod": "12",
                "UsePrevAtr": "False",
                "ShowPivotPoints": "False",
                "UseXTrend": "False",
                "UseXTrendMTF": "False",
                "XTrendMTFTimeframe": "m5",
                "UseAdx": "True",
                "AdxThreshold": "15",
                "UseEma": "False",
                "EmaPeriod": "50",
                "EnableCircuitBreaker": "True",
                "CircuitBreakerBuffer": "0.08",
                "ShowCBLevels": "True",
                "CBLineStyle": "Line",
                "CBLineOpacity": "100",
                "ShowLabels": "True",
                "BuyLabelOffset": "0.01",
                "SellLabelOffset": "0.4",
                "ShowExitLabels": "True",
                "ShowEntryExitMarkers": "True",
                "ShowTradeLineConnections": "True",
                "ShowSupportResistance": "False",
                "PlotOnClosedBar": "True",
                "ShowStatistics": "False",
                "MaxTradesToDisplay": "30",
                "TablePosition": "0",
                "TableOpacity": "80"
            }
        }
    
    def validate_parameters(self, parameters: Dict) -> Dict:
        """
        Validate and sanitize optimization parameters before generating configs
        """
        validated = {}
        
        # Define valid ranges for parameters
        validation_rules = {
            'pivot_period': {'min': 2, 'max': 15, 'type': int, 'default': 5},
            'atr_factor': {'min': 0.8, 'max': 2.0, 'type': float, 'default': 1.2},
            'atr_period': {'min': 10, 'max': 40, 'type': int, 'default': 12},
            'adx_threshold': {'min': 5, 'max': 25, 'type': int, 'default': 15},
            'ema_period': {'min': 20, 'max': 250, 'type': int, 'default': 50},
            'circuit_breaker_buffer': {'min': 0.01, 'max': 0.20, 'type': float, 'default': 0.08},
            'reentry_cooldown_bars': {'min': 0, 'max': 5, 'type': int, 'default': 2},
            'reentry_window_bars': {'min': 0, 'max': 20, 'type': int, 'default': 15}
        }
        
        for param, value in parameters.items():
            if param in validation_rules:
                rule = validation_rules[param]
                
                # Type validation
                try:
                    if rule['type'] == int:
                        value = int(value)
                    elif rule['type'] == float:
                        value = float(value)
                except (ValueError, TypeError):
                    value = rule['default']
                
                # Range validation
                if 'min' in rule and value < rule['min']:
                    value = rule['min']
                if 'max' in rule and value > rule['max']:
                    value = rule['max']
                
                validated[param] = value
            else:
                # Keep boolean and string parameters as-is
                validated[param] = value
        
        return validated
    
    def get_parameter_summary(self, parameters: Dict) -> str:
        """
        Get a human-readable summary of the parameters
        """
        summary_lines = []
        
        # Core parameters
        if 'pivot_period' in parameters:
            summary_lines.append(f"Pivot Period: {parameters['pivot_period']}")
        if 'atr_factor' in parameters:
            summary_lines.append(f"ATR Factor: {parameters['atr_factor']}")
        if 'atr_period' in parameters:
            summary_lines.append(f"ATR Period: {parameters['atr_period']}")
        
        # Filters
        filters_active = []
        if parameters.get('use_adx', False):
            filters_active.append(f"ADX ({parameters.get('adx_threshold', 15)})")
        if parameters.get('use_ema', False):
            filters_active.append(f"EMA ({parameters.get('ema_period', 50)})")
        if parameters.get('use_xtrend', False):
            filters_active.append("XTrend")
        
        if filters_active:
            summary_lines.append(f"Filters: {', '.join(filters_active)}")
        else:
            summary_lines.append("Filters: None")
        
        # Risk management
        if parameters.get('enable_circuit_breaker', False):
            summary_lines.append(f"Circuit Breaker: {parameters.get('circuit_breaker_buffer', 0.08):.2%}")
        
        if parameters.get('allow_reentry', False):
            summary_lines.append(f"Re-entry: {parameters.get('reentry_cooldown_bars', 2)}|{parameters.get('reentry_window_bars', 15)} bars")
        
        return "\n".join(summary_lines)
