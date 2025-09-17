"""
Pepperstone Position Sizing Module
Accurate simulation of Pepperstone broker position sizing rules
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional

class PepperstonePositionSizer:
    """
    Accurate position sizing simulation for Pepperstone broker
    Based on official Pepperstone documentation and rules
    """
    
    def __init__(self):
        # Pepperstone-specific multipliers and rules
        self.asset_rules = {
            # Crypto pairs (like BTC-USD, ETH-USD)
            'crypto': {
                'point_value': 1.0,  # 1 point = $1 per unit
                'leverage': 100,     # 100:1 leverage
                'min_position': 0.01,
                'position_step': 0.01,
                'formula': 'crypto'
            },
            
            # Forex majors (like EUR-USD)
            'forex_major': {
                'point_value': 1.0,  # 1 pip = $1 per 10k units
                'leverage': 500,     # 500:1 leverage  
                'min_position': 1000,
                'position_step': 1000,
                'formula': 'forex'
            },
            
            # Gold (GC or XAUUSD)
            'gold': {
                'point_value': 1.0,  # 1 point = $1 per unit
                'leverage': 200,     # 200:1 leverage
                'min_position': 0.01,
                'position_step': 0.01,
                'formula': 'gold'
            }
        }
        
    def identify_asset_type(self, symbol: str) -> str:
        """Identify asset type from symbol name"""
        symbol = symbol.upper()
        
        if any(crypto in symbol for crypto in ['BTC', 'ETH', 'LTC', 'XRP', 'ADA']):
            return 'crypto'
        elif symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF']:
            return 'forex_major'
        elif any(gold in symbol for gold in ['GC', 'XAUUSD', 'GOLD']):
            return 'gold'
        else:
            # Default to forex major for unknown pairs
            return 'forex_major'
    
    def calculate_position_size(self, 
                              symbol: str,
                              entry_price: float, 
                              stop_loss_price: float, 
                              risk_amount: float,
                              account_balance: float = 10000) -> Dict:
        """
        Calculate position size using Pepperstone's exact formulas
        
        Args:
            symbol: Trading instrument (e.g., 'BTCUSD', 'EURUSD')
            entry_price: Entry price for the trade
            stop_loss_price: Stop loss price
            risk_amount: Dollar amount willing to risk
            account_balance: Account balance for leverage calculations
            
        Returns:
            Dict with position details
        """
        
        asset_type = self.identify_asset_type(symbol)
        rules = self.asset_rules[asset_type]
        
        # Calculate stop loss in points/pips
        stop_loss_points = abs(entry_price - stop_loss_price)
        
        # Apply Pepperstone's specific formulas
        if rules['formula'] == 'crypto':
            # Crypto: Position Size = Risk$ / Stop_Loss_Points
            # 1 point = $1 per unit
            raw_position_size = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
            
        elif rules['formula'] == 'forex':
            # Forex: Position Size = (Risk$ × 10,000) / (Stop_Loss_Pips × Point_Value)
            # 1 pip = $1 per 10k units for majors
            stop_loss_pips = stop_loss_points * 10000  # Convert to pips
            raw_position_size = (risk_amount * 10000) / stop_loss_pips if stop_loss_pips > 0 else 0
            
        elif rules['formula'] == 'gold':
            # Gold: Position Size = Risk$ / Stop_Loss_Points
            # 1 point = $1 per unit
            raw_position_size = risk_amount / stop_loss_points if stop_loss_points > 0 else 0
            
        else:
            raw_position_size = 0
        
        # Apply broker constraints
        position_size = self._apply_broker_constraints(raw_position_size, rules)
        
        # Calculate trade value and required margin
        trade_value = position_size * entry_price
        required_margin = trade_value / rules['leverage']
        
        # Check if account can handle the position
        max_position_by_margin = (account_balance * 0.8) / (entry_price / rules['leverage'])
        position_size = min(position_size, max_position_by_margin)
        
        # Recalculate final values
        final_trade_value = position_size * entry_price
        final_required_margin = final_trade_value / rules['leverage']
        actual_risk = position_size * stop_loss_points
        
        return {
            'position_size': position_size,
            'trade_value': final_trade_value,
            'required_margin': final_required_margin,
            'actual_risk': actual_risk,
            'stop_loss_points': stop_loss_points,
            'asset_type': asset_type,
            'leverage': rules['leverage'],
            'valid': position_size > 0 and final_required_margin <= account_balance * 0.8
        }
    
    def _apply_broker_constraints(self, raw_size: float, rules: Dict) -> float:
        """Apply Pepperstone's minimum position size and step constraints"""
        if raw_size <= 0:
            return 0
        
        # Round to minimum step size
        min_size = rules['min_position']
        step_size = rules['position_step']
        
        if raw_size < min_size:
            return 0  # Position too small
        
        # Round down to nearest step
        steps = int((raw_size - min_size) / step_size)
        adjusted_size = min_size + (steps * step_size)
        
        return max(adjusted_size, min_size)
    
    def validate_trade_parameters(self, 
                                symbol: str,
                                entry_price: float,
                                stop_loss_price: float, 
                                risk_percent: float,
                                account_balance: float) -> Tuple[bool, str]:
        """
        Validate if trade parameters will generate valid positions
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        
        if entry_price <= 0:
            return False, "Invalid entry price"
        
        if stop_loss_price <= 0:
            return False, "Invalid stop loss price"
        
        if risk_percent <= 0 or risk_percent > 10:
            return False, "Risk percent should be between 0-10%"
        
        if account_balance <= 0:
            return False, "Invalid account balance"
        
        # Calculate position to check validity
        risk_amount = account_balance * (risk_percent / 100)
        result = self.calculate_position_size(symbol, entry_price, stop_loss_price, risk_amount, account_balance)
        
        if not result['valid']:
            return False, f"Insufficient margin. Required: ${result['required_margin']:.2f}, Available: ${account_balance * 0.8:.2f}"
        
        if result['position_size'] == 0:
            return False, "Position size too small after broker constraints"
        
        return True, "Valid trade parameters"


def test_position_sizer():
    """Test the position sizer with sample data"""
    sizer = PepperstonePositionSizer()
    
    # Test cases based on your diagnostic data
    test_cases = [
        {
            'symbol': 'BTCUSD',
            'entry_price': 115000.0,
            'stop_loss_price': 114965.0,  # 35 points stop loss (ATR)
            'risk_amount': 200.0,  # 2% of $10k account
            'account_balance': 10000.0
        },
        {
            'symbol': 'EURUSD', 
            'entry_price': 1.0850,
            'stop_loss_price': 1.0820,  # 30 pip stop loss
            'risk_amount': 200.0,
            'account_balance': 10000.0
        },
        {
            'symbol': 'GCUSD',  # Gold
            'entry_price': 2650.0,
            'stop_loss_price': 2620.0,  # 30 point stop loss
            'risk_amount': 200.0,
            'account_balance': 10000.0
        }
    ]
    
    print("🔍 PEPPERSTONE POSITION SIZING TEST")
    print("=" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {case['symbol']}")
        print("-" * 40)
        
        result = sizer.calculate_position_size(
            case['symbol'],
            case['entry_price'],
            case['stop_loss_price'],
            case['risk_amount'],
            case['account_balance']
        )
        
        print(f"Entry Price: ${case['entry_price']:,.2f}")
        print(f"Stop Loss: ${case['stop_loss_price']:,.2f}")
        print(f"Risk Amount: ${case['risk_amount']:,.2f}")
        print(f"Position Size: {result['position_size']:.6f} units")
        print(f"Trade Value: ${result['trade_value']:,.2f}")
        print(f"Required Margin: ${result['required_margin']:,.2f}")
        print(f"Actual Risk: ${result['actual_risk']:,.2f}")
        print(f"Valid Trade: {'✅ YES' if result['valid'] else '❌ NO'}")
        print(f"Asset Type: {result['asset_type']}")
        print(f"Leverage: {result['leverage']}:1")
        
        # Validation
        is_valid, message = sizer.validate_trade_parameters(
            case['symbol'],
            case['entry_price'],
            case['stop_loss_price'],
            case['risk_amount'] / case['account_balance'] * 100,
            case['account_balance']
        )
        print(f"Validation: {'✅' if is_valid else '❌'} {message}")


if __name__ == "__main__":
    test_position_sizer()
