"""
Risk Manager
Advanced risk management module for trading operations
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskManager:
    """Advanced risk management for trading operations"""
    
    def __init__(self,
                 max_position_size: float = 100.0,  # Max position in USDT
                 max_daily_loss: float = 50.0,      # Max daily loss in USDT
                 max_drawdown: float = 20.0,        # Max drawdown percentage
                 risk_per_trade: float = 2.0,       # Risk per trade percentage
                 max_open_positions: int = 3,       # Max concurrent positions
                 correlation_threshold: float = 0.7  # Max correlation between positions
                 ):
        """
        Initialize risk manager
        
        Args:
            max_position_size: Maximum position size in USDT
            max_daily_loss: Maximum daily loss in USDT
            max_drawdown: Maximum portfolio drawdown (%)
            risk_per_trade: Risk per trade as percentage of balance
            max_open_positions: Maximum number of open positions
            correlation_threshold: Maximum correlation between positions
        """
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown
        self.risk_per_trade = risk_per_trade
        self.max_open_positions = max_open_positions
        self.correlation_threshold = correlation_threshold
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_reset_date = datetime.now().date()
        
        # Portfolio tracking
        self.portfolio_high = 0.0
        self.current_drawdown = 0.0
        self.open_positions = {}
        
        logger.info("🛡️ Risk Manager initialized")
        logger.info(f"   Max position size: ${self.max_position_size}")
        logger.info(f"   Max daily loss: ${self.max_daily_loss}")
        logger.info(f"   Max drawdown: {self.max_drawdown}%")
        logger.info(f"   Risk per trade: {self.risk_per_trade}%")
    
    def check_trade_approval(self, 
                           signal: Dict[str, Any],
                           current_balance: float,
                           symbol: str,
                           position_size: float) -> Dict[str, Any]:
        """
        Check if trade should be approved based on risk rules
        
        Args:
            signal: Trading signal
            current_balance: Current account balance
            symbol: Trading symbol
            position_size: Requested position size
            
        Returns:
            Dict with approval status and reasoning
        """
        try:
            # Reset daily tracking if new day
            self._reset_daily_tracking()
            
            # Check individual risk factors
            checks = {
                'position_size': self._check_position_size(position_size),
                'daily_loss': self._check_daily_loss(),
                'drawdown': self._check_drawdown(current_balance),
                'open_positions': self._check_max_positions(symbol),
                'correlation': self._check_correlation(symbol),
                'signal_quality': self._check_signal_quality(signal)
            }
            
            # Determine overall approval
            approved = all(check['approved'] for check in checks.values())
            
            # Calculate adjusted position size
            adjusted_size = self._calculate_adjusted_position_size(
                position_size, current_balance, checks
            )
            
            result = {
                'approved': approved,
                'adjusted_position_size': adjusted_size,
                'risk_score': self._calculate_risk_score(checks),
                'checks': checks,
                'reasoning': self._generate_reasoning(checks, approved)
            }
            
            if approved:
                logger.info(f"✅ Trade approved for {symbol}: ${adjusted_size:.2f}")
            else:
                logger.warning(f"❌ Trade rejected for {symbol}: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in trade approval: {e}")
            return {
                'approved': False,
                'adjusted_position_size': 0.0,
                'risk_score': 1.0,
                'checks': {},
                'reasoning': f"Risk check error: {e}"
            }
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position in risk tracking"""
        try:
            if position_data.get('closed', False):
                # Position closed
                if symbol in self.open_positions:
                    closed_position = self.open_positions.pop(symbol)
                    pnl = position_data.get('pnl', 0.0)
                    self.daily_pnl += pnl
                    self.daily_trades += 1
                    
                    logger.info(f"📈 Position closed: {symbol} (PnL: ${pnl:.2f})")
            else:
                # Position opened/updated
                self.open_positions[symbol] = {
                    'entry_time': position_data.get('entry_time', datetime.now()),
                    'entry_price': position_data.get('entry_price', 0.0),
                    'position_size': position_data.get('position_size', 0.0),
                    'stop_loss': position_data.get('stop_loss', 0.0),
                    'take_profit': position_data.get('take_profit', 0.0)
                }
                
                logger.info(f"📊 Position updated: {symbol}")
            
            # Update portfolio metrics
            self._update_portfolio_metrics()
            
        except Exception as e:
            logger.error(f"❌ Error updating position: {e}")
    
    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        try:
            self._reset_daily_tracking()
            
            status = {
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'daily_loss_limit_remaining': max(0, self.max_daily_loss + self.daily_pnl),
                'current_drawdown': self.current_drawdown,
                'drawdown_limit_remaining': max(0, self.max_drawdown - self.current_drawdown),
                'open_positions_count': len(self.open_positions),
                'max_positions_remaining': max(0, self.max_open_positions - len(self.open_positions)),
                'portfolio_high': self.portfolio_high,
                'risk_level': self._get_current_risk_level()
            }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting risk status: {e}")
            return {}
    
    def emergency_stop_check(self, current_balance: float) -> bool:
        """Check if emergency stop should be triggered"""
        try:
            self._reset_daily_tracking()
            
            # Daily loss limit exceeded
            if self.daily_pnl <= -self.max_daily_loss:
                logger.error(f"🚨 EMERGENCY STOP: Daily loss limit exceeded (${self.daily_pnl:.2f})")
                return True
            
            # Maximum drawdown exceeded
            self._update_portfolio_metrics(current_balance)
            if self.current_drawdown >= self.max_drawdown:
                logger.error(f"🚨 EMERGENCY STOP: Max drawdown exceeded ({self.current_drawdown:.2f}%)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in emergency stop check: {e}")
            return True  # Err on the side of caution
    
    def _reset_daily_tracking(self):
        """Reset daily tracking if new day"""
        current_date = datetime.now().date()
        if current_date != self.daily_reset_date:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_reset_date = current_date
            logger.info("🔄 Daily risk tracking reset")
    
    def _check_position_size(self, position_size: float) -> Dict[str, Any]:
        """Check if position size is within limits"""
        approved = position_size <= self.max_position_size
        return {
            'approved': approved,
            'value': position_size,
            'limit': self.max_position_size,
            'message': f"Position size: ${position_size:.2f} (limit: ${self.max_position_size})"
        }
    
    def _check_daily_loss(self) -> Dict[str, Any]:
        """Check daily loss limits"""
        remaining = self.max_daily_loss + self.daily_pnl
        approved = remaining > 0
        return {
            'approved': approved,
            'value': self.daily_pnl,
            'limit': -self.max_daily_loss,
            'remaining': remaining,
            'message': f"Daily PnL: ${self.daily_pnl:.2f} (limit: ${-self.max_daily_loss})"
        }
    
    def _check_drawdown(self, current_balance: float) -> Dict[str, Any]:
        """Check portfolio drawdown"""
        self._update_portfolio_metrics(current_balance)
        approved = self.current_drawdown < self.max_drawdown
        return {
            'approved': approved,
            'value': self.current_drawdown,
            'limit': self.max_drawdown,
            'message': f"Drawdown: {self.current_drawdown:.2f}% (limit: {self.max_drawdown}%)"
        }
    
    def _check_max_positions(self, symbol: str) -> Dict[str, Any]:
        """Check maximum open positions"""
        open_count = len(self.open_positions)
        # Allow if symbol already has position (update) or within limit
        approved = symbol in self.open_positions or open_count < self.max_open_positions
        return {
            'approved': approved,
            'value': open_count,
            'limit': self.max_open_positions,
            'message': f"Open positions: {open_count} (limit: {self.max_open_positions})"
        }
    
    def _check_correlation(self, symbol: str) -> Dict[str, Any]:
        """Check correlation with existing positions"""
        # Simplified correlation check - could be enhanced with actual price correlation
        approved = True  # For now, approve all
        correlation = 0.0
        
        # TODO: Implement actual correlation calculation with price data
        
        return {
            'approved': approved,
            'value': correlation,
            'limit': self.correlation_threshold,
            'message': f"Position correlation: {correlation:.2f} (limit: {self.correlation_threshold})"
        }
    
    def _check_signal_quality(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Check signal quality and confidence"""
        confidence = signal.get('confidence', 0.0)
        min_confidence = 0.6  # Minimum confidence threshold
        approved = confidence >= min_confidence
        
        return {
            'approved': approved,
            'value': confidence,
            'limit': min_confidence,
            'message': f"Signal confidence: {confidence:.2f} (min: {min_confidence})"
        }
    
    def _calculate_adjusted_position_size(self, 
                                        original_size: float,
                                        current_balance: float,
                                        checks: Dict[str, Any]) -> float:
        """Calculate risk-adjusted position size"""
        try:
            adjusted_size = original_size
            
            # Apply position size limit
            if not checks['position_size']['approved']:
                adjusted_size = min(adjusted_size, self.max_position_size)
            
            # Apply daily loss consideration
            if checks['daily_loss']['remaining'] < adjusted_size:
                adjusted_size = max(0, checks['daily_loss']['remaining'] * 0.5)
            
            # Apply drawdown consideration
            if self.current_drawdown > self.max_drawdown * 0.8:  # 80% of max drawdown
                adjusted_size *= 0.5  # Reduce size by half
            
            # Apply signal confidence scaling
            signal_confidence = checks['signal_quality']['value']
            confidence_scaling = max(0.5, signal_confidence)  # Scale down if low confidence
            adjusted_size *= confidence_scaling
            
            return max(0, adjusted_size)
            
        except Exception as e:
            logger.error(f"❌ Error calculating adjusted position size: {e}")
            return 0.0
    
    def _calculate_risk_score(self, checks: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-1, where 1 is highest risk)"""
        try:
            risk_factors = []
            
            # Position size risk
            pos_risk = min(1.0, checks['position_size']['value'] / self.max_position_size)
            risk_factors.append(pos_risk)
            
            # Daily loss risk
            daily_loss_ratio = abs(self.daily_pnl) / self.max_daily_loss
            risk_factors.append(min(1.0, daily_loss_ratio))
            
            # Drawdown risk
            drawdown_risk = self.current_drawdown / self.max_drawdown
            risk_factors.append(min(1.0, drawdown_risk))
            
            # Position count risk
            position_risk = len(self.open_positions) / self.max_open_positions
            risk_factors.append(position_risk)
            
            # Signal quality risk (inverse of confidence)
            signal_risk = 1.0 - checks['signal_quality']['value']
            risk_factors.append(signal_risk)
            
            # Weighted average risk score
            weights = [0.25, 0.25, 0.25, 0.15, 0.10]
            risk_score = sum(r * w for r, w in zip(risk_factors, weights))
            
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk score: {e}")
            return 0.5  # Medium risk as default
    
    def _generate_reasoning(self, checks: Dict[str, Any], approved: bool) -> str:
        """Generate human-readable reasoning for approval/rejection"""
        try:
            if approved:
                return "All risk checks passed"
            
            failed_checks = [
                check['message'] for check in checks.values() 
                if not check['approved']
            ]
            
            return f"Failed checks: {'; '.join(failed_checks)}"
            
        except Exception as e:
            logger.error(f"❌ Error generating reasoning: {e}")
            return "Risk assessment error"
    
    def _update_portfolio_metrics(self, current_balance: float = None):
        """Update portfolio-level risk metrics"""
        try:
            if current_balance is not None:
                # Update portfolio high watermark
                if current_balance > self.portfolio_high:
                    self.portfolio_high = current_balance
                
                # Calculate current drawdown
                if self.portfolio_high > 0:
                    self.current_drawdown = max(0, 
                        (self.portfolio_high - current_balance) / self.portfolio_high * 100
                    )
                else:
                    self.current_drawdown = 0.0
            
        except Exception as e:
            logger.error(f"❌ Error updating portfolio metrics: {e}")
    
    def _get_current_risk_level(self) -> str:
        """Get current risk level description"""
        try:
            # Calculate combined risk factors
            risk_factors = []
            
            # Daily loss factor
            if self.max_daily_loss > 0:
                daily_factor = abs(self.daily_pnl) / self.max_daily_loss
                risk_factors.append(daily_factor)
            
            # Drawdown factor
            drawdown_factor = self.current_drawdown / self.max_drawdown
            risk_factors.append(drawdown_factor)
            
            # Position count factor
            position_factor = len(self.open_positions) / self.max_open_positions
            risk_factors.append(position_factor)
            
            # Overall risk level
            avg_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 0
            
            if avg_risk < 0.3:
                return "LOW"
            elif avg_risk < 0.6:
                return "MEDIUM"
            elif avg_risk < 0.8:
                return "HIGH"
            else:
                return "CRITICAL"
                
        except Exception as e:
            logger.error(f"❌ Error determining risk level: {e}")
            return "UNKNOWN"
