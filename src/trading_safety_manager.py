"""
Production Trading Safety Manager - Comprehensive safety for live trading
Implements all safety checks and emergency stops for production trading
"""
import os
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ProductionTradingSafetyManager:
    """Comprehensive safety manager for production trading"""
    
    def __init__(self):
        self.last_heartbeat = datetime.now()
        self.trade_count_today = 0
        self.trade_count_hour = 0
        self.last_trade_time = None
        self.emergency_stop = False
        self.safety_violations = []
        self.model_validator = None
        
        # Safety limits
        self.max_trades_per_day = 50
        self.max_trades_per_hour = 10
        self.min_time_between_trades = 30  # seconds
        self.max_consecutive_losses = 5
        self.max_drawdown_percent = 15.0
        
        # Track consecutive losses
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        
        logger.info("🛡️ PRODUCTION TRADING SAFETY MANAGER INITIALIZED")
        logger.info(f"   📊 Max trades/day: {self.max_trades_per_day}")
        logger.info(f"   ⏱️ Max trades/hour: {self.max_trades_per_hour}")
        logger.info(f"   🚫 Max consecutive losses: {self.max_consecutive_losses}")
        logger.info(f"   📉 Max drawdown: {self.max_drawdown_percent}%")
    
    def pre_trade_safety_check(self, symbol: str, trade_amount: float) -> Dict[str, Any]:
        """Comprehensive pre-trade safety check"""
        check_result = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'trade_amount': trade_amount,
            'is_safe_to_trade': False,
            'safety_violations': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_failed': 0
        }
        
        logger.info(f"🔍 PRE-TRADE SAFETY CHECK: {symbol} ${trade_amount}")
        
        # Check 1: Emergency stop
        if self.emergency_stop:
            check_result['safety_violations'].append("EMERGENCY STOP ACTIVE")
            check_result['checks_failed'] += 1
            logger.error("❌ EMERGENCY STOP ACTIVE - NO TRADING ALLOWED")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # Check 2: Model validation
        try:
            if self.model_validator:
                model_status = self.model_validator.get_model_status()
                if not model_status.get('has_validated_model', False):
                    check_result['safety_violations'].append("NO VALIDATED MODEL AVAILABLE")
                    check_result['checks_failed'] += 1
                    logger.error("❌ No validated production model available")
                    return check_result
                else:
                    check_result['checks_passed'] += 1
                    logger.info("✅ Validated model available")
            else:
                check_result['warnings'].append("Model validator not initialized")
                logger.warning("⚠️ Model validator not initialized")
        except Exception as e:
            check_result['safety_violations'].append(f"Model validation failed: {e}")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Model validation error: {e}")
            return check_result
        
        # Check 3: Trade frequency limits
        now = datetime.now()
        
        # Daily limit
        if self.trade_count_today >= self.max_trades_per_day:
            check_result['safety_violations'].append(f"Daily trade limit exceeded: {self.trade_count_today}/{self.max_trades_per_day}")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Daily trade limit exceeded: {self.trade_count_today}/{self.max_trades_per_day}")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # Hourly limit  
        if self.trade_count_hour >= self.max_trades_per_hour:
            check_result['safety_violations'].append(f"Hourly trade limit exceeded: {self.trade_count_hour}/{self.max_trades_per_hour}")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Hourly trade limit exceeded: {self.trade_count_hour}/{self.max_trades_per_hour}")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # Time between trades
        if self.last_trade_time:
            time_since_last = (now - self.last_trade_time).total_seconds()
            if time_since_last < self.min_time_between_trades:
                check_result['safety_violations'].append(f"Trade too soon: {time_since_last:.1f}s < {self.min_time_between_trades}s")
                check_result['checks_failed'] += 1
                logger.error(f"❌ Trade too soon: {time_since_last:.1f}s < {self.min_time_between_trades}s")
                return check_result
            else:
                check_result['checks_passed'] += 1
        else:
            check_result['checks_passed'] += 1
        
        # Check 4: Consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            check_result['safety_violations'].append(f"Too many consecutive losses: {self.consecutive_losses}")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Too many consecutive losses: {self.consecutive_losses}")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # Check 5: Drawdown protection
        if self.current_drawdown >= self.max_drawdown_percent:
            check_result['safety_violations'].append(f"Maximum drawdown exceeded: {self.current_drawdown:.1f}%")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Maximum drawdown exceeded: {self.current_drawdown:.1f}%")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # Check 6: Trade amount validation
        if trade_amount <= 0:
            check_result['safety_violations'].append(f"Invalid trade amount: ${trade_amount}")
            check_result['checks_failed'] += 1
            logger.error(f"❌ Invalid trade amount: ${trade_amount}")
            return check_result
        else:
            check_result['checks_passed'] += 1
        
        # All checks passed
        if check_result['checks_failed'] == 0:
            check_result['is_safe_to_trade'] = True
            logger.info(f"✅ PRE-TRADE SAFETY CHECK PASSED - {check_result['checks_passed']} checks OK")
        else:
            logger.error(f"❌ PRE-TRADE SAFETY CHECK FAILED - {check_result['checks_failed']} violations")
        
        return check_result
    
    def record_trade_entry(self, symbol: str, trade_amount: float, entry_price: float):
        """Record a trade entry for safety tracking"""
        try:
            now = datetime.now()
            
            # Update counters
            self.trade_count_today += 1
            self.trade_count_hour += 1
            self.last_trade_time = now
            
            # Log entry
            logger.info(f"📈 TRADE ENTRY RECORDED:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Amount: ${trade_amount}")
            logger.info(f"   Entry Price: ${entry_price}")
            logger.info(f"   Daily Count: {self.trade_count_today}/{self.max_trades_per_day}")
            logger.info(f"   Hourly Count: {self.trade_count_hour}/{self.max_trades_per_hour}")
            
            # Send heartbeat
            self.heartbeat()
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade entry: {e}")
    
    def record_trade_exit(self, symbol: str, exit_price: float, profit_loss: float, exit_reason: str):
        """Record a trade exit and update safety metrics"""
        try:
            # Update consecutive losses
            if profit_loss < 0:
                self.consecutive_losses += 1
                logger.warning(f"📉 Loss recorded: ${profit_loss:.2f} (consecutive: {self.consecutive_losses})")
            else:
                self.consecutive_losses = 0  # Reset on profit
                logger.info(f"📈 Profit recorded: ${profit_loss:.2f} (streak reset)")
            
            # Update drawdown (simplified)
            if profit_loss < 0:
                self.current_drawdown += abs(profit_loss / 1000 * 100)  # Rough % calculation
            else:
                self.current_drawdown = max(0, self.current_drawdown - (profit_loss / 1000 * 100))
            
            # Log exit
            logger.info(f"📊 TRADE EXIT RECORDED:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Exit Price: ${exit_price}")
            logger.info(f"   P&L: ${profit_loss:.2f}")
            logger.info(f"   Exit Reason: {exit_reason}")
            logger.info(f"   Consecutive Losses: {self.consecutive_losses}")
            logger.info(f"   Current Drawdown: {self.current_drawdown:.1f}%")
            
            # Check for emergency stop conditions
            self.check_emergency_conditions()
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade exit: {e}")
    
    def check_emergency_conditions(self):
        """Check if emergency stop should be triggered"""
        try:
            emergency_triggers = []
            
            # Too many consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                emergency_triggers.append(f"Consecutive losses: {self.consecutive_losses}")
            
            # Excessive drawdown
            if self.current_drawdown >= self.max_drawdown_percent:
                emergency_triggers.append(f"Drawdown: {self.current_drawdown:.1f}%")
            
            # Model validation failure
            if self.model_validator:
                try:
                    status = self.model_validator.get_model_status()
                    if not status.get('has_validated_model', False):
                        emergency_triggers.append("Model validation failed")
                except Exception as e:
                    emergency_triggers.append(f"Model check error: {e}")
            
            # Trigger emergency stop if conditions met
            if emergency_triggers:
                self.trigger_emergency_stop(emergency_triggers)
                
        except Exception as e:
            logger.error(f"❌ Emergency condition check failed: {e}")
    
    def trigger_emergency_stop(self, reasons: List[str]):
        """Trigger emergency stop with reasons"""
        self.emergency_stop = True
        self.safety_violations.extend(reasons)
        
        logger.critical("🚨 EMERGENCY STOP TRIGGERED!")
        logger.critical("🚨 REASONS:")
        for reason in reasons:
            logger.critical(f"   • {reason}")
        
        # TODO: Send Discord notification
        # TODO: Close all open positions
        # TODO: Notify admin
    
    def heartbeat(self):
        """Send heartbeat signal"""
        self.last_heartbeat = datetime.now()
        
        # Log heartbeat every 5 minutes
        if not hasattr(self, 'last_heartbeat_log') or \
           (datetime.now() - getattr(self, 'last_heartbeat_log', datetime.min)).total_seconds() > 300:
            logger.info(f"💓 TRADING HEARTBEAT - System active at {self.last_heartbeat}")
            self.last_heartbeat_log = datetime.now()
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        now = datetime.now()
        health_status = {
            'timestamp': now.isoformat(),
            'is_healthy': True,
            'warnings': [],
            'errors': [],
            'metrics': {
                'trades_today': self.trade_count_today,
                'trades_hour': self.trade_count_hour,
                'consecutive_losses': self.consecutive_losses,
                'current_drawdown': self.current_drawdown,
                'emergency_stop': self.emergency_stop
            }
        }
        
        # Check heartbeat age
        if self.last_heartbeat:
            heartbeat_age = (now - self.last_heartbeat).total_seconds()
            if heartbeat_age > 3600:  # 1 hour
                health_status['errors'].append(f"Stale heartbeat: {heartbeat_age:.0f}s old")
                health_status['is_healthy'] = False
        
        # Check safety violations
        if self.safety_violations:
            health_status['errors'].extend(self.safety_violations)
            health_status['is_healthy'] = False
        
        # Check emergency stop
        if self.emergency_stop:
            health_status['errors'].append("Emergency stop active")
            health_status['is_healthy'] = False
        
        return health_status
    
    def reset_daily_counters(self):
        """Reset daily counters (should be called at midnight)"""
        self.trade_count_today = 0
        logger.info("🔄 Daily trade counter reset")
    
    def reset_hourly_counters(self):
        """Reset hourly counters (should be called every hour)"""
        self.trade_count_hour = 0
        logger.debug("🔄 Hourly trade counter reset")
    
    def manual_emergency_stop(self, reason: str = "Manual stop"):
        """Manually trigger emergency stop"""
        self.trigger_emergency_stop([f"Manual: {reason}"])
    
    def clear_emergency_stop(self, reason: str = "Manual clear"):
        """Clear emergency stop (admin only)"""
        self.emergency_stop = False
        self.safety_violations = []
        self.consecutive_losses = 0
        self.current_drawdown = 0.0
        
        logger.warning(f"⚠️ EMERGENCY STOP CLEARED: {reason}")
        logger.warning("⚠️ ALL SAFETY METRICS RESET")
    
    def set_model_validator(self, validator):
        """Set the model validator instance"""
        self.model_validator = validator
        logger.info("🔗 Model validator linked to safety manager")

# Global safety manager instance
_safety_manager = None

def get_safety_manager():
    """Get singleton safety manager instance"""
    global _safety_manager
    if _safety_manager is None:
        _safety_manager = ProductionTradingSafetyManager()
    return _safety_manager

def main():
    """Test the safety manager"""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    safety = ProductionTradingSafetyManager()
    
    # Test safety check
    check_result = safety.pre_trade_safety_check("BTCUSDT", 100.0)
    print(f"Safety check result: {check_result['is_safe_to_trade']}")
    
    # Test trade recording
    if check_result['is_safe_to_trade']:
        safety.record_trade_entry("BTCUSDT", 100.0, 50000.0)
        safety.record_trade_exit("BTCUSDT", 50500.0, 10.0, "Take Profit")
    
    # Test health check
    health = safety.check_system_health()
    print(f"System health: {health['is_healthy']}")

if __name__ == "__main__":
    main()
