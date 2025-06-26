#!/usr/bin/env python3
"""
Auto-Culling System for Underperforming Trading Models

This module automatically pauses or triggers retraining for models that
consistently underperform based on configurable criteria.
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.trading_stats import get_stats_manager
from src.model_training.incremental_trainer import IncrementalTrainer
from src.discord_notifications import send_trader_notification

logger = logging.getLogger(__name__)

class AutoCullingSystem:
    """Automatic model performance monitoring and culling system"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.culling_config_file = os.path.join(data_dir, "auto_culling_config.json")
        self.paused_models_file = os.path.join(data_dir, "paused_models.json")
        
        # Default configuration
        self.config = {
            "enabled": True,
            "check_interval_minutes": 60,  # Check every hour
            "min_trades_before_culling": 10,
            "win_rate_threshold": 0.50,    # 50% minimum win rate
            "max_consecutive_losses": 5,
            "max_total_loss": 100.0,       # $100 maximum loss
            "auto_retrain": True,          # Automatically trigger retraining
            "pause_duration_hours": 24,    # Pause for 24 hours before retry
            "max_retrain_attempts": 3      # Maximum retraining attempts
        }
        
        self.paused_models = {}
        self.retrain_attempts = {}
        
        # Load configuration and state
        self._load_config()
        self._load_paused_models()
        
        os.makedirs(data_dir, exist_ok=True)
    
    def _load_config(self):
        """Load auto-culling configuration"""
        try:
            if os.path.exists(self.culling_config_file):
                with open(self.culling_config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
                logger.info("✅ Auto-culling config loaded")
        except Exception as e:
            logger.warning(f"Could not load culling config: {e}")
            self._save_config()
    
    def _save_config(self):
        """Save auto-culling configuration"""
        try:
            with open(self.culling_config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save culling config: {e}")
    
    def _load_paused_models(self):
        """Load paused models state"""
        try:
            if os.path.exists(self.paused_models_file):
                with open(self.paused_models_file, 'r') as f:
                    data = json.load(f)
                    self.paused_models = data.get('paused_models', {})
                    self.retrain_attempts = data.get('retrain_attempts', {})
                logger.info(f"✅ Loaded {len(self.paused_models)} paused models")
        except Exception as e:
            logger.warning(f"Could not load paused models: {e}")
    
    def _save_paused_models(self):
        """Save paused models state"""
        try:
            data = {
                'paused_models': self.paused_models,
                'retrain_attempts': self.retrain_attempts,
                'last_updated': datetime.utcnow().isoformat()
            }
            with open(self.paused_models_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save paused models: {e}")
    
    def is_model_paused(self, model_name: str) -> bool:
        """Check if a model is currently paused"""
        if model_name not in self.paused_models:
            return False
        
        pause_info = self.paused_models[model_name]
        pause_until = datetime.fromisoformat(pause_info['pause_until'])
        
        if datetime.utcnow() >= pause_until:
            # Pause period expired, remove from paused list
            del self.paused_models[model_name]
            self._save_paused_models()
            logger.info(f"🔓 Model {model_name} pause period expired")
            return False
        
        return True
    
    def pause_model(self, model_name: str, reason: str, duration_hours: Optional[float] = None):
        """Pause a model for a specified duration"""
        if duration_hours is None:
            duration_hours = self.config['pause_duration_hours']
        
        pause_until = datetime.utcnow() + timedelta(hours=duration_hours)
        
        self.paused_models[model_name] = {
            'reason': reason,
            'paused_at': datetime.utcnow().isoformat(),
            'pause_until': pause_until.isoformat(),
            'duration_hours': duration_hours
        }
        
        self._save_paused_models()
        
        logger.warning(f"⏸️ Model {model_name} paused for {duration_hours}h: {reason}")
        
        # Send notification
        send_trader_notification(f"⏸️ **Model Paused**: {model_name.upper()}\n"
                               f"**Reason**: {reason}\n"
                               f"**Duration**: {duration_hours}h\n"
                               f"**Resume Time**: {pause_until.strftime('%Y-%m-%d %H:%M UTC')}")
    
    def unpause_model(self, model_name: str):
        """Manually unpause a model"""
        if model_name in self.paused_models:
            del self.paused_models[model_name]
            self._save_paused_models()
            logger.info(f"🔓 Model {model_name} manually unpaused")
            send_trader_notification(f"🔓 **Model Unpaused**: {model_name.upper()}")
            return True
        return False
    
    def should_cull_model(self, model_name: str, performance: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a model should be culled based on performance"""
        if not self.config['enabled']:
            return False, "Auto-culling disabled"
        
        # Check minimum trades threshold
        if performance['total_trades'] < self.config['min_trades_before_culling']:
            return False, f"Insufficient trades ({performance['total_trades']})"
        
        # Check win rate
        if performance['win_rate'] < self.config['win_rate_threshold']:
            return True, f"Low win rate: {performance['win_rate']:.1%} < {self.config['win_rate_threshold']:.1%}"
        
        # Check consecutive losses
        if performance['consecutive_losses'] >= self.config['max_consecutive_losses']:
            return True, f"Too many consecutive losses: {performance['consecutive_losses']}"
        
        # Check total loss
        if performance['total_pnl'] < -self.config['max_total_loss']:
            return True, f"Excessive losses: ${performance['total_pnl']:.2f}"
        
        return False, "Performance within acceptable range"
    
    def attempt_retrain(self, model_name: str) -> bool:
        """Attempt to retrain a flagged model"""
        try:
            # Check retrain attempt limit
            attempts = self.retrain_attempts.get(model_name, 0)
            if attempts >= self.config['max_retrain_attempts']:
                logger.warning(f"❌ Max retrain attempts reached for {model_name}")
                return False
            
            # Increment attempt counter
            self.retrain_attempts[model_name] = attempts + 1
            self._save_paused_models()
            
            logger.info(f"🔄 Attempting retrain #{attempts + 1} for {model_name}")
            
            # Initialize trainer and check if retraining is needed
            trainer = IncrementalTrainer(model_name)
            should_retrain, reason = trainer.should_retrain(
                win_rate_threshold=self.config['win_rate_threshold'] * 100
            )
            
            if not should_retrain:
                logger.info(f"✅ {model_name} does not need retraining: {reason}")
                return True
            
            # Perform incremental retraining
            result = trainer.run_incremental_training(force=True)
            
            if result.get('success'):
                logger.info(f"✅ Retraining successful for {model_name}")
                
                # Reset model flags and attempt counter
                stats_mgr = get_stats_manager()
                stats_mgr.reset_model_flags(model_name)
                
                if model_name in self.retrain_attempts:
                    del self.retrain_attempts[model_name]
                    self._save_paused_models()
                
                send_trader_notification(f"✅ **Retraining Complete**: {model_name.upper()}\n"
                                       f"**Accuracy**: {result.get('accuracy', 0):.3f}\n"
                                       f"**Trades Used**: {result.get('trades_used', 0)}\n"
                                       f"**Model Resumed**")
                return True
            else:
                logger.error(f"❌ Retraining failed for {model_name}: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error during retraining {model_name}: {e}")
            return False
    
    def run_culling_check(self):
        """Run a complete culling check on all models"""
        if not self.config['enabled']:
            logger.info("Auto-culling disabled, skipping check")
            return
        
        logger.info("🔍 Running auto-culling performance check...")
        
        stats_mgr = get_stats_manager()
        flagged_models = []
        paused_models = []
        retrained_models = []
        
        # Check all models
        for model_name, performance in stats_mgr.models_performance.items():
            perf_dict = {
                'total_trades': performance.total_trades,
                'win_rate': performance.win_rate,
                'consecutive_losses': performance.consecutive_losses,
                'total_pnl': performance.total_pnl
            }
            
            # Skip if already paused
            if self.is_model_paused(model_name):
                continue
            
            # Check if model should be culled
            should_cull, reason = self.should_cull_model(model_name, perf_dict)
            
            if should_cull:
                flagged_models.append((model_name, reason))
                
                if self.config['auto_retrain']:
                    # Attempt retraining
                    if self.attempt_retrain(model_name):
                        retrained_models.append(model_name)
                    else:
                        # Retraining failed, pause the model
                        self.pause_model(model_name, f"Retraining failed: {reason}")
                        paused_models.append(model_name)
                else:
                    # Just pause without retraining
                    self.pause_model(model_name, reason)
                    paused_models.append(model_name)
        
        # Log results
        if flagged_models:
            logger.warning(f"🚩 Flagged {len(flagged_models)} underperforming models")
            for model, reason in flagged_models:
                logger.warning(f"  • {model}: {reason}")
        
        if retrained_models:
            logger.info(f"🔄 Successfully retrained {len(retrained_models)} models: {', '.join(retrained_models)}")
        
        if paused_models:
            logger.warning(f"⏸️ Paused {len(paused_models)} models: {', '.join(paused_models)}")
        
        if not flagged_models:
            logger.info("✅ All models performing within acceptable range")
        
        # Send summary notification if any action was taken
        if flagged_models:
            summary = f"🤖 **Auto-Culling Report**\n\n"
            summary += f"**Flagged Models**: {len(flagged_models)}\n"
            summary += f"**Retrained**: {len(retrained_models)}\n"
            summary += f"**Paused**: {len(paused_models)}\n\n"
            
            if paused_models:
                summary += f"**Paused Models**: {', '.join(paused_models)}\n"
            if retrained_models:
                summary += f"**Retrained Models**: {', '.join(retrained_models)}"
            
            send_trader_notification(summary)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current auto-culling system status"""
        return {
            'enabled': self.config['enabled'],
            'check_interval_minutes': self.config['check_interval_minutes'],
            'paused_models': len(self.paused_models),
            'paused_details': self.paused_models,
            'retrain_attempts': self.retrain_attempts,
            'config': self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update auto-culling configuration"""
        self.config.update(new_config)
        self._save_config()
        logger.info("✅ Auto-culling configuration updated")

# Global instance
_auto_culler = None

def get_auto_culler() -> AutoCullingSystem:
    """Get singleton auto-culling system instance"""
    global _auto_culler
    if _auto_culler is None:
        _auto_culler = AutoCullingSystem()
    return _auto_culler

def run_periodic_culling():
    """Run periodic culling checks (for background tasks)"""
    culler = get_auto_culler()
    
    while culler.config['enabled']:
        try:
            culler.run_culling_check()
            
            # Sleep for configured interval
            sleep_seconds = culler.config['check_interval_minutes'] * 60
            logger.info(f"💤 Auto-culling sleeping for {culler.config['check_interval_minutes']} minutes")
            time.sleep(sleep_seconds)
            
        except KeyboardInterrupt:
            logger.info("Auto-culling stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in periodic culling: {e}")
            time.sleep(300)  # Sleep 5 minutes on error

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-Culling System for Trading Models")
    parser.add_argument("--check", action="store_true", help="Run a single culling check")
    parser.add_argument("--daemon", action="store_true", help="Run periodic culling checks")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--unpause", type=str, help="Unpause a specific model")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    culler = get_auto_culler()
    
    if args.check:
        culler.run_culling_check()
    elif args.daemon:
        run_periodic_culling()
    elif args.status:
        status = culler.get_status()
        print(json.dumps(status, indent=2))
    elif args.unpause:
        if culler.unpause_model(args.unpause):
            print(f"✅ Model {args.unpause} unpaused")
        else:
            print(f"❌ Model {args.unpause} was not paused")
    else:
        print("Use --check, --daemon, --status, or --unpause <model_name>")
