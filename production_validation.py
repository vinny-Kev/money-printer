#!/usr/bin/env python3
"""
🚀 PRODUCTION READINESS VALIDATOR
Final comprehensive system check before live deployment
"""

import sys
import os
import importlib
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class ProductionValidator:
    """Comprehensive production readiness checker"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.passed_tests = []
        
    def log_error(self, message):
        self.errors.append(message)
        print(f"❌ {message}")
        
    def log_warning(self, message):
        self.warnings.append(message)
        print(f"⚠️ {message}")
        
    def log_success(self, message):
        self.passed_tests.append(message)
        print(f"✅ {message}")
        
    def test_critical_imports(self):
        """Test all critical module imports"""
        print("\n🔍 TESTING CRITICAL IMPORTS...")
        
        critical_modules = [
            ('pandas', 'Data processing'),
            ('numpy', 'Numerical computing'),
            ('sklearn', 'Machine learning'),
            ('binance', 'Binance API'),
            ('discord', 'Discord bot'),
            ('ta', 'Technical analysis'),
            ('requests', 'HTTP requests'),
            ('dotenv', 'Environment variables')
        ]
        
        for module_name, description in critical_modules:
            try:
                importlib.import_module(module_name)
                self.log_success(f"{module_name} - {description}")
            except ImportError as e:
                self.log_error(f"{module_name} missing - {description}: {e}")
                
    def test_core_modules(self):
        """Test core application modules"""
        print("\n🔍 TESTING CORE MODULES...")
        
        core_modules = [
            ('src.trading_bot.trade_runner', 'Trading execution'),
            ('src.data_collector.data_scraper', 'Data collection'),
            ('src.model_training.random_forest_trainer', 'Model training'),
            ('src.lightweight_discord_bot', 'Discord bot'),
            ('src.trading_safety', 'Trading safety'),
            ('src.safe_config', 'Configuration'),
            ('src.discord_notifications', 'Notifications')
        ]
        
        for module_name, description in core_modules:
            try:
                module = importlib.import_module(module_name)
                self.log_success(f"{description} module")
                
                # Test key functions exist
                if 'trade_runner' in module_name:
                    if hasattr(module, 'run_single_trade') and hasattr(module, 'get_usdt_balance'):
                        self.log_success("Trading functions available")
                    else:
                        self.log_error("Missing trading functions")
                        
                elif 'data_scraper' in module_name:
                    if hasattr(module, 'main'):
                        self.log_success("Data scraper main function available")
                    else:
                        self.log_error("Missing data scraper main function")
                        
                elif 'random_forest_trainer' in module_name:
                    if hasattr(module, 'main'):
                        self.log_success("Model trainer main function available")
                    else:
                        self.log_error("Missing model trainer main function")
                        
            except Exception as e:
                self.log_error(f"{description} module failed: {e}")
                
    def test_trading_safety(self):
        """Test trading safety systems"""
        print("\n🔍 TESTING TRADING SAFETY SYSTEMS...")
        
        try:
            from src.trading_safety import TradingSafetyManager
            from src.safe_config import get_config
            
            config = get_config()
            safety_mgr = TradingSafetyManager(config)
            
            # Test safety checks
            can_trade, reason = safety_mgr.can_trade_now()
            self.log_success(f"Safety manager initialized - Can trade: {can_trade}")
            
            # Test position sizing
            if hasattr(safety_mgr, 'calculate_position_size'):
                size = safety_mgr.calculate_position_size(1000, 0.7, 0.05)
                if 0 < size <= 1000:
                    self.log_success(f"Position sizing working - ${size:.2f}")
                else:
                    self.log_warning(f"Position sizing unusual - ${size:.2f}")
            
        except Exception as e:
            self.log_error(f"Trading safety failed: {e}")
            
    def test_environment_variables(self):
        """Test critical environment variables"""
        print("\n🔍 TESTING ENVIRONMENT VARIABLES...")
        
        critical_env_vars = [
            ('DISCORD_BOT_TOKEN', 'Discord bot authentication'),
            ('BINANCE_API_KEY', 'Binance API access'),
            ('BINANCE_SECRET_KEY', 'Binance API secret'),
        ]
        
        for var_name, description in critical_env_vars:
            value = os.getenv(var_name)
            if value:
                masked_value = f"{value[:8]}..." if len(value) > 8 else "***"
                self.log_success(f"{var_name} - {description} ({masked_value})")
            else:
                self.log_error(f"{var_name} missing - {description}")
                
        # Test optional but recommended vars
        optional_vars = [
            'DISCORD_WEBHOOK',
            'LIVE_TRADING',
            'RAILWAY_API_TOKEN'
        ]
        
        for var_name in optional_vars:
            value = os.getenv(var_name)
            if value:
                self.log_success(f"{var_name} configured")
            else:
                self.log_warning(f"{var_name} not set (optional)")
                
    def test_file_structure(self):
        """Test required file structure"""
        print("\n🔍 TESTING FILE STRUCTURE...")
        
        required_files = [
            'main.py',
            'requirements.txt',
            'Dockerfile.full',
            'railway.toml',
            'src/config.py',
            'src/safe_config.py',
            'src/trading_bot/trade_runner.py',
            'src/data_collector/data_scraper.py',
            'src/lightweight_discord_bot.py'
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.log_success(f"Required file: {file_path}")
            else:
                self.log_error(f"Missing required file: {file_path}")
                
    def test_basic_functionality(self):
        """Test basic functionality without real API calls"""
        print("\n🔍 TESTING BASIC FUNCTIONALITY...")
        
        try:
            # Test configuration loading
            from src.safe_config import get_config
            config = get_config()
            self.log_success("Configuration loading")
            
            # Test Discord notifications (mock)
            from src.discord_notifications import send_general_notification
            # Don't actually send notification during testing
            self.log_success("Discord notification system available")
            
            # Test technical indicators
            try:
                from src.trading_bot.technical_indicators import TechnicalIndicators
                import pandas as pd
                import numpy as np
                
                # Create dummy data
                df = pd.DataFrame({
                    'open': np.random.uniform(95, 105, 50),
                    'high': np.random.uniform(100, 110, 50),
                    'low': np.random.uniform(90, 100, 50),
                    'close': np.random.uniform(95, 105, 50),
                    'volume': np.random.uniform(1000, 10000, 50)
                })
                
                ti = TechnicalIndicators()
                result = ti.calculate_all_indicators(df)
                if 'rsi' in result.columns:
                    self.log_success("Technical indicators calculation")
                else:
                    self.log_warning("Technical indicators missing RSI")
                    
            except Exception as e:
                self.log_error(f"Technical indicators failed: {e}")
                
        except Exception as e:
            self.log_error(f"Basic functionality test failed: {e}")
            
    def run_all_tests(self):
        """Run all validation tests"""
        print("🚀" + "="*60 + "🚀")
        print("   MONEY PRINTER - PRODUCTION READINESS VALIDATION")
        print("🚀" + "="*60 + "🚀")
        print(f"📅 Validation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.test_critical_imports()
        self.test_environment_variables()
        self.test_file_structure()
        self.test_core_modules()
        self.test_trading_safety()
        self.test_basic_functionality()
        
        # Final report
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        
        print(f"✅ Passed Tests: {len(self.passed_tests)}")
        print(f"⚠️ Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n🚨 CRITICAL ERRORS FOUND:")
            for error in self.errors:
                print(f"  • {error}")
                
        if self.warnings:
            print("\n⚠️ WARNINGS:")
            for warning in self.warnings:
                print(f"  • {warning}")
                
        # Final verdict
        print("\n" + "="*60)
        if not self.errors:
            print("🎉 PRODUCTION READY!")
            print("✅ All critical tests passed. Safe to deploy.")
            if self.warnings:
                print("💡 Address warnings for optimal performance.")
            return True
        else:
            print("🛑 NOT READY FOR PRODUCTION!")
            print("❌ Critical errors must be fixed before deployment.")
            return False

def main():
    """Run production validation"""
    validator = ProductionValidator()
    
    try:
        is_ready = validator.run_all_tests()
        sys.exit(0 if is_ready else 1)
    except KeyboardInterrupt:
        print("\n👋 Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation crashed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
