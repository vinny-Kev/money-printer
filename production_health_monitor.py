#!/usr/bin/env python3
"""
Production System Health Monitor
Comprehensive health check for all production components
"""
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Set up logging for health monitor"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('system_health_monitor.log')
        ]
    )
    return logging.getLogger(__name__)

def check_data_pipeline():
    """Check data collection and storage pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING DATA PIPELINE")
    
    results = {
        'status': 'UNKNOWN',
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        from src.config import PARQUET_DATA_DIR, DATA_ROOT
        from src.storage.enhanced_storage_manager import EnhancedStorageManager
        
        # Check data directory
        if not PARQUET_DATA_DIR.exists():
            results['issues'].append(f"Data directory missing: {PARQUET_DATA_DIR}")
            results['status'] = 'CRITICAL'
            return results
        
        # Check for data files
        parquet_files = list(PARQUET_DATA_DIR.glob("**/*.parquet"))
        results['stats']['parquet_files'] = len(parquet_files)
        
        if len(parquet_files) == 0:
            results['issues'].append("No data files found")
            results['status'] = 'CRITICAL'
            return results
        
        # Check file ages
        recent_files = 0
        for file in parquet_files:
            age_hours = (datetime.now().timestamp() - file.stat().st_mtime) / 3600
            if age_hours < 24:
                recent_files += 1
        
        results['stats']['recent_files_24h'] = recent_files
        
        if recent_files == 0:
            results['warnings'].append("No recent data files (last 24h)")
        
        # Test storage manager
        try:
            storage = EnhancedStorageManager(local_backup_dir=str(PARQUET_DATA_DIR))
            files = storage.list_files()
            results['stats']['accessible_files'] = len(files)
            
            if len(files) > 0:
                results['status'] = 'HEALTHY'
                logger.info(f"✅ Data pipeline healthy: {len(files)} files accessible")
            else:
                results['status'] = 'DEGRADED'
                results['warnings'].append("Storage manager found no files")
                
        except Exception as e:
            results['issues'].append(f"Storage manager error: {e}")
            results['status'] = 'DEGRADED'
        
    except Exception as e:
        results['issues'].append(f"Data pipeline check failed: {e}")
        results['status'] = 'CRITICAL'
    
    return results

def check_model_pipeline():
    """Check model training and validation pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING MODEL PIPELINE")
    
    results = {
        'status': 'UNKNOWN',
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        from src.config import MODELS_DIR
        from src.model_validation import ProductionModelValidator
        
        # Check models directory
        if not MODELS_DIR.exists():
            results['issues'].append(f"Models directory missing: {MODELS_DIR}")
            results['status'] = 'CRITICAL'
            return results
        
        # Check for ensemble models
        ensemble_files = list(MODELS_DIR.glob("production_ensemble_*.joblib"))
        scaler_files = list(MODELS_DIR.glob("production_scaler_*.joblib"))
        feature_files = list(MODELS_DIR.glob("production_features_*.json"))
        
        results['stats']['ensemble_models'] = len(ensemble_files)
        results['stats']['scaler_files'] = len(scaler_files)
        results['stats']['feature_files'] = len(feature_files)
        
        # Check for complete model sets
        if ensemble_files and scaler_files and feature_files:
            # Test model validator
            try:
                validator = ProductionModelValidator()
                model_package = validator.load_validated_model()
                
                if model_package:
                    validation_info = model_package['validation']
                    results['stats']['model_validation'] = {
                        'tests_passed': validation_info['tests_passed'],
                        'tests_failed': validation_info['tests_failed'],
                        'model_type': validation_info['model_info']['ensemble_type'],
                        'feature_count': validation_info['model_info']['feature_count']
                    }
                    
                    if validation_info['is_valid']:
                        results['status'] = 'HEALTHY'
                        logger.info("✅ Model pipeline healthy: Production ensemble validated")
                    else:
                        results['status'] = 'DEGRADED'
                        results['issues'].extend(validation_info['validation_errors'])
                else:
                    results['status'] = 'CRITICAL'
                    results['issues'].append("Model validation failed")
                    
            except Exception as e:
                results['status'] = 'DEGRADED'
                results['issues'].append(f"Model validation error: {e}")
        else:
            results['status'] = 'CRITICAL'
            results['issues'].append("Incomplete model set - missing ensemble, scaler, or features")
        
        # Check for legacy models
        legacy_models = list(MODELS_DIR.glob("**/trained_model.pkl"))
        results['stats']['legacy_models'] = len(legacy_models)
        
        if len(legacy_models) > 0 and len(ensemble_files) == 0:
            results['warnings'].append("Only legacy models available - upgrade to ensemble recommended")
        
    except Exception as e:
        results['issues'].append(f"Model pipeline check failed: {e}")
        results['status'] = 'CRITICAL'
    
    return results

def check_trading_safety():
    """Check trading safety systems"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING TRADING SAFETY")
    
    results = {
        'status': 'UNKNOWN',
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        from src.trading_safety_manager import get_safety_manager
        
        # Initialize safety manager
        safety = get_safety_manager()
        
        # Check system health
        health = safety.check_system_health()
        results['stats']['safety_metrics'] = health['metrics']
        
        if health['is_healthy']:
            results['status'] = 'HEALTHY'
            logger.info("✅ Trading safety healthy")
        else:
            results['status'] = 'DEGRADED'
            results['issues'].extend(health['errors'])
            results['warnings'].extend(health['warnings'])
        
        # Test safety check
        test_check = safety.pre_trade_safety_check("BTCUSDT", 100.0)
        results['stats']['safety_check'] = {
            'is_safe': test_check['is_safe_to_trade'],
            'checks_passed': test_check['checks_passed'],
            'checks_failed': test_check['checks_failed']
        }
        
        if not test_check['is_safe_to_trade']:
            results['warnings'].append(f"Safety check failed: {test_check['safety_violations']}")
        
    except Exception as e:
        results['issues'].append(f"Trading safety check failed: {e}")
        results['status'] = 'CRITICAL'
    
    return results

def check_api_connectivity():
    """Check API connectivity and permissions"""
    logger = logging.getLogger(__name__)
    logger.info("🔍 CHECKING API CONNECTIVITY")
    
    results = {
        'status': 'UNKNOWN',
        'issues': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY
        
        if not BINANCE_API_KEY or not BINANCE_SECRET_KEY:
            results['issues'].append("Binance API credentials not configured")
            results['status'] = 'CRITICAL'
            return results
        
        # Test Binance connection
        try:
            from binance.client import Client
            client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET_KEY)
            
            # Test basic connectivity
            server_time = client.get_server_time()
            results['stats']['server_time'] = server_time['serverTime']
            
            # Test account access
            account_info = client.get_account()
            results['stats']['account_type'] = account_info.get('accountType', 'UNKNOWN')
            results['stats']['can_trade'] = account_info.get('canTrade', False)
            
            if account_info.get('canTrade', False):
                results['status'] = 'HEALTHY'
                logger.info("✅ API connectivity healthy")
            else:
                results['status'] = 'DEGRADED'
                results['warnings'].append("Account cannot trade")
                
        except Exception as e:
            results['issues'].append(f"Binance API error: {e}")
            results['status'] = 'CRITICAL'
        
    except Exception as e:
        results['issues'].append(f"API connectivity check failed: {e}")
        results['status'] = 'CRITICAL'
    
    return results

def generate_health_report():
    """Generate comprehensive health report"""
    logger = setup_logging()
    logger.info("🏥 STARTING PRODUCTION SYSTEM HEALTH CHECK")
    logger.info("=" * 60)
    
    # Run all checks
    checks = {
        'data_pipeline': check_data_pipeline(),
        'model_pipeline': check_model_pipeline(),
        'trading_safety': check_trading_safety(),
        'api_connectivity': check_api_connectivity()
    }
    
    # Generate summary
    overall_status = 'HEALTHY'
    total_issues = 0
    total_warnings = 0
    
    for check_name, result in checks.items():
        if result['status'] == 'CRITICAL':
            overall_status = 'CRITICAL'
        elif result['status'] == 'DEGRADED' and overall_status == 'HEALTHY':
            overall_status = 'DEGRADED'
        
        total_issues += len(result['issues'])
        total_warnings += len(result['warnings'])
    
    # Create report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'summary': {
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'checks_run': len(checks)
        },
        'checks': checks
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏥 PRODUCTION SYSTEM HEALTH REPORT")
    print("=" * 60)
    print(f"📅 Generated: {datetime.now()}")
    print(f"🎯 Overall Status: {overall_status}")
    print(f"🔍 Checks Run: {len(checks)}")
    print(f"❌ Issues: {total_issues}")
    print(f"⚠️ Warnings: {total_warnings}")
    print("=" * 60)
    
    for check_name, result in checks.items():
        status_emoji = {"HEALTHY": "✅", "DEGRADED": "⚠️", "CRITICAL": "❌", "UNKNOWN": "❓"}
        print(f"{status_emoji.get(result['status'], '❓')} {check_name.upper()}: {result['status']}")
        
        if result['issues']:
            for issue in result['issues']:
                print(f"   ❌ {issue}")
        
        if result['warnings']:
            for warning in result['warnings']:
                print(f"   ⚠️ {warning}")
        
        if result['stats']:
            print(f"   📊 Stats: {result['stats']}")
        print()
    
    # Production readiness assessment
    print("🚀 PRODUCTION READINESS ASSESSMENT")
    print("=" * 60)
    
    if overall_status == 'HEALTHY':
        print("🎉 SYSTEM IS PRODUCTION READY!")
        print("✅ All critical systems operational")
        print("💡 Recommended: Start live trading")
    elif overall_status == 'DEGRADED':
        print("⚠️ SYSTEM HAS ISSUES BUT CAN OPERATE")
        print("📋 Review warnings and resolve if possible")
        print("💡 Recommended: Monitor closely, fix issues")
    else:
        print("❌ SYSTEM NOT READY FOR PRODUCTION")
        print("🚨 Critical issues must be resolved")
        print("💡 Recommended: Fix all critical issues before trading")
    
    # Save report
    report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Detailed report saved: {report_file}")
    
    return report

if __name__ == "__main__":
    try:
        report = generate_health_report()
        
        # Exit with appropriate code
        if report['overall_status'] == 'CRITICAL':
            sys.exit(1)
        elif report['overall_status'] == 'DEGRADED':
            sys.exit(2)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        sys.exit(3)
