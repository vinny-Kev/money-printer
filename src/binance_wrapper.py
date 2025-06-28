"""
Enhanced Binance Client Wrapper - Production-ready market data collection
"""
import logging
from typing import List, Optional, Dict, Any
from binance.client import Client
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)

class EnhancedBinanceClient:
    """Enhanced Binance client with robust error handling and data processing"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Binance client"""
        try:
            self.client = Client(
                api_key=self.api_key,
                api_secret=self.api_secret,
                testnet=self.testnet
            )
            
            # Test connection
            self.client.ping()
            logger.info("✅ Binance client connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Binance client: {e}")
            self.client = None
            return False
    
    def test_connection(self) -> bool:
        """Test if the connection is working"""
        if not self.client:
            return False
        
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information including balances"""
        if not self.client:
            return None
        
        try:
            account_info = self.client.get_account()
            
            # Extract key information
            result = {
                'account_type': account_info.get('accountType'),
                'can_trade': account_info.get('canTrade'),
                'can_withdraw': account_info.get('canWithdraw'),
                'can_deposit': account_info.get('canDeposit'),
                'update_time': account_info.get('updateTime'),
                'balances': []
            }
            
            # Process balances
            for balance in account_info.get('balances', []):
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                
                if total_balance > 0:  # Only include non-zero balances
                    result['balances'].append({
                        'asset': balance['asset'],
                        'free': free_balance,
                        'locked': locked_balance,
                        'total': total_balance
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get account info: {e}")
            return None
    
    def get_historical_klines(self, symbol: str, interval: str, limit: int = 100) -> Optional[List]:
        """Get historical kline data"""
        if not self.client:
            logger.error("❌ Binance client not initialized")
            return None
        
        try:
            # Use the simple approach with just limit
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            logger.debug(f"📊 Retrieved {len(klines)} klines for {symbol} {interval}")
            return klines
            
        except Exception as e:
            logger.error(f"❌ Failed to get klines for {symbol} {interval}: {e}")
            return None
    
    def _interval_to_minutes(self, interval: str) -> Optional[int]:
        """Convert interval string to minutes"""
        interval_map = {
            '1m': 1,
            '3m': 3,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '2h': 120,
            '4h': 240,
            '6h': 360,
            '8h': 480,
            '12h': 720,
            '1d': 1440,
            '3d': 4320,
            '1w': 10080,
            '1M': 43200
        }
        return interval_map.get(interval)
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get symbol information"""
        if not self.client:
            return None
        
        try:
            exchange_info = self.client.get_exchange_info()
            
            for symbol_info in exchange_info['symbols']:
                if symbol_info['symbol'] == symbol:
                    return {
                        'symbol': symbol_info['symbol'],
                        'status': symbol_info['status'],
                        'base_asset': symbol_info['baseAsset'],
                        'quote_asset': symbol_info['quoteAsset'],
                        'is_spot_trading_allowed': symbol_info.get('isSpotTradingAllowed'),
                        'permissions': symbol_info.get('permissions', [])
                    }
            
            logger.warning(f"⚠️ Symbol {symbol} not found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get symbol info for {symbol}: {e}")
            return None
    
    def get_24hr_ticker_stats(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get 24hr ticker statistics"""
        if not self.client:
            return None
        
        try:
            ticker = self.client.get_24hr_ticker(symbol=symbol)
            
            return {
                'symbol': ticker['symbol'],
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'weighted_avg_price': float(ticker['weightedAvgPrice']),
                'prev_close_price': float(ticker['prevClosePrice']),
                'last_price': float(ticker['lastPrice']),
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'open_price': float(ticker['openPrice']),
                'high_price': float(ticker['highPrice']),
                'low_price': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'open_time': ticker['openTime'],
                'close_time': ticker['closeTime'],
                'count': int(ticker['count'])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get ticker stats for {symbol}: {e}")
            return None
    
    def klines_to_dataframe(self, klines: List, symbol: str, interval: str) -> pd.DataFrame:
        """Convert klines data to pandas DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        # Column names for klines data
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert timestamp columns to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Convert numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                         'quote_asset_volume', 'number_of_trades',
                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata
        df['symbol'] = symbol
        df['interval'] = interval
        df['collection_time'] = datetime.now()
        
        # Drop the 'ignore' column
        df = df.drop('ignore', axis=1)
        
        return df
    
    def get_server_time(self) -> Optional[Dict[str, Any]]:
        """Get server time"""
        if not self.client:
            return None
        
        try:
            server_time = self.client.get_server_time()
            return {
                'server_time': server_time['serverTime'],
                'server_time_datetime': pd.to_datetime(server_time['serverTime'], unit='ms'),
                'local_time': datetime.now(),
                'time_diff_ms': server_time['serverTime'] - int(datetime.now().timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"❌ Failed to get server time: {e}")
            return None
