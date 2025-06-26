#!/usr/bin/env python3
"""
Production WebSocket Manager

Handles WebSocket connections with reconnection logic, exponential backoff,
and automatic failover for market data feeds.
"""

import asyncio
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import websockets
import websocket
from collections import deque
import random

logger = logging.getLogger(__name__)

class WebSocketReconnectManager:
    """Production-ready WebSocket manager with exponential backoff"""
    
    def __init__(self, safety_manager, max_retries: int = 10):
        self.safety_manager = safety_manager
        self.max_retries = max_retries
        self.ws = None
        self.is_connected = False
        self.is_reconnecting = False
        self.retry_count = 0
        self.last_ping_time = None
        self.message_buffer = deque(maxlen=1000)  # Buffer recent messages
        self.connection_start_time = None
        self.total_reconnects = 0
        
        # Callbacks
        self.on_message_callback = None
        self.on_connect_callback = None
        self.on_disconnect_callback = None
        
        # Watchdog thread
        self.watchdog_thread = None
        self.should_stop = False
        
    def set_message_callback(self, callback: Callable[[Dict], None]):
        """Set callback for incoming messages"""
        self.on_message_callback = callback
    
    def set_connect_callback(self, callback: Callable[[], None]):
        """Set callback for successful connections"""
        self.on_connect_callback = callback
    
    def set_disconnect_callback(self, callback: Callable[[], None]):
        """Set callback for disconnections"""
        self.on_disconnect_callback = callback
    
    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = min(300, 2 ** self.retry_count)  # Max 5 minutes
        jitter = random.uniform(0.1, 0.3) * base_delay  # 10-30% jitter
        return base_delay + jitter
    
    def connect(self, url: str, ping_interval: int = 30):
        """Connect to WebSocket with error handling"""
        try:
            if self.is_connected:
                logger.warning("WebSocket already connected")
                return True
            
            logger.info(f"🔌 Connecting to WebSocket: {url}")
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self.message_buffer.append({
                        'timestamp': datetime.utcnow(),
                        'data': data
                    })
                    
                    # Update safety manager
                    self.safety_manager.update_data_timestamp()
                    
                    # Call user callback
                    if self.on_message_callback:
                        self.on_message_callback(data)
                        
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket error: {error}")
                self.safety_manager.handle_websocket_failure()
            
            def on_close(ws, close_status_code, close_msg):
                logger.warning(f"WebSocket closed: {close_status_code} - {close_msg}")
                self.is_connected = False
                self.connection_start_time = None
                
                if self.on_disconnect_callback:
                    self.on_disconnect_callback()
                
                if not self.should_stop:
                    self._schedule_reconnect(url, ping_interval)
            
            def on_open(ws):
                logger.info("✅ WebSocket connected successfully")
                self.is_connected = True
                self.is_reconnecting = False
                self.retry_count = 0
                self.connection_start_time = datetime.utcnow()
                
                if self.on_connect_callback:
                    self.on_connect_callback()
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            
            # Start connection in background thread
            def run_websocket():
                try:
                    self.ws.run_forever(
                        ping_interval=ping_interval,
                        ping_timeout=10,
                        ping_payload="ping"
                    )
                except Exception as e:
                    logger.error(f"WebSocket run_forever error: {e}")
            
            ws_thread = threading.Thread(target=run_websocket, daemon=True)
            ws_thread.start()
            
            # Start watchdog thread
            if not self.watchdog_thread or not self.watchdog_thread.is_alive():
                self.watchdog_thread = threading.Thread(target=self._watchdog, daemon=True)
                self.watchdog_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            self.safety_manager.handle_websocket_failure()
            return False
    
    def _schedule_reconnect(self, url: str, ping_interval: int):
        """Schedule reconnection with exponential backoff"""
        if self.should_stop or self.is_reconnecting:
            return
        
        if self.retry_count >= self.max_retries:
            logger.error(f"🚨 Max WebSocket retries ({self.max_retries}) reached - giving up")
            self.safety_manager.handle_websocket_failure()
            return
        
        self.is_reconnecting = True
        self.retry_count += 1
        self.total_reconnects += 1
        
        delay = self.get_backoff_delay()
        logger.warning(f"⏳ WebSocket reconnect #{self.retry_count} in {delay:.1f}s")
        
        def reconnect_after_delay():
            time.sleep(delay)
            if not self.should_stop:
                self.connect(url, ping_interval)
        
        thread = threading.Thread(target=reconnect_after_delay, daemon=True)
        thread.start()
    
    def _watchdog(self):
        """Watchdog thread to monitor connection health"""
        while not self.should_stop:
            try:
                if self.is_connected and self.connection_start_time:
                    # Check connection age
                    age = datetime.utcnow() - self.connection_start_time
                    
                    # Check for stale data
                    if self.message_buffer:
                        last_message_age = datetime.utcnow() - self.message_buffer[-1]['timestamp']
                        if last_message_age > timedelta(minutes=2):
                            logger.warning(f"⚠️ No WebSocket data for {last_message_age.total_seconds():.0f}s")
                            # Force reconnection if no data for too long
                            if last_message_age > timedelta(minutes=5):
                                logger.error("🔌 Forcing WebSocket reconnection due to stale data")
                                self.disconnect()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(30)
    
    def send_message(self, message: Dict) -> bool:
        """Send message to WebSocket"""
        try:
            if not self.is_connected or not self.ws:
                logger.warning("Cannot send message - WebSocket not connected")
                return False
            
            self.ws.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            return False
    
    def disconnect(self):
        """Gracefully disconnect WebSocket"""
        try:
            self.should_stop = True
            self.is_connected = False
            
            if self.ws:
                self.ws.close()
            
            logger.info("🔌 WebSocket disconnected")
            
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        stats = {
            "is_connected": self.is_connected,
            "is_reconnecting": self.is_reconnecting,
            "retry_count": self.retry_count,
            "total_reconnects": self.total_reconnects,
            "buffer_size": len(self.message_buffer),
            "connection_uptime": None,
            "last_message_age": None
        }
        
        if self.connection_start_time:
            stats["connection_uptime"] = (datetime.utcnow() - self.connection_start_time).total_seconds()
        
        if self.message_buffer:
            stats["last_message_age"] = (datetime.utcnow() - self.message_buffer[-1]['timestamp']).total_seconds()
        
        return stats

class BinanceWebSocketManager:
    """Binance-specific WebSocket manager with multiple streams"""
    
    def __init__(self, safety_manager):
        self.safety_manager = safety_manager
        self.connections: Dict[str, WebSocketReconnectManager] = {}
        self.subscribed_symbols = set()
        self.price_data = {}
        self.kline_data = {}
        
    def subscribe_to_prices(self, symbols: List[str]):
        """Subscribe to ticker price streams"""
        try:
            # Create ticker stream URL
            streams = [f"{symbol.lower()}@ticker" for symbol in symbols]
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            # Create connection manager
            conn_manager = WebSocketReconnectManager(self.safety_manager)
            conn_manager.set_message_callback(self._handle_price_message)
            
            # Connect
            if conn_manager.connect(stream_url):
                self.connections['prices'] = conn_manager
                self.subscribed_symbols.update(symbols)
                logger.info(f"✅ Subscribed to price streams for {len(symbols)} symbols")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to price streams: {e}")
            return False
    
    def subscribe_to_klines(self, symbols: List[str], interval: str = "1h"):
        """Subscribe to kline/candlestick streams"""
        try:
            # Create kline stream URL
            streams = [f"{symbol.lower()}@kline_{interval}" for symbol in symbols]
            stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
            
            # Create connection manager
            conn_manager = WebSocketReconnectManager(self.safety_manager)
            conn_manager.set_message_callback(self._handle_kline_message)
            
            # Connect
            if conn_manager.connect(stream_url):
                self.connections['klines'] = conn_manager
                logger.info(f"✅ Subscribed to kline streams for {len(symbols)} symbols")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to subscribe to kline streams: {e}")
            return False
    
    def _handle_price_message(self, data: Dict):
        """Handle incoming price ticker messages"""
        try:
            if 'c' in data and 's' in data:  # Current price and symbol
                symbol = data['s']
                price = float(data['c'])
                self.price_data[symbol] = {
                    'price': price,
                    'timestamp': datetime.utcnow(),
                    'volume': float(data.get('v', 0)),
                    'change_percent': float(data.get('P', 0))
                }
                
        except Exception as e:
            logger.error(f"Error handling price message: {e}")
    
    def _handle_kline_message(self, data: Dict):
        """Handle incoming kline messages"""
        try:
            if 'k' in data:
                kline = data['k']
                symbol = kline['s']
                
                # Only process closed klines
                if kline.get('x', False):  # x = is this kline closed?
                    self.kline_data[symbol] = {
                        'open': float(kline['o']),
                        'high': float(kline['h']),
                        'low': float(kline['l']),
                        'close': float(kline['c']),
                        'volume': float(kline['v']),
                        'timestamp': datetime.utcnow(),
                        'close_time': int(kline['T'])
                    }
                    
        except Exception as e:
            logger.error(f"Error handling kline message: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        price_info = self.price_data.get(symbol)
        if price_info:
            # Check if price is recent (within 30 seconds)
            age = (datetime.utcnow() - price_info['timestamp']).total_seconds()
            if age <= 30:
                return price_info['price']
        
        return None
    
    def get_latest_kline(self, symbol: str) -> Optional[Dict]:
        """Get latest kline data for symbol"""
        kline_info = self.kline_data.get(symbol)
        if kline_info:
            # Check if kline is recent (within 1 hour)
            age = (datetime.utcnow() - kline_info['timestamp']).total_seconds()
            if age <= 3600:  # 1 hour
                return kline_info
        
        return None
    
    def is_data_fresh(self, max_age_seconds: int = 60) -> bool:
        """Check if we have fresh data from WebSocket"""
        if not self.price_data:
            return False
        
        # Check if any symbol has recent data
        now = datetime.utcnow()
        for price_info in self.price_data.values():
            age = (now - price_info['timestamp']).total_seconds()
            if age <= max_age_seconds:
                return True
        
        return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get overall connection status"""
        status = {
            "connections": {},
            "subscribed_symbols": len(self.subscribed_symbols),
            "price_data_symbols": len(self.price_data),
            "kline_data_symbols": len(self.kline_data),
            "data_is_fresh": self.is_data_fresh()
        }
        
        for name, conn in self.connections.items():
            status["connections"][name] = conn.get_connection_stats()
        
        return status
    
    def disconnect_all(self):
        """Disconnect all WebSocket connections"""
        for conn in self.connections.values():
            conn.disconnect()
        
        self.connections.clear()
        self.subscribed_symbols.clear()
        logger.info("🔌 All WebSocket connections disconnected")
