"""
Test Binance data retrieval directly
"""
from src.binance_wrapper import EnhancedBinanceClient
from src.config import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_TESTNET

# Test the Binance client directly
client = EnhancedBinanceClient(
    api_key=BINANCE_API_KEY,
    api_secret=BINANCE_SECRET_KEY,
    testnet=BINANCE_TESTNET
)

print(f"🔗 Connection test: {client.test_connection()}")

# Test data retrieval
print("\n📊 Testing data retrieval...")
klines = client.get_historical_klines("BTCUSDT", "1m", 10)
print(f"Raw klines: {len(klines) if klines else 0} items")

if klines:
    print(f"First kline: {klines[0]}")
    
    # Test DataFrame conversion
    df = client.klines_to_dataframe(klines, "BTCUSDT", "1m")
    print(f"DataFrame: {len(df)} rows")
    print(df.head())
else:
    print("❌ No klines data retrieved")

# Test account info
print("\n💰 Testing account info...")
account = client.get_account_info()
if account:
    print(f"Account type: {account['account_type']}")
    print(f"Non-zero balances: {len([b for b in account['balances'] if b['total'] > 0])}")
else:
    print("❌ No account info retrieved")
