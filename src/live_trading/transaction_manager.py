import os
from dotenv import load_dotenv
import base58
import json

# For Solana
from solana.keypair import Keypair
from solana.rpc.api import Client

# Load environment variables
load_dotenv()

# --- Wallet Loaders ---

def load_solana_wallet():
    raw = os.getenv("SOLANA_PHANTOM_SECRET_KEY")
    if not raw:
        raise ValueError("SOLANA_PHANTOM_SECRET_KEY not found in .env")
    # If the key is a comma-separated list of ints
    if "," in raw:
        secret = bytes([int(x.strip()) for x in raw.split(",")])
    else:
        secret = base58.b58decode(raw.strip())
    return Keypair.from_secret_key(secret)

def load_ethereum_key():
    key = os.getenv("ETHEREUM_SECRET_KEY")
    if not key:
        raise ValueError("ETHEREUM_SECRET_KEY not found in .env")
    return key

def load_bitcoin_key():
    key = os.getenv("BTC_SECRET_KEY")
    if not key:
        raise ValueError("BTC_SECRET_KEY not found in .env")
    return key

# --- Jup API Call Placeholder ---

def jup_api_trade(symbol, side, amount, wallet):
    """
    Placeholder for Jup API trade call.
    symbol: str, e.g. "SOL"
    side: "buy" or "sell"
    amount: float
    wallet: Keypair or private key
    """
    # TODO: Implement actual Jup API call here
    print(f"[JUP] {side.upper()} {amount} {symbol} using wallet {str(wallet)[:8]}... (not implemented)")
    # Return a fake transaction ID for now
    return "SIMULATED_TX_ID"

# --- Transaction Functions ---

def buy_coin(symbol, amount):
    # For now, only Solana is implemented as an example
    if symbol.lower() == "sol":
        wallet = load_solana_wallet()
        tx_id = jup_api_trade(symbol, "buy", amount, wallet)
        print(f"[LIVE] Buy order for {amount} {symbol.upper()} sent. TX: {tx_id}")
    else:
        print(f"[LIVE] Buy for {symbol.upper()} not implemented yet.")

def sell_coin(symbol, amount):
    if symbol.lower() == "sol":
        wallet = load_solana_wallet()
        tx_id = jup_api_trade(symbol, "sell", amount, wallet)
        print(f"[LIVE] Sell order for {amount} {symbol.upper()} sent. TX: {tx_id}")
    else:
        print(f"[LIVE] Sell for {symbol.upper()} not implemented yet.")

# --- Utility: Show Wallet Info ---

def show_wallet_info():
    try:
        sol_wallet = load_solana_wallet()
        client = Client("https://api.mainnet-beta.solana.com")
        balance = client.get_balance(sol_wallet.public_key)['result']['value'] / 1e9
        print(f"SOL Public key: {sol_wallet.public_key}")
        print(f"SOL balance: {balance:.4f} SOL")
    except Exception as e:
        print(f"Solana wallet error: {e}")
    # You can add similar info for ETH/BTC if needed

if __name__ == "__main__":
    show_wallet_info()
    # Example usage:
    # buy_coin("sol", 0.01)
    # sell_coin("sol", 0.01)