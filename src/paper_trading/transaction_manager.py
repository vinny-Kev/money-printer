import os
from solana.keypair import Keypair
from solana.rpc.api import Client
from dotenv import load_dotenv
import base58
import json

load_dotenv()

def load_solana_wallet():
    raw = os.getenv("SOLANA_PRIVATE_KEY")
    if not raw:
        raise ValueError("SOLANA_PRIVATE_KEY not found in .env")
    
    # If the key is a comma-separated list of ints
    if "," in raw:
        secret = bytes([int(x.strip()) for x in raw.split(",")])
    else:
        secret = base58.b58decode(raw.strip())
    
    return Keypair.from_secret_key(secret)

SIMULATION_MODE = True  # Set to False for real trading

def buy_coin(symbol):
    if SIMULATION_MODE:
        print(f"[SIM] Buying {symbol} (simulation only)")
    else:
        print(f"Buying {symbol}... (implement real logic here)")

def sell_coin(symbol):
    if SIMULATION_MODE:
        print(f"[SIM] Selling {symbol} (simulation only)")
    else:
        print(f"Selling {symbol}... (implement real logic here)")

LIVE_STATS_PATH = "live_stats.json"

def update_live_stats(win, profit):
    stats = {"trades": 0, "wins": 0, "profit": 0.0}
    if os.path.exists(LIVE_STATS_PATH):
        with open(LIVE_STATS_PATH, "r") as f:
            stats = json.load(f)
    stats["trades"] += 1
    if win:
        stats["wins"] += 1
    stats["profit"] += profit
    with open(LIVE_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

def get_live_stats():
    if os.path.exists(LIVE_STATS_PATH):
        with open(LIVE_STATS_PATH, "r") as f:
            return json.load(f)
    return {"trades": 0, "wins": 0, "profit": 0.0}

if __name__ == "__main__":
    client = Client("https://api.mainnet-beta.solana.com")  # or "https://api.devnet.solana.com"
    keypair = load_solana_wallet()
    print(f"Public key: {keypair.public_key}")
    balance = client.get_balance(keypair.public_key)['result']['value'] / 1e9
    print(f"SOL balance: {balance:.4f} SOL")
