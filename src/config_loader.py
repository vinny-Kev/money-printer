import json
def load_wallets(path='config/wallets.json'):
    with open(path, 'r') as f:
        return json.load