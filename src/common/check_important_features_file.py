import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
FEATURE_PATH = os.path.join(BASE_DIR, "data/models/important_features.json")

print("Resolved FEATURE_PATH:", FEATURE_PATH)
print("Exists?", os.path.exists(FEATURE_PATH))
