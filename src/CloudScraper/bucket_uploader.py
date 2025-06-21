import os
import json
import time
import logging
import requests  # For sending Discord alerts
from datetime import datetime
from b2sdk.v2 import InMemoryAccountInfo, B2Api
from dotenv import load_dotenv
import random
import pandas as pd
from io import StringIO, BytesIO  # Use BytesIO for binary data
import backoff

# Load environment variables
load_dotenv()

# Logger setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load Backblaze credentials
B2_BUCKET_DATA = os.getenv("B2_BUCKET_DATA")
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")  # Discord webhook URL


# trip ko lang lmao
peasant_quotes = [
    "My lord! We're running out of storage!",
    "Forgive me, my lord — the bucket is nearly full!",
    "The bytes overflow, sire! The bytes!!",
    "We've uploaded too much, my lord. The gods of cloud are angry!",
    "Sire, should we burn the logs to make space?"
]
# Initialize B2 connection
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)

# Get buckets
data_bucket = b2_api.get_bucket_by_name(B2_BUCKET_DATA)

STATUS_FILE = "bucket_status.json"  # File to store bucket status
UPLOAD_TRACKER_FILE = "upload_tracker.json"  # File to track daily uploads
DAILY_UPLOAD_LIMIT_MB = 500  # Daily upload limit in MB
TOTAL_UPLOAD_LIMIT_GB = 8.5  # Total upload limit in GB (combined for both buckets)

# Global cache for bucket storage
_bucket_storage_cache = {}
_bucket_cache_timestamps = {}

API_COUNTER = {
    "class_c": 80,
    "limit": 2450,
    "date": datetime.utcnow().strftime("%Y-%m-%d")
}


def load_api_counter():
    if os.path.exists("api_counter.json"):
        with open("api_counter.json") as f:
            data = json.load(f)
        if data["date"] != datetime.utcnow().strftime("%Y-%m-%d"):
            data = {"class_c": 0, "limit": 2450, "date": datetime.utcnow().strftime("%Y-%m-%d")}
        return data
    return {"class_c": 0, "limit": 2450, "date": datetime.utcnow().strftime("%Y-%m-%d")}

# Initialize API_COUNTER
API_COUNTER = load_api_counter()

# Discord alert function
def send_discord_alert(message):
    """
    Send an alert message to a Discord channel via webhook.
    :param message: The message to send.
    """
    if not DISCORD_WEBHOOK:
        logger.warning("⚠️ No Discord webhook set. Skipping alert.")
        return

    try:
        quote = random.choice(peasant_quotes)
        payload = {
             "content": f"🏯 **{quote}**\n\n{message}"
        }
        response = requests.post(DISCORD_WEBHOOK, json=payload)
        if response.status_code == 204:
            logger.info("✅ Discord alert sent.")
        else:
            logger.warning(f"❌ Failed to send Discord alert: {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Exception while sending Discord alert: {e}")

def reset_daily_upload_tracker():
    """
    Reset the daily upload tracker at midnight.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    tracker = {"date": today, "uploaded_bytes": 0}
    with open(UPLOAD_TRACKER_FILE, "w") as f:
        json.dump(tracker, f)
    logger.info("✅ Daily upload tracker reset.")

def check_daily_upload_limit(file_size):
    """
    Check if the daily upload limit has been exceeded.
    :param file_size: Size of the file to be uploaded (in bytes).
    :return: True if the upload can proceed, False if the limit is exceeded.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        # Load the tracker file
        if os.path.exists(UPLOAD_TRACKER_FILE):
            with open(UPLOAD_TRACKER_FILE, "r") as f:
                tracker = json.load(f)
        else:
            tracker = {"date": today, "uploaded_bytes": 0}

        # Reset the tracker if the date has changed
        if tracker["date"] != today:
            reset_daily_upload_tracker()
            tracker = {"date": today, "uploaded_bytes": 0}

        # Check if the upload exceeds the daily limit
        total_uploaded_mb = (tracker["uploaded_bytes"] + file_size) / (1024 * 1024)
        if total_uploaded_mb > DAILY_UPLOAD_LIMIT_MB:
            msg = f"Daily limit exceeded: {total_uploaded_mb:.2f} MB / {DAILY_UPLOAD_LIMIT_MB} MB"
            logger.warning(msg)
            send_discord_alert(msg)
            return False
        elif total_uploaded_mb > 0.9 * DAILY_UPLOAD_LIMIT_MB:
            msg = f"⚠️ Daily upload nearing limit: {total_uploaded_mb:.2f} MB / {DAILY_UPLOAD_LIMIT_MB} MB"
            send_discord_alert(msg)

        # Update the tracker with the new upload size
        tracker["uploaded_bytes"] += file_size
        with open(UPLOAD_TRACKER_FILE, "w") as f:
            json.dump(tracker, f)

        return True
    except Exception as e:
        logger.error(f"❌ Error checking daily upload limit: {e}")
        return False

def check_total_upload_limit(file_size):
    """
    Check if the total upload limit (combined for both buckets) has been exceeded.
    :param file_size: Size of the file to be uploaded (in bytes).
    :return: True if the upload can proceed, False if the limit is exceeded.
    """
    if os.getenv("DEBUG_MODE") == "1":
        logger.info("Skipping total storage check (debug mode)")
        return True

    try:
        # Use cached storage values
        total_storage_used = 0
        for bucket in [data_bucket]:
            total_storage_used += get_cached_bucket_storage(bucket)

        # Convert storage used to GB
        total_uploaded_gb = (total_storage_used + file_size) / (1024 * 1024 * 1024)
        if total_uploaded_gb > TOTAL_UPLOAD_LIMIT_GB:
            msg = f"Total limit exceeded: {total_uploaded_gb:.2f} GB / {TOTAL_UPLOAD_LIMIT_GB} GB"
            logger.warning(msg)
            send_discord_alert(msg)
            update_bucket_status(is_full=True)
            return False
        elif total_uploaded_gb > 0.9 * TOTAL_UPLOAD_LIMIT_GB:
            msg = f"⚠️ Total storage nearing limit: {total_uploaded_gb:.2f} GB / {TOTAL_UPLOAD_LIMIT_GB} GB"
            send_discord_alert(msg)

        return True
    except Exception as e:
        logger.error(f"❌ Error checking total upload limit: {e}")
        return False

def upload_file(data, remote_filename=None, bucket_type="data", is_binary=True):
    """
    Upload data to the specified Backblaze bucket.
    :param data: Data to upload (string or bytes).
    :param remote_filename: Name of the file in the bucket.
    :param bucket_type: "data" for data bucket (default).
    :param is_binary: Whether the data is binary (True) or text (False).
    """
    if not remote_filename:
        raise ValueError("remote_filename must be provided.")

    # Check the daily upload limit
    file_size = len(data)
    if not check_daily_upload_limit(file_size):
        logger.error(f"❌ Upload of {remote_filename} canceled due to daily upload limit.")
        return

    # Check the total upload limit
    if not check_total_upload_limit(file_size):
        logger.error(f"❌ Upload of {remote_filename} canceled due to total upload limit.")
        return

    # Select the appropriate bucket (always data_bucket since logs_bucket is removed)
    bucket = data_bucket

    # Upload the file using safe_upload
    safe_upload(bucket, data, remote_filename)
    logger.info(f"✅ Uploaded {remote_filename} ({file_size / (1024 * 1024):.2f} MB) to bucket {bucket.name}")

def update_bucket_status(is_full):
    """
    Update the bucket status in a file.
    :param is_full: True if the bucket is full, False otherwise.
    """
    status = {"bucket_full": is_full}
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
    logger.info(f"✅ Bucket status updated: {'full' if is_full else 'available'}")

def check_bucket_storage():
    """
    Check the storage usage of the data bucket.
    :return: Tuple of (storage_used, storage_limit).
    """
    try:
        # Ensure data_bucket is initialized
        bucket = data_bucket
        if not bucket:
            raise ValueError("data_bucket is not initialized. Check your B2_BUCKET_DATA environment variable.")

        # Calculate storage used
        storage_used = get_cached_bucket_storage(bucket)
        storage_limit = TOTAL_UPLOAD_LIMIT_GB * 1024 * 1024 * 1024  # Convert GB to bytes

        # Check if the bucket is full
        is_full = storage_used >= storage_limit
        update_bucket_status(is_full)

        logger.info(f"Storage used: {storage_used / (1024 ** 3):.2f} GB / {TOTAL_UPLOAD_LIMIT_GB} GB")
        return storage_used, storage_limit

    except Exception as e:
        logger.error(f"❌ Error checking bucket storage: {e}")
        return 0, TOTAL_UPLOAD_LIMIT_GB * 1024 * 1024 * 1024

def get_cached_bucket_storage(bucket, refresh_interval=600):
    """
    Get the cached storage usage for a bucket. Refresh the cache if it is older than the refresh interval.
    :param bucket: The bucket to check.
    :param refresh_interval: Time in seconds before refreshing the cache.
    :return: Total storage used in bytes.
    """
    bucket_name = bucket.name
    current_time = time.time()

    # Check if cache exists and is still valid
    if bucket_name in _bucket_storage_cache:
        last_updated = _bucket_cache_timestamps.get(bucket_name, 0)
        if current_time - last_updated < refresh_interval:
            return _bucket_storage_cache[bucket_name]

    # Calculate storage used
    storage_used = 0
    for file_version, _ in safe_ls(bucket):
        storage_used += file_version.size

    # Cache the result and update the timestamp
    _bucket_storage_cache[bucket_name] = storage_used
    _bucket_cache_timestamps[bucket_name] = current_time
    return storage_used

def list_bucket_if_not_cached(bucket, force_refresh=False):
    """
    List the bucket only if the storage is not cached or a refresh is forced.
    :param bucket: The bucket to list.
    :param force_refresh: Force a refresh of the bucket listing.
    :return: Cached storage or the result of safe_ls().
    """
    cached = _bucket_storage_cache.get(bucket.name)
    if not force_refresh and cached:
        return cached
    return get_cached_bucket_storage(bucket)

def check_and_pause_if_bucket_full():
    """
    Check the bucket status and pause if the bucket is full.
    """
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r") as f:
                status = json.load(f)
            if status.get("bucket_full"):
                logger.warning("⚠️ Buckets are full. Pausing scraper for 24 hours...")
                send_discord_alert("⚠️ **Buckets Full**: Pausing scraper for 24 hours.")
                time.sleep(24 * 60 * 60)  # Pause for 24 hours
    except FileNotFoundError:
        logger.info("ℹ️ Bucket status file not found. Assuming buckets are available.")
    except Exception as e:
        logger.error(f"❌ Error checking bucket status: {e}")

def check_file_exists(bucket, file_name):
    """
    Check if a specific file exists in the bucket.
    :param bucket: The bucket to check.
    :param file_name: The name of the file to check.
    :return: True if the file exists, False otherwise.
    """
    try:
        bucket.get_file_info_by_name(file_name)
        return True
    except Exception:
        return False

def local_log(message, level="INFO"):
    log_folder = "local_logs"
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, f"log_{datetime.utcnow().date()}.txt")
    with open(log_path, "a") as f:
        f.write(f"{datetime.utcnow()} [{level}] {message}\n")


def save_api_counter():
    with open("api_counter.json", "w") as f:
        json.dump(API_COUNTER, f)

def safe_ls(bucket):
    if API_COUNTER["class_c"] >= API_COUNTER["limit"]:
        send_discord_alert("🚨 B2 Class C limit hit. Scraper halted.")
        raise SystemExit("API limit reached.")
    API_COUNTER["class_c"] += 1
    save_api_counter()
    return bucket.ls(recursive=True)

def save_batched_parquet():
    batch_data = []
    ohlcv_buffer = {}
    for symbol, data in ohlcv_buffer.items():
        if not data:
            continue
        df = pd.DataFrame(data)
        df["symbol"] = symbol
        batch_data.append(df)

    if batch_data:
        combined_df = pd.concat(batch_data)
        buffer = BytesIO()  # Use BytesIO for binary data
        combined_df.to_parquet(buffer, index=False, compression="snappy")
        buffer.seek(0)

        # Generate a filename
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M")
        remote_filename = f"daily_{timestamp}_batch.parquet"

        # Upload the file
        upload_file(buffer.getvalue(), remote_filename, bucket_type="data")
        logger.info(f"✅ Uploaded batched Parquet file: {remote_filename}")

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def safe_upload(bucket, data, remote_filename):
    """
    Safely upload data to the bucket with retries on failure.
    :param bucket: The bucket to upload to.
    :param data: The data to upload.
    :param remote_filename: The name of the file in the bucket.
    """
    bucket.upload_bytes(data, remote_filename)

# Example usage
if __name__ == "__main__":
    # Example: Upload a test file
    test_data = b"Test data for upload"
    upload_file(test_data, remote_filename="test_file.txt", bucket_type="data")

