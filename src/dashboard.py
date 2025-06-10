import streamlit as st
import importlib
import os
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression

# Auto-refresh every 60 seconds
st_autorefresh(interval=60 * 1000, key="train_stats_refresh")

st.title("Money Printer Dashboard")

# --- TRAINER SECTION ---
st.header("Model Trainer")
if st.button("Run Trainer Now", key="run_trainer_btn"):
    with st.spinner("Training model..."):
        trainer = importlib.import_module("trainer")
        if hasattr(trainer, "train_once"):
            msgs = trainer.train_once()
            for msg in msgs:
                st.info(msg)
        else:
            st.warning("No train_once() function found. Running continuous_train_loop() for one cycle.")
            trainer.continuous_train_loop()
    st.success("Training complete!")

TRAINER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trainer"))
train_stats_path = os.path.join(TRAINER_DIR, "train_stats.json")
TRADING_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trading_data"))
signals_path = os.path.join(TRADING_DATA_DIR, "latest_signals.json")

if os.path.exists(train_stats_path):
    with open(train_stats_path, "r") as f:
        train_stats = json.load(f)
    st.metric("Rows Trained", train_stats.get("rows_trained", 0))
    st.write("Last Trained:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(train_stats.get("timestamp", 0))))
    if "test_results" in train_stats:
        st.subheader("Test Set Performance for Each Model")
        df_test = pd.DataFrame(train_stats["test_results"]).T
        st.dataframe(df_test.style.highlight_max(axis=0, color="lightgreen").format("{:.3f}"))
    else:
        st.info("No test set results available yet.")
else:
    st.info("No training stats available yet.")

# Show latest signals/calls with confidence filter
if os.path.exists(signals_path):
    with open(signals_path, "r") as f:
        signals = json.load(f)
    if signals:
        df_signals = pd.DataFrame(signals)
        if "probability" in df_signals.columns:
            st.subheader("Latest Model Signals (Test Set)")
            conf = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
            filtered = df_signals[df_signals["probability"] >= conf]
            st.write(f"Showing {len(filtered)} of {len(df_signals)} signals with probability ≥ {conf}")
            st.dataframe(filtered)
        else:
            st.dataframe(df_signals)
    else:
        st.info("No signals to display yet.")
else:
    st.info("No signals to display yet.")

# --- PAPER TRADER SECTION ---
st.header("Paper Trading")
if st.button("Run Paper Trading Simulation", key="paper_trade_btn"):
    with st.spinner("Simulating paper trades..."):
        paper_trader = importlib.import_module("paper_trader")
        paper_trader.simulate_paper_trading()
    st.success("Paper trading simulation complete!")

paper_stats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/paper_stats.json"))
if os.path.exists(paper_stats_path):
    with open(paper_stats_path, "r") as f:
        paper_stats = json.load(f)
    st.metric("Paper Winrate", f"{paper_stats.get('winrate', 0.0):.2%}")
    st.metric("Paper Portfolio Gain", f"${paper_stats.get('gain', 0.0):.2f}")
    st.metric("Paper Trading Balance", f"${paper_stats.get('final_balance', 10.0):.2f}")
else:
    st.info("No paper trading stats available yet.")

trade_history_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/trades/paper_trades.json"))
if os.path.exists(trade_history_path):
    with open(trade_history_path, "r") as f:
        trade_history = json.load(f)
    if trade_history:
        st.subheader("Paper Trading Transaction History")
        df_trades = pd.DataFrame(trade_history)
        st.dataframe(df_trades)
    else:
        st.info("No paper trading transactions yet.")
else:
    st.info("No paper trading transactions yet.")

# --- SCRAPER SECTION ---
st.header("Data Scraper")

# Scraping interval control (default 300 seconds)
if "scrape_interval" not in st.session_state:
    st.session_state["scrape_interval"] = 300

scrape_interval = st.number_input(
    "Scraping interval (seconds)", min_value=60, max_value=3600, value=st.session_state["scrape_interval"], step=10
)
st.session_state["scrape_interval"] = scrape_interval

if st.button("Run Scraper Once", key="run_scraper_btn"):
    with st.spinner("Scraping data..."):
        try:
            scraper = importlib.import_module("data_scraper")
            scraper.main()
            st.success("Scraping complete!")
        except Exception as e:
            st.error(f"Scraper error: {e}")

if st.button("Start Looping Scraper", key="loop_scraper_btn"):
    with st.spinner(f"Starting continuous scraping every {scrape_interval} seconds..."):
        import threading
        scraper = importlib.import_module("data_scraper")
        t = threading.Thread(target=scraper.continuous_scrape_loop, args=(scrape_interval,), daemon=True)
        t.start()
        st.success(f"Looping scraper started in background (interval: {scrape_interval}s)!")

# --- LIVE TRADER SECTION (if implemented) ---
st.header("Live Trading")
if st.button("Start Live Trading", key="live_trade_btn"):
    with st.spinner("Simulating live trades..."):
        try:
            live_trader = importlib.import_module("live_trader")
            live_trader.simulate_live_trading()
            st.success("Live trading simulation complete!")
        except Exception as e:
            st.error(f"Live trader error: {e}")

live_stats_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/live_stats.json"))
if os.path.exists(live_stats_path):
    with open(live_stats_path, "r") as f:
        live_stats = json.load(f)
    st.metric("Live Winrate", f"{live_stats.get('winrate', 0.0):.2%}")
    st.metric("Live Portfolio Gain", f"${live_stats.get('gain', 0.0):.2f}")
else:
    st.info("No live trading stats available yet.")

# --- MODEL DATA FILE STATUS ---
st.header("Model Data File Status")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/models"))
model_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_model_data.csv")]

status_rows = []
now = time.time()
for fname in model_files:
    fpath = os.path.join(DATA_DIR, fname)
    symbol = fname.replace("_model_data.csv", "")
    mtime = os.path.getmtime(fpath)
    age_hours = (now - mtime) / 3600
    status = "Fresh" if age_hours < 24 else "Stale"
    status_rows.append({
        "Symbol": symbol,
        "Last Modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "Age (hours)": f"{age_hours:.1f}",
        "Status": status
    })

if status_rows:
    df_status = pd.DataFrame(status_rows)
    st.dataframe(df_status)
else:
    st.info("No model data files found.")

MAPPING_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/coin_mapping.json"))
with open(MAPPING_PATH, "r") as f:
    mapping = json.load(f)
all_symbols = set(mapping.keys())
existing_symbols = set([f.replace("_model_data.csv", "") for f in model_files])
missing_symbols = all_symbols - existing_symbols

if missing_symbols:
    st.warning(f"Missing model data for: {', '.join(sorted(missing_symbols))}")
else:
    st.success("All mapped coins have model data files.")

# Additional information for debugging and tracking
if 'all_df' in locals():
    st.info(f"Data shape: {all_df.shape}")
    st.info("Training model...")
    st.success(f"Model trained. Avg accuracy: {avg_accuracy:.2%}")
    st.warning(f"Not enough data to train ({len(all_df)}/{MIN_ROWS_TO_TRAIN})")

old_files = [row for row in status_rows if float(row["Age (hours)"]) >= 24]
if old_files:
    st.warning("The following files are older than 24 hours and will be deleted by cleanup:")
    st.dataframe(pd.DataFrame(old_files))
else:
    st.success("No old model files to clean up.")

if "autotrain_paused" not in st.session_state:
    st.session_state["autotrain_paused"] = False

if st.button("Pause Auto-Train" if not st.session_state["autotrain_paused"] else "Resume Auto-Train"):
    st.session_state["autotrain_paused"] = not st.session_state["autotrain_paused"]
    st.experimental_rerun()

if st.session_state["autotrain_paused"]:
    st.warning("Auto-training is paused.")
else:
    st.success("Auto-training is active.")

fresh_cutoff = datetime.now() - timedelta(hours=24)
coin_fresh_rows = []
for fname in model_files:
    fpath = os.path.join(DATA_DIR, fname)
    df = pd.read_csv(fpath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        fresh_count = (df["timestamp"] >= fresh_cutoff).sum()
    else:
        fresh_count = 0
    coin_fresh_rows.append({"Symbol": fname.replace("_model_data.csv", ""), "Fresh Rows": fresh_count})

st.subheader("Fresh Rows per Coin (last 24h)")
st.dataframe(pd.DataFrame(coin_fresh_rows))

X, y, closes = prepare_data(df)
if 'RSI' in X.columns:
    X = X.drop(columns=['RSI'])  # Drop RSI if not critical
# Now drop rows with NaNs in X or y
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]