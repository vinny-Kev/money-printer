"""
Quick test of the production data scraper
"""
from src.data_collector.production_data_scraper import ProductionDataScraper

# Test the scraper
scraper = ProductionDataScraper(
    symbols=['BTCUSDT'], 
    intervals=['1m'], 
    memory_only=True
)

# Run for 1 minute
results = scraper.run_timed_collection(hours=0.02, cycle_interval_minutes=0.5)

print(f"✅ Results: {results['total_files_saved']} files, {results['total_rows_collected']} rows")
print(f"🔗 Binance working: {scraper.binance is not None}")
