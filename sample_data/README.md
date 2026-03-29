# Sample Data Files for Testing Octane CLI
#
# These files contain realistic fake data for testing CLI commands
# that read from files. Use them as-is or modify to test edge cases.
#
# Usage:
#   octane extract batch sample_data/extract_urls.txt
#   octane portfolio import sample_data/schwab_positions.csv --broker Schwab
#   octane portfolio crypto import sample_data/crypto_coinbase.csv --exchange Coinbase

## Files

| File | Purpose | Command |
|------|---------|---------|
| `extract_urls.txt` | 5 URLs (YouTube, arXiv, web) for batch extraction | `octane extract batch sample_data/extract_urls.txt` |
| `schwab_positions.csv` | 10 stock positions (AAPL, NVDA, MSFT, etc.) | `octane portfolio import sample_data/schwab_positions.csv --broker Schwab` |
| `crypto_coinbase.csv` | 10 crypto trades (BTC, ETH, SOL, ADA) | `octane portfolio crypto import sample_data/crypto_coinbase.csv --exchange Coinbase` |
