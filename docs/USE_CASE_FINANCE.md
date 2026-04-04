# Use Case: Financial Analysis

> **Status: Beta** — Portfolio import and display work. AI-driven analysis requires BodegaOS running. Live brokerage feeds are in development.

Octane as a private financial research terminal — portfolio analysis, risk assessment, and investment research, all local.

---

## Portfolio Import

Octane does not connect to your brokerage directly (yet). Import positions from a CSV export:

```bash
# Export positions CSV from your brokerage (Schwab, Fidelity, etc.)
# Then import:
octane portfolio import ~/Downloads/positions.csv

# Confirm import
octane portfolio show
```

Expected CSV format (Schwab-compatible):
```
Symbol,Description,Quantity,Price,Market Value,Day Change,...
NVDA,NVIDIA Corp,100,870.00,87000.00,+1200.00,...
AAPL,Apple Inc,50,185.00,9250.00,+75.00,...
```

---

## Viewing Your Portfolio

```bash
# Current holdings table
octane portfolio show

# With risk metrics
octane portfolio show --risk

# JSON output (for piping)
octane portfolio show --json
```

---

## Risk Analysis

```bash
# Full risk analysis: Sharpe ratio, volatility, drawdown, correlation
octane portfolio risk

# Risk for a specific ticker in your portfolio
octane portfolio risk --symbol NVDA
```

---

## Dividend Analysis

```bash
# View dividend income from holdings
octane portfolio dividends

# Projected annual income
octane portfolio dividends --annual
```

---

## Tax Lots

```bash
# Tax lot analysis — short vs long-term gains
octane portfolio tax-lots

# JSON output for tax preparation
octane portfolio tax-lots --json
```

---

## AI-Driven Research on Your Holdings

Combine portfolio data with Octane's research pipeline:

```bash
# Research a holding
octane investigate "NVDA competitive position and growth outlook" --deep 6 --cite

# Compare two holdings
octane compare "NVDA vs AMD for AI infrastructure exposure" --deep 4 --cite

# Research macro context
octane investigate "impact of AI chip export restrictions on semiconductor ETFs" --deep 6
```

---

## Live Market Watch

```bash
# Watch a ticker with live price updates
octane watch NVDA
octane watch AAPL --interval 15

# Multiple tickers
octane watch NVDA AAPL MSFT GOOGL

# With price alerts
octane watch NVDA --alert-above 900 --alert-below 800
```

---

## Mission Control — Portfolio Dashboard

```bash
octane ui start
```

Navigate to `http://localhost:44480` and open the Portfolio tab.

Available charts (beta):
- Holdings table with live values
- Sector allocation donut
- Net worth timeline
- Dividend bar chart
- Correlation heatmap
- Sector radar
- Source distribution

---

## Crypto Holdings (Beta)

```bash
octane portfolio crypto
octane portfolio crypto --json
```

---

## What's Next

- **Live brokerage feeds**: Direct Schwab API integration (planned May 2026)
- **Options tracking**: Call/put positions and Greeks
- **Tax export**: Direct export to TurboTax-compatible format
- **Alerts via iMessage**: Portfolio alerts sent to your iPhone (experimental)

---

## Important Notes

- **No cloud storage**: All portfolio data stays in your local Postgres database
- **No brokerage access needed**: Works entirely from CSV exports
- **Audit trail**: Every portfolio command is logged to the secure audit log
- **Encryption**: Sensitive data can be stored in the Touch ID vault (`octane vault`)
