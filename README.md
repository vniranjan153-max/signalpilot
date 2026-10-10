# AI Trade Signals

A free, serverless **AI signal dashboard** for stocks, indices, crypto and commodities.
A scheduled GitHub Action downloads market data, trains a small machine-learning model, validates it with a
**walk-forward backtest**, and publishes `signals.json`. A static web page on **GitHub Pages** shows
**BUY / SELL / HOLD**, the probability of a rise, and, importantly, an honest **model-quality grade**.

> **Disclaimer:** educational project, **not financial advice**. Signals are statistical estimates and can be wrong.
> Backtests do not predict future results. Trading can lose money. Do your own research.

```
Yahoo Finance ─► features (RSI, MACD, SMA, ATR, volatility…) ─► Gradient-boosting classifier
      ─► walk-forward out-of-sample predictions ─► long-only backtest (with costs)
      ─► site/data/signals.json ─► static dashboard (GitHub Pages)
```

## Quick start (about 5 minutes)

1. **Create a GitHub repo** and push this folder:
   ```bash
   git init && git add . && git commit -m "AI trade signals"
   git branch -M main
   git remote add origin https://github.com/<you>/ai-trade-signals.git
   git push -u origin main
   ```
2. In the repo go to **Settings → Pages → Build and deployment → Source: GitHub Actions**.
3. Open the **Actions** tab → **Update signals and deploy** → *Run workflow*.
4. Your site goes live at `https://<you>.github.io/ai-trade-signals/` and refreshes every weekday at 16:00 IST.

## Run locally

```bash
pip install -r requirements.txt
python -m pytest -q                 # tests (includes a no-look-ahead check)
python -m signals.run --demo        # synthetic data, no internet needed
python -m signals.run               # live data from Yahoo Finance
python -m http.server -d site 8000  # open http://localhost:8000
```

## Choose your markets

Edit `config.yaml`. Use Yahoo tickers: `RELIANCE.NS` (NSE), `^NSEI` (NIFTY 50), `AAPL`, `BTC-USD`, `GC=F`.
You can also change the prediction horizon, BUY/SELL thresholds and assumed trading costs there.

## How it works

| Step | File | Notes |
|------|------|-------|
| Data | `signals/data.py` | Yahoo Finance daily bars; synthetic generator for demo/tests |
| Features | `signals/features.py` | Returns, RSI, MACD, trend, ATR, volatility, volume. **Only past data** is used per row |
| Model | `signals/model.py` | Small `HistGradientBoostingClassifier`, retrained every 21 days; training stops `horizon` days before each test block (purged) |
| Backtest | `signals/backtest.py` | Long-only, costs per position change, Sharpe, drawdown, vs buy & hold |
| Output | `signals/run.py` | Writes `site/data/signals.json` |
| UI | `site/` | Plain HTML/CSS/JS, no build step, no API keys |

### Reading the signals honestly
- **P(up)** is the estimated chance the price is higher in N trading days. Around 50% means the model has no view.
- **Model quality** is the out-of-sample AUC. **0.50 is a coin flip.** Below ~0.52 the page greys the card as *No edge*;
  treat those signals as noise. Genuine edges in liquid markets are small and fade over time.
- The synthetic demo data is random, so the demo correctly shows **No edge** everywhere.

## Ideas to extend
- Add news/sentiment features, India VIX, FII/DII flows, or macro data.
- Try LightGBM, calibrated probabilities, or ensembles; compare with simple benchmarks (buy & hold, SMA crossover).
- Add position sizing and a Telegram bot that posts the daily signals.
- Add more validation: multiple time periods, transaction-cost stress tests, paper-trade before real money.

## Limitations
Yahoo data can be delayed, adjusted or missing; the backtest ignores slippage on thin markets, taxes and
exchange rules; daily models cannot see intraday moves. Results will differ from live trading.

## License
MIT
