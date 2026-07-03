# AOR Simulation Dashboard

This is a Streamlit dashboard for the Art of Retention model.

## What it does

The dashboard lets you input bookmaker-level averages:

- Number of bets
- Turnover
- GGR
- Active users

Then it runs at least 200 Monte Carlo simulations for a 30-day period.

The model distributes betting activity across users using a truncated Gaussian distribution, applies AOR funnel assumptions, estimates growth in bet count, average stake and hold, and calculates:

- Average freebets per active user
- Program cost
- Gross incremental GGR
- Net incremental GGR
- ROI
- Break-even share across simulations

## Run locally

```bash
pip install -r requirements.txt
streamlit run aor_retention_dashboard.py
```

## Important model assumption

For this version, the exchange rate is:

```text
1 Coin = 1 freebet
```

So if a completed mission gives 2 Coins, the model treats that as 2 freebets.
The monetary cost is controlled separately via:

```text
Value of 1 freebet, €
Redemption rate
Freebet economic cost factor
```
