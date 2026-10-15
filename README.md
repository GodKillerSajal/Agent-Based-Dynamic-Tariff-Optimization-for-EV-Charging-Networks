# Agent-Based Dynamic Tariff Optimization for EV Charging Networks

**Context & Affiliation**: Society of Business Open Project 2026, IIT Roorkee  
**Domain**: Agentic AI, Time-Series Forecasting, Smart Grid Operations & Dynamic Pricing  
**Datasets Used**: 
- **UrbanEV (Shenzhen)**: 8,640 timestamps at 5-minute intervals across 248 charging stations (~2.14M records of occupancy, volume, duration, price).
- **Caltech Adaptive Charging Network (ACN)**: Session-level EV transaction logs.

---

## 1. Problem Statement
Fixed EV charging tariffs fail to balance fluctuating real-time grid demand, causing acute peak-hour charger congestion, long driver wait times, and suboptimal station revenue. This project develops an autonomous, multi-agent AI system that dynamically adjusts tariffs to shift charging loads away from peak periods while maximizing charging station revenue.

---

## 2. Multi-Agent Architecture
The system is structured as four cooperating autonomous agents:
1. **Demand Prediction Agent**: Ingests synchronized 5-minute interval metrics, calculates temporal lags and rolling occupancy indicators, and forecasts multi-station charging demand. Outperforms baseline SARIMA models in inference speed and RMSE.
2. **Congestion Prediction Agent**: Evaluates real-time charger utilization, queuing likelihood, and dwell-time distributions to identify bottleneck stations.
3. **Dynamic Tariff Pricing Agent**: Applies an elasticity-based pricing model ($P_{dynamic} = P_{base} 	imes (1 + lpha \cdot 	ext{Congestion})$) to dynamically modulate rates and incentivize off-peak charging.
4. **Monitoring Agent**: Tracks continuous operational KPIs, comparing baseline vs. dynamic revenues, station-level elasticity shifts, and peak-hour load shaving.

---

## 3. Key Quantitative Outcomes
- **15.6% Operator Revenue Lift**: Modeled across diverse price elasticity scenarios.
- **21.4% Peak-Hour Load Shaving**: Effectively shifted EV charging demand toward off-peak grid hours.
- **Synchronized 5-Minute Grid**: Processed over 2.1 million time-series data points with zero temporal leakage.

---

## 4. Key Files in this Folder
- `urbanev.ipynb`: Full exploratory data analysis, feature engineering, and agent modeling on the UrbanEV dataset.
- `acn.ipynb`: Session-level modeling and agent validation on the Caltech ACN dataset.

---

## 5. IIT Roorkee Resume Bullets (Strict 1-Line Format)
```text
• Designed a 4-agent collaborative AI system to forecast charging demand and optimize dynamic grid tariffs in real time.
• Processed 2.1M+ records across 248 stations (UrbanEV/ACN) on a 5-min grid, engineering lag & load metrics.
• Simulated dynamic pricing policies, achieving a 15.6% operator revenue lift and 21.4% peak load shaving.
```

## LangGraph working copy and measured evaluation

The workspace copy adds `ev_langgraph.py`, which wires the forecasting, station congestion proxy, pricing policy, and evaluation into a deterministic LangGraph state machine. The graph uses the available UrbanEV CSVs; it does not claim that the 2026 resume scenario figures are causal results.

Run from the `portfolio_projects` root after installing its shared environment:

```powershell
portfolio_projects\.venv\Scripts\python.exe portfolio_projects\expanded\ev-dynamic-tariff-langgraph\run_ev_eval.py
```

The run uses a chronological 70/15/15 split. The feature builder only uses values through the previous five-minute interval; the model is compared with a previous-day seasonal-naive forecast. MLflow records forecast and scenario metrics in `outputs/mlflow.db`, and the fitted forecaster is saved under `outputs/`.

### Measured working-copy evaluation

On the held-out final 15% of the available UrbanEV time window (1,253 five-minute intervals), the lagged HistGradientBoosting forecaster achieved **MAE 156.97** and **RMSE 239.86**, versus **MAE 497.76** and **RMSE 721.42** for the previous-day seasonal-naive baseline. The next-interval congestion proxy reached **0.999 precision, 0.202 recall, and 0.337 F1** against the train-only station occupancy p90 label. Its high precision comes with low recall, so this is not a reliable operational alert model yet.

Pricing scenarios estimated revenue changes of **+0.77%, +0.46%, and +0.15%** under assumed elasticities of -0.2, -0.5, and -0.8. These are counterfactual calculations, not observed revenue or causal effects. Existing resume figures (15.6% revenue lift and 21.4% peak-load shaving) are separate modeled results from the original project material and were not reproduced by this evaluation.

The UrbanEV files do not provide station capacities, queue labels, or observed responses to price changes, so the congestion score is a percentile proxy and the revenue/load changes are not causal estimates. Full metric details are in `outputs/evaluation.json`.
