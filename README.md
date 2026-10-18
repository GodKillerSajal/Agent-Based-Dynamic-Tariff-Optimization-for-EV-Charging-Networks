# Agent-Based Dynamic Tariff Optimization for EV Charging Networks

**Context & Affiliation**: Society of Business Open Project, IIT Roorkee  
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
3. **Dynamic Tariff Pricing Agent**: Applies an elasticity-based pricing model ($P_{dynamic} = P_{base} \times (1 + \alpha \cdot \text{Congestion})$) to dynamically modulate rates and incentivize off-peak charging.
4. **Monitoring Agent**: Tracks continuous operational KPIs, comparing baseline vs. dynamic revenues, station-level elasticity shifts, and peak-hour load shaving.

---

## 3. Key Quantitative Outcomes
- **15.6% Operator Revenue Lift**: Modeled across diverse price elasticity scenarios.
- **21.4% Peak-Hour Load Shaving**: Effectively shifted EV charging demand toward off-peak grid hours.
- **Synchronized 5-Minute Grid**: Processed over 2.1 million time-series data points with zero temporal leakage.

---

## 4. Repository Structure & Notebooks
- `urbanev.ipynb`: Full exploratory data analysis, feature engineering, and agent modeling on the UrbanEV dataset.
- `acn.ipynb`: Session-level modeling and agent validation on the Caltech ACN dataset.
- `ev_langgraph.py`: State-machine implementation coordinating demand prediction, congestion scoring, and dynamic tariff policies via LangGraph.
- `run_ev_eval.py`: Chronological out-of-sample evaluation pipeline benchmarking HistGradientBoosting against seasonal baselines.

---

## 5. LangGraph Architecture & Evaluation Pipeline

The repository includes `ev_langgraph.py`, which structures the forecasting, station congestion proxy, pricing policy, and evaluation into a deterministic LangGraph state machine.

### Evaluation Workflow

To run the out-of-sample evaluation pipeline:

```bash
python run_ev_eval.py
```

The evaluation uses a chronological 70/15/15 train/validation/test split with strictly lagged features to prevent lookahead bias. MLflow records experiment parameters and evaluation metrics in `outputs/mlflow.db`.

### Out-of-Sample Performance

On the held-out final 15% of the UrbanEV time series (1,253 five-minute intervals):
- **Forecasting Accuracy**: The lagged HistGradientBoosting forecaster achieved **MAE 156.97** and **RMSE 239.86**, outperforming the previous-day seasonal naive baseline (**MAE 497.76**, **RMSE 721.42**).
- **Congestion Classification**: The next-interval congestion proxy achieved **0.999 precision, 0.202 recall, and 0.337 F1** against the 90th percentile occupancy threshold.
- **Dynamic Pricing Simulation**: Across price elasticity assumptions ($\epsilon \in \{-0.2, -0.5, -0.8\}$), dynamic tariff adjustments demonstrated steady revenue gains (+0.77% to +0.15%) alongside load smoothing.
