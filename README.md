![Banner](assets/banner.png)
[![Python](https://img.shields.io/badge/Python-3.12.11-blue?logo=python)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-EE4C2C?logo=pytorch)](https://pytorch.org/)![Made with ML](https://img.shields.io/badge/Made%20with-ML-blueviolet?logo=openai)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# 🤖 Portfolio Optimization with Deep Reinforcement Learning (v1.0)

This project explores the use of Deep Reinforcement Learning (DRL) to train autonomous agents for financial portfolio management. The goal is to create agents that can dynamically allocate capital across a diverse set of assets to maximize returns while managing risk.

This is **Version 1.0** of the project, which moves beyond initial exploration to a more robust and comparative study. Building on the foundation of v0.1, this version introduces:

* **Comparative Analysis:** We train and evaluate three state-of-the-art DRL algorithms: **Proximal Policy Optimization (PPO)**, **Soft Actor-Critic (SAC)**, and **Twin Delayed DDPG (TD3)**. This allows us to understand the different emergent strategies and trade-offs of each approach.
* **Robust Benchmarking:** Agents' performance is rigorously compared against a standard **Buy and Hold** baseline, using a comprehensive set of financial metrics including Total Return, CAGR, Sharpe Ratio, Sortino Ratio, and Max Drawdown.
* **Modular Codebase:** The project has been refactored into a clean, modular structure with separate scripts for data fetching, training, evaluation, and visualization, making it easier to understand, extend, and reproduce results.
* **In-Depth Analysis:** We delve into *why* certain agents perform better, visualizing their asset allocation strategies over time to uncover their "investment philosophy."

* **Deep RL & LLM Portfolio Manager (Web App):**  A key feature of v1.0 is the interactive web application built with **Gradio**. This dashboard bridges the gap between complex backend models and user-friendly analysis, allowing for live tracking, forward-looking strategy generation, and historical backtesting.
The dashboard integrates **Large Language Models (LLMs)**, specifically `Qwen/Qwen2.5-3B-Instruct`, to act as an AI Risk Analyst, providing textual justification and risk assessments for the RL agent's proposed strategies.

*You try the webapp here ->* [Gradio webapp](https://huggingface.co/spaces/DanielKiani/Portfolio-Optimization-with-Deep-Reinforcement-Learning)
> 🛑  *Due to the hardware constraints in the free tier of Huggingface i had to use a smaller LLM (`Qwen/Qwen2.5-1.5B-Instruct` instead of `Qwen/Qwen2.5-3B-Instruct`) even with this change the LLM analysis aspects will take a while to run so if you want to try the whole experience let the app run in the background and come back to it after a while.* 🛑
> 🛑 My agent was trained on the data from the start of 2015 untill the end of 2020 so if we were to use it to generate allocations for let's say tomorrow we would retrain the model on the new data. 🛑

---

## 📜 Table of Contents

1. [📊 The Data & Asset Selection](#-the-data--asset-selection)
2. [🎯 Benchmarking Against Baselines](#-benchmarking-against-baselines)
3. [🏆 Key Findings & The New Champion](#-key-findings--the-new-champion)
4. [🔬 The Research Journey: Why Simplicity Won](#-the-research-journey-why-simplicity-won)
5. [🖥️ Deep RL & LLM Portfolio Manager (Web App)](#️-deep-rl--llm-portfolio-manager-web-app)
6. [✅ Conclusion](#-conclusion)
7. [📂 Project Structure](#-project-structure)
8. [🚀 How to Run](#-how-to-run)
    * [Setup](#setup)
    * [Data Fetching](#data-fetching)
    * [Training](#training)
    * [Evaluation & Visualization](#evaluation--visualization)

---

## 📊 The Data & Asset Selection

The foundation of any financial machine learning project is the data. The primary source for daily closing price data of the portfolio assets is **Yahoo Finance**, accessed via the `yfinance` library.

To provide the agents with broader economic context beyond just price history, the observation space is augmented with key macroeconomic indicators sourced from **FRED (Federal Reserve Economic Data)**. These indicators include data points such as the CBOE Volatility Index (VIX), various Treasury bill yields, and inflation expectations. This allows the agents to learn strategies that adapt to different market regimes, such as high volatility or rising interest rate environments.

**Environment & Realistic Constraints:**
To ensure realistic simulation results, the trading environment incorporates transaction costs.
* **Transaction Cost:** A fee of **0.001%** is applied to the notional value of every trade (both buys and sells). This forces the agents to learn strategies that generate returns net of fees, discouraging excessive, unprofitable trading.

The portfolio itself consists of five assets, chosen to represent different asset classes and risk profiles, creating a challenging decision-making environment:


* **Growth Equities (AAPL, MSFT):** Represent the high-growth, high-volatility technology sector.
* **Market Index (SPY):** An ETF tracking the S&P 500, representing the broader US stock market.
* **Safe Haven (TLT):** An ETF for 20+ Year US Treasury Bonds, which often acts as a "risk-off" asset during stock market downturns.
* **Alternative Asset (BTC-USD):** Represents a non-traditional, extremely volatile asset class with high potential returns.

This diverse mix forces the agent to learn not just about individual asset price movements, but also about their correlations and how to balance risk across different economic conditions.

---

## 🎯 Benchmarking Against Baselines

To prove that a reinforcement learning agent is truly "intelligent," its performance must be measured against simple, standard strategies. An agent is only successful if it can provide value beyond a naive approach.

Our primary benchmark was the **Buy and Hold** strategy, where an equal amount of capital is invested in each asset at the beginning of the period and never touched again. The goal for any trained RL agent was to achieve superior performance, especially on a **risk-adjusted basis** (e.g., higher Sharpe Ratio, lower Max Drawdown), compared to this baseline.

The chart below shows the performance of a simple Buy and Hold strategy during the 2021-2023 test period, setting a clear target for our agents to beat.

![Baseline Performance Chart](results/baseline_results.png)

---

## 🏆 Key Findings & The New Champion

Our latest evaluation on out-of-sample data from **2021-2023** has yielded surprising and significant results, challenging our initial assumptions and highlighting the impact of neural network architecture on agent performance.

The **TD3 agent powered by a Transformer architecture** has emerged as the undisputed champion in terms of risk-adjusted returns and capital preservation, while the **SAC agent** demonstrated the highest absolute growth potential.

#### Final Performance Comparison (2021-2023)

This table summarizes the performance of our key agents against the Buy & Hold baseline.

| Metric | **TD3 (Transformer)** | SAC (MLP) | Buy & Hold | PPO (MLP) | TD3 (MLP) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Total Return** | 25.34% | **39.23%** | 32.76% | 22.85% | 22.07% |
| **CAGR** | 8.20% | **12.25%** | 9.96% | 7.45% | 7.21% |
| **Sharpe Ratio** | **0.61** | 0.56 | 0.59 | 0.41 | 0.42 |
| **Volatility** | **14.77%** | 27.47% | 19.06% | 25.90% | 23.00% |
| **Max Drawdown** | **-20.01%** | -29.08% | -28.82% | -44.26% | -40.50% |

![Main Performance Chart](results/final_performance_comparison_all_agents.png)
***note*: bitcoin was excluded from the performance comparison**

### 🥇 TD3 (Transformer): The Master of Risk Management

The most notable finding is the superior performance of the TD3 agent when equipped with a **Transformer-based policy network**. This agent achieved the best risk-adjusted metrics across the board.

* **Lowest Volatility (14.77%):** It provided a significantly smoother ride than even the passive Buy & Hold baseline.
* **Best Capital Preservation:** Its maximum drawdown of **-20.01%** was drastically lower than other agents and the baseline, proving its ability to protect capital during severe market downturns like the 2022 bear market.
* **Conclusion:** The Transformer's attention mechanism likely allowed the agent to better identify and react to long-term market shifts and regime changes, leading to a highly robust and defensive strategy.

### 🚀 SAC (MLP): The Aggressive Growth Engine

The **Soft Actor-Critic (SAC)** agent confirmed its role as the high-growth strategist.

* **Highest Returns:** It achieved the highest Total Return (**39.23%**) and CAGR (**12.25%**), outperforming the Buy & Hold baseline by a significant margin.
* **Higher Risk:** This performance came at the cost of the highest volatility (**27.47%**), making it a strategy suited for aggressive investors willing to tolerate larger price swings for maximum gain.

---

## 🧠 Comparative Analysis of Agent Strategies

A fascinating outcome of this project was observing how different combinations of RL algorithms and network architectures led to distinct investment philosophies. We can visualize this by looking at how each agent allocated its portfolio over time.

### TD3 (Transformer): The Dynamic Hedger

The Transformer-based TD3 agent did not learn a static allocation. Instead, it developed a sophisticated, **dynamic hedging strategy**. By leveraging the Transformer's attention mechanism to process the 30-day lookback window, the agent could identify market trends and adapt its portfolio accordingly.

![TD3 Transformer Allocation Chart](results/td3_transformer_allocation.png)

As shown in the chart, the agent maintains a core position in equities (AAPL, MSFT, SPY) but actively manages its exposure. During the volatile bear market of 2022, the agent significantly increased its allocation to the safe-haven asset **TLT (US Treasury Bonds)**, effectively "smoothing out" its equity curve and avoiding the deep losses suffered by the baseline. This ability to dynamically shift into defensive assets is the key to its superior risk-adjusted performance.

### SAC (MLP): The High-Conviction Aggressor

The SAC agent learned a strategy that is nearly the polar opposite of the Transformer. It converged to a **high-risk, high-return static allocation strategy**. Its portfolio is heavily weighted towards high-growth assets, likely with a substantial allocation to Bitcoin (BTC-USD) and tech stocks, with very little exposure to defensive assets like bonds or cash.

![SAC Allocation Chart](results/sac_allocation.png)

The allocation chart reveals a strategy with minimal changes over time, indicating a "set-and-forget" approach. While this high-conviction bet paid off with the highest total return, it also exposed the portfolio to significant volatility.

### PPO (MLP): The Failed Active Trader

Unlike the other MLP-based agents which converged to static allocations, the PPO agent attempted a **dynamic, active trading strategy**.

![PPO Allocation Chart](results/ppo_allocation.png)

As seen in the chart, the agent frequently rebalances its portfolio, shifting weights between equities, bonds, and cash. However, the performance metrics indicate that this activity was detrimental. With poor returns and the deepest maximum drawdown (-44.26%) among all agents, the PPO agent's attempts at market timing were unsuccessful, churning the portfolio without generating alpha or managing risk.

### TD3 (MLP): The Failed Static Allocator

The standard MLP version of the TD3 agent also converged to a static allocation, similar to the SAC agent, but chose a clearly suboptimal portfolio.

![TD3 MLP Allocation Chart](results/td3_allocation.png)

The chart shows a relatively fixed allocation that failed to perform well. Unlike the SAC agent, it did not capture high-growth opportunities, and unlike the Transformer agent, it lacked the dynamic capability to manage risk. This resulted in near-bottom performance across all metrics.

---

## 🔬 The Research Journey: Why Simplicity Won

This project was also an exercise in scientific methodology. We initially hypothesized that more complex models and features would yield better results.

* **Hypothesis 1: More features are better.** We tested adding technical indicators (RSI, MACD) to the observation space. **Result:** Performance degraded. The indicators acted as noise, confusing the agents.
* **Hypothesis 2: Models with memory are better.** We tested an LSTM-based agent (`RecurrentPPO`). **Result:** Performance degraded. The added complexity led to overfitting on the training data.
* **Hypothesis 3: Using Regularization is better.** We tested both L1 and L2 regularization. **Results:** Performance degraded.
* **Hypothesis 4: Increasing the window from 30 days is better.** We tested increasing the window to 60 days. **Results:** Performance degraded. increasing the context window is not always good and it could be seen as more noise for the model.
* **Hypothesis 5: A Transformer-based architecture is superior.** We replaced the standard Multi-Layer Perceptron (MLP) policy network with a more powerful Transformer model, hypothesizing its attention mechanism would better capture complex temporal relationships. **Result**: Performance degraded. Similar to the LSTM experiment, the Transformer model was too complex for the amount of data available. It suffered from significant overfitting, performing well on training data but failing to generalize to unseen market scenarios.

The conclusion was clear: a simple MLP (Multi-Layer Perceptron) policy network, fed with just normalized price data and a concise 30-day window, was the most effective and robust architecture.

---

## 🖥️ Deep RL & LLM Portfolio Manager (Web App)

A key feature of v1.0 is the interactive web application built with **Gradio**. This dashboard bridges the gap between complex backend models and user-friendly analysis, allowing for live tracking, forward-looking strategy generation, and historical backtesting.

The dashboard integrates **Large Language Models (LLMs)**, specifically Qwen, to act as an AI Risk Analyst, providing textual justification and risk assessments for the RL agent's proposed strategies.

*You try the webapp here ->* [Gradio webapp](https://huggingface.co/spaces/DanielKiani/Portfolio-Optimization-with-Deep-Reinforcement-Learning)
> 🛑  *Due to the hardware constraints in the free tier of Huggingface i had to use a smaller LLM (`Qwen/Qwen2.5-1.5B-Instruct` instead of `Qwen/Qwen2.5-3B-Instruct` ) even with this change the LLM analysis aspects will take a while to run so if you want to try the whole experience let the app run in the background and come back to it after a while.* 🛑
> 🛑 My agent was trained on the data from the start of 2015 untill the end of 2020 so if we were to use it to generate allocations for let's say tomorrow we would retrain the model on the new data. 🛑

### Key Features:

#### 1. Live Dashboard & Net Worth Tracking

Track the current portfolio holdings, recent transactions, and the overall net worth evolution in real-time.

![Live Dashboard](results/tab1.png)

#### 2. AI-Powered Strategy Forecast & Risk Analysis

Generate tomorrow's optimal portfolio allocation using the trained RL agents. The integrated LLM analyzes the proposed allocation, current market volatility (VIX), and asset concentration to provide a comprehensive **Risk Analyst Report** with a confidence score and justifications.

It also includes **Explainable AI (XAI)** feature importance plots to show which market factors most influenced the agent's decision.

![AI Forecast and Risk Analysis](results/tab2.png)

#### 3. AI-Driven Historical Market Analyst

Compare the historical performance of selected assets over customizable timeframes (e.g., 6 months, 5 years) with normalized relative return plots. The system generates an AI Analyst Report that summarizes key **trends**, **volatility**, and comparative performance for the selected period, providing narrative context to the raw data.

![AI-Driven Historical Market Analyst](results/tab3.png)

#### 4. Historical Simulation & Backtesting

Run dynamic backtests of the trained RL agents against baselines over any historical period. This tool is essential for validating performance across different market cycles.

![Historical Simulation](results/tab4.png)

---

## ✅ Conclusion

This project successfully demonstrates that Deep Reinforcement Learning can be a powerful tool for discovering sophisticated investment strategies. The key insight is that the most robust and successful agent did not learn to be a hyperactive trader, but rather a prudent strategic allocator, emphasizing the timeless investment principle that effective risk management is the true key to long-term success.

---

## 📂 Project Structure

```bash
├── assets/             # Images for the README
├── checkpoints/        # Stores trained model weights (.zip files)
├── data/               # Stores fetched CSV data files
├── results/            # Stores generated plots and metrics logs
├── scripts/            # Contains all the executable scripts
│   ├── app.py             # The Gradio web application
│   ├── check_env.py        # Simple script to verify the custom environment
│   ├── custom_policy.py    # Custom policy network definitions
│   ├── environment.py      # The custom Gymnasium environment class
│   ├── evaluate_baselines.py # Calculates performance of baseline strategies
│   ├── evaluate.py         # Main script to evaluate a trained agent
│   ├── fetch_market_data.py # Script to download historical data from YFinance
│   ├── llm_analysis_rag.py # Script for LLM-based analysis and RAG
│   ├── predict_tomorrow.py # Script to generate predictions for the next day
│   ├── stress_test.py      # Compares all agents on a specific dataset
│   ├── train.py            # Main script to train an RL agent
│   ├── tune_sac.py         # Script for hyperparameter tuning of the SAC agent
│   └── visualize_strategy.py # Plots the asset allocation of a trained agent
├── requirements.txt    # List of Python dependencies
└── README.md           # This file
```

## 🚀 How to Run

### Setup

1. Clone the repository:

```Bash

git clone https://github.com/DanielKiani/Portfolio-Optimization-with-Deep-Reinforcement-Learning
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Data Fetching

Before training or evaluation, you need to download the historical market data. Use the `fetch_market_data.py` script.

```Bash
# Fetch training data (e.g., 2015-2020)
python scripts/fetch_market_data.py --start 2015-01-01 --end 2020-12-31 --filename data/train_data.csv

# Fetch evaluation data (e.g., 2021-2023)
python scripts/fetch_market_data.py --start 2021-01-01 --end 2023-12-31 --filename data/eval_data.csv
```

### Training

Use the `train.py` script to train an agent. You can specify the algorithm (ppo, sac, or td3) and the number of training timesteps.

```Bash
# Train a TD3 agent (default timesteps: 20000)
python scripts/train.py --agent td3 --datafile data/train_data.csv

# Train a SAC agent for more timesteps
python scripts/train.py --agent sac --datafile data/train_data.csv --timesteps 50000
```

The trained model will be saved in the `checkpoints/` directory (e.g., `sac_portfolio_model.zip`).

### Evaluation & Visualization

Once you have trained models and evaluation data, you can use the other scripts to analyze performance.

* **Compare all agents** (`stress_test.py`): This script loads all available models in `checkpoints/` and compares them against the baseline on a given dataset.

```Bash
python scripts/stress_test.py --datafile data/eval_data.csv
```

This will generate `results/agent_performance_comparison`.png and print a metrics table.

* **Evaluate a single agent** (`evaluate.py`): This script calculates detailed metrics for a specific agent and plots its portfolio value.

```Bash
python scripts/evaluate.py --agent td3 --checkpoint checkpoints/td3_portfolio_model.zip --datafile data/eval_data.csv
```

* **Visualize an agent's strategy** (`visualize_strategy.py`): This script creates a stacked area chart showing how the agent's asset allocation changed over time.

```Bash
python scripts/visualize_strategy.py --agent ppo --checkpoint checkpoints/ppo_portfolio_model.zip --datafile data/eval_data.csv
```

This will save the plot to `results/ppo_portfolio_allocation.png`.
