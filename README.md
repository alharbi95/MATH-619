# A CNN-LSTM Hybrid Model for Multivariate Stock Price Forecasting Using Technical and Sentiment Indicators

> Deep Learning for Directional Forecasting in Bear Market Conditions

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Keras](https://img.shields.io/badge/Keras-3.13-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

**MATH-619 Capstone Project**
King Fahd University of Petroleum & Minerals — Mathematics Department
Master of Science in Data Science & Analytics

---

## Overview

This project develops and evaluates a **CNN-LSTM hybrid model with temporal self-attention** for next-day stock price direction forecasting (UP / DOWN). All evaluation is conducted on a genuine **bear-market test window** (September 2021 – September 2022) corresponding to the Federal Reserve's most aggressive rate-hike cycle since 1980 — among the hardest realistic test regimes in published deep learning finance research.

**Key Result:** **53.35% directional accuracy** on the bear-market test set, achieved by CNN-LSTM with temporal self-attention on the sentiment-enriched feature set (C3) — matching the Fischer & Krauss (2018) benchmark of 53.20% under significantly harder evaluation conditions.

---

## Headline Findings

| Finding | Result |
|---|---|
| Best accuracy on 2022 bear market | **53.35%** |
| Improvement from attention + sentiment combination | **+2.37 pp** |
| Total models trained | **84** |
| Test sequences (out-of-sample) | **231** |
| Total parameters | **29,921** |

### Three Insights

1. **Sentiment alone does not help.** Adding sentiment to plain CNN-LSTM dropped accuracy from 51.25% → 50.98%. By itself, sentiment is noise.

2. **Sentiment + Attention is the winning combination.** When attention is added together with sentiment, accuracy jumps to 53.35% — a +2.37 pp gain. Attention is what unlocks sentiment's value.

3. **Attention is conditional, not universal.** With technical features only, attention **hurts** by −2.07 pp. With sentiment added, attention **helps** by +2.37 pp. Attention earns its parameters only when the feature set is heterogeneous.

---

## Architecture

    Input (20 days × 16 features) → CNN → Temporal Attention → LSTM → Output (UP / DOWN)

- **CNN block:** Conv1D · 32 filters · kernel size 3 · local pattern extraction
- **Temporal Self-Attention:** weights informative timesteps (placed between CNN and LSTM with residual connection)
- **LSTM block:** 64 units · sequential memory
- **Output:** Sigmoid · binary classification (UP / DOWN)

Total parameters: **29,921** — kept deliberately small to control overfitting on the limited financial data.

---

## Data

| Source | Type | Coverage | Notes |
|---|---|---|---|
| TradingView | Technical (price, volume, indicators) | Jan 2015 – Sep 2022 | 5 stocks × 2,724 trading days |
| Kaggle (Equinox 2023) | Sentiment (news + social) | Sep 2021 – Sep 2022 | 252 days |

### Stocks Analyzed
AAPL · AMZN · GOOG · MSFT · TSLA (large-cap U.S. tech)

### Feature Conditions

| Tag | Name | Features | Count |
|---|---|---|---|
| C1 | Baseline | OHLCV: Open, High, Low, Close, Volume | 5 |
| C2 | + Technical | + EMA(20, 50), RSI, MACD, Bollinger Bands, VIX | 16 |
| C3 | + Sentiment | C2 + daily sentiment score | 17 |

---

## Results

### Mean Accuracy (%) on Bear-Market Test Set

| Architecture | C1 (OHLCV) | C2 (+ Technical) | C3 (+ Sentiment) |
|---|---|---|---|
| CNN-only | 50.74 | 49.87 | 50.43 |
| LSTM-only | 49.35 | 48.48 | 50.87 |
| CNN-LSTM | 51.08 | 51.25 | 50.98 |
| **CNN-LSTM + Attention** | 50.65 | 49.18 | **53.35 ★** |

### Effect of Attention

| Condition | Δ Accuracy | Interpretation |
|---|---|---|
| C2 (Technical only) | **−2.07 pp** (51.25 → 49.18) | Attention HURTS |
| C3 (with Sentiment) | **+2.37 pp** (50.98 → 53.35) | Attention HELPS |

### Benchmark Comparison

| Study | Test Regime | Accuracy |
|---|---|---|
| Fischer & Krauss (2018) | S&P 500, mixed | 53.20% |
| Lu et al. (2020) | Multi-stock, bull | 55.90% |
| **This Study** | **2022 Bear Market** | **53.35%** |

---

## Methodology

1. **Collect & align data** from TradingView and Kaggle
2. **Feature engineering & normalization** (16/17 features per condition)
3. **Sequence construction** (20-day sliding windows)
4. **Train models** with walk-forward cross-validation (4 expanding folds, COVID 2020 fold excluded)
5. **Evaluate on test set** (Sep 2021 – Sep 2022, 231 sequences, bear market)
6. **Interpret results** with Integrated Gradients

**Time Splits:**
- Training: Jan 2015 – May 2020
- Validation: May 2020 – Sep 2021
- Test: Sep 2021 – Sep 2022 (bear market, n = 231)

**Target:** binary direction — `1 = UP` if Close[t+1] > Close[t], else `0 = DOWN`

---

## Explainable AI (XAI)

Integrated Gradients reveals strong **temporal recency bias**: ~60% of the model's attribution falls on the last 5 trading days, with **t−1 and t−2 dominating** individual timestep importance.

**Strongest features:** RSI and MACD (momentum oscillators) — not raw prices.
**Older days (t−10 and earlier):** contribute almost nothing.

**Implication:** the model functions as a **short-horizon momentum detector**, best deployed for overnight strategies — not long-term allocation.

---

## Limitations

- **Sentiment data quality and cost.** The free Kaggle sentiment dataset has serious quality issues (AMZN/MSFT identical r = 1.00 forcing MSFT exclusion from C3, TSLA near-constant σ ≈ 0.057, GOOG with 14 missing values). Professional-grade alternatives (Bloomberg, Refinitiv, RavenPack) were beyond the academic budget.
- **Single evaluation window.** One 252-day bear-market window cannot tell us how the model behaves in bull, correction, or recovery regimes.
- **Universe scope.** Five large-cap U.S. tech stocks; findings may not transfer to mid-caps, international equities, or other sectors.

---

## Future Work

- Acquire **professional-grade sentiment** covering the full 2015–2022 training period.
- **Multi-regime evaluation** across bull, correction, bear, and recovery windows.
- Compare **richer attention mechanisms** — multi-head, cross-attention, full Transformer architectures.

---

## Repository Contents

    .
    ├── Code.ipynb                     # Main Jupyter notebook with full pipeline
    ├── Code.py                        # Python script export
    ├── Math619_g202415480_.pdf        # Full project report
    ├── poster/                        # A0 academic poster (PDF + PowerPoint)
    ├── data/                          # Input datasets (technical + sentiment)
    └── README.md                      # This file

---

## Reproduction

The pipeline uses **Python 3.10**, **TensorFlow 2.19**, **Keras 3.13**, **scikit-learn**, and **XGBoost**. All randomness is controlled by a fixed seed (seed=42).

    # Open the notebook
    jupyter notebook Code.ipynb

    # Or run as a script
    python Code.py

For full reproduction details, training configurations, and ablation studies, see the project report (Math619_g202415480_.pdf).

---

## Author

- **Abdullah Hussain Alharbi**

- Student ID: g202415480
- Master of Science in Data Science & Analytics
- King Fahd University of Petroleum & Minerals

### Supervision
- **Advisor:** Dr. Mousa Ahmad Al-Bashrawi
- **Co-advisor:** Mr. Mohammed Agbawi

---

## Citation

If this work is useful in your research, please cite:

    @misc{alharbi2025cnnlstm,
      author = {Alharbi, Abdullah Hussain},
      title  = {A CNN-LSTM Hybrid Model for Multivariate Stock Price Forecasting Using Technical and Sentiment Indicators},
      year   = {2025},
      school = {King Fahd University of Petroleum and Minerals},
      note   = {MATH-619 Capstone Project},
      url    = {https://github.com/alharbi95/MATH-619}
    }

---

## License

This project is released for academic and educational purposes only.

---

> *"Even small predictive edges can generate significant financial value."*
