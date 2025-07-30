# eth-anomaly-detec
# A Multi-Model Quantitative Analysis of the Ethereum Volatility Surface for Anomaly Detection

This repository contains the code and analysis for a comprehensive quantitative finance project focused on the Ethereum options market. The project demonstrates an end-to-end workflow: from ingesting and cleaning raw derivatives data to calibrating advanced financial models, identifying market anomalies, and quantifying model risk.

## Table of Contents
1.  [Project Objective](#project-objective)
2.  [Data Sources](#data-sources)
3.  [Quantitative Methodology](#quantitative-methodology)
4.  [Results and Key Findings](#results-and-key-findings)
    - [4.1. SVI Model Calibration and 3D Surface Construction](#41-svi-model-calibration-and-3d-surface-construction)
    - [4.2. Term Structure Analysis](#42-term-structure-analysis)
    - [4.3. The Volatility Risk Premium (VRP) Anomaly](#43-the-volatility-risk-premium-vrp-anomaly)
    - [4.4. Model Validation via Vega Comparison](#44-model-validation-via-vega-comparison)
    - [4.5. Heston Model Diagnostic](#45-heston-model-diagnostic)
5.  [Conclusion](#conclusion)
6.  [How to Run](#how-to-run)

---

## Project Objective

The goal of this project is to develop a robust framework for analyzing the Ethereum implied volatility (IV) surface to extract forward-looking risk signals and identify anomalous market structures. This is achieved through a multi-stage process that includes:
-   Constructing a smooth, arbitrage-free IV surface from raw market data.
-   Analyzing the term structures of volatility and skew to interpret market sentiment.
-   Quantifying the Volatility Risk Premium (VRP) by comparing implied vs. realized volatility.
-   Validating the model by analyzing option sensitivities (Greeks).
-   Stress-testing a dynamic stochastic volatility model (Heston) against the observed market state.

---

## Data Sources

The analysis relies on a rich, multimodal dataset:
-   **Derivatives Data:** A high-frequency snapshot of the entire ETH options order book (`ETH_options_data-*.csv`).
-   **Historical Price Data:** Daily OHLC data for ETH/USD (`ETH_day.csv`) for calculating realized volatility.
-   **Contextual Data:** Time-series data for on-chain volume (`volume-24h.csv`) and social media sentiment (`reddit_posts_sentiment_formal.csv`) were used in the initial exploratory phase to define the project's scope.

---

## Quantitative Methodology

1.  **SVI (Stochastic Volatility Inspired) Model:** Used to parameterize and fit a smooth, arbitrage-free curve to the discrete IV quotes for each expiration. Calibration is performed via a bounded non-linear optimization (L-BFGS-B).
2.  **Garman-Klass Volatility Estimator:** A high-efficiency method used to calculate historical realized volatility from daily OHLC data.
3.  **Heston Stochastic Volatility Model:** A dynamic, SDE-based model calibrated to the market via Fourier inversion pricing and a global optimizer (SLSQP) to diagnose the market's underlying risk dynamics.

---

## Results and Key Findings

### 4.1. SVI Model Calibration and 3D Surface Construction

The SVI model was successfully calibrated to the raw market data, transforming noisy, discrete quotes into a continuous and theoretically sound surface.


**Figure 1: SVI Smile Fit**
<img width="1014" height="630" alt="output" src="https://github.com/user-attachments/assets/8c005bd2-31b3-4847-ab85-d8fa5e3689ec" />
The model demonstrates a high-fidelity fit to the market data for a single expiration, accurately capturing the "volatility smile" phenomenon.



**Figure 2: 3D Implied Volatility Surface**
<img width="893" height="739" alt="output4" src="https://github.com/user-attachments/assets/c48a0aad-2893-4823-942e-49371b863066" />
By interpolating the fitted SVI parameters across all expirations, we construct the complete 3D implied volatility surface.


### 4.2. Term Structure Analysis

Deconstructing the 3D surface reveals key insights into the market's expectation of risk over time.

**Figure 3: Term Structures of Volatility and Skew**
<img width="1589" height="590" alt="output1" src="https://github.com/user-attachments/assets/fe4e7dce-97d5-46eb-919d-09eb2e82c2b6" />


-   **Volatility Backwardation (Left Plot):** The term structure is steeply inverted, with short-term IV (>70%) far exceeding long-term IV (~64%). This is an anomalous market state signaling acute, near-term fear or uncertainty.
-   **Negative Skew (Right Plot):** The SVI skew parameter (`ρ`) is strongly negative, indicating high demand for downside protection (puts) and confirming a significant "crash fear" priced into the market.

### 4.3. The Volatility Risk Premium (VRP) Anomaly

The project's key finding comes from comparing the market's *expectation* of volatility (IV) with historical *reality* (RV). The analysis was contextualized to the period following the "Black Thursday" crash of 2020.

**Figure 4: Volatility Risk Premium (VRP) Analysis**
<img width="1009" height="630" alt="output3" src="https://github.com/user-attachments/assets/1cad5ba7-f8af-4086-9638-fa1e80e435bd" />

The analysis uncovered a **profoundly negative Volatility Risk Premium averaging -42.26 percentage points**. The realized volatility (green line) from the recent past was dramatically higher than the implied volatility (blue line) being priced for the future. This rare anomaly signifies a market regime of extreme complacency or disbelief, where options were trading at a deep discount to recently observed risk.

### 4.4. Model Validation via Vega Comparison

To validate the SVI model, its implied Vega was compared against the exchange's benchmark.

**Figure 5: Vega Comparison**
<img width="841" height="547" alt="output2" src="https://github.com/user-attachments/assets/642833d7-1ca1-483a-ab48-8f43956da006" />


The results show a strong linear agreement, validating the SVI model's consistency. However, a slight, systematic bias was detected, quantitatively demonstrating the concept of **model risk** and its implications for practical hedging strategies.

### 4.5. Heston Model Diagnostic

A Heston stochastic volatility model was calibrated to the data. The optimizer pushed all key parameters to their absolute boundaries, and the final pricing error remained high. This served as a powerful diagnostic, concluding that the market state was so anomalous that it **stressed the descriptive capacity of a standard dynamic volatility model.**

---

## Conclusion

This project successfully executed an end-to-end quantitative workflow, moving from raw data to sophisticated model calibration and conclusive, data-driven insights. It identified a multi-faceted anomaly in the Ethereum market, characterized by backwardation, negative skew, and a profoundly negative Volatility Risk Premium. The analysis confirms a rare market regime of post-crash complacency and demonstrates the limitations of standard financial models under stressed conditions.

---

## How to Run

1.  **Environment Setup:** The project was developed in a Paperspace environment with an A6000 GPU. Key libraries include `pandas`, `numpy`, `scipy`, and `matplotlib`.
2.  **Data:** Place all dataset CSV files into the `/notebooks/datasets/` directory.
3.  **Execution:** The analysis is contained within a series of Jupyter Notebooks. Run the data preparation and `final_df` creation cell first, followed by the main analysis cell which performs the SVI fitting and generates all results.
