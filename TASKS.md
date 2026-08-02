# 📝 Project Task Log & Problem Solving
**Project:** NLP-Enhanced Intraday Volatility Prediction  
**Status:** ✅ Complete

This document tracks the core technical challenges addressed and solved during the development of this Capstone.

## 🟢 Phase 1: Data Infrastructure
| Task | Problem / Challenge | Solution Implemented |
| :--- | :--- | :--- |
| **Market Data Ingestion** | Need high-frequency (5-minute) data for S&P 500 without paying for Bloomberg. | Built a pipeline using `yfinance` to fetch recent 60-day intraday data for SPY. |
| **Target Variable Definition** | Volatility is latent (invisible); it cannot be downloaded directly. | **Engineered Feature:** Calculated Rolling Standard Deviation of Log Returns (Window=12) to create a "Realized Volatility" ground truth. |

## 🟡 Phase 2: Unstructured Data (NLP)
| Task | Problem / Challenge | Solution Implemented |
| :--- | :--- | :--- |
| **News Acquisition** | Historical news data is expensive and unstructured. | **Web Scraper:** Built a custom parser for FinViz using `BeautifulSoup` to extract headlines and timestamps. |
| **Sentiment Quantification** | Financial text is context-heavy; standard dictionaries (e.g., "Liability") misclassify financial terms. | **Deep Learning:** Deployed `ProsusAI/FinBERT`, a transformer model pre-trained on financial corpus, to score headlines (Pos/Neg/Neutral). |

## 🟠 Phase 3: Feature Engineering
| Task | Problem / Challenge | Solution Implemented |
| :--- | :--- | :--- |
| **Data Alignment** | News happens sporadically (irregular timestamps), while market data is continuous (5-min intervals). | **Resampling Algorithm:** Mapped news timestamps to the nearest future market candle (Forward Fill logic) to prevent look-ahead bias. |
| **Sequence Formatting** | LSTM models require 3D input tensors (Samples, Time Steps, Features). | **Sliding Window:** Created a function to transform 2D DataFrames into `(N, 12, 5)` tensors. |

## 🔴 Phase 4: Modeling & Evaluation
| Task | Problem / Challenge | Solution Implemented |
| :--- | :--- | :--- |
| **Model Selection** | Standard regression cannot capture temporal dependencies (memory) in volatility. | **LSTM Network:** Built a Recurrent Neural Network with PyTorch to learn time-dependent patterns. |
| **Overfitting Prevention** | Deep Learning models memorize noise in small financial datasets. | **Regularization:** Applied `Dropout=0.2` and Early Stopping logic via manual epoch monitoring. |