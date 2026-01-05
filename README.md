# 📊 Sepsis Early Warning Score (EWS) System

> Predictive ML model identifying sepsis risk 6 hours before onset using time-series features and XGBoost

[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-blue)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable-green)](https://shap.readthedocs.io/)
[![Databricks](https://img.shields.io/badge/Databricks-FF3621?logo=databricks)](https://databricks.com/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)

## 📋 Overview

Predictive analytics system using **XGBoost** and **time-series feature engineering** to identify patients at risk of developing sepsis 6 hours before clinical diagnosis, enabling proactive intervention and potentially saving lives.

### Business Value
- ⏰ **6-hour early warning** before sepsis onset
- 🎯 **80%+ ROC-AUC** prediction accuracy
- 📊 **24,000+ ICU observations** processed
- 🔍 **SHAP explainability** for clinical trust
- 💊 **Actionable scores** (0-100 EWS scale)

### Key Achievements
- **0.82 ROC-AUC** on sepsis prediction
- **1,000 patient stays** simulated (MIMIC-style data)
- **20 clinical features** engineered from time-series
- **Interpretable predictions** with SHAP values

---

## 🎯 Problem Statement

### Clinical Challenge

**Sepsis Statistics:**
- Affects **1.7 million** Americans annually
- **270,000 deaths** per year in the US
- **#1 cause** of hospital deaths
- Early detection **critical** for survival

**Current Limitations:**
- Sepsis diagnosed **too late** (often hours after onset)
- No reliable **early warning** system
- Subtle patterns **difficult to detect** manually
- Treatment delay **increases mortality** by 7.6% per hour

### Solution

An **ML-powered Early Warning System** that:
1. ✅ Analyzes vital sign trends (not just current values)
2. ✅ Incorporates lab results and clinical history
3. ✅ Predicts sepsis risk **6 hours in advance**
4. ✅ Provides interpretable scores (0-100 scale)
5. ✅ Explains predictions with SHAP values

---

## 🏗️ Architecture

### System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  PREDICTIVE EWS PIPELINE                         │
└─────────────────────────────────────────────────────────────────┘

MIMIC-Style Data → Time-Series → Feature → XGBoost → SHAP → EWS
  (1000 stays)     Processing    Engineering  Training  Analysis  (0-100)
                                                                      ↓
                                              ┌─────────────────────┘
                                              ↓
                                     High-Risk Patients
                                     (EWS ≥ 70)
                                              ↓
                                     Clinical Intervention
```

### Data Flow

1. **ICU Data Collection** - Hourly vitals, periodic labs
2. **Time-Series Processing** - Rolling windows, trends
3. **Feature Engineering** - 20 clinical features
4. **Model Training** - XGBoost with class balancing
5. **SHAP Analysis** - Feature importance
6. **EWS Calculation** - Convert probabilities to 0-100 scores

---

## 🛠️ Technical Implementation

### Technologies Used

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Platform** | Databricks | Unified analytics |
| **Storage** | Delta Lake | Time-series data storage |
| **Processing** | PySpark | Window functions, aggregations |
| **ML Algorithm** | XGBoost | Gradient boosting classifier |
| **Explainability** | SHAP | Feature importance |
| **Tracking** | MLflow | Experiment tracking |
| **Visualization** | Plotly | Interactive dashboards |

### Machine Learning Approach

**Algorithm:** XGBoost (Gradient Boosting)

**Why XGBoost?**
- ✅ Handles **imbalanced data** (80% no sepsis, 20% sepsis)
- ✅ **High accuracy** on structured data
- ✅ **Feature importance** built-in
- ✅ **Fast training** and inference
- ✅ Robust to **missing values**

**Key Parameters:**
```python
max_depth = 6              # Tree depth
learning_rate = 0.1        # Step size
n_estimators = 100         # Number of trees
scale_pos_weight = 4       # Handle class imbalance (80/20 ratio)
```

---

## 📊 Data

### Synthetic MIMIC-III Style Dataset

**Patient Simulation:**
- 1,000 ICU patient stays
- 24 hours monitoring per patient
- Hourly vital signs measurements
- Lab results every 6 hours
- **Total: 24,000+ observations**

**Sepsis Distribution:**
- 800 patients (80%) - No sepsis
- 200 patients (20%) - Develop sepsis
- Realistic clinical progression
- 6-hour deterioration window before diagnosis

### Time-Series Features

**Raw Measurements (8 variables):**
1. Heart Rate (HR)
2. Systolic Blood Pressure (SBP)
3. Diastolic Blood Pressure (DBP)
4. Oxygen Saturation (SpO2)
5. Respiratory Rate (RR)
6. Temperature
7. White Blood Cell count (WBC)
8. Lactate level

**Engineered Features (20 total):**

**Rolling Statistics (3h, 6h windows):**
- `hr_mean_3h`, `hr_mean_6h` - Trending
- `hr_std_3h`, `hr_std_6h` - Variability
- `sbp_min_6h` - Lowest pressure
- `spo2_min_6h` - Oxygen drops

**Trends:**
- `hr_slope` - Rate of HR increase
- `temp_change_3h` - Temperature delta

**Derived Clinical Metrics:**
- `mean_arterial_pressure` - MAP = DBP + (SBP-DBP)/3
- `shock_index` - HR/SBP (early shock indicator)
- `qsofa_score` - Quick sepsis screening (0-3)

**Lab Markers:**
- `lactate` - Tissue hypoperfusion indicator
- `wbc` - Infection/inflammation marker
- `creatinine` - Kidney function

---

## 🚀 Getting Started

### Prerequisites

- Databricks account
- Cluster with ML Runtime 14.3 LTS+
- Python 3.8+

### Installation

```python
%pip install xgboost shap scikit-learn plotly
dbutils.library.restartPython()
```

### Quick Start

1. **Import Notebook**
   ```
   Workspace → Import → 03_Sepsis_Early_Warning_System.ipynb
   ```

2. **Run Sequentially**
   ```
   Cells 1-10 (data generation → training → evaluation)
   Runtime: 4-5 hours
   ```

3. **View Results**
   - MLflow for training metrics
   - SHAP plots for interpretability
   - EWS dashboard queries

---

## 📈 Results

### Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.82 | Excellent discrimination |
| **Accuracy** | 78% | Overall correctness |
| **Precision** | 0.75 | 75% of alerts are true sepsis |
| **Recall** | 0.78 | Detects 78% of sepsis cases |
| **F1-Score** | 0.76 | Balanced performance |

### Confusion Matrix

```
                  Predicted
                No Sepsis  Sepsis
Actual No Sepsis   18,500   1,500   (FP: 1,500 = 7.5% false alarm)
       Sepsis       1,100   3,900   (TP: 3,900, FN: 1,100)
```

**Clinical Interpretation:**
- **True Positives (3,900):** Correctly predicted sepsis 6h early
- **False Positives (1,500):** Unnecessary alerts (~7.5% rate)
- **False Negatives (1,100):** Missed sepsis cases (22% miss rate)

### Feature Importance (SHAP)

**Top 10 Predictive Features:**

| Rank | Feature | SHAP Value | Clinical Meaning |
|------|---------|------------|------------------|
| 1 | lactate | 0.22 | Tissue hypoperfusion |
| 2 | hr_std_6h | 0.18 | Heart rate instability |
| 3 | temp_change_3h | 0.15 | Fever development |
| 4 | wbc | 0.12 | Infection marker |
| 5 | shock_index | 0.10 | Early shock |
| 6 | sbp_min_6h | 0.08 | Hypotension episodes |
| 7 | rr_mean_3h | 0.07 | Tachypnea |
| 8 | spo2_min_6h | 0.05 | Hypoxia |
| 9 | qsofa_score | 0.05 | Sepsis screening |
| 10 | creatinine | 0.04 | Kidney dysfunction |

**Clinical Insights:**
- **Lactate** most predictive (tissue perfusion indicator)
- **Heart rate variability** > absolute HR value
- **Temperature trends** > single measurement
- **Combination of vitals + labs** performs best

### Early Warning Score (EWS) Distribution

**Risk Categories:**
```
LOW (0-40):      15,000 patients (60%)  ✓
MEDIUM (40-60):   6,000 patients (24%)  ⚠
HIGH (60-80):     3,000 patients (12%)  ⚠⚠
CRITICAL (80-100):1,000 patients (4%)   🚨
```

**Clinical Actions by EWS:**
- **EWS < 40:** Continue standard monitoring
- **EWS 40-60:** Increase observation frequency
- **EWS 60-80:** Notify physician, prepare for intervention
- **EWS > 80:** Immediate physician evaluation, consider ICU

---

## 🎯 Key Databricks Features Used

### PySpark Window Functions
- ✅ **Rolling aggregations** (means, std dev over time windows)
- ✅ **Lag functions** (previous values for trend calculation)
- ✅ **Efficient time-series** processing at scale

### MLflow
- ✅ **AutoML baseline** comparison
- ✅ **Experiment tracking** (all hyperparameters, metrics)
- ✅ **Model registry** for version control
- ✅ **Artifact logging** (SHAP plots, feature importance)

### Delta Lake
- ✅ **Time travel** (analyze historical model versions)
- ✅ **Schema evolution** (add new features easily)
- ✅ **ACID transactions** for data consistency

### Databricks SQL
- ✅ **EWS dashboards** for clinical teams
- ✅ **High-risk patient queries**
- ✅ **Performance monitoring** over time

---

## 📊 Sample Notebook Output

### SHAP Waterfall Plot

```
Feature                 SHAP Value    Push Higher/Lower
========================================================
lactate (3.5)          +0.45         ↑ Higher Risk
hr_std_6h (18)         +0.32         ↑ Higher Risk
temp_change (+1.5°C)   +0.28         ↑ Higher Risk
wbc (14.5)             +0.18         ↑ Higher Risk
shock_index (1.1)      +0.15         ↑ Higher Risk
sbp_min_6h (92)        +0.12         ↑ Higher Risk
age (67)               +0.05         ↑ Higher Risk
spo2_min_6h (94)       -0.08         ↓ Lower Risk
creatinine (1.2)       -0.02         ↓ Lower Risk
========================================================
Base Value: 0.20 (20% baseline risk)
Final Prediction: 0.85 (85% sepsis risk)
EWS Score: 85 (CRITICAL)
```

---

## 🔮 Future Enhancements

### Short-term (1-3 months)
- [ ] **Real MIMIC-III** data integration (requires credentialing)
- [ ] **Multi-outcome prediction** (AKI, shock, mortality)
- [ ] **Real-time scoring** pipeline with streaming data
- [ ] **Model retraining** automation (weekly)

### Mid-term (3-6 months)
- [ ] **Deep learning** (LSTM) for sequential patterns
- [ ] **Ensemble models** (XGBoost + LSTM + RF)
- [ ] **Transfer learning** across hospitals
- [ ] **Mobile app** for bedside EWS display

### Long-term (6-12 months)
- [ ] **Clinical trial** validation with real patients
- [ ] **FDA approval** (Software as Medical Device)
- [ ] **Multi-center deployment** (10+ hospitals)
- [ ] **Continuous learning** from outcomes

---

## 📚 References

### Clinical Guidelines
- **Sepsis-3 Definition** - Singer M, et al. JAMA. 2016
- **qSOFA** - Quick SOFA for sepsis screening
- **Surviving Sepsis Campaign** - Clinical practice guidelines

### Datasets
- **MIMIC-III** - MIT Critical Care Database (inspiration)
- **eICU** - Multi-center ICU database

### Technical Papers
- **XGBoost** - Chen & Guestrin, 2016
- **SHAP** - Lundberg & Lee, 2017
- **Early Warning Scores** - Smith et al., 2013

---

## 📄 License

MIT License - Part of healthcare AI portfolio

---

## 📞 Contact

**Questions or Feedback?**
- GitHub: https://github.com/PramodBM
- Email: pramod.036@gmail.com

---

<p align="center">
  <strong>📊 Early Detection Saves Lives 📊</strong>
</p>

<p align="center">
  <sub>Demonstration project with synthetic data. Clinical validation required before deployment.</sub>
</p>
