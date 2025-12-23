# Project: Predicting 30-Day Hospital Readmission for Diabetic Patients

**Goal:** To build and evaluate a robust, interpretable classification model to predict 30-day hospital readmission using a real-world, messy clinical dataset.

## Project Summary

This project is an end-to-end data science workflow focused on realistic model development. The analysis began with a 100k-row dataset from the UCI Machine Learning Repository, which is known for its complexity, missing data, and severe class imbalance.

The core of this project was not just benchmarking models, but a deep-dive root cause analysis that:
1.  **Identified and corrected critical data leakage** in the raw data, which would have made a naive model completely useless.
2.  **Proved** that a simple, interpretable model (`LogisticRegression`) was significantly more robust and effective than a complex non-linear model (`RandomForest`) for this specific problem.
3.  **Tuned** the final model to a specific performance goal (high recall) by adjusting the decision threshold, demonstrating the critical trade-off between precision and recall.

## 1. Dataset

* **Source:** "Diabetes 130-US hospitals" (UCI Machine Learning Repository) [1]
* **Size:** 101,766 patient encounters
* **Challenge:** The data is extremely "dirty," with 48+ features, high missingness, complex categorical codes (ICD-9), and severe class imbalance (only 11% of patients are readmitted in <30 days).

## 2. The Data Science Workflow

This project followed a rigorous debugging and analysis path.

### Step 1: Data Cleaning & Feature Engineering
First, I cleaned the raw data and engineered meaningful features:
* **Cleaning:** Dropped high-null columns (`weight`, `A1Cresult`, `medical_specialty`) and standardized missing values.
* **Target Definition:** Created the binary target `readmitted_binary` (1 = `<30 days`, 0 = `>30` or `NO`).
* **Feature Engineering (Meds):** Statistically analyzed the 23 medication columns. I selected only high-signal drugs (e.g., `metformin`, `insulin`, `glipizide`, `glyburide`) that were prescribed to >10% of patients.
* **Feature Engineering (Diagnosis):** Grouped 70+ unique `diag_1` ICD-9 codes into 6 clinical categories (e.g., 'Diabetes', 'Circulatory', 'Respiratory') to reduce dimensionality, following the methodology of the original research paper [1].

### Step 2: The "Data Leakage" Trap (The Critical Finding)
A preliminary model's interpretation showed `discharge_disposition_id_11` (patient "Expired") as the #1 predictor.

This is a classic, fatal **data leakage** flaw. A model cannot use information about a patient's discharge (e.g., if they died) to predict an event *after* discharge (readmission).

**This finding became the core of the project.** I re-started by **removing the entire `discharge_disposition_id` column** to build a realistic, ethical, and non-trivial model.

### Step 3: Building an Honest, Interpretable Baseline
With the data leakage fixed, I built an interpretable baseline model.
* **Model:** `LogisticRegression(class_weight='balanced')`
* **Why:** `LogisticRegression` is simple, interpretable, and `class_weight='balanced'` directly addresses the 11% class imbalance.
* **Result:** This *honest* model achieved a strong, realistic baseline: **52% recall** for the 'Readmitted (1)' class. This proves the model is finding a real signal.

### Step 4: Investigating (and Debunking) Complex Models
I hypothesized that a complex, non-linear model (`RandomForest`) could improve performance. This hypothesis **failed**, which was a critical insight.
* **Test:** Even on the clean, non-leaking data, and using advanced imbalance techniques (`class_weight='balanced_subsample'`), the `RandomForest` model failed.
* **Result:** **1% recall** for the 'Readmitted (1)' class.
* **Conclusion:** The complex model was overfitting to the "noise" from the 100+ one-hot-encoded features. The simple, robust `LogisticRegression` was the correct and superior tool for this dataset.

### Step 5: Final Model Tuning & Interpretation (The Deliverable)
The final step was to tune the *honest, winning* model to demonstrate the precision-recall trade-off.
* **Action:** I adjusted the model's **decision threshold from 0.5 to 0.4**.
* **Final Result:** The recall for at-risk patients jumped from **52% to 89%**, at the expense of precision (which dropped from 17% to 13%).
* **Interpretation:** This tuning demonstrates how the model can be flexibly adapted to a business goal (e.g., "find as many at-risk patients as possible").
* **Insights:** An analysis of the model's coefficients showed the top *real* predictors of readmission are:
    1.  **`age_[0-10)`**: A strong negative driver, suggesting the specialized system of care for children is highly effective.
    2.  **`admission_source_id_3`**: A strong positive driver, acting as a proxy for patient acuity (e.g., "Transfer from a hospital/nursing facility").
    3.  **`number_inpatient`**: A logical positive driver, indicating that prior hospital visits are a key predictor of future visits.

## 3. References

[1] Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014). Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. *BioMed research international*, 2014.
