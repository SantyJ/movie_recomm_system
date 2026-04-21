# CS 550: Massive Data Mining - Recommender System Project

**Authors**: Santhosh, Mohammed Faisal Khan, Jacob Sze  
**Institution**: Rutgers University
**Course**: CS 550 Massive Data Mining  
**Dataset**: MovieLens 1M (`ml-latest-small`)

---

## Project Overview

This repository contains the complete implementation of **Option 1: Recommender System**. Rather than relying on popular monolithic black-box libraries, we developed a Recommender System from scratch using mathematical linear algebra primitives to predict user-item ratings and generate Top-10 recommendation lists.

Beyond establishing mathematical accuracy, this project directly addresses modern constraints in structural algorithmic safety, explicitly implementing the two **Trustworthiness** bonus rubrics:
1. **Explainability (Option A):** Generating human-readable rationale via a Content-Based Surrogate module.
2. **Controllability (Option C):** Providing interactive mechanisms allowing users to drop specific parameters post-prediction.

---

## Repository Structure

```text
recomm_system/
│
├── data_prep.py                # 1. Pipeline for 80/20 train-test splitting (Prevents Data Leakage)
│
├── baseline_cf.py              # 2. Algorithm 1: Memory-Based Cosine Similarity CF (Baseline)
├── my_svd_approach.py          # 3. Algorithm 2: Model-Based Singular Value Decomposition (SVD)
│
├── generate_plots.py           # Automates visualization of comparison metrics
│
├── app.py                      # Interactive Streamlit Demo (Visualizes the Trustworthiness modules)
│
├── project_report.tex          # Formatted ACM Academic LaTeX Report detailing logic/theory
└── README.md                   # Project documentation
```

---

## Installation & Setup

Ensure you have Python `3.9+` installed on your machine.

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/cs550-recommender-system.git
   cd cs550-recommender-system
   ```

2. **Install Required Libraries:**
   The project primarily utilizes deep linear-algebra packages and Streamlit for the front-end demo.
   ```bash
   pip install pandas numpy scipy scikit-learn matplotlib streamlit
   ```

3. **Data Availability:**
   To adhere to repository limits, the `ml-latest-small` directory (raw MovieLens CSVs) and the `processed/` matrix tensors are `.gitignore`d. 
   **Note to Graders:** Please ensure the standard `ml-latest-small` folder extracted from GroupLens is placed natively in the root directory before running module scripts.

---

## Execution Guide

To make grading seamless, we have centralized the execution stack into a single Python automation wrapper. This eliminates the need to run components separately.

Run the master project wrapper:
```bash
python run_project.py
```

This single command will sequentially execute:
1. **Data Preprocessing:** Splitting the dataset uniquely per-user (80/20) and securely holding out ground-truths to prevent data leakage.
2. **Metric Verification (Baseline):** Firing Cosine vectors to assert MAE and NDCG.
3. **Metric Verification (SVD):** Strictly validating SVD recommendation ordering using NDCG, Recall, and F1 limits.
4. **Graphing Automation:** Regenerating the comparative visualization charts locally based exclusively on the model outputs.

---

## The Trustworthiness Demo (Streamlit Interface)

The highlight of the project is the interactive UI which proves our findings for the Trustworthiness rubrics.

To launch the dashboard, execute:
```bash
streamlit run app.py
```

### Dashboard Features
* **Option A (Transparency):** Watch the AI generate dynamic, human-readable explanations justifying **why** the underlying math chose specific recommendations based on genre-DNA mapping against the user's high-rated history.
* **Option C (Controllability):** Interact with the "Filter Global Recommendations" sidebar to immediately strip specific domains (e.g., "Horror") dynamically from your SVD matrix results in real-time.
