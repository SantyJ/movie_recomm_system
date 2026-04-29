# CS 550: Massive Data Mining - Recommender System Project

**Authors**: Santhosh, Mohammed Faisal Khan, Jacob Sze  
**Institution**: Rutgers University
**Course**: CS 550 Massive Data Mining  
**Dataset**: MovieLens 1M (`ml-latest-small`)

---

## Project Overview

This repository contains the complete implementation of **Option 1: Recommender System**. Rather than relying on popular monolithic black-box libraries, we developed a Recommender System from scratch using mathematical linear algebra primitives to predict user-item ratings and generate Top-10 recommendation lists.

Beyond establishing mathematical accuracy, this project directly addresses modern constraints in structural algorithmic safety, explicitly implementing the three **Trustworthiness** bonus rubrics:
1. **Explainability (Option A):** Generating human-readable rationale via a Content-Based Surrogate module.
2. **Controllability (Option C):** Providing interactive mechanisms allowing users to drop specific parameters post-prediction.
3. **Robustness & Vulnerability (Option E):** Demonstrating "Mathematical Collateral Damage" against a Targeted Clone Swarm Data-Poisoning attack.

---

## Repository Structure

```text
recomm_system/
│
├── data_prep.py                # 1. Pipeline for 80/20 train-test splitting (Prevents Data Leakage)
│
├── baseline_cf.py              # 2. Algorithm 1: Memory-Based Cosine Similarity CF (Baseline)
├── my_svd_approach.py          # 3. Algorithm 2: Model-Based Singular Value Decomposition (SVD)
├── model.py                    # 4. SVD execution wrapper for Trustworthiness logic integration
│
├── verify_metrics.py           # Evaluation script for MAE and RMSE prediction errors
├── verify_ranking_metrics.py   # Evaluation script for Precision, Recall, F1, and NDCG
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

## Execution Guide: How to Verify the Results

To make grading seamless, we have centralized the execution stack into a single Python automation wrapper. This eliminates the need to run components separately.

Run the master project wrapper:
```bash
python run_project.py
```

This single command will sequentially execute:
1. **Data Preprocessing:** Splitting the dataset uniquely per-user (80/20) and securely holding out ground-truths to prevent data leakage.
2. **Metric Verification:** Firing SVD and Baseline arrays to assert MAE and RMSE floating-point accuracy errors.
3. **Ranking Verification:** Strictly validating recommendation ordering using NDCG, Recall, and F1 limits.
4. **Graphing Automation:** Regenerating the comparative visualization charts locally based exclusively on the current model run outputs.

### Optional Deep-Dive Scripts
For grading and evaluation purposes, the `run_project.py` wrapper covers the primary execution workflow. However, we have also included two standalone diagnostic scripts for deeper inspection:
* **`python verify_metrics.py`**: Executes an isolated user-by-user breakdown of the Top-10 ranking metrics for a small subset of sample users, outputting directly to `processed/metrics_verification_report.txt`.
* **`python generate_report.py`**: A secondary utility that compiles a highly-detailed, side-by-side mathematical analysis of the models. It generates a comprehensive text summary (`processed/model_comparison_results.txt`) explicitly breaking down the performance gap between the Baseline CF and SVD models across MAE, RMSE, and all Top-10 Ranking scores, alongside academic conclusions on the sparsity problem.

---

## The Trustworthiness Demo (Streamlit Interface)

The highlight of the project is the interactive UI which proves our findings for the Extra Credit Trustworthiness rubrics (Explainability, Controllability, and Shilling Attacks).

To launch the dashboard, execute:
```bash
streamlit run app.py
```

### Dashboard Features
* **Option A (Transparency):** Watch the AI generate dynamic, human-readable explanations justifying **why** the underlying math chose specific recommendations based on genre-DNA mapping against the user's high-rated history.
* **Option C (Controllability):** Interact with the "Filter Global Recommendations" sidebar to immediately strip specific domains (e.g., "Horror") dynamically from your SVD matrix results in real-time.
* **Option E (Robustness):** Engage the **Targeted Clone Attack (Shilling)** Simulation.
  * Turn the attack toggle **ON**.
  * Observe how the introduction of 500 perfectly-aligned "sleeper agent" bots drastically forces the mathematical optimization plane to shift. 
  * You will witness **Mathematical Collateral Damage**: untouched niche movies (like *Pirates of the Caribbean* or *Catwoman*) suddenly artificially surge up into the top rankings simply because they were caught within the localized density gravity of the clone attack.
