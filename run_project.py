import subprocess
import sys
import time
import os

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"Executing: {description}")
    print(f"Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        # Run process and stream output to console
        process = subprocess.run([sys.executable, script_name], check=True)
        elapsed = time.time() - start_time
        print(f"[{script_name} completed in {elapsed:.2f} seconds]\n")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to run {script_name}.")
        print("Please check the terminal output above for errors.")
        sys.exit(1)
        
def main():
    print("=========================================================")
    print("   CS 550: Massive Data Mining - Project Execution Suite ")
    print("=========================================================")
    print("This wrapper will sequentially execute the entire project")
    print("pipeline: Data Prep -> Baseline CF -> SVD Approach -> Plots")
    print("=========================================================\n")
    
    if not os.path.exists('ml-latest-small/movies.csv'):
        print("[CRITICAL WARNING] The 'ml-latest-small' dataset was not found!")
        print("Please ensure the raw MovieLens CSVs are placed in the root directory")
        print("before executing this framework.")
        sys.exit(1)
        
    time.sleep(2)
    
    # 1. Data Prep
    run_script('data_prep.py', 'Step 1/4: Preprocessing Data (80/20 per-user split)')
    
    # 2. Baseline
    run_script('baseline_cf.py', 'Step 2/4: Calculating Baseline CF (MAE & NDCG)')
    
    # 3. SVD
    run_script('my_svd_approach.py', 'Step 3/4: Calculating SVD Matrix Factorization (MAE & NDCG)')
    
    # 4. Generate Plots
    run_script('generate_plots.py', 'Step 4/4: Generating Visualization Artifacts')
    
    print("=========================================================")
    print("      PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("=========================================================")
    print("\nNext Steps:")
    print("To launch the interactive Trustworthiness Dashboard (Option A, C, & E),")
    print("please run the following command in your terminal:")
    print("\n   streamlit run app.py\n")

if __name__ == '__main__':
    main()
