import os
import time
import pandas as pd
from train import run_training

def main():
    k_values = [16, 32, 64, 96, 128, 207]
    
    out_dir = "outputs/experiments"
    # Ensure it's relative to the project root, so we check cwd
    # If run from src, it creates src/outputs/experiments
    # Let's write it to the project root assuming script is run from project root
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    print(f"Starting k-sweep experiment for k={k_values}")
    
    for k in k_values:
        print(f"\n{'='*40}")
        print(f"Running experiment for k = {k}")
        print(f"{'='*40}")
        
        start_time = time.time()
        
        # Override k
        res = run_training({"k": k})
        
        end_time = time.time()
        
        if res is not None:
            res["time_sec"] = round(end_time - start_time, 2)
            results.append(res)
        else:
            print(f"Experiment failed for k={k}")
            
    if not results:
        print("No results to save.")
        return
        
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(out_dir, "k_sweep_results.csv")
    df.to_csv(csv_path, index=False)
    
    # Save to Markdown
    md_path = os.path.join(out_dir, "k_sweep_results.md")
    md_str = df.to_markdown(index=False)
    with open(md_path, "w") as f:
        f.write(md_str)
        
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS")
    print("="*80)
    print(md_str)
    print("\nResults saved to:")
    print(f"- {csv_path}")
    print(f"- {md_path}")

if __name__ == "__main__":
    main()
