import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def plot_results():
    csv_path = "outputs/experiments/k_sweep_results.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define colors
    val_color = '#1f77b4'
    test_color = '#ff7f0e'
    
    # Plot MAE
    axes[0].plot(df['k'], df['best_val_mae'], marker='o', linewidth=2, color=val_color, label='Val MAE')
    axes[0].plot(df['k'], df['test_mae'], marker='s', linewidth=2, color=test_color, label='Test MAE')
    axes[0].set_title('Mean Absolute Error (MAE)', fontsize=14)
    axes[0].set_xlabel('k (number of eigenvectors)', fontsize=12)
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=11)
    
    # Plot RMSE
    axes[1].plot(df['k'], df['best_val_rmse'], marker='o', linewidth=2, color=val_color, label='Val RMSE')
    axes[1].plot(df['k'], df['test_rmse'], marker='s', linewidth=2, color=test_color, label='Test RMSE')
    axes[1].set_title('Root Mean Square Error (RMSE)', fontsize=14)
    axes[1].set_xlabel('k (number of eigenvectors)', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=11)

    # Plot MAPE
    axes[2].plot(df['k'], df['best_val_mape'], marker='o', linewidth=2, color=val_color, label='Val MAPE')
    axes[2].plot(df['k'], df['test_mape'], marker='s', linewidth=2, color=test_color, label='Test MAPE')
    axes[2].set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14)
    axes[2].set_xlabel('k (number of eigenvectors)', fontsize=12)
    axes[2].set_ylabel('MAPE', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(fontsize=11)
    
    plt.suptitle('GFT Traffic Forecasting: Error Metrics vs. Eigenvectors (k)', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Save to project dir
    out_path = "outputs/experiments/k_sweep_plot.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {out_path}")
    
    # Copy to brain dir for embedding
    brain_path = r"C:\Users\baii\.gemini\antigravity\brain\b5ec29eb-fc45-4c96-9976-f54bec0f2e56\k_sweep_plot.png"
    shutil.copy(out_path, brain_path)

if __name__ == "__main__":
    plot_results()
