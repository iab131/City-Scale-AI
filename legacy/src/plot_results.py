import os
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def create_plot(df, title, out_path, brain_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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
    
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved successfully to {out_path}")
    
    if brain_path:
        shutil.copy(out_path, brain_path)

def plot_results():
    csv_path = "outputs/experiments/k_sweep_results.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    brain_dir = r"C:\Users\baii\.gemini\antigravity\brain\b5ec29eb-fc45-4c96-9976-f54bec0f2e56"
    
    # 1. Full
    create_plot(
        df,
        'GFT Traffic Forecasting: Error Metrics vs. Eigenvectors (k)',
        "outputs/experiments/k_sweep_plot.png",
        os.path.join(brain_dir, "k_sweep_plot.png")
    )
    
    # 2. k = 1 to 16
    df_low = df[df['k'] <= 16]
    create_plot(
        df_low,
        'GFT Traffic Forecasting: Error Metrics (k=1 to 16)',
        "outputs/experiments/k_sweep_plot_low.png",
        os.path.join(brain_dir, "k_sweep_plot_low.png")
    )
    
    # 3. k = 16 to 207
    df_high = df[df['k'] >= 16]
    create_plot(
        df_high,
        'GFT Traffic Forecasting: Error Metrics (k=16 to 207)',
        "outputs/experiments/k_sweep_plot_high.png",
        os.path.join(brain_dir, "k_sweep_plot_high.png")
    )

if __name__ == "__main__":
    plot_results()
