import pandas as pd
import os

data_part1 = [
    {"k": 16, "best_val_mae": 3.13765, "best_val_rmse": 6.02841, "best_val_mape": 0.197251, "test_mae": 3.69934, "test_rmse": 6.99501, "test_mape": 0.26539, "time_sec": 45.66},
    {"k": 32, "best_val_mae": 3.64753, "best_val_rmse": 6.81448, "best_val_mape": 0.228476, "test_mae": 4.34135, "test_rmse": 7.97894, "test_mape": 0.315273, "time_sec": 47.19},
    {"k": 64, "best_val_mae": 4.18728, "best_val_rmse": 7.63228, "best_val_mape": 0.599586, "test_mae": 5.22681, "test_rmse": 9.24156, "test_mape": 0.862901, "time_sec": 49.53},
    {"k": 96, "best_val_mae": 4.53685, "best_val_rmse": 8.1627, "best_val_mape": 0.603542, "test_mae": 5.76178, "test_rmse": 9.97423, "test_mape": 0.944615, "time_sec": 51.98},
    {"k": 128, "best_val_mae": 4.83965, "best_val_rmse": 8.6422, "best_val_mape": 0.812364, "test_mae": 6.10793, "test_rmse": 10.5075, "test_mape": 1.20804, "time_sec": 53.93},
    {"k": 207, "best_val_mae": 5.21302, "best_val_rmse": 9.49904, "best_val_mape": 0.112206, "test_mae": 6.49366, "test_rmse": 11.63, "test_mape": 0.147406, "time_sec": 59.18}
]

data_part2 = [
    {"k": 1, "best_val_mae": 2.07098, "best_val_rmse": 4.54183, "best_val_mape": 0.145404, "test_mae": 2.46994, "test_rmse": 5.38630, "test_mape": 0.213069, "time_sec": 40.40},
    {"k": 2, "best_val_mae": 2.13606, "best_val_rmse": 4.65083, "best_val_mape": 0.142348, "test_mae": 2.63580, "test_rmse": 5.61092, "test_mape": 0.216040, "time_sec": 42.90},
    {"k": 4, "best_val_mae": 2.30863, "best_val_rmse": 4.86328, "best_val_mape": 0.626356, "test_mae": 2.76150, "test_rmse": 5.80360, "test_mape": 0.912040, "time_sec": 44.75},
    {"k": 8, "best_val_mae": 2.70943, "best_val_rmse": 5.42855, "best_val_mape": 0.187442, "test_mae": 3.20091, "test_rmse": 6.32413, "test_mape": 0.268779, "time_sec": 45.19},
    {"k": 12, "best_val_mae": 2.99537, "best_val_rmse": 5.84295, "best_val_mape": 0.175364, "test_mae": 3.50762, "test_rmse": 6.75862, "test_mape": 0.243978, "time_sec": 45.11},
]

df1 = pd.DataFrame(data_part1)
df2 = pd.DataFrame(data_part2)

df_combined = pd.concat([df2, df1]).sort_values('k').reset_index(drop=True)

out_dir = "outputs/experiments"
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(out_dir, "k_sweep_results.csv")
df_combined.to_csv(csv_path, index=False)

md_path = os.path.join(out_dir, "k_sweep_results.md")
md_str = df_combined.to_markdown(index=False)
with open(md_path, "w") as f:
    f.write(md_str)

print("Recovered data saved successfully.")
