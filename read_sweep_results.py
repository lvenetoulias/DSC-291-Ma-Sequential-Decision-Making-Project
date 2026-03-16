import pickle

with open("./results/sweep_results.pkl", "rb") as f:
    sweep = pickle.load(f)

for algo, params in sweep.items():
    print(f"\n{algo}:")
    for param_name, result in params.items():
        best_val = result["best_value"]
        print(f"  {param_name}: best={best_val}")
        for val, reg, std in zip(
            result["param_values"],
            result["mean_final_regret"],
            result["std_final_regret"],
        ):
            marker = " <--" if val == best_val else ""
            print(f"    {param_name}={val}  regret={reg:.1f} ± {std:.1f}{marker}")

        