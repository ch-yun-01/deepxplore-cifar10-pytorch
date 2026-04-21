import os
import subprocess
import sys


def run_command(cmd):
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    needed = [
        "checkpoints/model1.pth",
        "checkpoints/model2.pth",
        "checkpoints/model3.pth",
    ]

    if not all(os.path.exists(p) for p in needed):
        print("[INFO] Missing checkpoints. Training models first...")
        run_command([sys.executable, "train_models.py"])
    else:
        print("[INFO] Checkpoints already exist. Skip training.")

    print("[INFO] Running DeepXplore-style generation...")
    experiments = [
        # (transformation, weight_diff, weight_nc, step, seeds, grad_iter, threshold, target_model, output_dir)
        # ("light",    "1.0", "0.1", "0.05", "100", "100", "0.5", "0", "./results/exp1_light_baseline"),
        ("occl",     "1.0", "0.1", "0.05", "100", "100", "0.5", "0", "./results/exp2_occl"),
        ("blackout", "1.0", "0.1", "0.05", "100", "100", "0.5", "0", "./results/exp3_blackout"),
        ("light",    "2.0", "0.1", "0.05", "100", "100", "0.5", "0", "./results/exp4_high_weight_diff"),
        ("light",    "1.0", "0.5", "0.05", "100", "100", "0.5", "0", "./results/exp5_high_weight_nc"),
        ("light",    "1.0", "0.1", "0.05", "100", "100", "0.5", "1", "./results/exp6_target_model1"),
        ("light",    "1.0", "0.1", "0.05", "100", "100", "0.5", "2", "./results/exp7_target_model2"),
        ("light",    "1.0", "0.1", "0.01", "100", "100", "0.5", "0", "./results/exp8_small_step"),
        ("light",    "1.0", "0.1", "0.10", "100", "100", "0.5", "0", "./results/exp9_large_step"),
        ("occl",     "2.0", "0.5", "0.05", "100", "100", "0.5", "0", "./results/exp10_occl_high_both"),
        ("light",    "1.0", "0.1", "0.05", "100", "100", "0.3", "0", "./results/exp11_low_threshold"),
        ("light",    "1.0", "0.1", "0.05", "100", "100", "0.7", "0", "./results/exp12_high_threshold"),
        ("light",    "1.0", "0.1", "0.05", "200", "100", "0.5", "0", "./results/exp13_seeds200"),
        ("light",    "1.0", "0.1", "0.05", "300", "100", "0.5", "0", "./results/exp14_seeds300"),
    ]

    for (transform, w_diff, w_nc, step, seeds, grad_iter, threshold, target, out_dir) in experiments:
        print(f"\n[INFO] Running experiment: {out_dir}")
        run_command([
            sys.executable, "gen_diff_cifar10.py",
            transform, w_diff, w_nc, step, seeds, grad_iter, threshold,
            "--target_model", target,
            "--model1_path", "./checkpoints/model1.pth",
            "--model2_path", "./checkpoints/model2.pth",
            "--model3_path", "./checkpoints/model3.pth",
            "--output_dir", out_dir,
        ])
    print("[INFO] Done. Check generated_inputs/ for outputs.")


if __name__ == "__main__":
    main()