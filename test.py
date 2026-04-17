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
    os.makedirs("generated_inputs", exist_ok=True)

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
    run_command([
        sys.executable, "gen_diff_torch.py",
        "light",
        "1.0",
        "0.1",
        "0.01",
        "20",
        "30",
        "0.5",
        "--target_model", "0",
        "--model1_path", "./checkpoints/model1.pth",
        "--model2_path", "./checkpoints/model2.pth",
        "--model3_path", "./checkpoints/model3.pth",
        "--output_dir", "./generated_inputs"
    ])

    print("[INFO] Done. Check generated_inputs/ for outputs.")


if __name__ == "__main__":
    main()