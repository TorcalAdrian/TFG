#!/usr/bin/env python3
import os, sys, subprocess, csv, re, statistics as stats

REPEATS = 5  # number of repetitions for each dataset
DATASETS_DIR = "datasets"
RESULTS_DIR = "results"
RESULT_FILE = os.path.join(RESULTS_DIR, "final_results.csv")

def install_requirements():
    print("\n📦 Installing dependencies (only first time)...")
    reqs = ["torch", "faiss-cpu", "numpy", "pandas", "tqdm", "scikit-learn"]
    for pkg in reqs:
        subprocess.call([sys.executable, "-m", "pip", "install", pkg])

def run_once(dataset_path, seed):
    cmd = [
        sys.executable, "ae_kmeans_pipeline.py",
        "--data", dataset_path,
        "--min_freq", "3",
        "--latent_dim", "200",
        "--hidden", "800,400",
        "--epochs", "20",
        "--batch_size", "1024",
        "--lr", "1e-3",
        "--k_list", "4", "8", "16",
        "--kmeans_n_init", "20",
        "--use_faiss", "True",
        "--seed", str(seed)
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in proc.stdout:
        print(line.strip())
        out.append(line.strip())
    proc.wait()
    return out

def parse_output(output_lines):
    ae_train, enc = None, None
    km_entries = {}
    for ln in output_lines:
        if "AE training time:" in ln:
            m = re.search(r"([\d.]+)", ln); 
            if m: ae_train = float(m.group(1))
        if "Encoding time:" in ln:
            m = re.search(r"([\d.]+)", ln); 
            if m: enc = float(m.group(1))
        if "NMI=" in ln and "k=" in ln:
            mk = re.search(r"k=(\d+)", ln)
            mn = re.search(r"NMI=([\d.]+)", ln)
            mt = re.search(r"time=([\d.]+)", ln)
            if mk and mn and mt:
                k = int(mk.group(1))
                nmi = float(mn.group(1))
                kt = float(mt.group(1))
                km_entries[k] = (nmi, kt)
    ae_total = (ae_train or 0.0) + (enc or 0.0)
    return ae_total, km_entries

def write_summary(all_rows, writer):
    # all_rows = list of dicts with dataset, k, nmi, ae, km
    grouped = {}
    for r in all_rows:
        key = (r["dataset"], r["k"])
        grouped.setdefault(key, {"nmis": [], "aes": [], "kms": []})
        grouped[key]["nmis"].append(r["nmi"])
        grouped[key]["aes"].append(r["ae_time"])
        grouped[key]["kms"].append(r["km_time"])

    writer.writerow([])
    writer.writerow(["dataset", "k", "repeats", "NMI_mean", "NMI_std", "AE_mean(s)", "AE_std(s)", "KM_mean(s)", "KM_std(s)", "type"])
    for (dataset, k), vals in grouped.items():
        writer.writerow([
            dataset,
            k,
            len(vals["nmis"]),
            f"{stats.mean(vals['nmis']):.4f}",
            f"{(stats.pstdev(vals['nmis']) if len(vals['nmis'])>1 else 0.0):.4f}",
            f"{stats.mean(vals['aes']):.2f}",
            f"{(stats.pstdev(vals['aes']) if len(vals['aes'])>1 else 0.0):.2f}",
            f"{stats.mean(vals['kms']):.2f}",
            f"{(stats.pstdev(vals['kms']) if len(vals['kms'])>1 else 0.0):.2f}",
            "mean/std"
        ])

def main():
    print("=== Autoencoder + KMeans Repeated Experiments ===")
    install_requirements()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    data_files = [os.path.join(DATASETS_DIR, f) for f in os.listdir(DATASETS_DIR) if f.endswith(".data")]
    if not data_files:
        print(f"No .data files found in '{DATASETS_DIR}/'. Please add datasets and rerun.")
        sys.exit(1)

    all_rows = []
    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "k", "seed", "NMI", "AE_time(s)", "KMeans_time(s)", "type"])

        for ds_path in data_files:
            dataset_name = os.path.basename(ds_path)
            print(f"\n Running {dataset_name} ({REPEATS} repetitions)...")
            for r in range(REPEATS):
                seed = 42 + r
                print(f"\n Run {r+1}/{REPEATS} | seed={seed}")
                output_lines = run_once(ds_path, seed)
                ae_total, km_entries = parse_output(output_lines)
                for k, (nmi, km_time) in km_entries.items():
                    row = {
                        "dataset": dataset_name,
                        "k": k,
                        "seed": seed,
                        "nmi": nmi,
                        "ae_time": ae_total,
                        "km_time": km_time
                    }
                    all_rows.append(row)
                    writer.writerow([
                        dataset_name, k, seed,
                        f"{nmi:.4f}",
                        f"{ae_total:.2f}",
                        f"{km_time:.2f}",
                        "run"
                    ])

        # summary rows
        write_summary(all_rows, writer)

    print(f"\n All experiments done. Results saved to {RESULT_FILE}\n")

if __name__ == "__main__":
    main()
