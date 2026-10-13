"""Roofline-style summary from results/bench.csv. Usage: python scripts/plot.py [results/bench.csv]"""
import csv, sys, collections
path = sys.argv[1] if len(sys.argv) > 1 else "results/bench.csv"
rows = list(csv.DictReader(open(path)))
by_size = collections.defaultdict(list)
for r in rows: by_size[int(r["M"])].append(r)
print(f"GPU: {rows[0]['gpu']}\n")
for n in sorted(by_size):
    print(f"{n} x {n} x {n}")
    print(f"  {'stage':14} {'ms':>9} {'GFLOP/s':>10} {'% cuBLAS':>9} {'max rel err':>12}")
    for r in sorted(by_size[n], key=lambda r: r['stage']):
        print(f"  {r['stage']:14} {float(r['ms']):9.3f} {float(r['gflops']):10.1f} {float(r['pct_of_cublas']):9.1f} {r['max_rel_err']:>12}")
    print()
try:
    import matplotlib.pyplot as plt
    n = max(by_size)
    rs = sorted(by_size[n], key=lambda r: r['stage'])
    plt.figure(figsize=(8, 4))
    plt.bar([r['stage'] for r in rs], [float(r['gflops']) for r in rs])
    plt.ylabel("GFLOP/s"); plt.title(f"fp32 GEMM {n}^3 on {rows[0]['gpu']}")
    plt.xticks(rotation=20); plt.tight_layout(); plt.savefig("results/gflops.png", dpi=150)
    print("wrote results/gflops.png")
except ImportError:
    print("(matplotlib not installed; skipped plot)")
