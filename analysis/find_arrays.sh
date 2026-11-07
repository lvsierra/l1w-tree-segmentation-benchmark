# Lists every .npy under the benchmark scratch area with its length and modification date,
# so the right prediction / GT arrays can be copied into methods.tsv.
# Usage: bash find_arrays.sh [root]    
ROOT=${1:-/mnt/scratch/users/$USER/tree-seg}
find "$ROOT" -name "*.npy" -size +100M 2>/dev/null | sort | while read -r f; do
  python3 - "$f" <<'PY'
import sys, numpy as np, os, time
f = sys.argv[1]
try:
    a = np.load(f, mmap_mode="r")
    n = f"{a.shape[0]:,}" if a.ndim else "?"
    dt = str(a.dtype)
except Exception as e:
    n, dt = "unreadable", ""
t = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(f)))
print(f"{n:>14}  {dt:<6} {t}  {f}")
PY
done
