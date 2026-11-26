"""Compare tensors of two GGUF files (CPU only, memory-mapped)."""
import sys, collections
import numpy as np
sys.path.insert(0, "src/llama.cpp-olala/gguf-py")
from gguf import GGUFReader
a, b = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
ta = {t.name: t for t in a.tensors}; tb = {t.name: t for t in b.tensors}
print("tensors", len(ta), len(tb), "only in a", len(set(ta)-set(tb)), "only in b", len(set(tb)-set(ta)))
import re
groups = collections.defaultdict(lambda: [0, 0, 0.0, 0])
for n, x in ta.items():
    y = tb.get(n)
    if y is None: continue
    key = re.sub(r"\.\d+\.", ".N.", n)
    g = groups[key]; g[0] += 1
    if x.tensor_type != y.tensor_type or x.data.shape != y.data.shape:
        g[3] += 1; continue
    def dec(t):
        d = np.asarray(t.data)
        if t.tensor_type.name == "BF16":
            return (d.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return d.astype(np.float32) if d.dtype.kind == "f" else None
    same = np.array_equal(np.asarray(x.data), np.asarray(y.data))
    xa, ya = (dec(x), dec(y)) if not same else (None, None)
    if not same:
        g[1] += 1
        if xa is not None:
            g[2] = max(g[2], float(np.linalg.norm(xa - ya) / (np.linalg.norm(xa) + 1e-12)))
for k, (n, d, r, m) in sorted(groups.items()):
    if d or m: print(f"{k:55s} n={n:3d} differ={d:3d} max_rel={r:.4f} type_mismatch={m}")
print("unchanged groups:", sum(1 for v in groups.values() if not v[1] and not v[3]), "of", len(groups))
print("dtypes a:", collections.Counter(t.tensor_type.name for t in a.tensors)); print("dtypes b:", collections.Counter(t.tensor_type.name for t in b.tensors))
