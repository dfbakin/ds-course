# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: DS Course
#     language: python
#     name: ds-course
# ---

# %% [markdown]
# # Lesson 2 — Doing it by hand: NumPy for people who already know the theory
#
# ## Who this is for
#
# You know what standardisation is. You know what a confusion matrix is. You can
# explain gradient descent at a whiteboard.
#
# And then you open an editor and write `X.mean(axis=1)` when you needed
# `axis=0`, get a `(20000, 8)` where you wanted `(8,)`, and spend forty minutes
# on a bug that is one keyword argument.
#
# That gap is not a gap in understanding. It is a gap in **fluency with the
# array**, and it closes the same way every fluency gap closes: by doing the
# small thing many times until it stops costing attention.
#
# So this lesson has almost no new theory. Every concept here you already know.
# What we do is implement all of it with nothing but NumPy — no pandas, no
# scikit-learn — on the same telemetry dataset from lesson 1.
#
# ## The three ideas that cause 90% of the bugs
#
# If you take nothing else from this notebook, take these:
#
# 1. **`axis=k` means "collapse the k-th axis"** — not "work along rows" or
#    whatever you half-remember. `X.mean(axis=0)` on a `(20000, 8)` gives `(8,)`.
# 2. **Broadcasting compares shapes from the right**, and stretches any axis of
#    length 1. `keepdims=True` exists to keep that axis around so the stretch
#    works.
# 3. **Slicing gives a view; fancy indexing gives a copy.** Writing into a view
#    modifies the original array. This one silently corrupts data.
#
# ## The rule of this notebook
#
# Every function we write is immediately checked against the library that
# already does it. If your version and NumPy's disagree, you find out in the
# next cell — not in production.

# %% [markdown]
# ## §0. Setup
#
# We load the CSV with **`np.loadtxt`** instead of `pd.read_csv`, because doing
# without pandas for one lesson makes visible how much of it you can do with
# arrays alone.

# %%
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 20260729
rng = np.random.default_rng(RANDOM_SEED)

# The dataset lives with lesson 1; find it whether we run from here or the root.
CANDIDATES = [Path.cwd() / ".." / "lesson01" / "data",
              Path.cwd() / "lesson01" / "data",
              Path.cwd() / "data"]
DATA_DIR = next(p.resolve() for p in CANDIDATES
                if (p / "service_telemetry.csv").exists())
FIG_DIR = Path.cwd() / "figures"
if not FIG_DIR.exists():
    FIG_DIR = Path.cwd() / "lesson02" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

np.set_printoptions(precision=4, suppress=True, linewidth=100)
print("data:", DATA_DIR)

# %%
# Same palette as lesson 1, so the two notebooks read as one course.
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}

plt.rcParams.update({
    "figure.facecolor": INK["surface"], "axes.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"], "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"], "axes.titlecolor": INK["primary"],
    "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.8,
    "grid.linestyle": "-", "xtick.color": INK["muted"],
    "ytick.color": INK["muted"], "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.frameon": False, "legend.fontsize": 9,
    "lines.linewidth": 2.0, "font.size": 10, "figure.dpi": 110,
})


def finish(ax, title=None, xlabel=None, ylabel=None, legend=False):
    if title:
        ax.set_title(title, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    if legend:
        ax.legend()
    return ax


# %% [markdown]
# ### Loading without pandas
#
# `np.loadtxt` reads one dtype at a time, so we make three passes: the numeric
# features, the integer target, and the string columns. That constraint is the
# whole reason pandas exists — a DataFrame is, more or less, a set of arrays of
# different dtypes with a shared index.

# %%
CSV = DATA_DIR / "service_telemetry.csv"
header = CSV.read_text().split("\n", 1)[0].split(",")
print("columns:", header)

FEATURE_COLS = slice(4, 12)          # cpu_util ... temperature_c
FEATURES = header[FEATURE_COLS]

X_raw = np.loadtxt(CSV, delimiter=",", skiprows=1, usecols=range(4, 12))
y = np.loadtxt(CSV, delimiter=",", skiprows=1, usecols=12, dtype=int)
meta = np.loadtxt(CSV, delimiter=",", skiprows=1, usecols=(2, 3, 13), dtype=str)
node_id, phase, split = meta[:, 0], meta[:, 1], meta[:, 2]

print(f"\nX_raw {X_raw.shape}  {X_raw.dtype}")
print(f"y     {y.shape}  {y.dtype}   positive rate {y.mean():.4f}")
print(f"features: {FEATURES}")

# %% [markdown]
# ## §1. The array: shape, dtype, axis
#
# Three attributes answer almost every "why did that break" question.

# %%
print(f"shape  : {X_raw.shape}      -> 20000 rows, 8 columns")
print(f"ndim   : {X_raw.ndim}           -> a 2-D array (a matrix)")
print(f"dtype  : {X_raw.dtype}     -> 64-bit floats")
print(f"size   : {X_raw.size:,}       -> total elements = 20000 * 8")
print(f"nbytes : {X_raw.nbytes / 1e6:.1f} MB")

# %% [markdown]
# ### What `axis` actually means
#
# The phrasing that gets people into trouble is "axis=0 is rows". Use this
# instead:
#
# > **`axis=k` is the axis that disappears.**
#
# `X` has shape `(20000, 8)`. Collapsing axis 0 removes the 20000 and leaves
# `(8,)` — one number per **column**, i.e. per feature. Collapsing axis 1
# removes the 8 and leaves `(20000,)` — one number per **row**, i.e. per window.

# %%
print(f"X_raw.shape             = {X_raw.shape}")
print(f"X_raw.sum(axis=0).shape = {X_raw.sum(axis=0).shape}   <- axis 0 gone: per-feature")
print(f"X_raw.sum(axis=1).shape = {X_raw.sum(axis=1).shape}   <- axis 1 gone: per-window")
print(f"X_raw.sum().shape       = {X_raw.sum().shape}        <- both gone: a scalar")

print("\nmean of each feature (axis=0) — this is the one you almost always want:")
for name, m in zip(FEATURES, X_raw.mean(axis=0)):
    print(f"  {name:20s} {m:10.4f}")

# %% [markdown]
# Averaging *across* features (`axis=1`) would add a CPU fraction to a
# temperature in Celsius and divide by 8. It runs without error and means
# nothing. **NumPy will not stop you from computing nonsense** — the shapes are
# valid, so it complies.

# %%
nonsense = X_raw.mean(axis=1)
print(f"X_raw.mean(axis=1) -> shape {nonsense.shape}, first value {nonsense[0]:.4f}")
print("no error, no warning, no meaning")

# %% [markdown]
# ## §2. Indexing: slices, masks, and the copy/view trap
#
# ### Basic slicing — `[rows, columns]`

# %%
print("first 3 rows, all columns:")
print(X_raw[:3])

cpu = FEATURES.index("cpu_util")
lat = FEATURES.index("disk_latency_ms")

print(f"\ncolumn {cpu} (cpu_util), first 5 : {X_raw[:5, cpu]}")
print(f"last row, first 3 features       : {X_raw[-1, :3]}")
print(f"every 5000th row, two columns    :\n{X_raw[::5000][:, [cpu, lat]]}")

# %% [markdown]
# Note `X_raw[:5, cpu]` returns shape `(5,)`, not `(5, 1)`. **An integer index
# drops that axis; a slice keeps it.** This is the source of endless shape
# mismatches:

# %%
print(f"X_raw[:5, cpu].shape      = {X_raw[:5, cpu].shape}       <- int index: axis dropped")
print(f"X_raw[:5, [cpu]].shape    = {X_raw[:5, [cpu]].shape}     <- list index: axis kept")
print(f"X_raw[:5, cpu:cpu+1].shape = {X_raw[:5, cpu:cpu + 1].shape}    <- slice: axis kept")

# %% [markdown]
# ### Boolean masks
#
# A mask is an array of `True`/`False` the same length as the axis you index.
# This is how you write `WHERE` clauses in NumPy.

# %%
is_incident = y == 1
is_post = phase == "post_deploy"

print(f"mask dtype {is_incident.dtype}, shape {is_incident.shape}")
print(f"incidents           : {is_incident.sum():,}")
print(f"post-deploy         : {is_post.sum():,}")
print(f"both (& = AND)      : {(is_incident & is_post).sum():,}")
print(f"either (| = OR)     : {(is_incident | is_post).sum():,}")
print(f"not incident (~)    : {(~is_incident).sum():,}")

# Booleans are 0/1 under the hood, so .mean() of a mask is a *rate*.
print(f"\nincident rate, pre-deploy  : {y[~is_post].mean():.4f}")
print(f"incident rate, post-deploy : {y[is_post].mean():.4f}")

# %% [markdown]
# Two things that trip everyone up:
#
# * Use `&`, `|`, `~` — **not** `and`, `or`, `not`. The Python keywords try to
#   collapse the whole array to one truth value and raise `ValueError`.
# * Parenthesise: `&` binds tighter than `==`, so `a == 1 & b == 1` parses as
#   `a == (1 & b) == 1`. Always write `(a == 1) & (b == 1)`.

# %%
try:
    _ = is_incident and is_post
except ValueError as exc:
    print(f"`and` on arrays raises: ValueError({exc})")

# %% [markdown]
# ### Fancy indexing: index by position

# %%
picks = np.array([0, 10, 100, 1000])
print(f"rows {picks} of cpu_util: {X_raw[picks, cpu]}")

# Reordering columns is just an index list.
reordered = X_raw[:, [lat, cpu]]
print(f"\nreordered to [disk_latency, cpu_util], first 2 rows:\n{reordered[:2]}")

# np.flatnonzero turns a mask into the positions where it is True.
idx = np.flatnonzero(is_incident)
print(f"\nfirst 5 incident row positions: {idx[:5]}   (total {idx.size:,})")

# %% [markdown]
# ### The copy/view trap
#
# **Slicing returns a view** — a window onto the same memory. Writing through it
# changes the original. **Fancy indexing and boolean masks return copies.**
#
# This is the single most dangerous thing in NumPy, because the failure is
# silent: no error, just data that quietly changed underneath you.

# %%
demo = np.arange(10)

view = demo[2:5]          # slice -> VIEW
view[0] = 999
print(f"after writing into a slice      : {demo}   <- the original changed!")

demo = np.arange(10)
copy = demo[[2, 3, 4]]    # fancy index -> COPY
copy[0] = 999
print(f"after writing into a fancy index: {demo}   <- original safe")

print(f"\nview.base is demo : {view.base is not None}")
print(f"copy.base is None : {copy.base is None}")
print("\nRule: if you are going to modify it and you did not mean to alias, "
      "call .copy().")

# %% [markdown]
# ## §3. Broadcasting
#
# Broadcasting lets arrays of different shapes combine without you writing a
# loop or materialising a big intermediate. The rules, in full:
#
# > Align the shapes **from the right**. For each axis, they are compatible if
# > they are **equal**, or one of them is **1**. An axis of length 1 is stretched
# > to match. A missing axis on the left is treated as 1.
#
# That is the entire specification.

# %%
X_small = X_raw[:4]                     # (4, 8)
col_mean = X_raw.mean(axis=0)           # (8,)

print(f"X_small     {X_small.shape}")
print(f"col_mean    {col_mean.shape}")
print("            (4, 8)")
print("               (8,)   <- aligned right, missing axis treated as 1")
print("            -------")
print(f"result      {(X_small - col_mean).shape}\n")
print(X_small - col_mean)

# %% [markdown]
# ### The failure everyone hits
#
# Subtracting a **per-row** mean does not broadcast, because `(20000,)` aligns
# against the *columns* (8) and 20000 ≠ 8.

# %%
row_mean = X_raw.mean(axis=1)           # (20000,)
try:
    X_raw - row_mean
except ValueError as exc:
    print(f"ValueError: {exc}")

print("\n  (20000, 8)")
print("     (20000,)   <- 20000 lines up against 8. Mismatch.")

# %% [markdown]
# The fix is `keepdims=True`, which leaves the collapsed axis in place with
# length 1 — exactly the length broadcasting stretches.

# %%
row_mean_kept = X_raw.mean(axis=1, keepdims=True)      # (20000, 1)
print(f"without keepdims: {row_mean.shape}")
print(f"with keepdims   : {row_mean_kept.shape}")
print(f"(20000, 8) - (20000, 1) -> {(X_raw - row_mean_kept).shape}  works")

# The same thing by hand, if you prefer being explicit:
print(f"row_mean[:, None] -> {row_mean[:, None].shape}   (None inserts an axis)")
assert np.array_equal(row_mean[:, None], row_mean_kept)

# %% [markdown]
# ### Broadcasting builds matrices from vectors
#
# A column vector against a row vector produces a full 2-D grid. This is how you
# write pairwise operations with no loop at all.

# %%
a = np.array([1, 2, 3])
print(f"a[:, None] shape {a[:, None].shape}, a[None, :] shape {a[None, :].shape}")
print(f"outer product (3,1) * (1,3) -> {(a[:, None] * a[None, :]).shape}")
print(a[:, None] * a[None, :])

# %% [markdown]
# ## §4. Reductions, and standardising by hand
#
# You know the formula: $z = (x - \mu) / \sigma$, per column. The only question
# is which axis, and the answer is *the one you want to disappear*.

# %%
def standardize(X, mean=None, std=None):
    """Standardise columns to mean 0, std 1.

    Pass `mean`/`std` to reuse statistics computed on another array -- which is
    what you must do for validation and test data, or you have leaked.

    Returns (X_scaled, mean, std).
    """
    # TODO (§4)   z = (x - mean) / std, per COLUMN
    #
    # Which axis disappears? You want one mean per feature, so the 20000
    # must go: `axis=0`.
    #
    # If `mean`/`std` are passed in, use them instead of recomputing --
    # that is how you apply TRAIN statistics to val/test without leaking.
    #
    # Guard against a constant column: std == 0 would give inf. Replace
    # those with 1.0 (`np.where`) before dividing.
    #
    # return: (X_scaled, mean, std)
    raise NotImplementedError('standardize')


X_std, mu, sigma = standardize(X_raw)
print(f"means after scaling (want 0): {X_std.mean(axis=0)}")
print(f"stds  after scaling (want 1): {X_std.std(axis=0)}")
assert np.allclose(X_std.mean(axis=0), 0, atol=1e-12)
assert np.allclose(X_std.std(axis=0), 1, atol=1e-12)
print("\nstandardised correctly")

# %% [markdown]
# ### `ddof`, the silent discrepancy
#
# `np.std` divides by $n$; `pandas.Series.std` and `np.std(ddof=1)` divide by
# $n-1$. At n = 20000 the difference is invisible; on 10 rows it is not. Know
# which one you are using.

# %%
tiny = X_raw[:10, cpu]
print(f"np.std(ddof=0) : {tiny.std():.6f}   (population, NumPy default)")
print(f"np.std(ddof=1) : {tiny.std(ddof=1):.6f}   (sample, pandas default)")
print(f"relative gap   : {abs(tiny.std() / tiny.std(ddof=1) - 1):.2%}")

# %% [markdown]
# ### Other reductions you will reach for

# %%
print(f"min / max per feature shape : {X_raw.min(axis=0).shape}")
print(f"argmax over axis 0          : {X_raw.argmax(axis=0)}")
print(f"                              ^ row index of each column's max")
print(f"cumulative sum of y, last   : {np.cumsum(y)[-1]:,}  (= y.sum() = {y.sum():,})")

# `any` / `all` answer yes-or-no questions about a whole array.
print(f"\nany negative value?  {(X_raw < 0).any()}")
print(f"all strictly > 0?    {(X_raw > 0).all()}   <- surprising?")
zero_cols = [FEATURES[j] for j in range(X_raw.shape[1]) if (X_raw[:, j] == 0).any()]
print(f"...because these columns contain exact zeros: {zero_cols}")
print("Check `all` claims against the data, not against your assumptions.")

# %% [markdown]
# ## §5. Vectorisation: deleting the loop
#
# You know that array operations are faster than Python loops. What matters for
# implementation is the *translation habit*: a loop that computes one number per
# row is almost always a reduction along `axis=1`.
#
# Concretely: compute a weighted risk score for every window.

# %%
weights = rng.normal(size=X_std.shape[1])


def score_loop(X, w):
    """The obvious way — and the way to stop writing."""
    # TODO (§5)   the deliberately slow version
    #
    # Two nested Python loops: for each row i, accumulate sum_j X[i,j]*w[j].
    # Write it once so the vectorised version below has something to beat.
    #
    # return: 1-D array of length X.shape[0]
    raise NotImplementedError('score_loop')


def score_vectorised(X, w):
    """The same computation as a single matrix-vector product."""
    # TODO (§5)   the same thing in one expression
    #
    # hint: this is exactly what the `@` operator does.
    raise NotImplementedError('score_vectorised')


t0 = time.perf_counter()
slow = score_loop(X_std, weights)
t_slow = time.perf_counter() - t0

t0 = time.perf_counter()
fast = score_vectorised(X_std, weights)
t_fast = time.perf_counter() - t0

assert np.allclose(slow, fast)
print(f"loop       : {t_slow * 1000:8.1f} ms")
print(f"vectorised : {t_fast * 1000:8.1f} ms")
print(f"speed-up   : {t_slow / t_fast:8.0f}x   (identical results)")

# %% [markdown]
# The speed-up is real but secondary. The reason to write `X @ w` is that it is
# **one line that says what it means**, and there is no index to get wrong.
#
# ### The translation table
#
# | Loop shape | Array version |
# |---|---|
# | accumulate `x[i] * w[i]` | `X @ w` |
# | one number per row | `X.something(axis=1)` |
# | one number per column | `X.something(axis=0)` |
# | `if cond: a else: b`, elementwise | `np.where(cond, a, b)` |
# | count matches | `mask.sum()` |
# | running total | `np.cumsum(x)` |
# | clamp to a range | `np.clip(x, lo, hi)` |
#
# `np.where` deserves a demonstration, because it replaces the most common loop
# of all:

# %%
risk_band = np.where(X_raw[:, cpu] > 0.8, "high",
                     np.where(X_raw[:, cpu] > 0.5, "medium", "low"))
bands, counts = np.unique(risk_band, return_counts=True)
for band, n in zip(bands, counts):
    print(f"  {band:7s} {n:6,}   incident rate {y[risk_band == band].mean():.4f}")

# %% [markdown]
# ## §6. Sorting, ranking, and top-k
#
# `np.sort` gives you sorted *values*. `np.argsort` gives you the *indices* that
# would sort — and that is almost always the one you want, because it lets you
# reorder other arrays the same way.

# %%
latency = X_raw[:, lat]
order = np.argsort(latency)

print(f"np.sort   -> values : {np.sort(latency)[-5:]}")
print(f"np.argsort-> indices: {order[-5:]}")
print(f"same thing          : {latency[order][-5:]}")
print(f"\nthe y-labels of the 5 highest-latency windows: {y[order][-5:]}")

# %% [markdown]
# ### Top-k without sorting everything

# %%
def top_k_indices(values, k):
    """Indices of the k largest values, largest first.

    `np.argpartition` is O(n) and only guarantees that the k-th element is in
    its final place -- everything before it is larger, in arbitrary order. So we
    partition first, then sort just those k.
    """
    # TODO (§6)   indices of the k largest values, largest first
    #
    # `np.argpartition(values, -k)[-k:]` gets you the k largest in O(n),
    # but in ARBITRARY order. Sort just those k to order them, then reverse.
    #
    # return: array of k indices
    # hint: `values[part]` then `np.argsort(...)[::-1]`, and remember to
    #       index back into `part` -- not into `values`.
    raise NotImplementedError('top_k_indices')


top10 = top_k_indices(latency, 10)
print(f"top-10 latency indices : {top10}")
print(f"their values           : {latency[top10]}")
assert np.array_equal(latency[top10], np.sort(latency)[::-1][:10])
print("\nmatches a full sort")

# %% [markdown]
# ### Percentiles by hand
#
# Implementing `np.percentile` is a good exercise because it is *almost*
# trivial, and the "almost" is where the bugs live: the position is usually
# fractional, so you interpolate between neighbours.

# %%
def percentile_manual(values, q):
    """Linear-interpolation percentile — NumPy's default method."""
    # TODO (§6)   linear-interpolation percentile (NumPy's default method)
    #
    #   1. sort the values
    #   2. the target position is (q/100) * (n - 1)  -- usually FRACTIONAL
    #   3. interpolate between the neighbours either side of that position
    #
    # The fractional position is the whole exercise; `s[int(pos)]` is the
    # off-by-a-bit bug this is designed to make you meet.
    raise NotImplementedError('percentile_manual')


for q in [50, 90, 99, 99.9]:
    ours = percentile_manual(latency, q)
    theirs = np.percentile(latency, q)
    print(f"  p{q:<5} ours {ours:9.4f}   numpy {theirs:9.4f}   "
          f"diff {abs(ours - theirs):.2e}")
    assert np.isclose(ours, theirs)
print("\nmatches np.percentile")

# %% [markdown]
# ### Ranking with ties
#
# Ranks come up constantly (AUC is a rank statistic). The subtlety is ties,
# which must all receive the *average* of the ranks they span.

# %%
def rank_average(values):
    """Ranks 1..n, ties averaged. Equivalent to scipy.stats.rankdata."""
    # TODO (§6)   ranks 1..n, with ties sharing their average rank
    #
    #   1. `np.argsort` gives the order; scatter 1..n back through it to get
    #      ordinal ranks (`ranks[order] = np.arange(1, n+1)`)
    #   2. `np.unique(values, return_inverse=True, return_counts=True)`
    #      gives each row a group code and each group its size
    #   3. `np.bincount(inverse, weights=ranks)` sums ranks per group;
    #      divide by the counts and index back out with `inverse`
    #
    # Step 1 alone is wrong whenever two values are equal -- which on real
    # data is always.
    raise NotImplementedError('rank_average')


test_vals = np.array([10.0, 20.0, 20.0, 20.0, 30.0, 10.0])
print(f"values : {test_vals}")
print(f"ranks  : {rank_average(test_vals)}")
print("         (the three 20s span ranks 3,4,5 -> all get 4)")

from scipy import stats  # only to check our work

assert np.allclose(rank_average(latency), stats.rankdata(latency))
print("\nmatches scipy.stats.rankdata on the real data")

# %% [markdown]
# ## §7. Grouping without pandas
#
# `df.groupby("node_id").mean()` is the operation people miss most when they
# drop to NumPy. Here is how it actually works underneath.
#
# The key is `np.unique(..., return_inverse=True)`: it gives you the distinct
# labels *and* an integer code per row. Once every row has an integer code,
# `np.bincount` does the aggregation.

# %%
def group_mean(labels, values):
    """Mean of `values` within each distinct label. Returns (labels, means)."""
    # TODO (§7)   pandas' groupby().mean(), in three lines
    #
    #   1. `np.unique(labels, return_inverse=True)` -> distinct keys, and an
    #      integer code per row
    #   2. `np.bincount(codes, weights=values)` sums values into buckets
    #   3. `np.bincount(codes)` counts them; divide
    #
    # return: (keys, means)
    # Pass `minlength=keys.size` so an empty trailing group is not dropped.
    raise NotImplementedError('group_mean')


nodes, node_rate = group_mean(node_id, y)
print("incident rate by node")
for n, r in zip(nodes, node_rate):
    print(f"  {n}  {r:.4f}   {'#' * int(r * 200)}")

print(f"\nweighted average back to the global rate: "
      f"{np.bincount(np.unique(node_id, return_inverse=True)[1]) @ node_rate / y.size:.4f}")
print(f"actual global rate:                        {y.mean():.4f}")

# %% [markdown]
# `np.bincount(codes, weights=v)` is the whole trick: it sums `v` into buckets
# indexed by `codes`. Counting is the same call with no weights.
#
# ### Two-way grouping
#
# Cross-tabulating is the same idea with a composite code: `code_a * n_b +
# code_b` flattens a 2-D grouping into one integer, exactly like row-major
# indexing.

# %%
def crosstab_rate(labels_a, labels_b, values):
    """Mean of `values` for every (a, b) combination."""
    # TODO (§7)   the same idea for TWO grouping columns
    #
    # Get codes for both label arrays, then flatten the pair into a single
    # integer: `flat = code_a * n_b + code_b`. That is row-major indexing by
    # hand. bincount over `flat`, then `.reshape(n_a, n_b)`.
    #
    # return: (keys_a, keys_b, table)
    # Use `np.maximum(counts, 1)` in the denominator so an empty cell gives
    # 0 rather than a divide-by-zero warning.
    raise NotImplementedError('crosstab_rate')


ka, kb, table = crosstab_rate(node_id, phase, y)
print(f"{'':10s}" + "".join(f"{b:>14s}" for b in kb))
for name, row in zip(ka, table):
    print(f"{name:10s}" + "".join(f"{v:14.4f}" for v in row))

# %%
fig, ax = plt.subplots(figsize=(7.6, 4.0))
width = 0.38
xs = np.arange(len(ka))
for k, (b, colour) in enumerate(zip(kb, [C["orange"], C["blue"]])):
    ax.bar(xs + (k - 0.5) * width, table[:, k], width * 0.92, color=colour, label=b)
ax.axhline(y.mean(), color=INK["muted"], linewidth=1.2)
ax.annotate(f"fleet average {y.mean():.3f}", xy=(0, y.mean()), xytext=(0, 6),
            textcoords="offset points", fontsize=8.5, color=INK["muted"])
ax.set_xticks(xs)
ax.set_xticklabels(ka, rotation=30, ha="right")
finish(ax, "Incident rate by node and deploy phase — computed with bincount",
       ylabel="incident rate", legend=True)
fig.tight_layout()
fig.savefig(FIG_DIR / "01_group_rates.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# ## §8. Building a design matrix
#
# Now we assemble the matrix a model actually consumes — all with array
# operations.
#
# ### Column transforms

# %%
LOG_COLS = [FEATURES.index(c) for c in
            ["disk_latency_ms", "request_rate_rps", "error_rate_pct"]]


def apply_log_transform(X, cols):
    """log1p the given columns, leave the rest alone. Does not modify X."""
    # TODO (§8)   log1p the listed columns, leave the others alone
    #
    # Start from `X.copy()`. Without the copy you modify the CALLER's array
    # in place -- the view/copy trap from §2, in its most expensive form.
    #
    # Why log1p and not log: `error_rate_pct` is exactly 0 for many rows,
    # and log(0) = -inf.
    raise NotImplementedError('apply_log_transform')


X_log = apply_log_transform(X_raw, LOG_COLS)
print(f"{'feature':22s}{'raw skew':>12}{'logged skew':>14}")
for j, name in enumerate(FEATURES):
    marker = " <-" if j in LOG_COLS else ""
    print(f"{name:22s}{stats.skew(X_raw[:, j]):12.2f}"
          f"{stats.skew(X_log[:, j]):14.2f}{marker}")
assert not np.shares_memory(X_log, X_raw), "must not alias the input"

# %% [markdown]
# ### One-hot encoding
#
# Categorical → numeric, with nothing but `np.unique` and fancy indexing. The
# identity-matrix trick is worth remembering: row `k` of `np.eye(n)` *is* the
# one-hot vector for category `k`.

# %%
def one_hot(labels):
    """Encode string labels as a 0/1 matrix. Returns (matrix, category_names)."""
    # TODO (§8)   categorical labels -> 0/1 matrix
    #
    # `np.unique(labels, return_inverse=True)` gives an integer code per row.
    # Then the trick: row k of `np.eye(n)` IS the one-hot vector for
    # category k, so `np.eye(n)[codes]` encodes everything at once.
    #
    # return: (matrix of shape (len(labels), n_categories), keys)
    raise NotImplementedError('one_hot')


node_oh, node_keys = one_hot(node_id)
print(f"one-hot shape {node_oh.shape} for {node_keys.size} nodes")
print(f"first 3 rows:\n{node_oh[:3].astype(int)}")
print(f"\nevery row sums to 1 : {np.all(node_oh.sum(axis=1) == 1)}")
print(f"column sums = counts: {node_oh.sum(axis=0).astype(int)}")

# %% [markdown]
# ### Cross-features by broadcasting
#
# Lesson 1 built these with `itertools`. Here is the array-native way:
# `X[:, :, None] * X[:, None, :]` builds every pairwise product at once, then
# `np.triu_indices` picks the upper triangle (so we keep `x_i * x_j` once, not
# twice).

# %%
def add_cross_features(X):
    """Append every product x_i * x_j with i <= j. Returns (expanded, pairs)."""
    # TODO (§8)   append every product x_i * x_j with i <= j
    #
    #   1. `X[:, :, None] * X[:, None, :]` broadcasts (n,d,1) against (n,1,d)
    #      to give (n, d, d): every pairwise product for every row
    #   2. `np.triu_indices(d)` gives the upper-triangle (i, j) pairs,
    #      including the diagonal, so each product is kept once
    #   3. index with `outer[:, i, j]` and hstack onto the original
    #
    # return: (expanded, pairs) where pairs is list(zip(i, j))
    raise NotImplementedError('add_cross_features')


X_cross, pairs = add_cross_features(X_std)
print(f"{X_std.shape[1]} features -> {X_cross.shape[1]} "
      f"({len(pairs)} products added)")

# Verify a couple of columns against an explicit computation.
for k in [0, 5, len(pairs) - 1]:
    i, j = pairs[k]
    assert np.allclose(X_cross[:, X_std.shape[1] + k], X_std[:, i] * X_std[:, j])
print(f"spot-checked columns; e.g. product #{len(pairs) - 1} is "
      f"{FEATURES[pairs[-1][0]]} * {FEATURES[pairs[-1][1]]}")

# %% [markdown]
# That `(n, d, d)` intermediate is 20000 × 8 × 8 = 1.28 M floats — fine here,
# but it grows as $d^2$. **Broadcasting is not free**: it materialises the whole
# result even when the inputs are small, and here that result is a hundred times
# larger than the data you started with.

# %%
for d in [8, 100, 1000]:
    gb = 20000 * d * d * 8 / 1e9
    size = f"{gb * 1000:.1f} MB" if gb < 1 else f"{gb:,.0f} GB"
    print(f"  d = {d:5d}  ->  intermediate is {size:>10s}"
          f"   ({'fine' if gb < 1 else 'out of the question'})")

# %% [markdown]
# ## §9. Splitting the data by hand
#
# The dataset ships with a `split` column, but building one yourself is a
# five-line exercise that exposes exactly what `train_test_split` does.

# %%
def stratified_split(y, fractions=(0.6, 0.2, 0.2), seed=0):
    """Assign each row to a split, keeping the class balance in each.

    Returns an integer array: 0 = train, 1 = val, 2 = test.
    """
    # TODO (§9)   assign rows to train/val/test, preserving class balance
    #
    # For each class separately: take its row positions, shuffle them with a
    # seeded generator, and cut at the cumulative fractions. Doing it per
    # class is what makes it stratified.
    #
    # return: int array, 0 = train, 1 = val, 2 = test
    # hint: `np.split(idx, cuts)` splits at a list of cut positions;
    #       `(np.cumsum(fractions)[:-1] * idx.size).astype(int)` builds them.
    raise NotImplementedError('stratified_split')


ours = stratified_split(y, seed=RANDOM_SEED)
print(f"{'split':8s}{'n':>8}{'positive rate':>16}")
for part, name in enumerate(["train", "val", "test"]):
    m = ours == part
    print(f"{name:8s}{m.sum():8,}{y[m].mean():16.4f}")
print(f"\nglobal rate {y.mean():.4f} — stratification kept all three aligned")

# %% [markdown]
# For the rest of the notebook we use the **shipped** split, so results line up
# with lesson 1.

# %%
train = split == "train"
val = split == "val"
test = split == "test"
print(f"train {train.sum():,}   val {val.sum():,}   test {test.sum():,}")

# %% [markdown]
# ## §10. Fitting a model with nothing but NumPy
#
# You know both of these. Type them anyway — the indices are the point.
#
# ### The normal equation
#
# $\hat w = (X^\top X)^{-1} X^\top y$, implemented with `solve` rather than
# `inv`.

# %%
def fit_normal_equation(X, y):
    """Least squares with an intercept. Returns (weights, intercept)."""
    # TODO (§10)   w = (X^T X)^-1 X^T y
    #
    # Prepend a column of ones for the intercept (`np.hstack` with
    # `np.ones((n, 1))`), solve, then split the intercept back off the front.
    #
    # return: (weights, intercept)
    # hint: `np.linalg.solve(A, b)`, never `np.linalg.inv(A) @ b`.
    raise NotImplementedError('fit_normal_equation')


# Standardise using TRAIN statistics only, then apply them to val/test.
X_tr, mu_tr, sd_tr = standardize(X_log[train])
X_va, *_ = standardize(X_log[val], mu_tr, sd_tr)
X_te, *_ = standardize(X_log[test], mu_tr, sd_tr)
y_tr, y_va, y_te = y[train], y[val], y[test]

w, b = fit_normal_equation(X_tr, y_tr)
print(f"{'feature':22s}{'weight':>10}")
for name, wi in zip(FEATURES, w):
    print(f"{name:22s}{wi:10.4f}")
print(f"{'intercept':22s}{b:10.4f}")

# %% [markdown]
# ### Gradient descent

# %%
def fit_gradient_descent(X, y, lr=0.1, n_iters=2000):
    """Full-batch gradient descent on the MSE. Returns (w, b, loss_history)."""
    # TODO (§10)   full-batch gradient descent on the MSE
    #
    # Start at w = 0, b = 0, then repeat n_iters times:
    #   resid = X @ w + b - y            (n,)
    #   record mean(resid**2)
    #   w -= lr * (2/n) * (X.T @ resid)  (d,)
    #   b -= lr * (2/n) * resid.sum()
    #
    # Watch the shapes: `X.T @ resid` is (d,n)@(n,) -> (d,), matching w.
    #
    # return: (w, b, history)
    raise NotImplementedError('fit_gradient_descent')


w_gd, b_gd, hist = fit_gradient_descent(X_tr, y_tr, lr=0.3, n_iters=3000)
print(f"max |w_gd - w_exact| : {np.abs(w_gd - w).max():.2e}")
print(f"|b_gd - b_exact|     : {abs(b_gd - b):.2e}")
assert np.allclose(w_gd, w, atol=1e-6)
print("\ngradient descent reached the closed-form solution")

# %%
fig, ax = plt.subplots(figsize=(7.0, 4.0))
final = np.mean((X_tr @ w + b - y_tr) ** 2)
ax.plot(hist - final, color=C["blue"])
ax.set_yscale("log")
ax.set_xscale("log")
finish(ax, "Gradient descent: distance from the closed-form optimum",
       xlabel="iteration", ylabel="loss − optimal loss")
fig.tight_layout()
fig.savefig(FIG_DIR / "02_gd_convergence.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# ## §11. Metrics in pure NumPy
#
# ### The confusion matrix in one line
#
# Encode each row as a 2-bit number — `2 * y_true + y_pred` — and count. That
# maps `(0,0) -> 0`, `(0,1) -> 1`, `(1,0) -> 2`, `(1,1) -> 3`, so a `bincount`
# of length 4 reshaped to `(2, 2)` *is* the confusion matrix.

# %%
def confusion_matrix_numpy(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    # TODO (§11)   the 2x2 matrix [[TN, FP], [FN, TP]] in two lines
    #
    # Encode each row as a 2-bit number: `2 * y_true + y_pred`. That sends
    #   (0,0)->0   (0,1)->1   (1,0)->2   (1,1)->3
    # so `np.bincount(codes, minlength=4).reshape(2, 2)` IS the confusion
    # matrix, in exactly sklearn's layout.
    raise NotImplementedError('confusion_matrix_numpy')


scores_te = X_te @ w + b
pred_te = (scores_te >= 0.5).astype(int)
cm = confusion_matrix_numpy(y_te, pred_te)

(tn, fp), (fn, tp) = cm
print(f"                predicted 0   predicted 1")
print(f"  actual 0      {tn:>11,}   {fp:>11,}")
print(f"  actual 1      {fn:>11,}   {tp:>11,}")

from sklearn.metrics import confusion_matrix  # only to check our work

assert np.array_equal(cm, confusion_matrix(y_te, pred_te))
print("\nmatches sklearn")

# %%
print(f"accuracy  {(tp + tn) / cm.sum():.4f}")
print(f"precision {tp / (tp + fp):.4f}")
print(f"recall    {tp / (tp + fn):.4f}")

# %% [markdown]
# ### ROC and AUC with `argsort` and `cumsum`
#
# Sort by descending score; then sweeping the threshold down that list adds one
# object at a time to the "predicted positive" set. So the running true- and
# false-positive counts are just cumulative sums of the sorted labels.

# %%
def roc_curve_numpy(y_true, score):
    """ROC curve from scratch. Returns (fpr, tpr)."""
    # TODO (§11)   the ROC curve from argsort and cumsum
    #
    #   1. order by DESCENDING score (`np.argsort(-score, kind='mergesort')`)
    #   2. walking down that list adds one object at a time to 'predicted
    #      positive', so running TP/FP counts are `np.cumsum` of the sorted
    #      labels and of (1 - labels)
    #   3. tied scores must be committed to as a block: keep only the last
    #      index of each run of equal scores
    #      (`np.flatnonzero(np.diff(sorted_scores))`, plus the final index)
    #   4. divide by the totals, and prepend the (0, 0) origin
    #
    # return: (fpr, tpr)
    raise NotImplementedError('roc_curve_numpy')


def auc_numpy(y_true, score):
    """AUC as the Mann-Whitney statistic: P(random positive > random negative)."""
    # TODO (§11)   AUC as a rank statistic
    #
    #         U = R+ - n+ * (n+ + 1) / 2,      AUC = U / (n+ * n-)
    #
    # where R+ is the sum of the ranks of the positives. Reuse the
    # `rank_average` you wrote in §6 -- the averaged ties are exactly the
    # convention this formula needs.
    raise NotImplementedError('auc_numpy')


fpr, tpr = roc_curve_numpy(y_te, scores_te)
auc_trapz = np.trapezoid(tpr, fpr)
auc_rank = auc_numpy(y_te, scores_te)

from sklearn.metrics import roc_auc_score

print(f"AUC via trapezoid   : {auc_trapz:.10f}")
print(f"AUC via ranks       : {auc_rank:.10f}")
print(f"AUC via sklearn     : {roc_auc_score(y_te, scores_te):.10f}")
assert np.isclose(auc_trapz, roc_auc_score(y_te, scores_te))
assert np.isclose(auc_rank, roc_auc_score(y_te, scores_te))
print("\nthree routes, one number")

# %%
fig, ax = plt.subplots(figsize=(5.6, 5.4))
ax.plot([0, 1], [0, 1], color=INK["axis"], linewidth=1.2)
ax.plot(fpr, tpr, color=C["blue"], label=f"AUC = {auc_rank:.4f}")
ax.set_aspect("equal")
ax.set_xlim(-0.01, 1.01)
ax.set_ylim(-0.01, 1.01)
finish(ax, "ROC curve, built with argsort and cumsum",
       xlabel="false positive rate", ylabel="true positive rate", legend=True)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_roc_numpy.png", bbox_inches="tight", dpi=140)
plt.show()

# %% [markdown]
# ## §12. End to end
#
# Everything above, in one function: raw array in, test AUC out, no pandas and
# no scikit-learn anywhere in the path.

# %%
def run_pipeline(X_raw, y, train, test, log_cols, use_cross=False):
    """Full pipeline with NumPy only. Returns (test_auc, weights, intercept)."""
    X = apply_log_transform(X_raw, log_cols)
    X_tr, mean, std = standardize(X[train])
    X_te, *_ = standardize(X[test], mean, std)

    if use_cross:
        X_tr, _ = add_cross_features(X_tr)
        X_te, _ = add_cross_features(X_te)
        X_tr, m2, s2 = standardize(X_tr)
        X_te, *_ = standardize(X_te, m2, s2)

    w, b = fit_normal_equation(X_tr, y[train])
    return auc_numpy(y[test], X_te @ w + b), w, b


auc_plain, *_ = run_pipeline(X_raw, y, train, test, LOG_COLS, use_cross=False)
auc_cross, *_ = run_pipeline(X_raw, y, train, test, LOG_COLS, use_cross=True)

print(f"plain features  : test AUC {auc_plain:.4f}")
print(f"cross-features  : test AUC {auc_cross:.4f}")
print(f"difference      : {auc_cross - auc_plain:+.4f}")

# %% [markdown]
# Those are the same two models as lesson 1, reproduced with array operations
# alone. (They differ slightly in the third decimal because lesson 1 used
# `log` on two columns where we used `log1p` on three — a reminder that
# preprocessing choices are part of the model.)
#
# ## Cheat sheet
#
# The operations that cover most of what you will write:
#
# | Task | Call |
# |---|---|
# | per-column stat | `X.mean(axis=0)` |
# | per-row stat | `X.mean(axis=1)` |
# | keep the axis for broadcasting | `X.mean(axis=1, keepdims=True)` |
# | insert an axis | `v[:, None]` |
# | filter rows | `X[mask]` |
# | mask → positions | `np.flatnonzero(mask)` |
# | elementwise if/else | `np.where(cond, a, b)` |
# | sort, keep the mapping | `np.argsort(v)` |
# | k largest | `np.argpartition(v, -k)[-k:]` |
# | distinct values + codes | `np.unique(v, return_inverse=True)` |
# | group sum / count | `np.bincount(codes, weights=v)` |
# | one-hot | `np.eye(n)[codes]` |
# | all pairwise products | `X[:, :, None] * X[:, None, :]` |
# | stack columns | `np.hstack` / `np.column_stack` |
# | solve a linear system | `np.linalg.solve(A, b)` |
#
# ### The three rules again
#
# 1. **`axis=k` is the axis that disappears.**
# 2. **Broadcasting aligns from the right; length-1 axes stretch.** `keepdims`
#    keeps the axis so it can.
# 3. **Slices are views, fancy indexing copies.** `.copy()` when you mean to.
#
# ### If you are stuck
#
# Print `.shape` at every step. Nearly every NumPy bug is a shape you did not
# expect, and shapes are cheap to look at.

# %%
print("checks that ran in this notebook:")
for line in [
    "standardize        -> mean 0, std 1 exactly",
    "score_loop         == score_vectorised",
    "top_k_indices      == full sort",
    "percentile_manual  == np.percentile",
    "rank_average       == scipy.stats.rankdata",
    "add_cross_features == explicit products",
    "fit_gradient_descent == fit_normal_equation",
    "confusion_matrix_numpy == sklearn",
    "auc_numpy == trapezoid == sklearn",
]:
    print(f"  [ok] {line}")
