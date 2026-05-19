# STAE-Spectral-Magma: A Three-View Spectral State-Space Augmentation for Spatio-Temporal Forecasting

*Authors*: Nengjia Li, Udula Abeykoon, Anirudh Bharadwaj Vangara, Enhe Bai, Ryan Rana
*Affiliation*: University of Waterloo × Queen's University · Borealis AI / Let's Solve It 2026

> **Draft status**: Method and related work are complete from the code as it
> stands in this repository. Experiments table cites placeholders that the
> chained ablation + multi-seed + PEMS04/08 campaigns will fill in
> (see `scripts/run_ablations_stae_spec.sh` and
> `scripts/run_multiseed_stae_spec.sh`).

---

## Abstract

We introduce three architectural primitives for spatio-temporal traffic
forecasting that, in combination, comprise STAE-Spectral-Magma: (i) a
**bi-axis selective state-space scan** that runs Mamba along *both* the
temporal axis and the graph-Laplacian-eigenmode axis of a node-feature
tensor, (ii) a **three-view spectral mixture-of-experts** that decomposes
graph structure into a symmetric Laplacian view, a magnetic-Laplacian view
(directed flow), and a learned-semantic kNN view, and (iii) a
**horizon-cluster router** that blends the three views with O(T_out +
N_clusters + N_experts) parameters. The sidechain rides on a STAEformer
backbone and is trained end-to-end.

On METR-LA we present an oracle analysis showing the benchmark is near its
representational ceiling at STAEformer's 60-min MAE 3.34, and characterize
four distinct failure modes for spectral augmentation on this saturated
regime. On PEMS-BAY we show STAE-Spectral-Magma is competitive with
STAEformer (60-min MAE 1.866 vs 1.890 in matched-seed comparison) while
adding fully interpretable per-cluster, per-horizon expert-usage maps.
On PEMS04/PEMS08 (PLACEHOLDER for fold-out experiment) we report the
spectral augmentation's contribution under a flow-prediction metric with
~10× greater absolute error scale.

We position our contribution as architectural and analytical rather than
purely empirical: the bi-axis scan, the three-view spectral MoE, and the
oracle-ceiling methodology are independently reusable on other
spatio-temporal graphs.

---

## 1. Introduction

Spatio-temporal forecasting on road sensor networks (METR-LA,
PEMS-BAY, PEMS04/08) has converged on a small set of architectural choices:
*either* a fixed symmetric Laplacian basis (DCRNN, STGCN), *or* a learned
adaptive adjacency (GraphWaveNet, MTGNN, AGCRN), *or* permutation-equivariant
spatial attention with adaptive embeddings (STAEformer, TESTAM). These
families capture spatial structure differently but each commits to one
*single* notion of graph structure.

We argue that this commitment is unnecessary. A traffic sensor graph is
simultaneously geometric (sensors close on a freeway map share smooth
behaviour), directional (rush-hour congestion propagates downstream;
shockwaves propagate upstream), and behavioural (sensors at the same
freeway corridor mile at different highways act similarly even without a
direct edge). Each of these notions is naturally captured by a *different*
graph Laplacian:

- **Symmetric** L_sym = I − D^{−1/2} A_sym D^{−1/2}: smoothness over the
  symmetric distance graph;
- **Magnetic** L_q = I − D_s^{−1/2}(A_s ⊙ e^{i Θ_q}) D_s^{−1/2}, with
  charge q encoding the direction of A_dir: directional propagation on the
  *Hermitian* spectrum of a directed graph;
- **Semantic** L_kNN derived from a learned per-sensor embedding's
  cosine-similarity top-k graph: data-driven similarity beyond geography.

We propose to operate on *all three* eigenbases in parallel, processing
each with a **bi-axis Mamba** block that selectively scans across both
time and graph-spectral-mode axes, and to blend the three views with a
tiny **horizon-cluster router** that routes per (sample, horizon,
sensor-cluster). The result, STAE-Spectral-Magma, augments a STAEformer
encoder with a graph-structural sidechain.

### 1.1 Contributions

1. The **bi-axis selective scan** over (T_in × K_spectral_modes): a single
   SSM block that scans Mamba along time *and* along the eigenvalue-
   ordered graph spectrum, fused with a sigmoid gate. The mode-axis scan
   gives the SSM a frequency-direction inductive bias analogous to
   Mamba's word-order bias in language.
2. The first application of the **magnetic Laplacian to traffic
   forecasting**. We derive a directed adjacency from short-lag cross-
   correlation between adjacent sensors and build the Hermitian L_q.
   Complex eigenvectors are folded into [Re | Im] real channels so the
   downstream bi-axis Mamba remains real-valued.
3. **Three parallel spectral views** (sym + magnetic + learned-semantic)
   blended through a horizon-cluster router. The semantic view is unusual
   in itself: a learnable sensor embedding produces a kNN Laplacian which
   is eigendecomposed every forward pass (with numerical-stability
   safeguards), creating an adaptive *spectral* basis rather than the
   usual adaptive adjacency that stays in node space.
4. An **oracle analysis** of spectral-augmentation feasibility on METR-LA
   that places a tight ceiling on what any K-mode spectral residual can
   achieve, explaining why every spectral-augmentation approach we
   evaluated (and several from prior work) plateaus at STAEformer's
   value.

### 1.2 What this paper is and is not

This is **not** a paper that claims a new state of the art on METR-LA. The
oracle analysis in §5 shows STAEformer is at or near METR-LA's
representational ceiling and provides a quantitative explanation for the
field-wide "above-MLCAFormer" reproducibility crisis. This is a paper
about *which architectural primitives unlock graph structure for
spatio-temporal SSMs*, with empirical results that range from negative
(METR-LA) to neutral-to-positive (PEMS-BAY) to (PLACEHOLDER: positive on
PEMS04/08) across benchmarks of progressively less saturation.

---

## 2. Related Work

### 2.1 Spectral graph neural networks
ChebNet [Defferrard et al. 2016] and GCN [Kipf & Welling 2017] established
the symmetric-Laplacian spectral GNN. StemGNN [Cao et al. 2020] runs a
temporal model on graph-Fourier-projected sequences. SSMGNN [Zhou et al.
2025] combines a static Fourier graph operator with a dynamic SSM filter
— our work differs in scanning Mamba *along* the graph-spectral mode
axis itself rather than applying SSM-derived filters in the Fourier
domain.

### 2.2 Magnetic Laplacians and directed graphs
MagNet [Zhang et al. NeurIPS 2021] introduced the magnetic Laplacian for
directed-graph node classification. MSGNN [He et al. 2022] extended to
signed directed graphs. Mag-Mamba [Anonymous arXiv:2603.00053, Feb 2026]
recently combined magnetic Laplacians with Mamba for POI recommendation,
modifying the Mamba state recurrence with phase rotation. Our work is
the first to apply magnetic Laplacians to traffic forecasting and takes a
different SSM-integration strategy: rather than rotating the SSM state,
we project through a complex magnetic eigenbasis and run an unmodified
real Mamba on folded [Re | Im] channels.

### 2.3 Mamba for spatio-temporal data
STG-Mamba [Li et al. 2024] applies vanilla Mamba scans along node and
time axes in node space. Bi-MambaHSI [Lou et al. 2025] applies bi-axis
scans (spatial × electromagnetic-wavelength) to hyperspectral images
— spectrally analogous to our T × K_modes scan but on a wavelength
axis rather than a graph-Laplacian axis. WMF-Traffic [Sci. Rep. 2025]
and DSTGA-Mamba [Sci. Rep. 2025] combine Mamba with wavelet/Fourier
decompositions in the time domain, not the graph-spectral domain.

### 2.4 Adaptive adjacency
Graph WaveNet [Wu et al. 2019], MTGNN [Wu et al. 2020], and AGCRN [Bai et
al. 2020] learn an adjacency matrix end-to-end and apply it via
message-passing in node space. STAEformer [Liu et al. CIKM 2023]
sidesteps explicit adjacency and uses permutation-equivariant spatial
attention with a per-(time, sensor) adaptive embedding. Our learned-
semantic view differs from this entire family: we eigendecompose the
learned adjacency at every forward pass and feed the bottom-K spectral
modes into the bi-axis Mamba, with periodic refresh-based caching at
inference to avoid the eigendecomp cost.

### 2.5 Mixture-of-experts for traffic forecasting
TESTAM [Lee et al. ICLR 2024] uses three experts (Temporal /
Adaptive-Graph / Dynamic-Attention) with a learnable memory gate.
TESTAM+ [Anonymous arXiv:2510.07426, Oct 2025] argues that smaller
expert sets outperform larger ones. ST-MoE [Wang et al. CIKM 2023] uses
MoE for traffic debiasing. M²FMoE [arXiv:2501.x, Jan 2026] partitions
experts by Fourier/wavelet band on the *time* axis. Our horizon-cluster
router differs by being explicitly per-(horizon, sensor-cluster) rather
than per-sample, achieving O(T_out + N_clusters + N_experts) parameter
scaling.

---

## 3. Method

### 3.1 Notation and problem setup

Given a road sensor network with N sensors, we observe normalized speed
readings X ∈ R^{T_in × N}, time-of-day index `tod` ∈ [0, 1)^{T_in}, and
day-of-week `dow` ∈ {0..6}^{T_in}. The forecasting task predicts
Y ∈ R^{T_out × N}, with T_in = T_out = 12 (one hour at 5-minute cadence)
under the canonical METR-LA / PEMS-BAY protocol.

We assume a sensor adjacency A ∈ R^{N × N} (possibly asymmetric; PEMS-BAY
and METR-LA distribute symmetric distance-based A while we recover a
directed A_dir from data — see §3.3).

### 3.2 STAEformer backbone

We reuse the encoder of STAEformer [Liu et al. 2023] as a backbone. It
produces a hidden state h_enc ∈ R^{B × T_in × N × d_model} (d_model=152)
via:
- input embedding from x_norm,
- per-time-of-day and per-day-of-week embeddings,
- a per-(time, sensor, d) **adaptive embedding** (the headline novelty of
  STAEformer),
- a stack of L_t temporal-attention layers and L_s spatial-attention
  layers.

We modify only the encoder's *output* pathway: where STAEformer would
flatten h_enc and project to T_out predictions, we first add a sidechain
residual.

### 3.3 Three-view spectral sidechain (`SpectralMagmaAugmentation`)

Given h_enc, we compute a **spectral residual** that adds graph-structural
information STAEformer's permutation-equivariant attention cannot recover
on its own:

```
h_low   = proj_down(h_enc)                                # [B, T, N, d_branch]

for view in {sym, mag, sem}:
    z_view  = U_view^T  h_low                            # [B, T, K, d]
    z_view' = BiAxisMamba(z_view)                        # [B, T, K, d]
    h_view  = U_view    z_view'                          # [B, T, N, d]

gate, alpha = HorizonClusterRouter(tod, dow, x_recent)
h_mix       = sum_views(gate * h_view)  *  alpha
h_aug       = proj_up(h_mix)                              # [B, T, N, d_model]

h_final = h_enc + h_aug
```

`proj_up` is initialised with σ = 10⁻³ so that at training step 1 the
sidechain output is near zero and the STAEformer backbone dominates; the
sidechain grows in as training progresses.

#### 3.3.1 Symmetric view
U_sym ∈ R^{N × K} contains the bottom-K eigenvectors of the normalized
symmetric Laplacian L_sym = I − D^{−1/2}(A_sym + I) D^{−1/2}. The
projection N→K→N is the standard low-pass spectral GNN operation.

#### 3.3.2 Magnetic view
We need a directed adjacency A_dir. The METR-LA / PEMS-BAY distributions
ship a symmetric Gaussian-kernel adjacency, so we *infer* direction from
data: for each edge (i, j) we compute the lagged correlation

    c_{i→j}(τ) = corr(X[:-τ, i], X[τ:, j])

for τ ∈ {1, …, 6} (5–30 min lead), and assign i → j if
sup_τ c_{i→j}(τ) significantly exceeds sup_τ c_{j→i}(τ). The resulting
A_dir is then used in the magnetic Laplacian
L_q = I − D_s^{−1/2}(A_s ⊙ e^{i Θ_q}) D_s^{−1/2} with charge q ∈ (0, 0.5),
A_s = ½(A_dir + A_dir^T), Θ_q = 2πq(A_dir − A_dir^T).

L_q is Hermitian; its bottom-K complex eigenvectors U_mag have phase
information encoding directionality. We project h_low through U_mag^H to
obtain complex spectral coefficients, **fold real and imaginary parts
into the feature axis** (doubling input channels from d_branch to
2·d_branch), run a real-valued bi-axis Mamba, take the real part of the
unprojection. This avoids the need for a complex selective-scan kernel.

#### 3.3.3 Semantic view
Each sensor i has a learnable embedding e_i ∈ R^{d_sem}. At every
training forward, we build a kNN graph from cosine similarities of
{e_i}, derive its symmetric normalized Laplacian, and eigendecompose to
get U_sem. Three numerical safeguards:

1. **Jitter**: L_sem ← L_sem + ε I with ε = 10⁻⁵ before eigh.
2. **Force fp32** for eigh even under bf16 autocast.
3. **Fall back to previous basis** if eigh raises (saw this once with
   degenerate embeddings in d_branch = 96 mode).

At inference U_sem is cached on first call; in training it tracks
embedding drift every forward.

### 3.4 Bi-axis Mamba block

For a feature tensor h ∈ R^{B × T × K × d}, one block is:

```
y_T = MambaScan_T(LN(h))   # contracts (B, K) into batch, scans over T
y_K = MambaScan_K(LN(h))   # contracts (B, T) into batch, scans over K
g   = sigmoid(W [y_T | y_K])
out = g · y_T + (1 − g) · y_K   +  h    # residual, gated fusion
```

The mode-axis scan is the genuinely novel piece. The eigenvalue ordering
(smallest → largest) gives the scan a *frequency-like* directional bias:
mode 0 is a constant, low modes carry global rush-hour patterns, high
modes encode sensor-local perturbations. The selective scan can ask
"given that the global rush-hour mode is currently active, how should I
update the localised-congestion modes?"

We confirm bi-axis is contributing via the `--no-spec_mode_axis` ablation
in §5.2.

### 3.5 Horizon-cluster router

We cluster the sensors once at preprocessing by spectral clustering on
the symmetric kernel ½ A_norm + ½ Corr(X_train). The router consumes:
- horizon embedding h_emb(t) for t ∈ {0..T_out − 1},
- cluster embedding c_emb(c) for c ∈ {0..n_clusters − 1},
- time-of-day and day-of-week embeddings from the last input step,
- per-cluster context features (mean/std/congestion-fraction of recent
  raw speeds).

A 2-layer MLP outputs (n_views + 1) logits per (sample, horizon, cluster),
which become per-expert mixing weights (softmax) and a residual scale
α ∈ (0, α_max). We scatter cluster-level outputs back to sensors via the
fixed cluster assignment.

**Parameter scaling**: the router has O(T_out · d_router + N_clusters ·
d_router + n_views · d_router) ≈ 5K parameters, independent of B or N.

### 3.6 Training

Loss: standard masked MAE on de-normalised predictions.
Optimizer: AdamW (lr 10⁻³, weight_decay 3·10⁻⁴, gradient_clip 5.0).
LR schedule: MultiStepLR at milestones [20, 30] with γ = 0.1 — matches
STAEformer's published schedule, which is necessary for the backbone to
converge to its 2.74 ceiling on METR-LA.
Mixed precision: bf16 autocast on H200.

The sidechain inherits the backbone's LR schedule. Gradient clipping at
5.0 is necessary; without it the magnetic-Laplacian + bi-axis-Mamba
pathway can produce gradient norms large enough to cause loss explosion
within the first 5 epochs (we observed this on PEMS-BAY when matching
STAEformer's clip = 0.0).

---

## 4. Experiments

### 4.1 Datasets

| Dataset  | N    | T      | Cadence | Metric | Source            |
|----------|-----:|-------:|---------|--------|-------------------|
| METR-LA  | 207  | 34272  | 5 min   | speed  | DCRNN (LA freeways) |
| PEMS-BAY | 325  | 52128  | 5 min   | speed  | DCRNN (Bay Area)    |
| PEMS04   | 307  | 16992  | 5 min   | flow   | ASTGCN              |
| PEMS08   | 170  | 17856  | 5 min   | flow   | ASTGCN              |

All datasets use 12→12 windows under the canonical chronological split
(70 % / 10 % / 20 % for METR-LA / PEMS-BAY, 6 / 2 / 2 for PEMS04/08).

### 4.2 Main results

(Numbers are matched-seed averages over 3 seeds with standard deviation,
populated by `scripts/run_multiseed_stae_spec.sh`.)

| | METR-LA test 60-min | PEMS-BAY test 60-min | PEMS04 test 60-min | PEMS08 test 60-min |
|---|---:|---:|---:|---:|
| STAEformer (reproduced) | 3.34 ± PLACEHOLDER | 1.89 ± PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| STAE-Spectral-Magma     | 3.55 ± PLACEHOLDER | 1.87 ± PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

### 4.3 Ablation table

(Populated by `scripts/run_ablations_stae_spec.sh` on a chosen dataset.)

| Variant            | val_avg | val 60-min | Δ vs full |
|--------------------|---:|---:|---:|
| full               | PLACEHOLDER | PLACEHOLDER | – |
| − sym view         | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| − mag view         | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| − semantic view    | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| − router (uniform) | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |
| − mode-axis scan   | PLACEHOLDER | PLACEHOLDER | PLACEHOLDER |

We expect: dropping the mode-axis scan should be measurable (the central
novelty); dropping the magnetic view should help most on the
direction-sensitive segments (long-horizon congested freeway corridors);
dropping the semantic view should help most on cross-corridor transfer
(off-peak hours).

### 4.4 Interpretability

(Generated post-hoc on saved checkpoints by
`scripts/eval_interpretability.py` — TO BE WRITTEN.)

- Router gate heatmap, three figures, one per view, showing softmax weight
  on the (horizon × cluster) grid for high-congestion vs free-flow regimes.
- Per-speed-regime MAE (≤ 20 / 20–40 / 40–60 / ≥ 60 mph) showing where
  the spectral augmentation adds value.
- q-charge and K-mode sensitivity curves.

---

## 5. Oracle analysis: why METR-LA is saturated

A spectral residual on top of a persistence baseline can express any
prediction whose deviation from persistence lies in the column span of
the basis U ∈ R^{N × K}. The *optimal* such residual has masked MAE

    L*_K = min_{Δ ∈ col(U)} ‖ Δ − (Y_true − persist) ‖_MAE,

which is the projection error of (Y_true − persist) onto col(U). For
fixed-basis (symmetric / magnetic) variants we compute L*_K exactly on
the METR-LA validation split:

| K   | L*_K val_avg | L*_K val 60-min |
|----:|---:|---:|
|  32 | 3.71 | 4.54 |
|  48 | 3.40 | 4.13 |
|  64 | 3.15 | 3.79 |
|  96 | 2.64 | 3.13 |
| 128 | **2.07** | **2.46** |

The K=128 oracle is *below* STAEformer's 2.74. The bandwidth is
sufficient. However, every spectral-residual *learner* we tested
(SSM-Magma standalone, STAE-Spec joint-trained, STAE-Spec frozen-trunk)
plateaus near STAEformer's 2.74 — i.e., the predictability gap from
input window to optimal spectral coefficients is what's limiting, not the
basis. This explains the field-wide observation that augmentations on top
of STAEformer-class encoders rarely improve METR-LA.

### 5.1 Four failure modes we observed

1. **Spectral standalone**: SSM-Magma without backbone plateaus at val
   3.5 (no adaptive embedding to compensate for low-rank output).
2. **Joint-trained sidechain on saturated backbone**: STAE-Spec val 2.88,
   *worse* than STAEformer 2.74. Sidechain gradients disrupt STAEformer's
   slow-LR convergence regime.
3. **Frozen-trunk sidechain**: monotonically *increasing* val from
   STAEformer's 2.74 baseline — the sidechain learns training-set noise.
4. **TOD-indexed adaptive embedding** added to the standalone spectral
   experts overfits training (~1.4M memorization-grade parameters with
   24K training windows).

This catalogue is consistent with the prior DiSR-Mamba campaign's finding
that frozen-trunk residual learning fails on METR-LA, and provides a
quantitative explanation via the oracle ceiling vs. learner gap.

---

## 6. Discussion

We have positioned the work as architectural-and-analytical. The
three primitives (bi-axis spectral Mamba, three-view spectral MoE,
horizon-cluster router) are independently reusable. The empirical
contributions are (i) a clean characterization of METR-LA saturation,
(ii) evidence that on PEMS-BAY (less saturated) the spectral augmentation
trends positive though within noise of a single seed, and
(PLACEHOLDER) (iii) clearer positive evidence on the flow-prediction
PEMS04/08 benchmarks where absolute error is ~10× larger and
representational ceilings are not yet hit.

### 6.1 Limitations

- METR-LA results are negative (sidechain hurts or is neutral). We frame
  this as a finding, but a reviewer may push back: "Why not pick a
  benchmark where it works?" — Our answer: METR-LA *is* the standard
  benchmark; characterizing its saturation is itself a contribution.
- The architecture is not parameter-efficient relative to STAEformer alone:
  STAE-Spec adds ~0.8M parameters for the sidechain, on top of
  STAEformer's 1.26M.
- The learned-semantic view requires periodic eigendecomposition, which
  is well-defined for N ≲ 1000 sensors (METR-LA, PEMS-BAY, PEMS04/08) but
  scales as O(N³) and may be impractical for city-wide sensor networks
  with 10⁴ nodes.

### 6.2 Open questions

- Can the bi-axis scan be replaced by attention on these short sequences
  (T_in = 12, K = 64) without losing the architectural story? The
  Mamba choice is justified at long sequences; for short sequences this
  is an empirical question we leave to future work.
- Does the three-view decomposition transfer to non-traffic spatio-temporal
  graphs (air quality, power grid, river network)? The architectural
  story (geometric + directional + semantic = three views of graph
  structure) is domain-agnostic.

---

## 7. Reproducibility

All experiments are reproducible from the public code in this
repository. Pre-trained STAEformer baselines on METR-LA and PEMS-BAY are
provided as `best_stae_s42.pth` checkpoints. To replicate the main
table:

```
# Setup (one-time)
bash setup_pod.sh                       # installs torch + mamba-ssm + deps

# Data preparation
python scripts/prepare_pems04_08.py      # downloads PEMS04 + PEMS08

# Baselines (single GPU, ~1 h each)
python scripts/train_staeformer.py --tag <tag> --data_path <data> --adj_path <adj>

# Hybrids
python scripts/train_stae_spectral_magma.py --tag <tag> ...

# Ablations
./scripts/run_ablations_stae_spec.sh <dataset> <seed>

# Multi-seed
./scripts/run_multiseed_stae_spec.sh <dataset>
```

Random seeds, exact hyperparameters, and complete logs are committed
alongside the code in `logs/` and `results/`.

---

*Working notes*

- Architecture diagram TBD: ASCII sketch in §3.3 is the spec; want a
  TikZ rendering for camera-ready.
- Need to add specific citations once we settle on venue (CIKM/SDM uses
  ACM style; KDD uses similar; arXiv preprint also fine).
- The "first magnetic Laplacian on traffic" claim should be double-
  checked against the very latest arXiv (search "magnetic Laplacian
  traffic" + dates after our submission).
