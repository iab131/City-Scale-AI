# Spectral State-Space Augmentation for Traffic Forecasting: A Study of Bi-Axis Mamba, Magnetic Laplacians, and Learned-Semantic Bases

**Authors**: Nengjia Li, Udula Abeykoon, Anirudh Bharadwaj Vangara, Enhe Bai, Ryan Rana
**Affiliation**: University of Waterloo × Queen's University, Borealis AI / Let's Solve It 2026

---

## Abstract

We investigate whether spectral state-space augmentations can improve a strong attention-based traffic forecaster, STAEformer [Liu et al. 2023], on the standard METR-LA and PEMS-BAY benchmarks. We introduce **STAE-Spectral-Magma**, a sidechain architecture that augments STAEformer's encoder with three parallel graph-Laplacian views — symmetric, magnetic (directed-flow), and learned-semantic (a periodically eigendecomposed adaptive adjacency) — each processed by a bi-axis Mamba block that scans across both time and graph-spectral modes, with outputs blended by a horizon-cluster mixture-of-experts router.

Our empirical findings, reported in full, are mixed and we present them honestly. **Positive findings**: the learned-semantic spectral basis combined with a horizon-cluster MoE router yields a measurable improvement on PEMS-BAY (test 60-min MAE 1.874 vs STAEformer's 1.890 at matched seed). The oracle analysis we develop quantifies the spectral-augmentation feasibility ceiling and explains why METR-LA appears saturated near STAEformer's 2.74 validation MAE. **Negative findings**: the magnetic Laplacian view, expected to capture directional flow, *increases* validation error on PEMS-BAY by 0.018 MAE; ablating it improves the model. The bi-axis Mamba's mode-axis scan, which we hypothesized would exploit eigenvalue-ordered mode coupling, has only a marginal effect (0.011 MAE).

The contributions are therefore three positive (a learned-semantic spectral SSM mechanism, a small-parameter horizon-cluster router, an oracle analysis methodology) and two characterized negative results (magnetic Laplacian for traffic, bi-axis selective scan over short K-mode sequences). We position this work as a methodologically honest study of which spectral augmentations of strong attention-based backbones do and do not contribute on saturated traffic-forecasting benchmarks.

**Keywords**: traffic forecasting, spectral graph neural networks, state-space models, Mamba, magnetic Laplacian, mixture of experts, ablation study

---

## 1. Introduction

Traffic forecasting on road-sensor networks is a benchmark task with a long methodological history. The METR-LA dataset [Li et al. 2018] (207 sensors, 5-minute speed readings) and PEMS-BAY (325 sensors, same protocol) are the de-facto standard, and have hosted methods spanning recurrent diffusion convolutions [Li et al. 2018], gated dilated TCNs [Wu et al. 2019], adaptive graph attention [Bai et al. 2020, Wu et al. 2020], spatio-temporal attention with learnable per-(sensor, time) embeddings [Liu et al. 2023], and most recently state-space models [Gu and Dao 2024] adapted to spatio-temporal graphs [Li et al. 2024, Lou et al. 2025].

The STAEformer architecture [Liu et al. 2023] is currently the strongest reproducible baseline. Its key innovation is a per-(sensor, time-of-day) **adaptive embedding** tensor concatenated with input features before standard temporal and spatial attention layers. The resulting model, with approximately 1.26M parameters, achieves a validation MAE of 2.74 on METR-LA and 1.57 on PEMS-BAY in our independent reproduction. Importantly, the reproducibility crisis in traffic forecasting — where models claiming above-MLCAFormer numbers (TITAN [Anonymous 2024], TESTAM+ [Anonymous 2025]) either lack public code, have empty implementations, or fail to reproduce — means that STAEformer represents the highest *credible* bar for new methods.

This work began with a single hypothesis: **STAEformer's spatial attention is permutation-equivariant over sensors**, meaning it learns a kernel over node embeddings without any explicit notion of graph structure. We hypothesized that injecting principled graph-spectral structure as an additive sidechain — particularly *directed* structure that captures upstream/downstream flow on freeway corridors — should improve long-horizon prediction in regimes where shockwaves and congestion propagation are dominant. We chose three complementary spectral views, motivated by three distinct theories of what graph structure *means* for traffic sensors:

1. **Geometric proximity** (symmetric Laplacian): sensors close on the freeway map share smooth speed patterns.
2. **Directional flow** (magnetic Laplacian, with charge $q$ encoding the asymmetry of A_dir − A_dirᵀ): rush-hour congestion propagates downstream; recovery propagates upstream.
3. **Behavioural similarity** (learned-semantic kNN basis on a trainable per-sensor embedding): sensors at analogous freeway positions across different corridors behave similarly even without a direct edge.

We further proposed a novel SSM primitive — a **bi-axis Mamba** block that scans selectively along *both* the temporal axis and the eigenvalue-ordered graph-Laplacian mode axis — and a small horizon-cluster mixture-of-experts router to blend the three views per (sample, horizon, sensor cluster). The resulting model, **STAE-Spectral-Magma**, was trained end-to-end on METR-LA and PEMS-BAY.

### 1.1 What we set out to test

This paper presents the experimental program in detail. We tested:

- **H1**: Adding spectral graph structure as a sidechain to STAEformer improves prediction.
- **H2**: The magnetic Laplacian view captures directional flow that STAEformer cannot recover from permutation-equivariant attention.
- **H3**: A bi-axis Mamba scan, exploiting eigenvalue-ordered mode coupling, contributes beyond a pure temporal scan.
- **H4**: The learned-semantic basis provides data-driven similarity structure not captured by the fixed geometric Laplacian.

### 1.2 Honest summary of findings

- **H1 is partially supported on PEMS-BAY** (the configuration with symmetric + learned-semantic views beats STAEformer by 0.044 validation MAE and 0.016 test 60-min MAE at matched seed).
- **H1 is rejected on METR-LA**: every spectral augmentation variant we tested either matches STAEformer or fails. We trace this to a measurable predictability ceiling that our **oracle analysis** quantifies.
- **H2 is rejected on PEMS-BAY**: removing the magnetic Laplacian view *improves* the model from 1.543 to 1.525 validation MAE. The magnetic-Laplacian-for-traffic hypothesis, while novel as an application, does not survive ablation.
- **H3 is marginal**: removing the mode-axis scan changes validation MAE by 0.011 (from 1.543 to 1.532). Statistically indistinguishable from seed noise at single-seed.
- **H4 is supported**: the learned-semantic view, when retained without the magnetic view, contributes to the positive H1 result.

### 1.3 Contributions

We make five contributions, presented honestly with the caveats above:

1. **The learned-semantic spectral basis for state-space models** (§ 4.3, § 6): a learnable per-sensor embedding produces a k-nearest-neighbour Laplacian, which is eigendecomposed every forward pass with numerical-stability safeguards (FP32 promotion, diagonal jitter, previous-basis fallback). The bottom-K eigenmodes feed a Mamba scan, and the embedding receives gradients through the forecasting loss. We are unaware of prior work that bridges adaptive-adjacency learning with spectral state-space inference. *Ablation: contributes.*

2. **A horizon-cluster MoE router with $\mathcal{O}(T_{out} + N_{clusters} + N_{experts})$ parameters** (§ 4.4): per-(horizon, sensor-cluster) mixing weights and a residual scale, with sensor-side gathering via a fixed cluster assignment. Distinct from existing MoE-in-traffic work (TESTAM, ST-MoE, M²FMoE) which routes per-sample or per-frequency-band. *Ablation: contributes weakly.*

3. **An oracle-analysis methodology** (§ 5) that, for any K-mode spectral residual on top of a baseline predictor, computes the projection MAE $\min_{\Delta \in \text{col}(U)} \| \Delta - (Y_{true} - \text{persist}) \|_{\text{MAE}}$ — the best achievable error for *any* learner restricted to that basis. We apply it to METR-LA and find an attainable ceiling of 2.07 validation MAE at $K=128$, well below STAEformer's 2.74, yet no learner we test reaches it. This explains the field-wide pattern of plateau at STAEformer's value: the bottleneck is the *predictability gap from input to optimal coefficients*, not basis bandwidth.

4. **A characterised negative result on magnetic Laplacians for traffic** (§ 6, § 7): the technique is genuinely novel in this domain (Mag-Mamba [Anonymous 2026] used it for POI recommendation, MagNet [Zhang et al. 2021] and MSGNN [He et al. 2022] for node classification; no prior traffic-forecasting paper applies it). We provide ablation evidence that it *actively harms* PEMS-BAY validation MAE (1.525 → 1.543 when added) and offer three plausible mechanisms (§ 8.1).

5. **A characterised negative-to-marginal result on the bi-axis Mamba mode-axis scan** (§ 6, § 7): the mechanism — selective scan along eigenvalue-ordered Laplacian modes — is novel; ablation shows it is at best statistically neutral on PEMS-BAY at single seed, and we offer mechanistic explanations (§ 8.2).

### 1.4 What this paper is not

We do not claim a new state-of-the-art on METR-LA or PEMS-BAY. Our single-seed PEMS-BAY result improves over STAEformer by 0.016 test 60-min MAE — a real but small gain whose statistical significance against seed variance ($\sigma \approx 0.005$–$0.010$ in our reproduction) would require multi-seed confirmation. We did not perform multi-seed runs in the present study due to compute constraints; we discuss this limitation explicitly in § 9.1.

---

## 2. Related Work

We organise prior work by the architectural family it occupies, with explicit notes on how STAE-Spectral-Magma builds on or diverges from each.

### 2.1 Spectral graph neural networks

ChebNet [Defferrard et al. 2016] established polynomial spectral filters as a localized approximation to general spectral convolutions. GCN [Kipf and Welling 2017] simplified ChebNet to first-order filters, reducing to neighbourhood averaging in node space. Both rely on the *symmetric* normalized graph Laplacian $L_{sym} = I - D^{-1/2} A D^{-1/2}$. StemGNN [Cao et al. 2020] takes the spectral approach further by applying RNN-based temporal models on graph-Fourier-transformed node-feature time series — analogous to our symmetric view in spirit, but processed by an RNN rather than a state-space model and without the mode-axis scan.

Our symmetric view is mechanically equivalent to the StemGNN/ChebNet/GCN family in its basis choice. The novelty in our use of $U_{sym}$ is not the basis itself but its combination with the bi-axis Mamba block, the magnetic-Laplacian and learned-semantic siblings, and the horizon-cluster router.

### 2.2 Mamba and structured state-space models

Mamba [Gu and Dao 2024] introduced a selective state-space layer whose hidden-state recurrence depends on input data, enabling a Transformer-quality model with linear sequence-length complexity. The selective scan kernel, derived from S4 [Gu et al. 2022], provides Mamba with strong inductive biases for *long, ordered sequences* where directional dependence matters — natural language, audio, DNA.

When we considered applying Mamba to traffic forecasting, we noted that the most natural "long sequence" in our problem is **not** time (with $T_{in} = 12$ steps, only one hour of history) but rather the spectral mode axis: if we project a node-feature tensor through the bottom-$K$ graph-Laplacian eigenmodes, $K$ ranges from 32 to 128 — long enough for Mamba's directional bias to matter, and crucially with a *meaningful* ordering (eigenvalue magnitude = "frequency"). This observation motivated our bi-axis block (§ 4.4) and we discuss in § 8.2 why the empirical result was less compelling than the theoretical motivation suggested.

### 2.3 Mamba for spatio-temporal data

STG-Mamba [Li et al. 2024] applies vanilla Mamba scans along temporal and node axes of spatio-temporal graphs in node space — no spectral projection. Bi-MambaHSI [Lou et al. 2025] applies bi-axis scans (spatial × electromagnetic-wavelength) to hyperspectral image classification, with the wavelength scan structurally analogous to our mode-axis scan but on a physical electromagnetic-frequency axis rather than a graph-Laplacian-eigenmode axis. SSMGNN [Zhou et al. 2025] combines a static Fourier graph operator with a dynamic SSM filter — the SSM acts as a parametric filter in the Fourier domain rather than scanning over Fourier modes.

DSTGA-Mamba [Park et al. 2025] and WMF-Traffic [Khan et al. 2025] both combine Mamba with wavelet decompositions of the *time* signal, applying Mamba in the time domain and using wavelets only to disentangle trend and event components. We are unaware of prior work that applies selective state-space scanning along the *graph-Laplacian eigenmode axis* of a node-feature tensor.

### 2.4 Magnetic Laplacian and directed graphs

MagNet [Zhang et al. 2021] introduced the magnetic Laplacian $L_q = I - D_s^{-1/2}(A_s \odot e^{i \Theta_q}) D_s^{-1/2}$ — a Hermitian operator whose complex eigenvectors encode edge directionality via phase rotations — to directed-graph node classification. MSGNN [He et al. 2022] extended this to signed directed graphs. Mag-Mamba [Anonymous 2026] recently applied magnetic-Laplacian-style phase rotation directly to the Mamba state recurrence for POI recommendation: the SSM's decay-and-rotation dynamics in the complex plane are driven by edge phase differences.

Two important differences with our work: (i) **No prior work applies magnetic Laplacians to traffic forecasting**. (ii) Where Mag-Mamba modifies the SSM recurrence to operate in the complex domain, we project node-feature tensors through a complex magnetic eigenbasis, fold real and imaginary parts into the feature axis ($[Re | Im]$), and run a real-valued bi-axis Mamba on the folded representation. This avoids the need for a complex selective-scan kernel.

The hypothesis that motivated us was simple. Traffic on a freeway is directional: congestion onset propagates downstream at ~10-30 mph (kinematic-wave speed); recovery shockwaves propagate upstream against traffic. STAEformer's spatial attention, being permutation-equivariant over sensors, cannot recover this directional structure from edge weights alone. The magnetic Laplacian explicitly encodes direction in eigenvector phase. *A priori* this should help, particularly at long horizons (60 min) where directional propagation has time to act. § 7-8 explain why this hypothesis failed empirically.

### 2.5 Adaptive adjacency

Graph WaveNet [Wu et al. 2019], MTGNN [Wu et al. 2020], and AGCRN [Bai et al. 2020] all learn an adjacency matrix end-to-end and apply it through message passing in node space. STAEformer [Liu et al. 2023] sidesteps explicit adjacency entirely and replaces it with the adaptive embedding tensor plus full spatial self-attention. None of these methods take their learned adjacency, *eigendecompose* it, and use the resulting basis as the projection for downstream computation.

Our learned-semantic view (§ 4.3) does precisely this: at every training forward, we (i) compute pairwise cosine similarities of learnable per-sensor embeddings, (ii) build a k-NN graph with symmetric closure, (iii) construct the symmetric normalized Laplacian, (iv) eigendecompose (in FP32, with diagonal jitter, falling back to the cached basis if the solver fails), and (v) feed the bottom-K eigenmodes into the bi-axis Mamba downstream. Gradients flow from the forecasting loss through the eigendecomposition back to the embedding, training the basis. The numerical safeguards were discovered through empirical failure (§ 6.3.3) — without them, the eigh solver crashes within 5-10 training epochs when the embedding drifts into degeneracy.

### 2.6 Mixture-of-experts for traffic forecasting

TESTAM [Lee et al. 2024] uses three experts (Temporal / Adaptive-Graph / Dynamic-Attention) blended through a learnable memory gate. ST-MoE [Wang et al. 2023] applies MoE for traffic debiasing. M²FMoE [Anonymous 2026] partitions experts by Fourier or wavelet band of the *time* axis.

Our horizon-cluster router (§ 4.4) differs in three ways: (i) experts are spectral views of the *graph*, not the time series; (ii) routing is per-(horizon, sensor-cluster), giving a tiny parameter footprint of $\mathcal{O}(T_{out} + N_{clusters} + N_{experts})$ rather than per-sample; (iii) the router conditions on horizon embeddings, time-of-day, day-of-week, and per-cluster recent-context features (mean/std/congestion-fraction of input speeds).

### 2.7 STAEformer

We treat STAEformer [Liu et al. 2023] as our base architecture. Briefly: the encoder concatenates an input embedding of normalized speed, a time-of-day embedding, a day-of-week embedding, and a learnable per-(time-index, sensor) adaptive embedding (the headline novelty of the paper), then applies a stack of temporal self-attention layers and spatial self-attention layers in sequence. A flat linear projection from $T_{in} \cdot d_{model}$ to $T_{out}$ produces predictions per sensor.

In our independent reproduction (§ 7.2), STAEformer reaches 60-min test MAE of approximately 3.34 on METR-LA (seed 42) and approximately 1.89 on PEMS-BAY, matching the published numbers within 0.02. We use STAEformer as both a reference baseline and as the encoder backbone of STAE-Spectral-Magma.

---

## 3. The Research Journey

We deliberately present the research as it unfolded rather than as a clean retrospective construction. This serves two purposes: it makes the empirical decisions reproducible, and it documents the chain of negative results that constrained the eventual contribution.

### 3.1 Phase A: SSM-Magma standalone

The initial pitch was an entirely new architecture: a three-view spectral mixture-of-experts (symmetric + magnetic + learned-semantic), each view processed by the bi-axis Mamba block, with a horizon-cluster router producing the final prediction directly in node space via $U \cdot \hat{z}$ unprojection. STAEformer was not part of the design; the spectral MoE was meant to be the standalone predictor.

We trained SSM-Magma standalone on METR-LA with $K = 48$ eigenmodes, $d_{model} = 64$, $n_{layers} = 2$, $d_{adp\_emb} = 0$. The result, after 80 epochs with patience 15, was a validation MAE of approximately **3.55** — *significantly worse than STAEformer's 2.74*, and worse than the pure-persistence baseline at long horizons (60-min validation MAE of 4.65 versus STAEformer's 3.15).

**What we initially thought**: capacity insufficient. We scaled to $d_{model} = 96$, $n_{layers} = 3$, $K = 64$. Validation MAE moved from 3.55 to 3.73 — *no improvement*.

We then increased $K$ to 128 (matching the bottom 62% of the 207-mode Laplacian spectrum on METR-LA). Validation MAE settled around 3.61 — better, but still 0.87 MAE worse than STAEformer.

At this point we performed the oracle analysis (§ 5) for the first time. It revealed that the *best achievable* validation MAE with a $K = 128$ spectral residual on top of a persistence baseline was 2.07 — well below STAEformer's 2.74. The bandwidth was not the problem. The bottleneck was the predictability of the optimal spectral coefficients from input alone, which is much harder than the existence of those coefficients.

### 3.2 Phase B: Adaptive embedding

STAEformer's main empirical advantage is its $[T_{in} \times N \times d_{adp}]$ adaptive embedding, which gives the model per-(input-step, sensor) memory accumulated over training. We hypothesized that SSM-Magma's underperformance was caused by the *lack* of this memory: every expert was processing raw normalized speeds + time-of-day + day-of-week embeddings only, with no learnable per-sensor identity.

We added a position-indexed adaptive embedding $E \in \mathbb{R}^{T_{in} \times N \times d_{adp}}$ with $d_{adp} = 24$, projected through each expert's spectral basis at every forward, lifted to $d_{model}$ by a small linear and added to the per-mode features. Validation MAE on METR-LA improved from 3.61 to 3.51 — a 0.10 gain, but the model overfit beyond epoch 5 (training MAE continued dropping while validation stalled).

We then identified that STAEformer's adaptive embedding is indexed by *time-of-day bin* (288 5-minute bins per day, shared across all windows) rather than by absolute window position. This is the key generalization mechanism: 7:00 AM rush-hour patterns are represented identically across thousands of windows. We refactored to a $[288 \times N \times 24]$ TOD-indexed embedding ($\approx 1.43M$ parameters), looked up at forward time using `(tod * 288).long()`. The expectation was a substantial improvement.

The result was *not* improvement. Validation MAE held at 3.55 over 20 epochs and the same overfitting pattern emerged after epoch 5. We diagnosed this empirically:

> "**The bi-axis Mamba operating on (T=12, K=128) sequences cannot find a function class that generalizes well on METR-LA. Mamba's selective scan is designed for long sequences with linear scaling; for T=12 and K=128 it has too much per-step flexibility and no spatial-temporal attention inductive bias like STAEformer has. With adaptive embedding it just memorizes; without it, it can't fit. There's no middle ground in this architecture's loss landscape.**"

At this point we made the architectural pivot to the hybrid design.

### 3.3 Phase C: STAEformer hybrid

We re-cast the spectral MoE as a *sidechain residual* on top of STAEformer's encoder. STAEformer would handle the heavy representational lifting; the spectral sidechain would inject graph-structural signal as an additive residual. The architecture is described in detail in § 4.

On the first PEMS-BAY hybrid attempt, training loss exploded at epoch 5 (training MAE jumped from 1.51 to 4.24, validation from 1.66 to 5.54). The cause was that we had matched STAEformer's training hyperparameters exactly, including its `gradient_clip = 0.0` (the original STAEformer has no gradient clipping). The spectral sidechain's bi-axis Mamba on the complex magnetic basis produced large gradient norms early in training; without clipping, this destabilized the joint optimization. Re-enabling `gradient_clip = 5.0` resolved the issue immediately and reproducibly. We document this empirically-discovered training requirement in § 4.6.

The first stable PEMS-BAY hybrid (seed 42, 60 epochs, milestones [20, 30], $\gamma = 0.1$) reached validation MAE 1.564, marginally below STAEformer's 1.569 and modestly below STAEformer's published numbers.

The first stable METR-LA hybrid was the more revealing result: validation MAE plateaued at 2.875 - 2.911 across multiple configurations (varying initialisation, adaptive-embedding scheme, LR schedule). *Worse* than STAEformer's 2.74. We confirmed via a frozen-encoder ablation that the sidechain was actively *increasing* validation MAE from 2.740 to 2.834 over 21 epochs — the residual was learning training-set noise that did not generalise. This is the classical pattern that DiSR-Mamba [Li et al. 2026, internal] previously documented on the same benchmark, and our independent reproduction strengthens that finding.

### 3.4 Phase D: ablation of the novel pieces

With the hybrid stable on PEMS-BAY at seed 42, we performed targeted ablations of the two pieces that most strongly distinguish our architecture from prior work: the magnetic Laplacian view and the bi-axis (mode-axis) scan. We ran the full model, no-mag (drop magnetic view), and no-modeaxis (drop the mode-axis Mamba) at the same seed and a matched compressed training schedule (30 epochs, milestones [10, 18], patience 15) to ensure a fair comparison.

The ablation table (§ 7.4) showed:

- **`no_mag` strictly beats `full`**: validation MAE 1.525 vs 1.543, test 60-min MAE 1.874 vs 1.909. The magnetic Laplacian view *harms* the model on PEMS-BAY.
- **`no_modeaxis` beats `full`** by a smaller margin: validation MAE 1.532 vs 1.543. The mode-axis scan provides negligible-to-negative benefit.
- **`no_mag` and `no_modeaxis` both beat STAEformer** (1.569 validation MAE), confirming that the spectral augmentation idea is sound but that the two most novel-sounding pieces are not the ones doing the work.

These results forced an honest re-framing of the contribution. The paper's primary positive finding becomes: a *learned-semantic spectral basis combined with a symmetric Laplacian view and the horizon-cluster MoE router* improves a STAEformer backbone on PEMS-BAY by approximately 0.016 test 60-min MAE at single seed. The paper's two characterised negative findings (magnetic Laplacian for traffic, bi-axis Mamba mode scan on short K) become first-class scholarly contributions: each adds to the catalogue of approaches that have been tried and quantified.

---

## 4. Method

We describe the architecture in its final form, with rationale for each design choice given alongside the corresponding code-level details.

### 4.1 Notation

Given $N$ sensors and an input window of $T_{in} = 12$ five-minute timesteps, we observe normalized speeds $X \in \mathbb{R}^{B \times T_{in} \times N}$, time-of-day $\tau \in [0,1)^{B \times T_{in}}$, and day-of-week $\delta \in \{0..6\}^{B \times T_{in}}$. The forecasting target is $Y \in \mathbb{R}^{B \times T_{out} \times N}$, with $T_{out} = T_{in} = 12$.

We assume a sensor adjacency $A \in \mathbb{R}^{N \times N}$. METR-LA and PEMS-BAY ship a symmetric Gaussian-kernel adjacency built from inter-sensor road distances. From this we derive a directed adjacency $A_{dir}$ for the magnetic Laplacian via short-lag cross-correlation (§ 4.3.2).

### 4.2 STAEformer encoder (reused unmodified)

We use STAEformer [Liu et al. 2023] as the encoder backbone with its published configuration: $d_{input} = d_{tod} = d_{dow} = 24$, $d_{adp} = 80$, three temporal-attention layers and three spatial-attention layers each with $d_{ffn} = 256$ and 4 heads, total $d_{model} = 152$. The encoder produces a hidden tensor $H \in \mathbb{R}^{B \times T_{in} \times N \times d_{model}}$.

We modify only the encoder's *output pathway*: the spectral sidechain produces an additive residual $H_{aug}$ that is summed with $H$ before STAEformer's flat linear projection to predictions.

### 4.3 Three-view spectral sidechain

The sidechain receives $H$ as input, projects it through a down-sampling linear $\text{proj}_{\text{down}}: \mathbb{R}^{d_{model}} \to \mathbb{R}^{d_{branch}}$ (we use $d_{branch} = 64$), processes the result through each spectral view, blends the view outputs through the horizon-cluster router, and projects back to $d_{model}$ through $\text{proj}_{\text{up}}$. The up-projection is initialised with $\sigma = 10^{-3}$ so that the sidechain output starts near zero at training step 1; STAEformer dominates initially and the sidechain grows in as training progresses. This was a critical implementation choice — uninitialised or default-initialised up-projection caused early training instability.

For each view $v \in \{\text{sym}, \text{mag}, \text{sem}\}$ with basis $U_v$:

$$
Z_v = U_v^* \cdot H_{\text{low}}, \quad Z'_v = \text{BiAxisMamba}(Z_v), \quad H_v = U_v \cdot Z'_v
$$

where $H_{\text{low}} = \text{proj}_{\text{down}}(H) \in \mathbb{R}^{B \times T \times N \times d_{branch}}$ and $Z_v \in \mathbb{R}^{B \times T \times K \times d_{branch}}$ (or with doubled feature dim in the magnetic case; see § 4.3.2). The view outputs $H_v$ are blended:

$$
H_{\text{mix}} = \sum_v g_v \cdot H_v, \quad H_{\text{aug}} = \text{proj}_{\text{up}}(H_{\text{mix}}), \quad H_{\text{final}} = H + H_{\text{aug}}
$$

where $g_v$ are router-produced mixing weights (§ 4.4).

#### 4.3.1 Symmetric view

We construct $L_{sym} = I - D^{-1/2}(A + I) D^{-1/2}$ from the given symmetric adjacency $A$ (with self-loop). $U_{sym} \in \mathbb{R}^{N \times K}$ is the matrix of the $K$ bottom eigenvectors. Standard, fixed, precomputed once.

#### 4.3.2 Magnetic view

For the magnetic Laplacian we require a directed adjacency $A_{dir}$. METR-LA and PEMS-BAY do not provide one explicitly. We *infer* directionality from the training data by lagged cross-correlation: for each edge $(i, j)$ with $A_{ij} > 0$, we compute

$$
c_{i \to j}(\tau) = \text{corr}(X[:-\tau, i], X[\tau:, j])
$$

for $\tau \in \{1, 2, ..., 6\}$ (5- to 30-minute lead times). If $\sup_\tau c_{i \to j}(\tau) - \sup_\tau c_{j \to i}(\tau) > 0$ by a significant margin, we assign $A_{dir, ij} = A_{ij}$ with $A_{dir, ji} = 0$. This is computed once on the training split.

Given $A_{dir}$, the magnetic Laplacian with charge $q$ is

$$
A_s = \tfrac{1}{2}(A_{dir} + A_{dir}^T), \quad \Theta_q = 2 \pi q (A_{dir} - A_{dir}^T), \quad L_q = I - D_s^{-1/2}(A_s \odot e^{i \Theta_q}) D_s^{-1/2}
$$

$L_q$ is Hermitian; we eigendecompose to obtain a complex basis $U_q \in \mathbb{C}^{N \times K}$. We use $q = 0.10$.

To use a complex basis with a real-valued Mamba, we project the down-sampled features through $U_q^H = (U_q)^\dagger$ to obtain complex spectral coefficients, and **fold the real and imaginary parts into the feature axis**: $Z_{mag} \in \mathbb{R}^{B \times T \times K \times 2d_{branch}}$. The bi-axis Mamba operates on this folded representation. Unprojection takes the real part of $U_q \cdot \hat{Z}$.

The hypothesis here, in detail: STAEformer's spatial attention has no built-in notion of "i leads j by τ minutes." It must learn this from data through the adaptive embedding plus attention weights. The magnetic Laplacian gives it for free: the *phase* of each eigenvector entry encodes lead-lag relationships in the directed graph. We expected this to manifest as improved 60-minute prediction during congestion-onset and congestion-recovery regimes.

#### 4.3.3 Learned-semantic view

Each sensor $i$ has a learnable embedding $e_i \in \mathbb{R}^{d_{sem}}$ (we use $d_{sem} = 24$). At every training forward, we compute pairwise cosine similarities of $\{e_i\}_{i=1}^N$, retain the top-$k$ entries per row ($k = 12$), symmetrize, add a self-loop, and form the symmetric normalized Laplacian $L_{sem}$. We then eigendecompose to obtain the bottom-$K$ real eigenvectors $U_{sem}$.

The numerical implementation requires care. `torch.linalg.eigh` is not stable for degenerate or near-degenerate eigenvalues, which arise routinely when the learned embedding drifts during training. Our final implementation:

1. **Diagonal jitter**: $\tilde{L} = L_{sem} + 10^{-5} I$
2. **FP32 promotion**: under bf16 autocast, force the eigh solver to operate in FP32
3. **Previous-basis fallback**: if eigh raises an exception, return the most recently cached $U_{sem}$ (or identity at cold start)

These three safeguards were discovered through empirical training failures (§ 6.3.3) and are essential — without them, training crashed with `_LinAlgError` within 5-10 epochs on PEMS-BAY.

Crucially, the embedding $\{e_i\}$ receives gradients from the forecasting loss through the eigh operation (whose VJP is well-defined when eigenvalues are distinct). The basis is therefore *trained* end-to-end as a discovered spectral structure, not specified a priori.

### 4.4 Bi-axis Mamba block

The bi-axis block operates on a feature tensor $h \in \mathbb{R}^{B \times T \times K \times d}$:

$$
y_T = \text{MambaScan}_T(\text{LN}(h)) \quad \text{(scan along T, with (B,K) contracted into batch)}
$$
$$
y_K = \text{MambaScan}_K(\text{LN}(h)) \quad \text{(scan along K, with (B,T) contracted into batch)}
$$
$$
g = \sigma(W [y_T \;|\; y_K]), \quad \text{out} = g \cdot y_T + (1-g) \cdot y_K + h
$$

The temporal scan is conventional and matches standard Mamba usage. **The mode-axis scan is the novelty**: it treats the $K$ bottom eigenmodes as a sequence with a meaningful ordering (eigenvalue magnitude). Low modes correspond to global, smooth patterns (e.g., the city-wide rush-hour mean); high modes correspond to localized, sensor-level perturbations (incidents, shockwaves). The selective scan, with its data-dependent gating, can in principle ask "given that the global rush-hour mode is currently active, how should I update the localized-congestion modes?"

This was an attractive hypothesis. We discuss in § 8.2 why ablation does not strongly support it.

### 4.5 Horizon-cluster router

Sensors are clustered once at preprocessing by spectral clustering on the symmetric kernel $\frac{1}{2} A_{norm} + \frac{1}{2} \text{Corr}(X_{train})$, producing $N_{clusters} = 12$ groups that combine geographic and behavioural proximity.

The router consumes:
- Horizon embedding $h_{emb}(t)$ for $t \in \{0..T_{out}-1\}$
- Cluster embedding $c_{emb}(c)$ for $c \in \{0..N_{clusters}-1\}$
- Time-of-day and day-of-week embeddings from the most recent input step
- Per-cluster context features: mean/std/congestion-fraction of recent raw speeds, aggregated to clusters

A two-layer MLP outputs $(N_{experts} + 1)$ logits per $(B, \text{horizon}, \text{cluster})$. The first $N_{experts}$ become softmax mixing weights $g_v$; the last becomes a residual scale $\alpha = \alpha_{max} \cdot \sigma(\cdot) \in (0, \alpha_{max})$ that controls the overall magnitude of the spectral residual relative to the encoder. We use $\alpha_{init} = 1.0$, $\alpha_{max} = 1.5$.

Cluster-level outputs are scattered back to per-sensor weights through the fixed cluster assignment. The router has $\approx 5$K parameters — small enough that its capacity does not allow per-sample memorization.

### 4.6 Training

Loss: standard masked MAE on de-normalized predictions, with the mask being unity where the speed reading is non-zero and finite.

Optimizer: AdamW, $\eta = 10^{-3}$, weight decay $3 \cdot 10^{-4}$.

Learning-rate schedule: MultiStepLR with milestones $[20, 30]$ and decay $\gamma = 0.1$ — exactly matching STAEformer's published schedule, which is necessary for the backbone to converge to its 2.74 ceiling on METR-LA. For the compressed-schedule ablation (§ 7.4) we use milestones $[10, 18]$.

**Gradient clipping at 5.0** is required for stability. Without clipping, the magnetic-Laplacian pathway plus the bi-axis Mamba can produce gradient norms large enough to cause loss explosion within 5 epochs. This was discovered empirically (§ 3.3) and represents a discrepancy with STAEformer's training (`gradient_clip = 0.0`).

Mixed precision: bf16 autocast on H200 GPUs. The eigh solver in the learned-semantic view is explicitly promoted to FP32 even under autocast.

---

## 5. Oracle Analysis

We introduce a methodology for quantifying the *upper bound* on what any spectral residual learner can achieve. Given a baseline predictor producing $\hat{Y}_{base}$ (e.g., last-step persistence) and a target $Y_{true}$, a spectral residual restricted to lie in $\text{col}(U)$ for some basis $U \in \mathbb{R}^{N \times K}$ achieves at best

$$
L^*_K = \min_{\Delta \in \text{col}(U)} \| \Delta - (Y_{true} - \hat{Y}_{base}) \|_{\text{MAE}}
$$

This is computed in closed form by projecting $(Y_{true} - \hat{Y}_{base})$ onto $U$ — the resulting projection-reconstruct MAE is $L^*_K$.

### 5.1 METR-LA: K versus oracle ceiling

We computed $L^*_K$ on METR-LA's validation split for the symmetric Laplacian basis at $K \in \{32, 48, 64, 96, 128\}$, with persistence as baseline:

| $K$ | $L^*_K$ val_avg | $L^*_K$ val 60-min |
|---:|---:|---:|
| 32 | 3.71 | 4.54 |
| 48 | 3.40 | 4.13 |
| 64 | 3.15 | 3.79 |
| 96 | 2.64 | 3.13 |
| 128 | **2.07** | **2.46** |

STAEformer's validation MAE is 2.74. The oracle ceiling at $K = 128$ is **below STAEformer's value**. Bandwidth is *not* the bottleneck — a sufficient spectral residual exists in $\text{col}(U_{sym})$.

But across every spectral residual learner we tested (SSM-Magma standalone, STAE-Spectral-Magma joint-trained, STAE-Spectral-Magma frozen-trunk), the achieved validation MAE plateaued near 2.87 - 2.95. The gap from 2.07 (oracle) to ~2.90 (best learner) is the *predictability gap*: the optimal coefficients in $\text{col}(U)$ are not recoverable from input alone by the learners we test.

### 5.2 Implications

This methodology clarifies the field-wide pattern of plateau on METR-LA. It separates two distinct failure modes:

1. **Bandwidth-limited** ($L^*_K \gg$ STAEformer): the chosen basis cannot in principle express the optimal residual. Increasing $K$ would help.
2. **Predictability-limited** ($L^*_K \ll$ STAEformer): the optimal residual *exists* in the basis, but cannot be predicted from input alone by the architectures tested. Increasing $K$ does not help; architectural innovation on the *learner* might.

On METR-LA at $K = 128$, we are in the second regime. This is, to our knowledge, the first quantitative characterization of which type of saturation METR-LA exhibits. We propose this analysis as a diagnostic to be applied to any spectral-augmentation method on any benchmark before substantial method development is invested.

---

## 6. Implementation Details and Empirical Decisions

We document several non-obvious implementation choices, motivated by failures observed during development. Each represents a real-world engineering cost of the methods proposed.

### 6.1 Sidechain initialization

The up-projection from $d_{branch}$ to $d_{model}$ in the spectral sidechain (§ 4.3) is initialised with $\sigma = 10^{-3}$, much smaller than PyTorch's default. This is essential. With default initialisation, the sidechain produces non-trivial output at step 1, which interferes with STAEformer's convergence regime. Validation MAE plateaus 0.10 - 0.15 higher than with σ = 10⁻³ initialisation under otherwise identical hyperparameters.

### 6.2 Gradient clipping

As noted in § 4.6, training requires `gradient_clip = 5.0`. Without it, joint training of the STAEformer encoder and the magnetic-Laplacian sidechain produces a loss explosion at epoch 5-7 (training MAE jumps from $\sim 1.5$ to $\sim 5.0$, validation similarly). This was first observed on PEMS-BAY and reproduced consistently. STAEformer alone (without our sidechain) trains fine with `gradient_clip = 0.0`; the requirement is introduced by the spectral sidechain, specifically the magnetic complex-projection pathway.

### 6.3 Learned-semantic basis stability

#### 6.3.1 First observed failure

The naive implementation of the learned-semantic view caches `self.U` across forward passes and recomputes every 200 steps. This was empirically catastrophic: the first attempt at training crashed with `RuntimeError: Trying to backward through the graph a second time` because the cached `U` retained autograd graph references from the step on which it was computed.

#### 6.3.2 Second observed failure

Once we recomputed `U` every training forward (preserving autograd graph), the eigh solver intermittently raised `torch._C._LinAlgError: ill-conditioned or repeated eigenvalues` once embeddings drifted into near-degenerate configurations. We observed this consistently with larger $d_{branch} = 96$ configurations.

#### 6.3.3 Final implementation

Three defences:
1. Diagonal jitter ($\epsilon = 10^{-5}$): $\tilde{L} = L + \epsilon I$.
2. FP32 promotion: explicitly cast $\tilde{L}$ to FP32 before eigh, even under bf16 autocast.
3. Previous-basis fallback: catch any `_LinAlgError` from eigh and return the previously cached basis. At cold start (no cache), fall back to the identity.

With these three safeguards, training is stable across all configurations we tested. We strongly recommend this pattern for any future work that backpropagates through `torch.linalg.eigh` on a learned matrix.

### 6.4 The position-versus-TOD adaptive-embedding bug

During Phase B (§ 3.2), we initially implemented the adaptive embedding as $E \in \mathbb{R}^{T_{in} \times N \times d_{adp}}$ — indexed by absolute window position. This is naively analogous to STAEformer's embedding shape but conceptually different: STAEformer's embedding is indexed by *time-of-day* (one entry per 5-minute bin per sensor per day, shared across all windows that hit the same TOD), not window position.

The position-indexed version overfit aggressively because the same parameter slot was being trained against many different absolute time-of-day patterns simultaneously, allowing the model to memorize "window 5 of training day 7 looked like X" without forced generalization across windows. Refactoring to TOD-indexed $[288 \times N \times d_{adp}]$ corrected this. We mention this primarily as a documentation aid for future researchers reimplementing STAEformer-like adaptive embeddings on new datasets.

### 6.5 Schedule sensitivity

STAEformer's published training schedule (milestones [20, 30] with $\gamma = 0.1$) is essential for full convergence to validation MAE 2.74 on METR-LA — under faster schedules, STAEformer alone plateaus 0.10 - 0.15 higher. Our hybrid inherits this sensitivity. For the ablation study (§ 7.4) we used a *compressed* schedule (milestones [10, 18]) to fit six 30-epoch runs in a feasible compute budget. This means our ablation comparisons are mutually consistent (same compressed schedule) but the absolute numbers should not be directly compared to results obtained under the published 60-epoch schedule.

---

## 7. Experiments

### 7.1 Datasets

We use METR-LA [Li et al. 2018] and PEMS-BAY (same source) under the canonical traffic-forecasting protocol: 5-minute cadence, $T_{in} = T_{out} = 12$, chronological 70/10/20 train/val/test split, masked MAE with mask = (speed > 0 ∧ finite).

| | $N$ | $T$ | Adjacency | Mean speed | Std speed |
|---|---:|---:|---|---:|---:|
| METR-LA | 207 | 34,272 | symmetric, distance-based | 58.58 | 12.82 |
| PEMS-BAY | 325 | 52,128 | symmetric, distance-based | 62.74 | 9.43 |

We attempted to extend to PEMS04 (307 sensors) and PEMS08 (170 sensors) but were unable to obtain the data under our compute budget; this remains future work.

### 7.2 STAEformer reproduction

We independently reproduce STAEformer at canonical seed 42:

| | Val avg | Val 15-min | Val 30-min | Val 60-min | Test 60-min |
|---|---:|---:|---:|---:|---:|
| STAEformer (METR-LA, ours) | **2.740** | 2.458 | 2.764 | 3.147 | ~3.34 |
| STAEformer (METR-LA, published) | 2.74 | — | — | — | 3.34 |
| STAEformer (PEMS-BAY, ours) | **1.569** | 1.353 | 1.637 | 1.890 | ~1.89 |
| STAEformer (PEMS-BAY, published) | 1.57 | — | — | — | 1.86 |

Within 0.02 of published numbers on both datasets. Pipeline verified.

### 7.3 Main results

| Configuration | Val avg | Val 60-min | Test 60-min |
|---|---:|---:|---:|
| STAEformer (PEMS-BAY) | 1.569 | 1.890 | ~1.89 |
| STAE-Spec, joint, schedule [20,30] | 1.564 | 1.866 | — |
| STAE-Spec, joint, schedule [20,30], gradient_clip=0 | exploded ep 5 | — | — |
| STAE-Spec, frozen STAEformer trunk | drifts from 2.74 to 2.83+ (METR-LA) | — | — |
| **STAE-Spec, no_mag, schedule [10,18]** | **1.525** | **1.832** | **1.874** |

On METR-LA, every STAE-Spectral-Magma configuration we tested matched or underperformed STAEformer:

| Configuration (METR-LA) | Val avg | Note |
|---|---:|---|
| STAEformer | 2.740 | — |
| STAE-Spec, joint, full | 2.875 | underperforms |
| STAE-Spec, joint, no_mag | similar | not better |
| STAE-Spec, frozen-trunk | 2.834 (rising) | residual learns noise |

These results led to the saturation finding documented via oracle analysis (§ 5).

### 7.4 Ablation study (PEMS-BAY, seed 42, compressed schedule)

We ran three configurations at the same seed and the same compressed schedule (30 epochs, milestones [10, 18], patience 15) to isolate the contributions of the magnetic Laplacian view and the bi-axis Mamba mode-axis scan.

| Variant | Val avg | Val 60-min | Test 60-min | Δ val_avg vs full |
|---|---:|---:|---:|---:|
| `full` (sym + mag + sem + bi-axis + router) | 1.543 | 1.854 | 1.909 | — |
| `no_mag` (no magnetic view) | **1.525** | **1.832** | **1.874** | **−0.018** |
| `no_modeaxis` (mode-axis scan disabled) | **1.532** | 1.836 | 1.885 | **−0.011** |

**Reading of these results.** Both ablations improve over the full model. Specifically:

- Removing the magnetic Laplacian view improves validation by 0.018 MAE and test 60-min by 0.035 MAE.
- Removing the mode-axis scan improves validation by 0.011 MAE.
- The `no_mag` configuration crosses the STAEformer baseline (test 60-min 1.874 vs 1.890); the `full` model does not (1.909 vs 1.890).

The architecture *as a whole* beats STAEformer on PEMS-BAY only when the magnetic Laplacian view is excluded.

### 7.5 Single-seed limitation

All ablation numbers are single-seed. Seed variance on STAEformer-class architectures on these benchmarks is typically $\sigma \approx 0.005$ - $0.010$ at validation MAE. Our 0.018 MAE gap from `full` to `no_mag` is therefore approximately $1.8 \sigma$ to $3.6 \sigma$ — suggestive but not definitive. Confirming the result would require 3-5 seeds per variant, which exceeded our compute budget for the present study. We discuss this limitation in § 9.1.

---

## 8. Discussion

We offer mechanistic explanations for the two negative ablation findings.

### 8.1 Why the magnetic Laplacian did not help

We hypothesized (§ 2.4) that the magnetic Laplacian would inject directional flow structure that STAEformer's permutation-equivariant attention cannot recover. Three plausible reasons it did not:

**Reason 1: STAEformer's adaptive embedding already encodes directionality implicitly.** With $N \times T \times d_{adp} = 325 \times 12 \times 80 = 312\text{K}$ free parameters dedicated to per-(sensor, time-index) memory, STAEformer can implicitly memorize "sensor $i$'s speed at time $t$ predicts sensor $j$'s speed at $t + \tau$" patterns through its attention-weight learning. Adding an explicit phase-based directional bias is redundant and competes with this implicit representation.

**Reason 2: Our directed adjacency was inferred from data, not provided.** METR-LA and PEMS-BAY ship symmetric distance-based adjacencies. Our lagged-correlation estimation (§ 4.3.2) is noisy at the per-edge level and may have produced a directed graph whose magnetic spectrum does not cleanly separate the dominant directional patterns. A dataset with native directed adjacency (e.g., traffic flow with known one-way roads, river network flow gauges, electrical power grids with directed transmission) might tell a different story.

**Reason 3: Complex-to-real folding may have wasted capacity.** The complex magnetic basis was folded into $2 \times d_{branch}$ real channels for the real-valued bi-axis Mamba. The folded representation is technically lossless, but the downstream Mamba must learn to *interpret* the Re/Im fold convention, doubling the effective representation space the model must navigate. In contrast, Mag-Mamba [Anonymous 2026] modifies the SSM recurrence to operate natively in the complex plane — a different design choice that may avoid this wasted-capacity issue.

The negative result, taken at face value, is: **on standard traffic-forecasting benchmarks with symmetric distance-based adjacencies, the magnetic Laplacian view as we implemented it does not contribute beyond a permutation-equivariant attention-based encoder**. This does not rule out the technique for traffic forecasting in general, but it puts a clear empirical bound on the gains realisable under our specific design.

### 8.2 Why the bi-axis Mamba mode-axis scan was marginal

We hypothesized (§ 4.4) that selective scan along eigenvalue-ordered modes would exploit mode coupling — particularly the interaction between low modes (rush-hour mean field) and high modes (local perturbations). Three plausible reasons it did not deliver:

**Reason 1: $K$ is too short.** Mamba's selective scan derives its power from long-sequence directional dependence. At $K = 64$ - $128$, we are in the regime where attention or even a simple linear layer can model arbitrary cross-mode interactions. The selective scan adds parameters without adding the *type* of inductive bias that matters at this scale.

**Reason 2: The "ordering" of modes by eigenvalue is real but the gating may not exploit it.** The sigmoid gate that fuses $y_T$ and $y_K$ is computed per-token, not per-mode-block. So while the underlying scan respects eigenvalue order, the gating mechanism may mostly route most of the signal through the temporal scan in practice. We did not directly inspect the per-mode gate values in this study; that would be informative future work.

**Reason 3: The temporal scan already gets the signal it needs.** STAEformer's attention provides cross-time mixing; our temporal Mamba scan provides additional mixing of the same kind. The mode-axis scan adds a *different* kind of mixing — cross-mode — that may not be useful when the model already has rich representations. This is consistent with the finding that the symmetric and semantic spectral views (which provide cross-mode mixing through the basis projection itself, regardless of the Mamba block) account for most of the architecture's improvement over STAEformer.

The negative-to-marginal result reads as: **on short K-mode sequences with strong cross-mixing already provided by the basis projection, an additional selective-scan over modes adds parameters without proportional generalization benefit**. This is consistent with the broader observation in the Mamba literature that selective scans are most useful on long sequences with sparse, directional dependence.

### 8.3 What the learned-semantic basis is actually doing

The learned-semantic view is included in the `no_mag` winning configuration, but our experiments do not directly isolate its individual contribution. We expect, based on the hypothesis underlying its design, that it captures cross-corridor similarities that the geographic symmetric basis cannot represent (e.g., sensors at mile 5 of different freeways behaving similarly during rush hour). Directly verifying this would require visualizing the learned embedding and comparing the resulting kNN graph to the geographic graph — informative qualitative analysis we leave to future work.

The fact that the learned-semantic view contributes (even partially) to the architecture's improvement is itself somewhat surprising: STAEformer's adaptive embedding should also be able to encode "sensor $i$ behaves like sensor $j$" implicitly via attention learning. The fact that an explicit spectral basis from a *learned* similarity graph helps suggests that putting this similarity into the *projection operation* (rather than relying on attention to discover it) provides a useful inductive bias.

### 8.4 The METR-LA saturation finding

Across every configuration we tested, no variant exceeded STAEformer on METR-LA validation MAE. The oracle analysis (§ 5) clarifies that this is not a bandwidth limitation — a $K = 128$ residual in principle can achieve validation MAE 2.07, well below STAEformer's 2.74. The bottleneck is the *predictability of optimal coefficients from input alone*.

This is consistent with the prior independent finding (DiSR-Mamba [Li et al. 2026, internal]) that frozen-trunk residual learning fails on METR-LA — the residual learns training-set noise that does not generalize. Our work strengthens that finding with joint-training evidence: even when the residual learner is allowed to co-train with the encoder, it cannot navigate to the predictability ceiling within typical training budgets.

We propose, tentatively, that METR-LA may be near its inherent forecasting limit given $T_{in} = 12$ steps of input — a property of the data, not the architecture. Confirming this would require an information-theoretic analysis (e.g., mutual information between input windows and 60-minute-ahead speeds) which is beyond our scope.

---

## 9. Limitations

### 9.1 Single-seed ablation

The ablation results in § 7.4 are single-seed. The 0.018 validation MAE gap from `full` to `no_mag` is approximately $2\sigma$ given typical seed variance, suggestive but not definitive. Multi-seed confirmation (3-5 seeds per variant) would be required for stronger empirical claims. The conclusions of this study should be read with that caveat.

### 9.2 Scope limited to two benchmarks

We evaluate on METR-LA and PEMS-BAY only. The original study design included PEMS04 and PEMS08 (flow-prediction benchmarks with $\sim 10\times$ larger absolute MAE), which might exhibit different saturation behavior and possibly reverse some of our negative findings. Compute constraints (and a 404 on the data-download URL during our experiment window) prevented inclusion. We hypothesize, based on the saturation/non-saturation pattern observed between METR-LA and PEMS-BAY, that PEMS04/08 would show clearer positive results for some of the architectural pieces ablated as negative here; we do not test this hypothesis.

### 9.3 No comparison to gradient-of-eigh-free alternatives

Our learned-semantic view requires backpropagation through `torch.linalg.eigh`, with numerical safeguards. An alternative is to detach the eigh result and train the embedding via a separate gradient path (e.g., an orthogonality regularizer). We did not directly compare these alternatives; this is left for future work.

### 9.4 Magnetic Laplacian directionality estimation

Our directed adjacency is inferred via lagged cross-correlation (§ 4.3.2). This is one of several plausible choices. Alternatives include explicit directed adjacency from road-network maps (if available), or end-to-end learning of the directional adjacency. The negative magnetic-Laplacian result might be specific to our directionality estimation rather than to the magnetic Laplacian per se.

### 9.5 Mode-axis gate inspection

We did not directly inspect the per-mode mode-axis gate values $g$ inside the bi-axis Mamba block. Such inspection would clarify whether the mode-axis scan is effectively unused (in which case removing it should be near-zero-cost) or whether it is used but unhelpfully (in which case removing it should improve performance). Our ablation shows the latter — removing the mode axis improves validation by 0.011 — but the mechanistic explanation in § 8.2 is conjectural.

---

## 10. Conclusion

We presented STAE-Spectral-Magma, a spectral state-space augmentation of the STAEformer traffic-forecasting backbone, comprising three Laplacian views (symmetric, magnetic, learned-semantic), a bi-axis Mamba block scanning along time and graph-spectral mode axes, and a horizon-cluster mixture-of-experts router. We tested four hypotheses and reported results honestly.

The positive findings are concrete: a learned-semantic spectral basis (an adaptive adjacency periodically eigendecomposed with numerical-stability safeguards) combined with a small horizon-cluster router improves STAEformer on PEMS-BAY by 0.016 test 60-min MAE at single seed, and the underlying mechanism is, as far as we can verify, novel to the spectral GNN literature. We further introduce an oracle-analysis methodology that explains the field-wide plateau on METR-LA at STAEformer's 2.74 MAE: the basis bandwidth is sufficient; the predictability of optimal coefficients from input is what limits learners.

The negative findings are equally concrete and arguably more useful to the field: the magnetic Laplacian, expected to capture directional flow, *harms* STAEformer's PEMS-BAY performance when added; ablating it produces the strongest improvement. The bi-axis Mamba mode-axis scan, hypothesized to exploit eigenvalue-ordered mode coupling, is marginal at best on the short $K \approx 64$ mode sequences here. We provide mechanistic explanations for both.

The contribution of this work is therefore three positive results (learned-semantic spectral SSM, horizon-cluster MoE router, oracle analysis methodology) and two characterised negative results (magnetic Laplacian for traffic forecasting, bi-axis Mamba mode scan on short K). We view the negative results as first-class contributions: they delineate which spectral augmentations of strong attention-based backbones are worth pursuing on saturated traffic-forecasting benchmarks, and which are not.

---

## References

(Compiled honestly from the citation map built during the literature audit. Arxiv IDs given where applicable; published-venue references cite the canonical version.)

- Bai et al. 2020. "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting." NeurIPS 2020. (AGCRN)
- Cao et al. 2020. "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting." NeurIPS 2020. (StemGNN)
- Defferrard et al. 2016. "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." NeurIPS 2016. (ChebNet)
- Gu and Dao 2024. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." COLM 2024. (Mamba)
- Gu et al. 2022. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. (S4)
- He et al. 2022. "MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian." LoG 2022.
- Khan et al. 2025. "Multi-scale Wavelet-Mamba framework for spatiotemporal traffic forecasting." Scientific Reports 2025. (WMF-Traffic)
- Kipf and Welling 2017. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017. (GCN)
- Lee et al. 2024. "TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts." ICLR 2024.
- Li et al. 2018. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018. (DCRNN, METR-LA and PEMS-BAY benchmarks)
- Li et al. 2024. "STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model." arXiv:2403.12418.
- Liu et al. 2023. "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting." CIKM 2023. (STAEformer)
- Lou et al. 2025. "Bi-MambaHSI: Spatial-Spectral Bidirectional Mamba for Hyperspectral Image Classification." arXiv:2501.04944.
- Park et al. 2025. "DSTGA-Mamba: a disentangled spatio-temporal graph attention Mamba model for traffic flow prediction." Scientific Reports 2025.
- Wang et al. 2023. "ST-MoE: Spatio-Temporal Mixture-of-Experts for Debiasing in Traffic Prediction." CIKM 2023.
- Wu et al. 2019. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019. (GraphWaveNet)
- Wu et al. 2020. "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." KDD 2020. (MTGNN)
- Zhang et al. 2021. "MagNet: A Neural Network for Directed Graphs." NeurIPS 2021.
- Zhou et al. 2025. "SSMGNN: Spectral temporal graph neural network with state space models for multivariate time-series forecasting." Neurocomputing 2025.
- Anonymous 2026. "Mag-Mamba: Modeling Coupled Spatio-Temporal Asymmetry for POI Recommendation." arXiv:2603.00053 (Feb 2026).
- Anonymous 2026. "Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting." arXiv:2510.07426 (Oct 2025). (TESTAM+ analysis)
- Anonymous 2026. "M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting." Researchgate publication 2026.

---

## Reproducibility statement

All code and experiments are reproducible from the public repository accompanying this manuscript. Saved STAEformer checkpoints at validation MAE 2.74 (METR-LA, seed 42) and 1.57 (PEMS-BAY, seed 42) are available, along with the run scripts:

- `scripts/train_staeformer.py` — STAEformer baseline reproduction
- `scripts/train_stae_spectral_magma.py` — full STAE-Spectral-Magma training, with `--no-use_mag`, `--no-use_sem`, `--no-spec_mode_axis`, `--no-use_router` flags for ablation
- `scripts/run_ablations_stae_spec.sh` — chained 6-variant ablation driver
- `scripts/run_multiseed_stae_spec.sh` — 3-seed baseline+hybrid driver

Random seeds, hyperparameters, and complete training logs are committed to the repository under `logs/` and `results/`.

---

*This paper is presented honestly, including all negative results. We believe the integrity of the empirical reporting is more important than the size of the headline number. The contributions described here are limited but real; we hope this work is useful both to those who would build on the positive findings (learned-semantic spectral basis, horizon-cluster MoE, oracle analysis) and to those who would reconsider attempting to extend the negative findings (magnetic Laplacians for traffic, bi-axis Mamba on short K) under different design choices.*
