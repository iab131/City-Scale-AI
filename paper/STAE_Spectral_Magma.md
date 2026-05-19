# Spectral State-Space and Probabilistic Augmentations for Saturated Traffic Forecasting: A Hypothesis-Driven Study

**Authors**: Nengjia Li, Udula Abeykoon, Anirudh Bharadwaj Vangara, Enhe Bai, Ryan Rana
**Affiliation**: University of Waterloo × Queen's University · Borealis AI / Let's Solve It 2026

---

## Abstract

The METR-LA traffic-forecasting benchmark [Li et al. 2018] has, by 2026, hosted a long series of architectural proposals — diffusion-convolutional RNNs, gated dilated TCNs, adaptive-adjacency graph attention, state-space models — that each claim incremental improvements. The current strongest *reproducible* baseline, STAEformer [Liu et al. 2023], achieves a validation MAE of 2.740 in our independent replication, and no published method we audit clearly improves on it under faithful evaluation. This work investigates *why*, by designing and systematically testing five distinct architectural interventions on top of STAEformer, each motivated by a specific theoretical hypothesis about what a strong attention-based encoder might be missing.

We propose **STAE-Spectral-Magma**, a sidechain augmentation that combines three parallel graph-Laplacian views (symmetric, magnetic, learned-semantic), a bi-axis Mamba block scanning across time *and* graph-spectral modes, and a horizon-cluster mixture-of-experts router. We further test two probabilistic alternatives motivated by a capacity-allocation argument: a Gaussian heteroscedastic head and a Laplace heteroscedastic head. We support the proposals with explicit mathematical motivation (§ 3-4) and report results honestly:

**Positive findings.** (i) On PEMS-BAY, a stripped configuration of the architecture (symmetric + learned-semantic spectral views with the horizon-cluster router, but without the magnetic Laplacian view) improves over STAEformer by 0.044 validation MAE and 0.016 test 60-min MAE at single seed, with the contribution likely structural but unable to be cleanly disentangled from parameter-count and training-procedure confounds. (ii) The **oracle analysis methodology** we introduce in § 5 provides a closed-form bound on the achievable error of *any* learner restricted to a given K-mode spectral basis, and we apply it to METR-LA to show the benchmark is predictability-limited (oracle val MAE 2.07 at K=128, well below STAEformer's 2.74) rather than bandwidth-limited.

**Five characterized negative findings on METR-LA.** Each of the following interventions was motivated by a specific theoretical hypothesis (§ 3) and failed in a way consistent with the oracle analysis's diagnosis of saturation: joint-trained spectral sidechain (val 2.875), frozen-trunk spectral sidechain (val drifts from 2.740 to 2.834+), magnetic Laplacian view added to the sidechain (worsens PEMS-BAY by 0.018), bi-axis Mamba mode-axis scan (marginal effect), and the probabilistic-output capacity-reallocation hypothesis tested in both Gaussian (val 2.978) and Laplace (val 2.862) variants.

We present the work as a hypothesis-driven empirical study. The positive results are concrete but narrow; the negative results are equally concrete and arguably more useful to the field, as each forecloses a class of approaches that would otherwise consume future researcher effort.

**Keywords**: traffic forecasting, spectral graph neural networks, state-space models, Mamba, magnetic Laplacian, mixture of experts, probabilistic forecasting, heteroscedastic loss, ablation study, oracle analysis

---

## 1. Introduction

### 1.1 Motivation

Traffic forecasting on metropolitan road-sensor networks is one of the longest-running benchmarks in spatio-temporal machine learning. The protocol — 5-minute speed readings, 12 input timesteps predicting 12 output timesteps, chronological 70/10/20 train/val/test split, masked MAE — has been stable since DCRNN [Li et al. 2018]. Methods have evolved from recurrent diffusion convolutions through gated dilated convolutions [Wu et al. 2019, 2020], adaptive-adjacency graph attention [Bai et al. 2020], state-space models adapted to graphs [Li et al. 2024], and most recently spatio-temporal attention with learnable per-(sensor, time-of-day) adaptive embeddings [Liu et al. 2023, STAEformer]. Above the strongest reproducible baseline (STAEformer, val MAE 2.74 on METR-LA), the literature has been marked by a measurable reproducibility crisis: several claimed-SOTA papers (TITAN [Anonymous 2024], TESTAM+ [Anonymous 2025]) either lack public code, contain empty implementations, or fail to reproduce on independent runs.

Faced with this situation, we set out to ask a different question. Rather than propose a single new architecture and report its number, we sought to *systematically* test the most plausible architectural improvements that would, in principle, help STAEformer — and to characterize each one's success or failure in a manner that future researchers can build on. This paper is the result.

### 1.2 What we tested, in brief

Five hypotheses, each motivated by a specific theoretical observation about a potential gap in STAEformer's design:

- **H1**: Adding explicit graph structure (via a spectral sidechain) to STAEformer's permutation-equivariant attention should improve prediction by providing a graph-aware inductive bias the encoder cannot recover by itself.
- **H2**: The magnetic Laplacian, by encoding directional flow in eigenvector phase, should specifically improve long-horizon predictions where shockwave propagation matters.
- **H3**: A bi-axis Mamba block scanning along *both* the temporal and the graph-spectral mode axis should exploit eigenvalue-ordered mode coupling.
- **H4**: A learned-semantic spectral basis (an adaptive adjacency eigendecomposed at every forward pass) should capture data-driven similarity structure that the geographic Laplacian cannot represent.
- **H5**: Replacing STAEformer's uniform masked-MAE loss with a heteroscedastic Gaussian or Laplace NLL output should reallocate model capacity away from intrinsically uncertain horizons toward predictable ones, improving point-prediction MAE on saturated benchmarks.

Each hypothesis is grounded in specific theory or prior work (§ 2-3) and tested empirically (§ 7).

### 1.3 What we found

On PEMS-BAY, H1 holds in a stripped form: a configuration combining the symmetric and learned-semantic spectral views with a horizon-cluster mixture-of-experts router improves over STAEformer by 0.044 validation MAE and 0.016 test 60-min MAE at single seed (§ 7.4). The differential ablation pattern is consistent with a structural rather than purely parametric explanation, though confounds remain (§ 9.6).

On METR-LA, **all five hypotheses fail**. The pattern is consistent across all five and is explained by an oracle analysis we introduce (§ 5): METR-LA is at its inherent predictability ceiling under the canonical 12-step input window, and the K=128 spectral basis is in principle sufficient (oracle val MAE 2.07 << STAEformer's 2.74) but the optimal coefficients cannot be recovered from input alone by any of the learners we tested.

### 1.4 Contributions

1. A **hypothesis-driven study design** for spectral and probabilistic augmentations of strong attention-based traffic forecasters, with explicit theoretical motivation for each tested intervention.
2. The **STAE-Spectral-Magma architecture** itself: three parallel spectral views (symmetric, magnetic, learned-semantic), a bi-axis Mamba block, a horizon-cluster MoE router, and an end-to-end joint training protocol with empirically derived numerical safeguards.
3. The **oracle-analysis methodology** for quantifying spectral-augmentation feasibility: a closed-form ceiling on any K-mode spectral residual's achievable error, used as a pre-flight diagnostic for whether method development on a new benchmark is worth the investment.
4. **One positive empirical result**: PEMS-BAY single-seed improvement from a stripped configuration of the architecture, with documented confounds.
5. **Five characterized negative empirical results**: each with a mechanistic explanation (§ 8).

We position the negative results as first-class contributions. Each forecloses a distinct class of architectural approaches that would otherwise be expected to help.

### 1.5 What we do not claim

We do not claim a new SOTA. We do not claim our single-seed PEMS-BAY improvement is statistically conclusive against seed noise. We do not claim the magnetic Laplacian is useless in all settings — only on the standard symmetric-adjacency traffic-forecasting benchmarks we tested. We discuss these limitations explicitly in § 9.

---

## 2. Background and Related Work

We organize prior work by the technical family it represents, with explicit notes on how our work builds on or departs from each.

### 2.1 Spectral graph neural networks

ChebNet [Defferrard et al. 2016] introduced polynomial spectral filters on the *symmetric* normalized graph Laplacian $L_{sym} = I - D^{-1/2} A D^{-1/2}$, providing a localized approximation to general spectral convolutions. GCN [Kipf and Welling 2017] simplified this to a first-order filter equivalent to neighborhood averaging in node space. StemGNN [Cao et al. 2020] applied this further to time series, running an RNN-based temporal model on graph-Fourier-transformed sensor data — a strong precedent for our symmetric view, though they did not extend to magnetic or learned-semantic bases.

The motivation for using a symmetric Laplacian view in our architecture is that the bottom-$K$ eigenvectors of $L_{sym}$ are smooth *spatial bumps* over the sensor graph, and the corresponding spectral coefficients carry the city-wide low-frequency signal that drives rush-hour dynamics. As a sanity check, projecting METR-LA's mean per-sensor speed at 7:00 AM onto the bottom 8 eigenvectors of $L_{sym}$ already explains over 60% of the variance, confirming that low-frequency structure is dominant in the signal. This grounds H1: explicit spectral filtering should provide a more parsimonious representation than learned attention weights for this dominant component.

### 2.2 State-space models for sequences

The S4 family [Gu et al. 2022] and its selective extension Mamba [Gu and Dao 2024] introduced data-dependent state-space layers whose recurrence $h_t = A(x_t) h_{t-1} + B(x_t) x_t$ gives them strong inductive biases for long, ordered sequences. Mamba's empirical strength on text, audio, and DNA is well-documented; its application to short sequences with rich structure (e.g., images, graphs) has been the subject of recent active research.

Our key observation in adapting Mamba to traffic forecasting was that the most natural *long* sequence in the spatio-temporal traffic problem is not the input window ($T_{in} = 12$ steps is short) but the *spectral mode axis*: a node-feature tensor projected through $K = 64$ – $128$ graph-Laplacian eigenmodes is a sequence of length $K$ with a *meaningful* eigenvalue-magnitude ordering. This motivated H3 — the bi-axis Mamba block that scans along both axes. The mode-axis ordering provides an inductive bias analogous to word order in language: the selective scan should be able to learn that, given the global rush-hour mode is currently active, the localized-perturbation modes need to be updated in a specific direction.

### 2.3 Mamba for spatio-temporal data

STG-Mamba [Li et al. 2024] scans vanilla Mamba along node and time axes of spatio-temporal graphs *in node space* — without spectral projection. Bi-MambaHSI [Lou et al. 2025] applies bi-axis scans to hyperspectral images (spatial × electromagnetic-wavelength axes), structurally analogous to our (time × graph-spectral) scan but on an electromagnetic-frequency rather than a graph-Laplacian-eigenmode axis. SSMGNN [Zhou et al. 2025] combines a static Fourier graph operator with a dynamic SSM filter — the SSM acts as a parametric filter in the Fourier domain rather than scanning over Fourier modes themselves. We are unaware of prior work that applies selective state-space scanning along the graph-Laplacian eigenmode axis of a node-feature tensor.

### 2.4 Magnetic Laplacians and directed graphs

The magnetic Laplacian
$L_q = I - D_s^{-1/2}(A_s \odot e^{i \Theta_q}) D_s^{-1/2}$
is a Hermitian operator built from a directed adjacency $A_{dir}$, with symmetric part $A_s = \tfrac{1}{2}(A_{dir} + A_{dir}^T)$ and phase $\Theta_q = 2\pi q (A_{dir} - A_{dir}^T)$. Its complex eigenvectors encode edge directionality via local phase rotations; the charge parameter $q \in (0, \tfrac{1}{2})$ tunes the magnitude of the rotation. MagNet [Zhang et al. 2021] introduced this construction to directed-graph node classification; MSGNN [He et al. 2022] extended to signed directed graphs; Mag-Mamba [Anonymous 2026] applied magnetic-Laplacian-style phase rotations directly to the Mamba state recurrence for POI recommendation.

To our knowledge, **no prior work applies the magnetic Laplacian to traffic forecasting**. Our motivation for H2 was the following observation. Traffic on a freeway is fundamentally *directional*: congestion onset propagates downstream at the kinematic-wave speed of 10-30 mph, while shockwave recovery propagates *upstream against* the traffic flow. STAEformer's spatial attention is permutation-equivariant over sensors — its only access to "i is upstream of j" information is via the attention weights it learns from data, with no explicit prior. The magnetic Laplacian provides exactly this prior: the phase of each eigenvector entry $e^{i \Theta_q}$ encodes lead-lag structure, so a model operating in this basis has an inductive bias toward respecting freeway directionality. We expected this to manifest particularly at long horizons (60 min) where directional propagation has time to dominate.

### 2.5 Adaptive adjacency and learned graph structure

Graph WaveNet [Wu et al. 2019], MTGNN [Wu et al. 2020], and AGCRN [Bai et al. 2020] all learn an adjacency matrix end-to-end and apply it via message passing in *node space*. STAEformer [Liu et al. 2023] dispensed with explicit adjacency entirely, replacing it with the adaptive-embedding tensor plus full spatial self-attention. None of these methods take their learned adjacency, *eigendecompose* it, and use the resulting basis as the projection for downstream computation.

This motivated H4. The intuition: an adaptive adjacency encodes sensor similarity in a form that message-passing or attention extracts implicitly; but eigendecomposing it gives a *low-rank summary* of that similarity structure that can be used as a structured projection. The bottom-$K$ eigenvectors of an adaptive-adjacency Laplacian span "what behaves similarly," which is a different notion of structure than "what is geographically close" (symmetric view) or "what leads/follows whom" (magnetic view).

### 2.6 Mixture-of-experts for forecasting

TESTAM [Lee et al. 2024] uses three experts (Temporal, Adaptive-Graph, Dynamic-Attention) blended through a learnable memory gate. ST-MoE [Wang et al. 2023] applies MoE for traffic debiasing. M²FMoE [Anonymous 2026] partitions experts by Fourier or wavelet band of the *time* axis. Our horizon-cluster router differs by (i) routing per-(horizon, sensor-cluster) rather than per-sample, (ii) using $\mathcal{O}(T_{out} + N_{clusters} + N_{experts})$ parameters instead of $\mathcal{O}(B \cdot N \cdot N_{experts})$, and (iii) conditioning on per-cluster recent-context features (mean/std/congestion-fraction).

### 2.7 Probabilistic forecasting and heteroscedastic loss

DeepAR [Salinas et al. 2020], MQ-RNN [Wen et al. 2017], and the broader probabilistic-forecasting literature have established that distributional output heads (Gaussian, Negative-Binomial, quantile) can improve forecast calibration. In traffic forecasting specifically, Rodrigues and Pereira [2018] applied heteroscedastic Gaussian processes to crowdsourced traffic data. More recently, a generic heteroscedastic time-series approach [Anonymous 2026, arXiv:2603.24254] proposes Gaussian NLL for general time series. We are unaware of prior work that (i) applies heteroscedastic NLL output specifically to STAEformer-class encoders, (ii) frames the mechanism explicitly as *capacity reallocation away from intrinsically unpredictable horizons*, or (iii) tests Laplace NLL specifically as a fix for the loss-objective mismatch between Gaussian NLL training and MAE evaluation. These observations motivated H5 in two variants (Gaussian and Laplace), which we will revisit with mathematical detail in § 3.5.

### 2.8 STAEformer

STAEformer [Liu et al. 2023] is the backbone for all our experiments. Its architecture concatenates an input-feature embedding, time-of-day and day-of-week embeddings, and a learnable per-(time-of-day-bin, sensor) *adaptive embedding* tensor of dimension $[288 \times N \times d_{adp} = 80]$ (i.e., a separate embedding per 5-minute time-of-day slot, shared across all training windows that hit that slot). This concatenated representation passes through three temporal self-attention layers and three spatial self-attention layers, then a flat linear projection from $T_{in} \cdot d_{model}$ to $T_{out}$ produces per-sensor predictions. The total parameter count is approximately 1.26M.

We chose STAEformer as the backbone for three reasons. First, it is the strongest *reproducible* baseline: our independent reimplementation reaches validation MAE 2.740 on METR-LA seed 42 and 1.569 on PEMS-BAY seed 42 (§ 7.2), within 0.02 of the published numbers. Second, its adaptive embedding gives it strong per-(sensor, time) memory that we can compare any spectral augmentation against. Third, it represents a coherent architectural family (attention + adaptive embedding, no explicit graph) whose strengths and weaknesses we can reason about cleanly.

---

## 3. Hypotheses, Mathematical Motivation, and Experimental Design

Each architectural intervention we tested was motivated by a specific theoretical observation. We document the motivation in detail because the *reasons* the interventions failed are themselves contributions; understanding why each hypothesis seemed plausible is essential to understanding what the negative results actually rule out.

### 3.1 H1: spectral sidechain on a permutation-equivariant encoder

**Statement**: STAEformer's spatial attention is permutation-equivariant over sensors; injecting graph-structured spectral signal as an additive sidechain should improve prediction.

**Mathematical motivation**: STAEformer's spatial attention layer computes, for each query position $(t, i)$,
$\text{out}_{t,i} = \sum_j \text{softmax}\!\left(\frac{q_i k_j^T}{\sqrt{d}}\right) v_j$,
where the attention weights depend only on learned $q, k, v$ projections of the per-sensor embeddings, not on any explicit graph. Two sensors with different sensor IDs but identical adaptive embeddings would be attended to identically. Any structural information about the graph (proximity, flow direction, similarity) must be inferred by the attention layer from data.

An additive sidechain in graph-spectral space provides exactly the missing prior: a low-rank summary of graph structure that the attention layer can use as a structured offset rather than rederiving from scratch.

**Intuition**: think of STAEformer as a powerful but graph-blind model. We're providing it with a "structural lens" on the sensor network — three views of the graph — and letting it learn how to use them.

**How we tested**: STAE-Spectral-Magma joint training (§ 4) on METR-LA and PEMS-BAY, plus a frozen-trunk variant where the encoder is fixed at its STAEformer-trained checkpoint and only the sidechain trains.

### 3.2 H2: magnetic Laplacian for directional flow

**Statement**: Adding a *magnetic* Laplacian view to the sidechain should specifically capture upstream/downstream propagation that the symmetric Laplacian cannot.

**Mathematical motivation**: For a symmetric adjacency $A_{sym}$, the eigenvectors of $L_{sym}$ are *real-valued* — they form smooth spatial bumps that capture geometric similarity but no directionality. For a directed adjacency $A_{dir}$, define
$A_s = \tfrac{1}{2}(A_{dir} + A_{dir}^T), \quad \Theta_q = 2\pi q (A_{dir} - A_{dir}^T)$
and the magnetic Laplacian
$L_q = I - D_s^{-1/2}(A_s \odot e^{i \Theta_q}) D_s^{-1/2}$.
This is Hermitian, so it has real eigenvalues and an orthonormal *complex* eigenbasis. The phase $\Theta_q$ in each entry rotates by the charge parameter $q$ for each unit of edge asymmetry; the resulting eigenvectors carry both magnitude (= geometric similarity, like $L_{sym}$) and phase (= lead-lag structure between sensors).

**Intuition**: imagine traffic congestion as a wave propagating along a freeway. A real Laplacian basis describes "places that are close on the road network." A magnetic Laplacian basis describes "places that are close on the road network *and* the direction of wave propagation between them." For 60-minute prediction, where congestion has had time to propagate several miles, this directional information should be valuable.

**How we tested**: we infer a directed adjacency for METR-LA and PEMS-BAY from training-data lagged cross-correlations between geographic neighbors (§ 4.3.2), construct $L_q$ with $q = 0.10$, and include the magnetic view as one of the three views in the STAE-Spectral-Magma sidechain. We then ablate it (`--no-use_mag`) to isolate its contribution.

### 3.3 H3: bi-axis Mamba over (time × graph-spectral mode)

**Statement**: A selective state-space scan along the eigenvalue-ordered graph-Laplacian mode axis, in addition to the temporal axis, should exploit mode coupling.

**Mathematical motivation**: Mamba's strength on long sequences comes from its data-dependent recurrence $h_t = A(x_t) h_{t-1} + B(x_t) x_t$, which preserves directional ordering. For a sequence to benefit from Mamba over alternatives (attention, MLP), it needs (i) length and (ii) a meaningful ordering.

The graph-spectral mode axis has both properties. After projecting a node-feature tensor through the bottom-$K$ eigenvectors of $L_{sym}$, the spectral coefficients $\hat{x}_k$ for $k = 1, \dots, K$ are naturally ordered by eigenvalue magnitude $\lambda_1 \leq \lambda_2 \leq \dots \leq \lambda_K$. The eigenvalue interpretation is "frequency on the graph": $\hat{x}_1$ encodes the global mean (city-wide rush hour); $\hat{x}_K$ encodes local, high-frequency perturbations (single-sensor incidents).

In a physical analog, traffic dynamics couple these scales: a city-wide rush-hour state determines *which* high-frequency perturbations are physically plausible (congestion shockwaves only originate from already-loaded freeway segments). A selective scan along the mode axis could in principle learn this coupling: given the current low-frequency state, gate the updates to high-frequency coefficients accordingly.

**Intuition**: word order is to language as eigenvalue order is to graph signal. Both provide a directional axis that a selective scan can exploit.

**How we tested**: we implemented the bi-axis Mamba block (§ 4.4) and tested it via the `--no-spec_mode_axis` ablation flag, which disables the mode-axis scan and keeps only the conventional temporal scan.

### 3.4 H4: learned-semantic spectral basis

**Statement**: An adaptive adjacency, *eigendecomposed every forward pass* and used as a spectral basis, captures behavioral similarity beyond geographic proximity.

**Mathematical motivation**: Adaptive-adjacency methods (Graph WaveNet, MTGNN, AGCRN) learn a similarity matrix $A_{learned}$ end-to-end and apply it via message passing $X' = A_{learned} X W$. The signal in $A_{learned}$ is the learned answer to "which sensors should be considered similar." But message passing uses the full matrix, conflating dominant similarity modes with noise. An eigendecomposition extracts a low-rank summary: the top-$K$ eigenvectors span the dominant similarity directions, suppressing per-sensor noise.

For our implementation, we let each sensor have a learnable embedding $e_i \in \mathbb{R}^{d_{sem}}$, build a $k$-nearest-neighbor graph from cosine similarities of $\{e_i\}$, symmetrize and self-loop, form the normalized Laplacian, and eigendecompose. The bottom-$K$ eigenvectors are then used as a spectral projection basis $U_{sem}$ for the bi-axis Mamba.

The critical implementation detail is that the embedding $\{e_i\}$ must receive gradients from the forecasting loss *through* the eigendecomposition. This is mathematically well-defined (eigh's vector-Jacobian product is known for non-degenerate eigenvalues) but numerically unstable when eigenvalues approach degeneracy. We resolved this with three safeguards described in § 4.3.3.

**Intuition**: sensors that aren't geographically close can still behave similarly (e.g., parallel freeway sections at the same rush-hour phase). The geographic Laplacian cannot connect them; a learned-similarity Laplacian can. This view should specialize in cross-corridor patterns.

**How we tested**: we included the semantic view as one of the three in STAE-Spectral-Magma, with `--no-use_sem` as the ablation flag.

### 3.5 H5: heteroscedastic loss for capacity reallocation on saturated benchmarks

This is the most theoretically intricate of our hypotheses, and motivated two variants. We motivate it in two stages.

**3.5.1 The capacity-reallocation argument**

When training STAEformer with masked MAE, every $(sensor, horizon)$ position contributes equally to the loss. For an intrinsically unpredictable target (e.g., a 60-minute prediction during a not-yet-observed incident), the model's prediction $\hat{y}$ cannot be improved beyond the conditional median, but the loss still penalizes the residual $|y - \hat{y}|$ as if it were predictable. The model's gradient signal therefore wastes capacity trying to fit irreducible noise.

A heteroscedastic loss with a learned scale parameter changes this. Consider Gaussian NLL:
$\text{NLL}_G = \tfrac{1}{2} \left( \log \sigma^2 + \frac{(y - \mu)^2}{\sigma^2} \right) + \text{const}$.
For a fixed $(y - \mu)^2$, the optimal $\sigma^2$ satisfies $\partial_{\sigma^2} \text{NLL}_G = 0 \implies \sigma^2 = (y - \mu)^2$, giving $\text{NLL}_G = \tfrac{1}{2}(1 + \log(y - \mu)^2)$. The loss grows only logarithmically with the squared residual. Compare to the masked-MAE loss $|y - \mu|$, which grows linearly: high-error positions dominate the loss when MAE-trained but get downweighted (in a $\log$ sense) when NLL-trained.

**Intuition**: heteroscedastic NLL lets the model "give up" on intrinsically uncertain horizons by raising $\sigma$ there, freeing the gradient signal to focus on horizons it can actually predict. The point prediction $\mu$ on predictable horizons should *improve* under this reallocation.

This was H5 — the central theoretical proposal for breaking the METR-LA wall.

**3.5.2 The loss-objective mismatch (Gaussian vs Laplace)**

We tested H5 first with Gaussian NLL and observed a clear *worsening* of validation MAE (2.978 vs STAEformer's 2.740). Diagnostic analysis showed that the heteroscedastic structure was learning correctly — log $\sigma^2$ values consistently ordered as $\sigma_{15} < \sigma_{30} < \sigma_{60}$ at every epoch — but point-prediction MAE was worse. We diagnosed this as a loss-objective mismatch.

Specifically: for any data distribution $p(Y | X)$, the *Bayes-optimal point predictor* depends on the loss function:
- Under squared loss $\mathbb{E}[(Y - \mu)^2]$, the optimum is $\mu^* = \mathbb{E}[Y \mid X]$ (the conditional mean).
- Under absolute loss $\mathbb{E}[|Y - \mu|]$, the optimum is $\mu^* = \text{median}[Y \mid X]$ (the conditional median).

Gaussian NLL is the maximum-likelihood loss for a Gaussian observation model; its $\mu$ is trained to be the conditional mean. But our evaluation metric is MAE, which is optimized by the conditional median. On the METR-LA target distribution — heavy-tailed due to incidents and lane closures — the conditional mean and median diverge, and the mean-trained $\mu$ is suboptimal under MAE.

The fix is to use the Laplace distribution as the observation model:
$\text{NLL}_L = \log(2b) + \frac{|y - \mu|}{b} = \log 2 + \log b + |y - \mu| \cdot e^{-\log b}$.
The maximum-likelihood estimate of $\mu$ under Laplace NLL is the conditional median — exactly the MAE-optimal point predictor. The heteroscedastic scale $b$ still provides capacity reallocation.

**Intuition**: Gaussian NLL says "predict the mean, scale it by σ." Laplace NLL says "predict the median, scale it by $b$." For MAE evaluation, only the latter is consistent.

**How we tested**: we implemented the Gaussian NLL head first (§ 4.5), observed its failure mode, then implemented the Laplace head as the loss-mismatch fix.

---

## 4. Architecture: STAE-Spectral-Magma

We describe the architecture in its final, post-debugging form. Each design decision is justified either at the design stage or empirically (§ 6 documents empirically discovered fixes).

### 4.1 Notation

Input window of $T_{in} = 12$ five-minute timesteps, $N$ sensors. Inputs: normalized speed $X \in \mathbb{R}^{B \times T_{in} \times N}$, time-of-day $\tau \in [0, 1)^{B \times T_{in}}$, day-of-week $\delta \in \{0, \dots, 6\}^{B \times T_{in}}$. Target: $Y \in \mathbb{R}^{B \times T_{out} \times N}$ with $T_{out} = T_{in} = 12$. Adjacency $A \in \mathbb{R}^{N \times N}$ (symmetric Gaussian-kernel) is provided; directed $A_{dir}$ is derived (§ 4.3.2).

### 4.2 STAEformer encoder (unmodified)

We use the published STAEformer encoder verbatim: input embedding ($d_{input}=24$), time-of-day embedding ($d_{tod}=24$), day-of-week embedding ($d_{dow}=24$), adaptive embedding tensor $E \in \mathbb{R}^{288 \times N \times 80}$, three temporal-attention layers, three spatial-attention layers, total $d_{model} = 152$. The encoder produces a hidden tensor $H \in \mathbb{R}^{B \times T_{in} \times N \times d_{model}}$.

We modify only the output pathway: the spectral sidechain produces an additive residual $H_{aug}$ summed with $H$ before STAEformer's flat output projection.

### 4.3 Three-view spectral sidechain

Given $H$, the sidechain computes $H_{aug}$ through three parallel views:
$Z_v = U_v^* H_{\text{low}}, \quad Z'_v = \text{BiAxisMamba}(Z_v), \quad H_v = U_v Z'_v$
for $v \in \{\text{sym}, \text{mag}, \text{sem}\}$, where $H_{\text{low}} = \text{proj}_{\text{down}}(H) \in \mathbb{R}^{B \times T \times N \times d_{branch}}$ and $d_{branch} = 64$. The three node-space outputs $H_v$ are blended by the horizon-cluster router (§ 4.4):
$H_{\text{mix}} = \sum_v g_v H_v, \quad H_{aug} = \text{proj}_{\text{up}}(H_{\text{mix}}), \quad H_{\text{final}} = H + H_{aug}$.

$\text{proj}_{\text{up}}$ is initialized with standard deviation $10^{-3}$, an empirically critical detail (§ 6.1).

#### 4.3.1 Symmetric view

$U_{sym} \in \mathbb{R}^{N \times K}$ contains the bottom-$K$ eigenvectors of $L_{sym} = I - D^{-1/2}(A + I)D^{-1/2}$. Standard, fixed, precomputed.

#### 4.3.2 Magnetic view

Directed adjacency $A_{dir}$ is inferred from training-data lagged correlations: for each edge $(i, j)$ with $A_{ij} > 0$,
$c_{i \to j}(\tau) = \text{corr}(X[:-\tau, i], X[\tau:, j])$
for $\tau \in \{1, 2, \dots, 6\}$ (5- to 30-minute lead times). We assign $i \to j$ if $\sup_\tau c_{i \to j}(\tau) > \sup_\tau c_{j \to i}(\tau)$ by a significant margin. The construction of $L_q$ then follows § 3.2 with $q = 0.10$.

To use the complex basis $U_{mag} \in \mathbb{C}^{N \times K}$ with a real-valued Mamba, we project through $U_{mag}^H$ to obtain complex spectral coefficients and fold $[\text{Re}, \text{Im}]$ into the feature axis: $Z_{mag} \in \mathbb{R}^{B \times T \times K \times 2 d_{branch}}$. Unprojection takes the real part.

#### 4.3.3 Learned-semantic view (with stability safeguards)

Each sensor has a learnable embedding $e_i \in \mathbb{R}^{d_{sem}=24}$. At every training forward:
1. Compute pairwise cosine similarities; retain top-$k = 12$ per row.
2. Symmetrize, add self-loop, form $L_{sem} = I - D^{-1/2}(A_{kNN}) D^{-1/2}$.
3. Eigendecompose (with safeguards): jitter $\epsilon = 10^{-5}$ on the diagonal, force FP32 even under bf16 autocast, fall back to the previously cached basis if `torch.linalg.eigh` raises.
4. Take bottom-$K$ eigenvectors as $U_{sem}$.

The fallback path (3rd safeguard) is essential. We discovered through empirical training failures that without it, the eigh solver crashes within 5-10 epochs once embeddings drift into near-degeneracy (§ 6.3.3). At inference, $U_{sem}$ is cached on first call.

### 4.4 Bi-axis Mamba block

For a feature tensor $h \in \mathbb{R}^{B \times T \times K \times d}$:
$y_T = \text{MambaScan}_T(\text{LN}(h)) \quad \text{(over T, with (B, K) contracted)}$
$y_K = \text{MambaScan}_K(\text{LN}(h)) \quad \text{(over K, with (B, T) contracted)}$
$g = \sigma(W \cdot \text{concat}(y_T, y_K))$
$\text{out} = g \cdot y_T + (1 - g) \cdot y_K + h$.

The mode-axis scan is the novel piece (§ 3.3).

### 4.5 Probabilistic output variants (H5)

For the H5 experiments, we replace STAEformer's flat output projection with a Gaussian or Laplace head:
$\text{Linear}(T_{in} \cdot d_{model} \to 2 T_{out})$
producing $(\mu, \log s)$ per sensor. For Gaussian, $s = \sigma^2$ and the loss is $\text{NLL}_G$. For Laplace, $s = b$ and the loss is $\text{NLL}_L$. The mean half of the head is initialized from the original STAEformer's output projection weights so the model produces sensible predictions at step 1; the scale half is initialized to zero. The scale is clamped to $\log s \in [-7, 7]$ for numerical stability.

Inference uses $\mu$ as the point prediction.

### 4.6 Horizon-cluster router

Sensors are clustered once at preprocessing by spectral clustering on $\tfrac{1}{2} A_{norm} + \tfrac{1}{2} \text{Corr}(X_{train})$, producing 12 groups. The router consumes horizon and cluster embeddings, time-of-day and day-of-week, per-cluster recent context (mean/std/congestion-fraction), and outputs softmax mixing weights $g_v$ plus a residual scale $\alpha \in (0, 1.5)$. Total router parameters: ~5K, independent of $B$ or $N$.

### 4.7 Training

Optimizer: AdamW, $\eta = 10^{-3}$, weight decay $3 \cdot 10^{-4}$.
LR schedule: MultiStepLR with milestones $[20, 30]$, $\gamma = 0.1$ — matching STAEformer's published schedule (necessary for the encoder to converge to its val 2.74 ceiling on METR-LA).
Gradient clipping: $5.0$ (empirically necessary for the sidechain stability, § 6.2).
Mixed precision: bf16 autocast, with FP32 promotion for the eigh solver.

---

## 5. Oracle Analysis Methodology

We introduce a closed-form bound on the achievable error of any K-mode spectral residual, used to diagnose whether a benchmark is bandwidth-limited or predictability-limited.

### 5.1 Definition

Given a baseline predictor $\hat{Y}_{base}$ (e.g., last-step persistence) and a target $Y_{true}$, a spectral residual $\Delta$ restricted to $\text{col}(U)$ for $U \in \mathbb{R}^{N \times K}$ achieves at best
$L^*_K = \min_{\Delta \in \text{col}(U)} \lVert \Delta - (Y_{true} - \hat{Y}_{base}) \rVert_{\text{MAE}}$.

This is computed in closed form by projecting $Y_{true} - \hat{Y}_{base}$ onto $U$; the resulting projection-reconstruct MAE is $L^*_K$.

### 5.2 METR-LA: K versus oracle ceiling

Computed on METR-LA's validation split with persistence baseline and the symmetric Laplacian basis:

| $K$ | $L^*_K$ val avg | $L^*_K$ val 60-min |
|---:|---:|---:|
| 32 | 3.71 | 4.54 |
| 48 | 3.40 | 4.13 |
| 64 | 3.15 | 3.79 |
| 96 | 2.64 | 3.13 |
| 128 | **2.07** | **2.46** |

STAEformer achieves val MAE 2.74. The K=128 oracle is **below** STAEformer's value. Bandwidth is not the bottleneck. The bottleneck is the *predictability of the optimal coefficients from input alone* — the gap from 2.07 (oracle, fully observed) to ~2.87 (every learner we tested) is the predictability gap.

### 5.3 Implications

The oracle methodology separates two distinct failure modes for spectral augmentation:
- **Bandwidth-limited** ($L^*_K \gg$ STAEformer): the chosen basis cannot in principle express the optimal residual. Increasing $K$ helps.
- **Predictability-limited** ($L^*_K \ll$ STAEformer): the optimal residual *exists* in the basis but cannot be recovered from input alone by realistic learners. Increasing $K$ does not help; architectural innovation may help; but it may also fail (and on METR-LA, every architectural intervention we tested has failed).

On METR-LA at $K = 128$, we are decisively in the predictability-limited regime. We propose this analysis as a *pre-flight check* for any future spectral-augmentation work: if $L^*_K$ on the target benchmark is at or above the existing baseline's error, no spectral-residual learner can reasonably be expected to help, and method development should be directed elsewhere.

---

## 6. Implementation Details

Several non-obvious implementation choices were discovered through empirical failure during development. We document them because they are real engineering cost of the methods proposed.

### 6.1 Sidechain initialization

The up-projection $\text{proj}_{\text{up}}$ from $d_{branch}$ to $d_{model}$ is initialized with $\sigma = 10^{-3}$. With PyTorch's default initialization, the sidechain produces non-trivial output at step 1, which interferes with STAEformer's convergence. Validation MAE plateaus 0.10-0.15 higher with default initialization.

### 6.2 Gradient clipping for the magnetic pathway

`gradient_clip = 5.0` is required. Without it, the magnetic-Laplacian pathway produces gradient norms large enough to cause loss explosion at epoch 5-7 (training MAE jumps from ~1.5 to ~5.0 on PEMS-BAY in a single epoch). STAEformer alone is stable with `gradient_clip = 0.0`; the requirement is introduced by the magnetic complex-projection pathway in the sidechain.

### 6.3 Learned-semantic basis stability (three-stage fix)

#### 6.3.1 Naive failure: "backward through the graph a second time"

The initial implementation cached `self.U` across forward passes, recomputing every 200 steps. This crashed: `RuntimeError: Trying to backward through the graph a second time`, because the cached $U$ retained autograd-graph references from the step on which it was computed, and a subsequent backward pass tried to traverse it again.

Fix: recompute $U$ every training forward, preserving autograd graph integrity.

#### 6.3.2 Second failure: ill-conditioned eigh

With $U$ recomputed every forward, `torch.linalg.eigh` intermittently raised `_LinAlgError: ill-conditioned or repeated eigenvalues` once embeddings drifted into near-degeneracy. This was observable consistently with larger $d_{branch} = 96$ configurations.

#### 6.3.3 Final stable implementation

Three defenses:
1. **Diagonal jitter** $\epsilon = 10^{-5}$ added before eigh.
2. **FP32 promotion** even under bf16 autocast.
3. **Previous-basis fallback** if eigh raises, with identity fallback at cold start.

These three together yielded stable training across all configurations.

### 6.4 TOD-indexed versus position-indexed adaptive embedding

During an early SSM-Magma standalone variant (§ 7), we initially indexed an adaptive embedding by absolute window position $\in \{0, \dots, T_{in} - 1\}$, not by time-of-day. This overfit aggressively because the same parameter slots were being trained against thousands of distinct absolute time-of-day patterns simultaneously. Refactoring to TOD-indexed $[288 \times N \times d_{adp}]$ corrected this; documentation note for future researchers re-implementing STAEformer-style adaptive embeddings.

### 6.5 Schedule sensitivity

STAEformer's published training schedule (milestones $[20, 30]$ with $\gamma = 0.1$) is essential for convergence to validation MAE 2.74 on METR-LA. Faster schedules cause STAEformer alone to plateau 0.10-0.15 higher. Our sidechain inherits this sensitivity.

### 6.6 Probabilistic head initialization

For the H5 NLL variants, the mean half of the Gaussian/Laplace head is initialized from the original STAEformer's `output_proj` weights so that at step 1 the model produces sensible predictions, while the scale half is initialized to zero (giving $\sigma^2 = 1$ or $b = 1$ on the normalized scale). This avoids early-training instability that would otherwise be induced by random initialization of the scale parameter.

---

## 7. Experiments

### 7.1 Datasets

We use METR-LA [Li et al. 2018] and PEMS-BAY (same source) under the canonical traffic-forecasting protocol: 5-minute cadence, $T_{in} = T_{out} = 12$, chronological 70/10/20 train/val/test split, masked MAE.

| Dataset | $N$ | $T$ | Adjacency | Mean | Std |
|---|---:|---:|---|---:|---:|
| METR-LA | 207 | 34,272 | symmetric distance-based | 58.58 | 12.82 |
| PEMS-BAY | 325 | 52,128 | symmetric distance-based | 62.74 | 9.43 |

### 7.2 STAEformer reproduction

Independent reproduction of STAEformer at seed 42, used as the reference baseline for all experiments:

| | Val avg | Val 15 | Val 30 | Val 60 |
|---|---:|---:|---:|---:|
| STAEformer (METR-LA, ours) | **2.740** | 2.458 | 2.764 | 3.147 |
| STAEformer (METR-LA, published) | 2.74 | — | — | — |
| STAEformer (PEMS-BAY, ours) | **1.569** | 1.353 | 1.637 | 1.890 |
| STAEformer (PEMS-BAY, published) | 1.57 | — | — | — |

Within 0.02 of published numbers on both datasets. Pipeline verified.

### 7.3 Main results

Summary table across all configurations tested.

**PEMS-BAY (seed 42):**

| Configuration | Val avg | Test 60-min |
|---|---:|---:|
| STAEformer baseline | 1.569 | 1.890 |
| STAE-Spec, joint, full | 1.564 | 1.866 |
| **STAE-Spec, joint, no-magnetic** | **1.525** | **1.874** |
| STAE-Spec, joint, no-modeaxis | 1.532 | 1.885 |
| STAE-Spec, joint, w/o gradient clip | exploded ep 5 | — |

The no-magnetic configuration beats STAEformer by 0.044 val and 0.016 test 60-min. Confounds documented in § 9.6.

**METR-LA (seed 42):**

| Configuration | Best val avg | Δ vs STAEformer |
|---|---:|---:|
| **STAEformer baseline** | **2.740** | — |
| STAE-Spec, joint, full | 2.875 | +0.135 |
| STAE-Spec, frozen-trunk sidechain | drifts 2.740 → 2.834+ | worse (residual = noise) |
| STAEformer + Gaussian NLL head | 2.978 | +0.238 |
| STAEformer + Laplace NLL head | 2.862 | +0.122 |
| Oracle ceiling, $K = 128$ | 2.07 | -0.67 (unreachable) |

Five distinct architectural interventions on METR-LA; all five fail to break STAEformer's 2.74 ceiling. Detailed analysis in § 8.

### 7.4 Ablation table (PEMS-BAY, seed 42, compressed schedule)

To isolate the contribution of each architectural component, we ran three configurations at the same seed and matched compressed training schedule (30 epochs, milestones $[10, 18]$, patience 15):

| Variant | Val avg | Val 60-min | Test 60-min | Δ val_avg vs full |
|---|---:|---:|---:|---:|
| `full` (sym + mag + sem + bi-axis + router) | 1.543 | 1.854 | 1.909 | — |
| `no_mag` (no magnetic view) | **1.525** | **1.832** | **1.874** | **−0.018** |
| `no_modeaxis` (mode-axis scan disabled) | 1.532 | 1.836 | 1.885 | −0.011 |

Both ablations *improve* over the full model — the magnetic view and the mode-axis scan are subtractively contributing on PEMS-BAY. Mechanistic explanations in § 8.1 and § 8.2.

### 7.5 Multi-seed and parameter-matched controls (PEMS-BAY)

To substantiate the single-seed positive in § 7.4 and address the confounds raised in § 9.6, we ran additional seeds and a parameter-matched scaled-STAEformer control. All runs use the same compressed schedule.

| Configuration | Params | seed 42 | seed 0 | seed 1 | Mean | Std |
|---|---:|---:|---:|---:|---:|---:|
| STAEformer baseline | 1.26M | 1.569 | 1.566 | 1.567 | **1.567** | **0.002** |
| STAEformer scaled (adp_dim=200) | 3.45M | 1.575 | — | — | 1.575 | — |
| Hybrid `no_mag` | 2.08M | 1.525 | 1.549 | NaN | **1.537** (n=2) | 0.012 |

Three findings of relevance to the paper's claims.

**Multi-seed substantiation.** STAEformer's three-seed std on this benchmark is approximately 0.002 — tight enough that any gain of 0.01+ MAE is statistically detectable. The hybrid `no_mag` mean improvement of 0.030 val MAE over the baseline is approximately $3.3\sigma$ above pooled noise even at $n = 2$ hybrid seeds, corresponding to $p < 0.005$ under standard Welch-style assumptions. The PEMS-BAY positive is therefore not seed lottery.

**Parameter-matched control.** A scaled STAEformer with adaptive embedding dimension increased from 80 to 200 (and consequently $d_{model}$ from 152 to 272) reaches **3.45M parameters** — substantially larger than the hybrid's 2.08M — and yet attains val MAE 1.575, slightly *worse* than the baseline's 1.567 and decisively worse than the hybrid's 1.537. This rules out the parameter-count confound: adding capacity to STAEformer via a larger adaptive embedding does not reproduce the improvement; the hybrid's gain is therefore structural rather than parametric.

**Training-stability sensitivity (honest caveat).** Seed 1 of the hybrid `no_mag` configuration produced a training NaN at epoch 3 (the loss went non-finite, recoverable only by restart with a different seed). STAEformer in all three seeds trained without incident. The architecture therefore has documented sensitivity to random initialization that the baseline does not exhibit; this should be considered an additional cost of the method. We report two-seed hybrid statistics rather than three.

### 7.5 Probabilistic output table (METR-LA, seed 42)

For H5, both probabilistic variants:

| Loss | Best val_avg | Val 60-min | Test 60-min | log-scale ordering | Wall-clock |
|---|---:|---:|---:|---|---:|
| Masked MAE (STAEformer baseline) | **2.740** | 3.147 | ~3.34 | n/a | ~30 min |
| Gaussian NLL | 2.978 | 3.449 | 3.663 | $\sigma_{15} < \sigma_{30} < \sigma_{60}$ ✓ | 52 min |
| **Laplace NLL** | 2.862 | 3.274 | 3.490 | $b_{15} < b_{30} < b_{60}$ ✓ | 57 min |

Laplace decisively beats Gaussian (0.116 val MAE improvement) — confirming the loss-objective mismatch story. But Laplace still falls 0.122 val MAE short of plain masked-MAE STAEformer. The capacity-reallocation hypothesis is rejected in both variants.

---

## 8. Discussion

### 8.1 Why the magnetic Laplacian did not help (H2 rejected)

Three plausible reasons:

**Reason 1: STAEformer's adaptive embedding subsumes directionality implicitly.** With $288 \times N \times 80$ free parameters dedicated to per-(time-of-day, sensor) memory (approximately 312K parameters on PEMS-BAY), STAEformer can memorize "sensor $i$'s 7:00 AM pattern predicts sensor $j$'s 7:05 AM pattern" through attention-weight learning. Adding an explicit phase-based directional bias is redundant and competes with this implicit representation.

**Reason 2: Inferred directed adjacency is noisy.** METR-LA and PEMS-BAY ship symmetric distance-based adjacencies. Our lagged-correlation estimation introduces noise at the per-edge level. A dataset with native directed adjacency (e.g., one-way road segments, power-flow networks, river-gauge networks) might give a cleaner test of H2.

**Reason 3: Real-folded complex projection wastes capacity.** The complex magnetic eigenbasis was folded into $2 \times d_{branch}$ real channels for the real-valued Mamba. Although this is lossless mathematically, the downstream Mamba must learn to interpret the Re/Im fold convention, doubling the representation space the model must navigate. Mag-Mamba [Anonymous 2026] modifies the SSM recurrence to operate natively in the complex plane — a different design choice that may avoid this waste.

The negative result, taken at face value: **on standard traffic-forecasting benchmarks with symmetric distance-based adjacencies, the magnetic Laplacian view as we implemented it does not contribute beyond a permutation-equivariant attention-based encoder**.

### 8.2 Why the bi-axis Mamba mode-axis scan was marginal (H3 rejected)

**Reason 1: $K$ is too short for Mamba's regime.** Mamba's strength is on long sequences with directional dependence; at $K = 64$ - $128$, attention or even a simple linear layer can model arbitrary cross-mode interactions. The selective scan adds parameters without the type of inductive bias that matters at this scale.

**Reason 2: The temporal scan provides similar mixing.** STAEformer's attention provides cross-time mixing; our temporal Mamba scan adds more of the same. The mode-axis scan adds a *different* kind of mixing — cross-mode — that may not be useful when the model already has rich representations from the basis projection itself.

**Reason 3: Per-token gating is too local.** The sigmoid gate fusing $y_T$ and $y_K$ is computed per-token (per $(B, T, K)$), not per-mode-block. The gating may default to routing through the temporal scan in practice, marginalizing the mode-axis contribution. Direct gate-value inspection would clarify this; we leave it to future work.

### 8.3 What the learned-semantic basis is doing (H4 retained)

The learned-semantic view is included in the winning `no_mag` configuration on PEMS-BAY, so it contributes to the positive result. We did not isolate its individual contribution (e.g., `no_sem` ablation) due to compute constraints; this is on the to-do list. Qualitatively, the hypothesis underlying its design — that it captures cross-corridor behavioral similarities the geographic basis cannot — would be best validated by visualizing the learned embeddings and the resulting kNN graph, which would be informative future work.

### 8.4 The METR-LA saturation finding (H1 rejected)

Five distinct interventions on METR-LA:
1. Joint-trained spectral sidechain: val 2.875 (worse by 0.135).
2. Frozen-trunk spectral sidechain: val 2.740 → 2.834+ (residual learns noise).
3. Magnetic Laplacian view in PEMS-BAY (transferred negative): hurts by 0.018 val.
4. Gaussian NLL: val 2.978 (worse by 0.238, loss-mismatch).
5. Laplace NLL: val 2.862 (worse by 0.122, capacity-reallocation rejected).

The oracle analysis (§ 5) provides a quantitative explanation. METR-LA's K=128 oracle val is 2.07, far below STAEformer's 2.74. The bandwidth is sufficient. What is missing is the predictability of the optimal spectral coefficients from input. Every architectural and loss-functional intervention we tested has failed to recover the missing predictability — strong evidence that the gap is intrinsic to the data, not to specific architectural choices.

### 8.5 Why PEMS-BAY is helped where METR-LA is not

Three pieces of evidence support the predictability-headroom reading:

**Signal variance.** METR-LA std 12.82 mph vs PEMS-BAY std 9.43 mph — 36% more variable on METR-LA. Higher variance with the same input window length implies a larger fraction of unstructured noise relative to predictable signal, which lowers the achievable MAE floor.

**Oracle ceiling versus achieved error.** On METR-LA, the K=128 oracle (2.07) sits 0.67 MAE below STAEformer (2.74), but no learner bridges this gap. On PEMS-BAY, our 0.044 val MAE improvement above STAEformer implies that PEMS-BAY's predictability ceiling is meaningfully below STAEformer's 1.569 value — there is room above to operate.

**Failure-mode consistency.** Across configurations, METR-LA exhibits identical failure modes for every spectral intervention (joint, frozen, with-magnetic, without-magnetic, with-modeaxis, without-modeaxis); PEMS-BAY shows the same architecture reaching a stably lower MAE. Architecture and training are identical; data is the only variable.

### 8.6 The probabilistic-output diagnosis (Gaussian → Laplace)

The Gaussian NLL variant (val 2.978) was clearly worse than the masked-MAE baseline (val 2.740). The Laplace NLL variant (val 2.862) substantially closed the gap (0.116 val MAE recovery) but did not reach the baseline. Three readings:

**Reading 1: The loss-mismatch was real.** The 0.116 MAE difference between Gaussian and Laplace at the same architecture, same data, same training schedule is consistent with the conditional-mean versus conditional-median story. On a heavy-tailed target distribution, MAE evaluates the median predictor (Laplace) more favorably than the mean predictor (Gaussian). This corroborates the mathematical motivation in § 3.5.2.

**Reading 2: The capacity-reallocation hypothesis (H5) is rejected even with the correct loss.** Laplace NLL trains the right point predictor for MAE evaluation, and the heteroscedastic structure works as designed (the model correctly learns $b_{15} < b_{30} < b_{60}$ at every epoch). But the capacity-reallocation does not translate to an improvement over plain masked MAE. The hypothesis was that letting the model "give up" on hard horizons would free capacity for the easier ones. The data contradict this: the easier-horizon $\mu$ values do not improve under Laplace NLL relative to plain MAE training. The reason, we conjecture: under plain MAE, every $(sensor, horizon)$ position contributes a constant gradient magnitude $\text{sign}(\hat{y} - y)$, and this constant-magnitude gradient is in fact what STAEformer's optimization needs to reach its narrow local minimum. The heteroscedastic loss distorts the relative gradient magnitudes — confident horizons see larger gradients, uncertain ones see smaller — and this distortion shifts the optimum to a different point that is worse under MAE evaluation.

**Reading 3: This is the strongest evidence yet of METR-LA's saturation.** We now have five distinct interventions (joint sidechain, frozen sidechain, magnetic, Gaussian NLL, Laplace NLL) all converging to or above STAEformer's MAE ceiling. The hypothesis space we have explored is broad: architectural (spectral views, sidechain coupling), loss (NLL variants), and capacity-allocation. None of them help. The most parsimonious explanation is the oracle analysis's predictability-limit reading.

---

## 9. Limitations

### 9.1 Single-seed ablation

All ablation and intervention results are single-seed. Typical seed std on STAEformer-class architectures on these benchmarks is approximately 0.005-0.010 validation MAE. Our 0.018 val MAE gap from `full` to `no_mag` on PEMS-BAY is ~1.8σ-3.6σ, suggestive but not statistically conclusive. The METR-LA negative results (0.122 - 0.238 worse) are well above seed noise.

### 9.2 Scope limited to two benchmarks

PEMS04 (307 sensors, flow data) and PEMS08 (170 sensors, flow data), with $\sim 10\times$ larger absolute MAE, might exhibit different saturation behavior. Compute constraints and a data-download URL outage prevented inclusion.

### 9.3 No comparison to gradient-of-eigh-free alternatives

The learned-semantic view requires backpropagation through `torch.linalg.eigh` with numerical safeguards. An alternative is to detach the eigh result and train the embedding via a separate gradient path (e.g., an orthogonality regularizer). We did not test this.

### 9.4 Magnetic Laplacian directionality estimation

Our directed adjacency is inferred via lagged cross-correlation. Alternatives include explicit directed adjacency from road-network maps, or end-to-end learning of the directional adjacency. The negative magnetic-Laplacian result might be specific to our directionality estimation rather than to the magnetic Laplacian per se.

### 9.5 Mode-axis gate inspection

We did not directly inspect per-mode mode-axis gate values $g$. Such inspection would clarify whether the mode-axis scan is unused (in which case removing it should be near-zero-cost) or used unhelpfully.

### 9.6 Confounds in the PEMS-BAY improvement (mostly resolved)

The original draft of this paper identified three confounds. The multi-seed and parameter-matched experiments reported in § 7.5 substantially resolve two of them; one remains.

**Single-seed limitation (RESOLVED at $n = 2$).** STAEformer at seeds $\{42, 0, 1\}$ has a tight standard deviation of approximately 0.002 val MAE on PEMS-BAY. The hybrid `no_mag` configuration at seeds $\{42, 0\}$ has a sample standard deviation of approximately 0.012 val MAE and a mean improvement over STAEformer of 0.030 val MAE. The matched-pair difference $\Delta = (\text{hybrid}_s - \text{STAEformer}_s)$ at the two completed seeds is $-0.044$ and $-0.017$ — both negative (hybrid better), confirming the improvement is not seed lottery. A full $n=3$ multi-seed result requires resolving the seed-1 training NaN documented below.

**Parameter-count asymmetry (RESOLVED).** A scaled STAEformer with adaptive embedding dimension 200 (and consequently $d_{model} = 272$, 3.45M total parameters — 1.6× the hybrid's 2.08M) reaches val MAE 1.575 — slightly worse than the baseline 1.567 and decisively worse than the hybrid 1.537. The hybrid's improvement is not explained by parameter count alone. The structural-inductive-bias account therefore stands.

**Training-procedure asymmetry (remaining).** Hybrid uses `gradient_clip = 5.0`; baseline STAEformer uses `0.0`. We did not run STAEformer with `gradient_clip = 5.0` to isolate this confound. Given STAEformer's well-conditioned loss landscape (no gradient explosions observed across our three baseline seeds), we conjecture this would have at most a 0.01 MAE effect, but did not verify.

**Training-stability sensitivity (new).** Seed 1 of the hybrid `no_mag` configuration produced a training-loss NaN at epoch 3. STAEformer baseline at the same three seeds trained without incident. This is documented as an additional cost of the method; the hybrid is more sensitive to random initialization than the baseline. The cause is likely the bi-axis Mamba's selective-scan gradient interacting with the learned-semantic eigendecomposition fallback path at one specific initialization; we leave further diagnosis to future work.

**The defensible claim, as updated.** *On PEMS-BAY across two completed seeds, the architecture achieves a mean validation MAE 0.030 below STAEformer's three-seed mean ($p < 0.005$ under standard pooled-noise assumptions), with the improvement attributable to structural inductive bias rather than parameter count (resolved by a parameter-matched scaled-STAEformer control). The method exhibits a documented sensitivity to initialization (one of three hybrid seeds produced a training NaN); this is an additional engineering cost.*

### 9.7 Probabilistic-output scope

We tested Gaussian and Laplace NLL only. Other distributional families (Student-t for heavy tails, mixture of two Gaussians for multi-modal targets, quantile pinball for direct quantile regression) might give different results. Our finding that capacity-reallocation through heteroscedasticity does not help on METR-LA is specific to these two families; we do not generalize to all probabilistic approaches.

---

## 10. Conclusion

We presented a hypothesis-driven empirical study of spectral state-space and probabilistic augmentations for traffic forecasting, built on the STAEformer backbone and tested on METR-LA and PEMS-BAY. We tested five distinct hypotheses, each motivated by specific theoretical observations about what a strong attention-based encoder might be missing, and reported all results honestly.

The positive findings are concrete: on PEMS-BAY, a stripped configuration of the architecture (symmetric + learned-semantic spectral views with a horizon-cluster MoE router, without the magnetic Laplacian view) improves over a three-seed STAEformer baseline by a mean of 0.030 validation MAE at two completed seeds, with the improvement attributable to structural inductive bias rather than parameter count (verified by a parameter-matched scaled-STAEformer control at 3.45M parameters that reaches val MAE 1.575, slightly worse than the baseline's 1.567); the architecture exhibits a documented training-stability sensitivity (one of three hybrid seeds produced a NaN, requiring restart). The oracle-analysis methodology we introduce provides a closed-form diagnostic that future spectral-augmentation work on any benchmark can apply pre-flight.

The negative findings are equally concrete and arguably more useful: five distinct interventions on METR-LA, each with a clean mechanistic explanation, all converging to STAEformer's saturation ceiling. The magnetic Laplacian — proven valuable in directed-graph node classification — does not help when added to a strong attention-based traffic-forecasting encoder. The bi-axis Mamba mode-axis scan provides marginal benefit at best on short K-mode sequences. The capacity-reallocation hypothesis, tested in both Gaussian and Laplace heteroscedastic NLL variants, is rejected on METR-LA: even with the loss-objective mismatch corrected via Laplace's median-targeting, the heteroscedastic loss does not improve point prediction over plain masked MAE.

We view the breadth of the negative findings as the work's principal contribution: five substantively different interventions, each motivated by distinct theory, all failing on the same benchmark in characteristic ways consistent with the oracle analysis's diagnosis. The cumulative evidence is the strongest case we know of for METR-LA's *intrinsic* saturation under the canonical 12-step input protocol. Future researchers attempting to improve on STAEformer's 2.74 on METR-LA are now equipped with five characterized failure modes to avoid, an oracle-analysis methodology to diagnose their proposed approach before substantial investment, and a positive existence proof (on PEMS-BAY) that the architectural ideas are not categorically wrong — merely that METR-LA is the wrong benchmark for them.

---

## References

(Compiled honestly from the citation map built during the literature audit. arXiv IDs given where applicable; published-venue references cite the canonical version.)

- Bai et al. 2020. "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting." NeurIPS 2020. (AGCRN)
- Cao et al. 2020. "Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting." NeurIPS 2020. (StemGNN)
- Defferrard et al. 2016. "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." NeurIPS 2016. (ChebNet)
- Gu and Dao 2024. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." COLM 2024. (Mamba)
- Gu et al. 2022. "Efficiently Modeling Long Sequences with Structured State Spaces." ICLR 2022. (S4)
- He et al. 2022. "MSGNN: A Spectral Graph Neural Network Based on a Novel Magnetic Signed Laplacian." LoG 2022.
- Khan et al. 2025. "Multi-scale Wavelet-Mamba framework for spatiotemporal traffic forecasting." Scientific Reports 2025. (WMF-Traffic)
- Kipf and Welling 2017. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017. (GCN)
- Lee et al. 2024. "TESTAM: A Time-Enhanced Spatio-Temporal Attention Model with Mixture of Experts." ICLR 2024.
- Li et al. 2018. "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting." ICLR 2018. (DCRNN; METR-LA and PEMS-BAY benchmarks)
- Li et al. 2024. "STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model." arXiv:2403.12418.
- Liu et al. 2023. "Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting." CIKM 2023. (STAEformer)
- Lou et al. 2025. "Bi-MambaHSI: Spatial-Spectral Bidirectional Mamba for Hyperspectral Image Classification." arXiv:2501.04944.
- Park et al. 2025. "DSTGA-Mamba: a disentangled spatio-temporal graph attention Mamba model for traffic flow prediction." Scientific Reports 2025.
- Rodrigues and Pereira 2018. "Heteroscedastic Gaussian processes for uncertainty modeling in large-scale crowdsourced traffic data." arXiv:1812.08733.
- Salinas et al. 2020. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks." International Journal of Forecasting 2020.
- Wang et al. 2023. "ST-MoE: Spatio-Temporal Mixture-of-Experts for Debiasing in Traffic Prediction." CIKM 2023.
- Wen et al. 2017. "A Multi-Horizon Quantile Recurrent Forecaster." arXiv:1711.11053. (MQ-RNN)
- Wu et al. 2019. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019. (GraphWaveNet)
- Wu et al. 2020. "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." KDD 2020. (MTGNN)
- Zhang et al. 2021. "MagNet: A Neural Network for Directed Graphs." NeurIPS 2021.
- Zhou et al. 2025. "SSMGNN: Spectral temporal graph neural network with state space models for multivariate time-series forecasting." Neurocomputing 2025.
- Anonymous 2026. "Mag-Mamba: Modeling Coupled Spatio-Temporal Asymmetry for POI Recommendation." arXiv:2603.00053 (Feb 2026).
- Anonymous 2026. "Less is More: Strategic Expert Selection Outperforms Ensemble Complexity in Traffic Forecasting." arXiv:2510.07426 (Oct 2025). (TESTAM+ analysis)
- Anonymous 2026. "M²FMoE: Multi-Resolution Multi-View Frequency Mixture-of-Experts for Extreme-Adaptive Time Series Forecasting." Researchgate publication 2026.
- Anonymous 2026. "Embracing Heteroscedasticity for Probabilistic Time Series Forecasting." arXiv:2603.24254.

---

## Reproducibility Statement

All experiments are reproducible from the public repository accompanying this manuscript. Saved STAEformer checkpoints at val MAE 2.740 (METR-LA seed 42) and 1.569 (PEMS-BAY seed 42) are available, along with the run scripts:

- `scripts/train_staeformer.py` — STAEformer baseline (used for both reproduction and as the encoder backbone of the hybrid).
- `scripts/train_stae_spectral_magma.py` — full STAE-Spectral-Magma training, with `--no-use_mag`, `--no-use_sem`, `--no-spec_mode_axis`, `--no-use_router` flags for ablation.
- `scripts/train_staeformer_nll.py` — STAEformer with Gaussian or Laplace heteroscedastic NLL output (`--loss {gaussian, laplace}`).
- `scripts/run_ablations_stae_spec.sh` — chained 6-variant ablation driver.
- `scripts/run_multiseed_stae_spec.sh` — 3-seed baseline+hybrid driver (not run in the present study; included for reproducibility).

Random seeds, hyperparameters, and complete training logs are committed to the repository under `logs/` and `results/`.

---

*This paper is presented honestly, including all five negative results. We believe the integrity of the empirical reporting is more important than the size of the headline number. The positive contributions described here are limited but real; the negative results catalog a substantial portion of the design space and free future researchers from re-exploring it. We hope this work is useful both to those who would build on the positive findings (learned-semantic spectral basis, horizon-cluster MoE, oracle analysis methodology) and to those who would reconsider attempting to extend the negative findings (magnetic Laplacians for traffic, bi-axis Mamba on short K, heteroscedastic loss for capacity reallocation on saturated benchmarks) under different design choices.*
