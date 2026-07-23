# A Generative Denoiser for `denoisers`: Research Survey & Design Proposal

**Recommendation up front:** add a **waveform-domain Schrödinger-bridge / data-to-data flow-matching denoiser** (`FlowUNet1D`) that extends the existing `UNet1D` with a time embedding, trains with a **data-prediction (x₀) objective reusing the repo's existing L1 + multi-resolution STFT losses**, and samples in **1–8 steps starting from the noisy waveform**. At 1 step it degenerates to (a regularized version of) the repo's current predictive denoiser — so the worst case is parity, and the upside is multi-step generative refinement, better perceptual quality, and better out-of-distribution robustness. No GAN discriminator, no gaussian-noise prior, no vocoder.

______________________________________________________________________

## 1. Why generative at all?

The repo's `UNet1D`/`WaveUNet` are *predictive* (discriminative) denoisers: they regress the single MMSE-ish estimate of clean speech. Predictive models win on sample-aligned metrics (SI-SDR) but:

- they over-smooth and leave "dull"/muffled residuals at low SNR,
- they degrade *below the unprocessed input* on signal quality when conditions are out-of-distribution (shown repeatedly, e.g. SGMSE+ TASLP 2023 mismatched-condition tables; CDiffuSE keeps PESQ 1.66 on mismatched CHiME-4 while DEMUCS collapses 2.65→1.38),
- they cannot *reconstruct* content that the noise destroyed — only attenuate.

Generative denoisers model the full posterior p(clean | noisy) and consistently win on perceptual/non-intrusive metrics (DNSMOS, NISQA, MOS) and OOD robustness, at the cost of some SI-SDR and a hallucination risk. The field's consensus recipe (2024–2026) is to *combine* the two: predictive anchor + generative refinement.

## 2. Survey: the four families

### 2.1 Score-based diffusion (SGMSE lineage)

| System                                                     | Domain             | Steps (NFE)   | VB-DMD PESQ / SI-SDR                    | Notes                                                                                              |
| ---------------------------------------------------------- | ------------------ | ------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------- |
| SGMSE (Interspeech 2022, arXiv:2203.17004)                 | complex STFT       | 50 PC         | 2.28 / 16.2                             | OUVE SDE: OU drift toward noisy y + VE noise                                                       |
| SGMSE+ (TASLP 2023, arXiv:2208.05830)                      | complex STFT       | 60 (RTF 1.77) | 2.93 / 17.3                             | NCSN++ 65M; best generalization of its era                                                         |
| CDiffuSE (ICASSP 2022, arXiv:2202.05256)                   | **waveform**       | 50–200        | 2.52 / 12.4                             | gaussian-endpoint DDPM; **no-op at 48 kHz** (EARS: POLQA 1.81 vs noisy 1.71, 42 s/audio-s)         |
| StoRM (TASLP 2023, arXiv:2212.11851)                       | complex STFT       | 10–20         | 2.93 / 18.8                             | predictive stage → diffusion "regeneration"; kills vocalizing/breathing artifacts, 10× fewer steps |
| UNIVERSE / UNIVERSE++ (arXiv:2206.03065, 2406.12194)       | waveform 16/24 kHz | 4–8           | ++: 2.93 / – (WER 2.9% w/ phoneme LoRA) | rich conditioner → few steps; hallucination fixed with CTC loss                                    |
| EARS 48 kHz benchmark (Interspeech 2024, arXiv:2406.06185) | —                  | —             | —                                       | SGMSE+ retuned (1534-pt STFT) is best 48 kHz system, but RTF ≈ 2.6                                 |

Documented failure modes (survey: Lemercier et al., IEEE SPM 2024, arXiv:2402.09821): vocalizing/breathing artifacts, speech inpainted into noise-only regions, phonetic confusions at negative SNR, systematically lower SI-SDR than predictive baselines, and step-count collapse below ~10 NFE without fixes.

Two important negative results from the design-space study (Gonzalez et al., TASLP 2024, arXiv:2312.04370): SGMSE+'s quality is **not** attributable to its clean→noisy OU drift, and gaussian prior mismatch is not per-se harmful — what matters is preconditioning, loss weighting, and where sampling starts.

### 2.2 Flow matching / Schrödinger bridges — the current frontier

The decisive 2024–2026 development: replace "gaussian noise → clean, conditioned on noisy" with a **data-to-data bridge whose endpoints are (clean, noisy)**. Reverse sampling starts *at the actual noisy signal*.

| System                                                            | Domain       | NFE   | VB-DMD PESQ / SI-SDR @ NFE=1 | Notes                                                                                                                                                                                                   |
| ----------------------------------------------------------------- | ------------ | ----- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SB for SE (NVIDIA, Interspeech 2024, arXiv:2407.16074)            | complex STFT | 3–50  | —                            | beats SGMSE+ by +0.5 PESQ / +3 dB SI-SDR / ~half WER on WSJ0-CHiME3; data-prediction loss + time-domain L1                                                                                              |
| SB-PESQ (ICASSP 2025, arXiv:2409.10753)                           | complex STFT | 1–16  | 3.45 / ~14                   | nearly flat from NFE 1→16 — DP-trained bridges are quasi-one-step                                                                                                                                       |
| SBCTM (Sony, arXiv:2507.11925)                                    | complex STFT | **1** | **3.56** / 12.2              | consistency-trajectory distillation; RTF 0.045; dropped GAN aux loss for stability                                                                                                                      |
| SB-RF (Xiaomi, arXiv:2606.05575)                                  | complex STFT | **1** | 3.39 / **19.5**              | from scratch, no distillation; best 1-step fidelity balance                                                                                                                                             |
| ROSE-CD (WASPAA 2025, arXiv:2507.05688)                           | complex STFT | **1** | 3.49 / 17.8                  | consistency distillation, 54× faster than teacher                                                                                                                                                       |
| MeanFlowSE (Xiamen, arXiv:2509.14858)                             | complex STFT | **1** | 2.94 / 19.98                 | MeanFlow identity, no distillation; needs curriculum                                                                                                                                                    |
| MeanFlowSE (NWPU, arXiv:2509.23299)                               | VAE latent   | **1** | DNS OVRL 3.368, WER 8.5%     | RTF 0.013, 40.7M params; WavLM conditioning critical                                                                                                                                                    |
| ARFSE (arXiv:2606.20001)                                          | complex STFT | 1–5   | 3.00 / 19.9                  | time-embedding-free rectified flow, RTF 0.02                                                                                                                                                            |
| "Rethinking Flow & Diffusion Bridges" (AAAI-26, arXiv:2602.18355) | —            | 1–5   | —                            | **unifying analysis**: with a data-prediction loss, a bridge model is an *augmented predictive model*; 1-step sampling ≡ prediction; TF-GridNet-backbone bridge at 2.2M params matches/beats NCSN++ 65M |

Key facts, corroborated across ≥6 independent groups:

1. **Data-to-data beats gaussian-endpoint for paired enhancement** — on fidelity metrics and, critically, on low-NFE robustness. Gaussian-prior score models collapse at 1 step (PESQ ≈ 1.0–1.8); bridges degrade gracefully or not at all.
1. **1-step inference is solved** (four independent routes: DP-bridges, MeanFlow, consistency-trajectory distillation, straightened rectified flow).
1. **The "Rethinking" ceiling**: in-domain and with DP loss, the bridge ≈ a noise-curriculum-regularized predictive model. The *genuine* generative payoff is OOD robustness and perceptual naturalness — exactly the axes where predictive denoisers hurt.
1. Training is stable (plain regression on x₀ targets over a noise curriculum) — no discriminators, no score-matching schedule pathologies. Aux PESQ-style losses Goodhart (SI-SDR collapse) unless balanced.

### 2.3 GANs

| System                                                                                    | Domain      | SR     | Notes                                                                                                                                                                                                                                                                                      |
| ----------------------------------------------------------------------------------------- | ----------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| SEGAN (2017, arXiv:1703.09452) → MetricGAN+ (2021) → CMGAN (TASLP 2024, arXiv:2209.11112) | wave → STFT | 16 kHz | metric-discriminator line; strong PESQ (CMGAN 3.41 @ 1.83M) but stuck at 16 kHz                                                                                                                                                                                                            |
| DEMUCS denoiser (Interspeech 2020, arXiv:2006.12847)                                      | waveform    | 16 kHz | **pure regression: L1 + multi-res STFT — the losses this repo already uses**                                                                                                                                                                                                               |
| HiFi-GAN denoise (Su et al. 2020, arXiv:2006.05694), HiFi-GAN-2 (WASPAA 2021)             | waveform    | 48 kHz | adversarial + deep-feature matching; studio quality                                                                                                                                                                                                                                        |
| HiFi++ (ICASSP 2023, arXiv:2203.13086)                                                    | waveform    | 48 kHz | generator later reused by FINALLY                                                                                                                                                                                                                                                          |
| **FINALLY** (NeurIPS 2024, arXiv:2410.05920)                                              | waveform    | 48 kHz | MOS 4.63, RTF 0.03; proves LS-GAN + deterministic generator is **mode-seeking** → hallucinates less than distribution-covering diffusion (PhER 0.14 vs UNIVERSE 0.20); 3-stage training: regression pretrain → adversarial finetune (MS-STFT discriminators, LR warm-up) → 48 kHz upsample |
| DeepFilterGAN (arXiv:2505.23515)                                                          | full-band   | 48 kHz | real-time predictive→GAN regeneration                                                                                                                                                                                                                                                      |

GANs are the strongest *single-pass waveform 48 kHz* systems today, but every report flags adversarial training fragility (replay buffers, LR warm-ups, discriminator-choice sensitivity; SBCTM removed its GAN loss for exactly this reason). The winning GAN recipe is always **adversarial fine-tuning on top of a regression-pretrained model** — i.e., a *stage*, not a family. It composes with any base model, including a bridge.

### 2.4 Latent / token generative restoration (context)

DiTSE (Adobe, arXiv:2504.09381; 48 kHz VAE-latent DiT, studio MOS 4.34, only system to reduce WER below noisy input), AnyEnhance (arXiv:2501.15417; DAC-token masked LM, 44.1 kHz), MaskSR/MaskSR2, Miipher-2 (arXiv:2505.04457), Resemble Enhance (open source: complex-STFT U-Net denoiser stage + latent CFM enhancer stage). These get the best absolute MOS at full band, but require a pretrained codec/VAE + SSL encoder (WavLM/w2v-BERT) and risk speaker drift / content hallucination without semantic conditioning. **Too much external machinery for this repo's scope today** — but note that *every* production-grade system is two-stage (predictive/regression first, generative second).

### 2.5 The waveform-at-48 kHz question

No published system does raw-waveform *score diffusion* SE at 48 kHz — CDiffuSE's fullband failure (gaussian-endpoint, ε-prediction, 200 steps) is the cautionary tale, and full-band systems went STFT (SGMSE+/EARS) or latent (DiTSE). **But the failure ingredients are precisely what a bridge removes:**

- gaussian prior mismatch → bridge starts *at* the noisy waveform;
- ε/score prediction (fine-structure modeling burden in waveform) → **x₀-prediction**, whose target is exactly the clean waveform the repo already regresses;
- 50–200 NFE cost at 48 kHz sample rates → 1–8 NFE.

Meanwhile, single-pass waveform conv-U-Nets at 48 kHz demonstrably work (HiFi-GAN-2, FINALLY, this repo's own `unet1d-vctk-48khz`). A 1–8-step bridge over that same backbone is k forward passes of a conv U-Net — linear cost, no attention over 48 k tokens. The "Rethinking" equivalence (1-step DP-bridge ≡ predictive model) means the design is anchored to a regime *proven* in this repo, and extends it continuously. This is genuinely novel territory at 48 kHz (worth stating in the README/paper-trail), with a clean fallback (§5.6).

## 3. Recommendation

**Family: Schrödinger-bridge-style conditional flow matching with a data-prediction objective ("bridge-FM").** Reasons, in order:

1. **Strict superset of the current repo.** At NFE=1 the sampler evaluates `f(y, y, t=1)` — a predictive denoiser with an extra input channel and time embedding. The ICFM study (arXiv:2508.20584) validated exactly this "direct data prediction" trick at PESQ 3.05 @ 1 step. Nothing is lost relative to `UNet1D`; multi-step refinement and stochastic sampling are gained.
1. **Loss reuse.** The DP objective is `distance(x̂₀, x₀)` averaged over bridge times — the repo's `nn.L1Loss` + `MultiResolutionSTFTLoss` apply verbatim as that distance. No new loss code, no discriminator, no differentiable-PESQ Goodharting.
1. **Training stability.** It is plain regression over a noise curriculum. Every GAN alternative carries documented instability; every gaussian-prior diffusion alternative carries low-NFE collapse and hallucination baggage that bridges empirically reduce (NVIDIA SB: ~half the WER of SGMSE+).
1. **Inference cost fits the repo's deployment story** (ONNX export, CPU-viable): k ∈ {1,…,8} U-Net passes, user-selectable at inference time with one knob — quality/latency dial for free.
1. **Best-supported upgrade paths.** Adversarial fine-tune stage (FINALLY recipe) and consistency-trajectory distillation (SBCTM) both apply *on top of* a trained bridge without changing the architecture.

Why not the alternatives as the *first* generative model:

- **Pure diffusion (SGMSE+-style):** dominated by bridges on every axis relevant here (NFE, prior mismatch, WER); needs 30–60 NFE and correctors; hallucination-prone.
- **GAN-first:** best single-pass quality, but adversarial training fragility is a poor fit for a lean OSS codebase, and it forecloses the multi-step generative capability. Better as a later fine-tuning stage.
- **Latent/codec generative:** highest ceiling, heaviest dependency footprint (codec + SSL encoder + possibly vocoder); contradicts the repo's end-to-end waveform identity.

## 4. Proposed formulation (precise)

Endpoints: `x₀ = clean`, `x₁ = y = noisy` (both waveforms, `(B, 1, T)`).

**Forward (training) marginal** — Brownian-bridge with exact Dirac endpoints:

```
x_t = (1 − t)·x₀ + t·y + σ(t)·ε,   ε ~ N(0, I),   σ(t) = σ_max·√(t(1−t))
```

with `t ~ U(0, 1)` per sample and `σ_max ≈ 0.1` (waveforms live in [−1, 1]; tune on val DNSMOS/SI-SDR — larger σ_max = more generative diversity, smaller = more predictive). This is the SE-Bridge/Thunder-style bridge; the SB-VE schedule (arXiv:2407.16074) is a drop-in alternative behind the same config field.

**Network & objective** — data prediction:

```
x̂₀ = f_θ(concat(x_t, y), t)
L = L1(x̂₀, x₀) + MultiResolutionSTFT(x̂₀, x₀)        # existing repo losses, unchanged
```

Optionally weight by a simple SNR-aware factor later; start unweighted (D3/arXiv:2512.10382 shows plain DP + aux time-domain losses is competitive when preconditioned; our U-Net input is already bounded).

**Sampler** — deterministic bridge resampling (DDIM-like), N steps on a grid `1 = t_N > … > t_0 = 0`:

```
x ← y
for k = N … 1:
    x̂₀ ← f_θ(concat(x, y), t_k)
    x  ← (1 − t_{k−1})·x̂₀ + t_{k−1}·y            # + σ(t_{k−1})·ε  if stochastic=True
return x̂₀ from the last iteration
```

- `N = 1` ⇒ exactly one forward pass, `x̂₀ = f_θ((y, y), 1)` — predictive mode.
- `stochastic=False` default (mode-seeking, fewer hallucinations, cf. FINALLY's argument); `stochastic=True` exposes diversity.
- Later steps dominate the output (Rethinking, arXiv:2602.18355), so small N is principled, not a hack.

## 5. Repo integration design (no code yet — shapes and contracts)

### 5.1 New module: `src/denoisers/modeling/flowunet1d/`

Follows the existing `unet1d/` layout exactly:

- **`config.py` — `FlowUNet1DConfig(PretrainedConfig)`**, `model_type = "flowunet1d"`. Fields = all existing `UNet1DConfig` fields (channels, kernel_size, num_groups, dropout, activation, max_length, sample_rate, norm_type) **plus**:
  - `time_embed_dim: int = 512` (Gaussian-Fourier features → 2-layer MLP)
  - `sigma_max: float = 0.1` (bridge noise scale), `schedule: str = "brownian"` (`"brownian" | "sb_ve"`)
  - `num_inference_steps: int = 4`, `stochastic_sampling: bool = False`
  - `in_channels: int = 2` (x_t ‖ y)
- **`modules.py`**: reuse `DownBlock1D`/`MidBlock1D`/`UpBlock1D` with one addition — a FiLM hook (`scale, shift = MLP(t_emb)`) applied after each block's normalization. The blocks already exist; this is an additive change or thin wrappers.
- **`model.py` — `FlowUNet1DModel(PreTrainedModel)`**:
  - `forward(x_t, y, t) -> x0_hat` (training path, single evaluation),
  - `denoise(y, num_steps=None) -> audio` (inference path implementing §4's sampler; this is what `publish`/ONNX export wraps for N=1).
  - Keep the `Tanh` output head (bounded x̂₀ matches bounded targets) and the input-skip concat trick from `UNet1D.out_conv`.

### 5.2 Lightning integration

New `FlowMatchingLightningModule` (sibling of `DenoisersLightningModule`, same metrics/W&B plumbing):

- `training_step`: sample `t`, `ε` → build `x_t` from `batch.audio`, `batch.noisy` → `x̂₀` → existing L1 + MRSTFT against `batch.audio`. (The `Batch(noisy, audio, lengths)` dataset contract is untouched.)
- `validation_step`: run the full sampler at `num_inference_steps` (and log an `N=1` variant to watch the predictive floor); reuse SNR/SDR/SI-SNR/SI-SDR metrics, DNSMOS, `log_audio_batch`, `plot_image_from_audio` unchanged.
- **EMA on by default** — the module already supports `use_ema`; every bridge/FM paper uses EMA ~0.999.
- Optimizer: existing AdamW 1e-4 + exponential decay is in line with the literature (NVIDIA SB uses Adam 1e-4, EMA 0.999).

### 5.3 Training data & schedule

- Same VCTK + transforms pipeline (the `transforms.py` noise/reverb/EQ augmentations *are* the degradation model). Recommend enabling the fuller augmentation set for this model since generative robustness is the point.
- Start at 24 kHz configs for iteration speed; the architecture is sample-rate-agnostic. Promote to 48 kHz once val curves are healthy.
- Sanity milestone 1: with `σ_max = 0`, `t` fixed at 1, the model must match a freshly trained `UNet1D` baseline (it is one, modulo the extra channel). This de-risks the integration before any generative behavior is tested.

### 5.4 Evaluation protocol

- Keep SI-SDR/SNR but treat them as *fidelity guards*, not targets — generative outputs trade SI-SDR for perceptual quality by design.
- Primary: DNSMOS (already wired) at N ∈ {1, 2, 4, 8}; report the N-vs-quality curve.
- Add a WER probe (any small ASR, e.g. whisper-tiny, eval-only script) — the literature's canonical hallucination detector. Optional but strongly recommended before publishing checkpoints.
- OOD check: train on VCTK-noise recipe, evaluate on a held-out noise family (e.g. disable one augmentation class at train time, test with it on) — this is where the bridge should visibly beat the predictive baseline.

### 5.5 Roadmap (each phase independently shippable)

1. **Phase 1 — bridge-FM core** (this proposal): `FlowUNet1D` + sampler + Lightning module; ship `flowunet1d-vctk-24khz`, then 48 kHz.
1. **Phase 2 — StoRM-style regeneration (optional, cheap):** initialize the bridge from the *existing pretrained* `unet1d-vctk-48khz` output instead of raw `y` (one config flag: endpoints become `(x₀, D(y))`). Reuses published repo assets; documented to remove artifacts and cut steps further.
1. **Phase 3 — adversarial fine-tune (optional):** FINALLY-recipe MS-STFT discriminators on top of the frozen-ish bridge at N=1, for single-pass studio quality. Contained blast radius: a training stage, not an architecture.
1. **Phase 4 — one-step guarantee (optional):** MeanFlow-style average-velocity or SBCTM-style consistency-trajectory distillation if N=1 quality without it proves insufficient.

### 5.6 Risks & fallbacks

| Risk                                                            | Signal                                           | Mitigation                                                                                                                                                                                                                                          |
| --------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Waveform bridge underperforms at 48 kHz (unpublished territory) | 48 kHz val DNSMOS ≤ predictive baseline at all N | Phase-2 regeneration mode (predictive front-end absorbs the fullband burden); or complex-STFT front-end behind the same config (SGMSE+-EARS settings: 1534-pt FFT, hop 384) — the U-Net becomes 2-channel-complex-in/out, everything else unchanged |
| Hallucination at very low SNR                                   | WER probe ↑ vs input                             | deterministic sampling default; smaller σ_max; Phase 2 anchoring                                                                                                                                                                                    |
| SI-SDR regression alarms users                                  | SI-SDR drop > ~2 dB vs `UNet1D`                  | document the tradeoff; N=1 deterministic mode as the "fidelity" preset                                                                                                                                                                              |
| JVP/curriculum complexity creep (MeanFlow route)                | —                                                | not needed in Phase 1; plain DP-bridge is curriculum-free                                                                                                                                                                                           |

## 6. Key references

Bridges/FM: NVIDIA SB (Interspeech 2024, arXiv:2407.16074); SB-PESQ (ICASSP 2025, arXiv:2409.10753); SBCTM (arXiv:2507.11925); SB-RF (arXiv:2606.05575); ROSE-CD (WASPAA 2025, arXiv:2507.05688); MeanFlowSE ×2 (arXiv:2509.14858, arXiv:2509.23299); COSE (arXiv:2509.15952); MeanSE (arXiv:2509.21214); ARFSE (arXiv:2606.20001); ICFM/DDP (arXiv:2508.20584); Rethinking bridges (AAAI-26, arXiv:2602.18355); SE-Bridge (arXiv:2305.13796); Stream.FM (arXiv:2512.19442).
Diffusion: SGMSE (arXiv:2203.17004); SGMSE+ (TASLP 2023, arXiv:2208.05830); CDiffuSE (arXiv:2202.05256); StoRM (arXiv:2212.11851); UNIVERSE/++ (arXiv:2206.03065, 2406.12194); EARS (arXiv:2406.06185); design space (arXiv:2312.04370); SPM survey (arXiv:2402.09821); Diffusion Buffer (arXiv:2506.02908).
GAN: SEGAN (arXiv:1703.09452); MetricGAN+ (arXiv:2104.03538); CMGAN (arXiv:2209.11112); DEMUCS (arXiv:2006.12847); HiFi-GAN denoise (arXiv:2006.05694); HiFi-GAN-2 (WASPAA 2021); HiFi++ (arXiv:2203.13086); FINALLY (NeurIPS 2024, arXiv:2410.05920); DeepFilterGAN (arXiv:2505.23515).
Latent/token: DiTSE (arXiv:2504.09381); AnyEnhance (arXiv:2501.15417); MaskSR (arXiv:2406.02092); Miipher-2 (arXiv:2505.04457); Resemble Enhance (github.com/resemble-ai/resemble-enhance).

*Caveats: 2026-dated arXiv IDs are preprints; several few-step results (SBM, VoiceBridge, HyFlowSE) were verified only at abstract level. Cross-paper VB-DMD numbers are self-reported and comparable to ~±0.05 PESQ. Two unrelated papers are both named "FlowSE"; citations above disambiguate by group.*
