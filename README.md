# BSETD: Bayesian Spectral Emotion Transition Discovery

Anonymous code release accompanying the paper *Bayesian Spectral Emotion Transition Discovery from Multi-Annotator Disagreement*. All identifying information (author names, institutions, repository URLs) has been removed for double-blind review.

## Overview

BSETD is a two-stage framework for discovering emotion-transition structure in multi-rater dialogue corpora.

- **Stage 1** — Hierarchical Dirichlet--Multinomial (DM) posterior estimation with empirical-Bayes (EB) concentration fitting (Minka 2000 fixed point), soft-label outer-product transition counts, and BH-FDR-controlled significance over the 42 off-diagonal cells.
- **Stage 2** — Symmetrized normalized graph Laplacian $\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ with two-sided band-limited reconstruction $\mathbf{A}^{\mathrm{lo}} = \mathbf{P}_{\mathrm{lo}} \mathbf{A} \mathbf{P}_{\mathrm{lo}}$ and $\mathbf{A}^{\mathrm{hi}} = \mathbf{P}_{\mathrm{hi}} \mathbf{A} \mathbf{P}_{\mathrm{hi}}$, where $\mathbf{P}_{\mathrm{lo,hi}} = \mathbf{U}_{\mathrm{lo,hi}} \mathbf{U}_{\mathrm{lo,hi}}^\top$ are the orthogonal projectors onto the low- and high-frequency eigen-subspaces.

## Directory Structure

```
.
├── README.md                       # this file
├── LICENSE                         # MIT (to be released after de-anonymization)
├── requirements.txt                # pinned Python dependencies
├── reproduce_main.sh               # one-shot reproduction script
├── bsetd/                          # implementation
│   ├── __init__.py
│   ├── stage1_dirichlet.py         # Stage 1: EB-EM DM posterior + BH-FDR
│   ├── stage2_spectral.py          # Stage 2: symmetrized Laplacian spectral decomposition
│   ├── synthetic_ablation.py       # 144-run synthetic ground-truth recovery
│   ├── bootstrap_ci.py             # Dialog-level cluster bootstrap (B=1000)
│   ├── bootstrap_ci_cross_corpus.py# Cross-corpus Pearson CIs
│   ├── permutation_robustness.py   # Within-dialog permutation null + 6-slice subpop
│   ├── per_dialog_extension.py     # Per-dialog hierarchical posterior (τ-controlled)
│   ├── directional_spectral.py     # Chung directed-graph Laplacian comparison
│   ├── rater_count_ablation.py     # Rater-count sensitivity (R=2,3,4)
│   ├── downstream_prediction.py    # BSETD-lift + logistic regression downstream
│   ├── emotionic_baseline.py       # EmotionIC-style 2-stream GRU baseline
│   ├── bert_baseline.py            # Frozen DistilBERT next-emotion baseline
│   ├── run_meld_bsetd.py           # MELD cross-corpus driver
│   ├── run_dailydialog_hardlabel.py# DailyDialog one-hot driver
│   ├── run_m3ed_bsetd.py           # M3ED Chinese cross-lingual driver
│   ├── run_friends_vs_emotionpush.py # EmotionLines intra-domain split
│   ├── dailydialog_llm_softlabel.py# GPT-5.4-mini AnnoLLM-style soft labels
│   ├── dailydialog_batch_split.py  # OpenAI Batch API chunking utility
│   └── dailydialog_aggregate_chunks.py # Batch output aggregation
├── data_processed/
│   ├── emotionlines_softlabels_v2_bsetd.parquet  # 29,245 utts × 5-rater real soft labels
│   └── emotionlines_overall_stats.json
├── experiments/
│   ├── stage1_emotionlines/        # Stage 1 outputs (total/inertia/contagion .npz)
│   ├── stage2_emotionlines/        # Stage 2 outputs (.npz + summary .json)
│   └── cross_corpus_bootstrap.json # Cross-corpus Pearson + 95% CIs
└── figures/
    ├── fig_pipeline_bsetd.pdf       # Fig. 1
    └── fig_hero_bsetd.pdf           # Fig. 2 (heatmap + chord + Sankey)
```

## Reproducing the Main Results

The pre-processed EmotionLines parquet and the corresponding Stage 1/2 outputs are bundled. Run:

```bash
bash reproduce_main.sh
```

Or, equivalently, run the stages individually from the repository root:

```bash
# Stage 1 on EmotionLines (Friends + EmotionPush)
python -m bsetd.stage1_dirichlet \
    --input data_processed/emotionlines_softlabels_v2_bsetd.parquet \
    --out experiments/stage1_emotionlines/

# Stage 2 on the Stage 1 posterior
python -m bsetd.stage2_spectral \
    --stage1-npz experiments/stage1_emotionlines/stage1_total.npz \
    --out experiments/stage2_emotionlines/

# Dialog-level cluster bootstrap (B=1000)
python -m bsetd.bootstrap_ci

# Within-dialog permutation null + subpopulation robustness
python -m bsetd.permutation_robustness

# Synthetic ground-truth recovery (144 grid points)
python -m bsetd.synthetic_ablation
```

Cross-corpus and ablation drivers:

```bash
python -m bsetd.run_meld_bsetd                # MELD
python -m bsetd.run_dailydialog_hardlabel     # DailyDialog (one-hot)
python -m bsetd.run_m3ed_bsetd                # M3ED (Chinese)
python -m bsetd.run_friends_vs_emotionpush    # EmotionLines intra-domain
python -m bsetd.rater_count_ablation          # R=2,3,4 sub-sampling
python -m bsetd.per_dialog_extension          # τ-controlled per-dialog posterior
python -m bsetd.directional_spectral          # Chung directed Laplacian
python -m bsetd.downstream_prediction         # BSETD-lift + LogReg
python -m bsetd.emotionic_baseline            # 2-stream GRU baseline
```

For DailyDialog-LLM (GPT-5.4-mini virtual annotators), an `OPENAI_API_KEY` environment variable is required. Soft-label generation is the only step that requires network access; all other experiments run offline on a single CPU core.

## Dependencies

- Python 3.10+
- numpy, scipy, pandas, scikit-learn, pyarrow
- (optional) torch, transformers — for DistilBERT/GRU downstream baselines
- (optional) openai — for DailyDialog-LLM annotation only

See `requirements.txt` for pinned versions.

## Computational Cost

All experiments run end-to-end on a single CPU core:

| Stage | EmotionLines (29,245 utt) | MELD (13,708) | DailyDialog (102,979) |
|---|---|---|---|
| Stage 1 | ~1 min | ~30 s | ~3 min |
| Stage 2 | <1 s | <1 s | <1 s |
| Bootstrap (B=1,000) | ~20 min | ~10 min | ~30 min |
| Synthetic ablation | ~90 min (144 configurations) | -- | -- |

No GPU is required for BSETD itself. GPU is only used by the optional DistilBERT downstream baseline.

## Expected Outputs

Top transition pairs on EmotionLines (Stage 1, ranked by $|\log_2 \mathrm{lift}|$):

| Source → Target | $\bar T_{jk}$ | lift | $\log_2$ lift |
|---|---|---|---|
| disgust → anger | 0.101 | 1.92 | +0.94 |
| joy → anger | 0.028 | 0.54 | −0.90 |
| anger → joy | 0.083 | 0.54 | −0.89 |
| anger → disgust | 0.080 | 1.81 | +0.86 |

Stage 2 inertia / contagion indices (EmotionLines):

| Emotion | Inertia $I_j$ | Contagion $C_j$ |
|---|---|---|
| neutral | 0.378 | 0.025 |
| joy | 0.338 | 0.011 |
| sadness | 0.227 | 0.010 |
| fear | 0.162 | 0.046 |
| anger | 0.206 | 0.028 |
| surprise | 0.177 | 0.031 |
| disgust | 0.123 | 0.052 |

Cross-corpus pairwise Pearson correlations across the 42 off-diagonal cells:

| | EL | MD | DD | M3 | LLM |
|---|---|---|---|---|---|
| EL | --- | 0.910 | 0.944 | 0.854 | 0.976 |
| MD | | --- | 0.916 | 0.814 | 0.952 |
| DD | | | --- | 0.795 | 0.979 |
| M3 | | | | --- | 0.847 |

## Data Availability

- **EmotionLines** (Friends + EmotionPush): provided in `data_processed/` as a soft-label parquet derived from the publicly released vote distributions (Hsu et al., LREC 2018).
- **MELD** (Poria et al., ACL 2019): publicly available; soft labels are reconstructed via Dirichlet smoothing.
- **DailyDialog** (Li et al., IJCNLP 2017): publicly available; used as one-hot soft labels.
- **DailyDialog-LLM**: virtual-annotator labels generated by GPT-5.4-mini under the protocol described in `bsetd/dailydialog_llm_softlabel.py`.
- **M3ED** (Zhao et al., ACL 2022): publicly available Chinese television-drama corpus; data and license retrieved from the original authors.

## License

Anonymous submission. An open-source license (MIT) will be applied with the de-anonymized version.

## Citation

A citation entry will be provided after de-anonymization.
