# Continual-learning rules shape representational drift

Reproduce the paper figures: train CNN/RNN continual-learning models, measure
representational drift on a fixed probe set, and aggregate across seeds.

## Setup

```bash
conda create -n drift python=3.11
conda activate drift
pip install -r requirements.txt
```

A CUDA GPU is required for training. Plots prefer Liberation Sans and fall back
to DejaVu Sans if it is missing (optional: `fonts-liberation` on Debian/Ubuntu).

Hyperparameters and the paper seed list live in [`paper_config.json`](paper_config.json).
The reader entry point is [`run_paper.py`](run_paper.py).

## Data (CNN only, Figures 2–4)

Datasets are not bundled. RNN experiments need no external download.

Place these ILSVRC2012 files in `cnn/data/` (ImageNet license required):

| File | Role |
|------|------|
| `ILSVRC2012_img_train.tar` | Training images |
| `ILSVRC2012_img_val.tar` | Official validation images |
| `ILSVRC2012_devkit_t12.tar.gz` | Labels / metadata |

Then build the subset used by the code (one-time; tens of GB of free disk):

```bash
python cnn/tools/process_imagenet.py \
  --raw_root cnn/data \
  --out_root cnn/data/imagenet1k-100-processed \
  --n_classes 100
```

This writes `train/`, `val/`, and `test/` under the out root. Classes are the
first 100 ImageNet-1k `wnid`s in lexicographic order (see Methods). Keep this
out-root path: it is the ImageNet loader default.

## Single-seed test

Full paper defaults use **10 seeds** and run sequentially on one process.
Start with one seed:

```bash
python run_paper.py cnn --seeds 0
python run_paper.py rnn --seeds 0
```

Re-run with `--force` only if you need to overwrite existing outputs.

## Reproduce the paper figures

```bash
python run_paper.py cnn          # Figs. 2–3
python run_paper.py cnn_anchor   # Fig. 4
python run_paper.py rnn          # Fig. 5 (+ appendix temporal panels)
```

Wall-clock is large on a single GPU (especially `cnn` and `cnn_anchor`). Prefer
the single-seed commands above first; on a cluster, run one experiment line per job
and allocate GPUs accordingly.

### Figure → output map

| Paper figure | Command | Primary outputs |
|--------------|---------|-----------------|
| Fig. 2 (accuracy / similarity / gap) | `cnn` | `cnn/experiments/paper_cnn_aggregate_report/<method>/accuracy_matrix.pdf`, `similarity_matrix_layer*.pdf`, `gap_drift_sample_pv.pdf` |
| Fig. 3 (UMAP) | `cnn` | per-run `cnn/experiments/paper_cnn_<method>_seed*/drift_analysis/sample_umap/` |
| Fig. 4 (anchoring) | `cnn_anchor` | `cnn/experiments/paper_anchor_report/task1_fwd_acc_vs_lambda.pdf`, `fwd_acc_vs_final_drift.pdf` |
| Fig. 5 (RNN grid) | `rnn` | `rnn/experiments/paper_rnn_aggregate_report/<method>/accuracy_matrix.pdf`, `pearson_matrix_fdgo.pdf`, `vector_drift_fdgo.pdf` |
| Appendix (temporal) | `rnn` | `.../temporal_similarity/cross_checkpoint_pearson_fdgo_fix1.pdf`, `cross_checkpoint_pearson_fdgo_stim1_go1.pdf` |

`<method>` is one of `normal`, `ewc`, `lwf`, `replay` (CNN) or `normal`, `replay` (RNN).

## Layout

```text
run_paper.py         Reader entry point
paper_config.json    Paper hyperparameters and seeds
cnn/                 CNN training, analysis, aggregation, ImageNet tools
rnn/                 RNN training, analysis, aggregation
```
