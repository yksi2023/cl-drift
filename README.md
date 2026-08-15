# Continual-learning rules shape representational drift

Code to reproduce the paper figures. It trains convolutional and recurrent
networks under continual-learning rules, measures how a fixed probe
representation changes across tasks, and writes paper panels into `figures/`.

The entry point is [`run_paper.py`](run_paper.py). Paper settings live in
[`paper_config.json`](paper_config.json). Figure 1 is a schematic and is not
generated here.

## Setup

```bash
conda create -n drift python=3.11
conda activate drift
pip install -r requirements.txt
```

A CUDA GPU is required for training. Plots use Liberation Sans when available
and fall back to DejaVu Sans (optional: `fonts-liberation` on Debian/Ubuntu).

## Data (CNN only)

RNN experiments generate their tasks in code. CNN experiments (Figs. 2–4 and S1)
need ImageNet. Datasets are not bundled.

Place these ILSVRC2012 files in `cnn/data/` (ImageNet license required):

| File | Role |
|------|------|
| `ILSVRC2012_img_train.tar` | Training images |
| `ILSVRC2012_img_val.tar` | Official validation images |
| `ILSVRC2012_devkit_t12.tar.gz` | Labels / metadata |

Build the 100-class subset used by the code (one-time; tens of GB of free disk):

```bash
python cnn/tools/process_imagenet.py \
  --raw_root cnn/data \
  --out_root cnn/data/imagenet1k-100-processed \
  --n_classes 100
```

This writes `train/`, `val/`, and `test/` under the out root. Classes are the
first 100 ImageNet-1k `wnid`s in lexicographic order. Keep this path: it is the
ImageNet loader default.

## Pipeline test

`--smoke` checks the install, data path, and figure export on a short run
(one seed; a reduced anchor-λ grid). It does not reproduce the paper figures.

```bash
python run_paper.py rnn --smoke          # no ImageNet; fastest
python run_paper.py cnn --smoke
python run_paper.py cnn_anchor --smoke
```

## Reproduce the paper figures

```bash
python run_paper.py cnn          # Figs. 2, 3, S1
python run_paper.py cnn_anchor   # Fig. 4
python run_paper.py rnn          # Figs. 5, S2
```

Defaults match the paper (10 seeds). CNN methods are naive sequential training,
EWC, LwF, and replay; the RNN comparison is naive training vs replay. Jobs run
sequentially on one process. CNN ImageNet training is slow on a single GPU; on a
cluster, run one command per job.

| Paper figure | Command | Output |
|--------------|---------|--------|
| Fig. 2 CNN method grid | `cnn` | `figures/figure2/` |
| Fig. 3 UMAP | `cnn` | `figures/figure3/` |
| Fig. 4 anchoring | `cnn_anchor` | `figures/figure4/` |
| Fig. 5 RNN method grid | `rnn` | `figures/figure5/` |
| Fig. S1 sample similarity / CKA | `cnn` | `figures/figure_s1/` |
| Fig. S2 RNN temporal structure | `rnn` | `figures/figure_s2/` |

## Layout

```text
run_paper.py         Reader entry point
paper_config.json    Paper hyperparameters
figures/             Paper panels (created at runtime)
cnn/                 CNN training, analysis, ImageNet tools
rnn/                 RNN training, analysis
```
