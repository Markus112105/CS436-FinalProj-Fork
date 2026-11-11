# CS 436 Final Project

**Question:** Does adding momentum to SGD change VGG11-BN first-layer energy profiles, CIFAR-10 accuracy, and training time?

## Experiment

1. Train VGG11-BN with SGD (momentum 0) and SGD with momentum 0.9 for 30 epochs across three matched seeds: six runs total.
2. Compare first-layer energy profiles, training-loss curves, test accuracy, and training time.
3. Present the findings in a four-page IEEE report.

## Results

We trained all six runs in one Google Colab session on a Tesla T4 GPU. The measurements are in `results/` and the figure is [docs/energy_and_loss.pdf](docs/energy_and_loss.pdf).

## Requirements

Python 3.14, `uv`, and VS Code with Python/Jupyter notebook support. The project uses PyTorch, torchvision, NumPy, Matplotlib, tqdm, ipykernel, and nbclient; the setup command pins the tested package versions.

## Run locally

From the repository root, set up Python 3.14:

```sh
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
  torch==2.14.1 torchvision==0.29.1 numpy==2.5.3 \
  matplotlib==3.11.2 tqdm==4.70.1 ipykernel==7.3.0 nbclient==0.11.0
```

Run the tests:

```sh
.venv/bin/python -m unittest discover -s CNNs-First-Layer -p test_utils.py
```

## Documentation

The [report skeleton](docs/final_project_report.tex), [skeleton PDF](docs/final_project_report.pdf), [project requirements](docs/report_requirements.pdf), and [original paper](docs/original_paper.pdf) are in `docs/`. The skeleton has empty sections; the completed report must be four pages including references.

Build the report from `docs/`:

```sh
pdflatex -interaction=nonstopmode -halt-on-error final_project_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_project_report.tex
```

## Attribution and contributions

Based on [Chowers and Weiss's code](https://github.com/RheaChowers/CNNs-First-Layer) for their [ICML 2023 paper](https://proceedings.mlr.press/v202/chowers23a.html).

We initially forgot to fork and started in a separate repository; this fork preserves the original repository history and carries over our project work.

**Markus Massina (GitHub: Markus112105):** project planning, baseline fixes and tests, local setup, and report skeleton.

**Michael Truong (GitHub: Michael-Truong111):** planned contributions: optimizer experiment implementation, results analysis, and report revisions. Both partners must contribute their own work and commits.
