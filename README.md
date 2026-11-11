# CS 436 Final Project

**Question:** Does adding momentum to SGD change VGG11-BN first-layer energy profiles, CIFAR-10 accuracy, and training time?

## Experiment

1. Train VGG11-BN with SGD (momentum 0) and SGD with momentum 0.9 for 30 epochs across three matched seeds: six runs total.
2. Compare first-layer energy profiles, training-loss curves, test accuracy, and training time.

## Results

Momentum raised mean test accuracy from 54.54% to 78.56% with the same training time

The experiment is in the [notebook](CNNs-First-Layer/optimizer_experiment.ipynb). We trained all six runs in one Google Colab session on a Tesla T4 GPU. The measurements are in `results/` and the figure is [docs/energy_and_loss.pdf](docs/energy_and_loss.pdf).

## Requirements

Python 3.14, `uv`, Jupyter notebook, PyTorch, torchvision, NumPy, Matplotlib, tqdm, ipykernel, and nbclient

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
if u want to redo the analysis without training, open the notebook and run sections 1, 2, 5, and 6

## Documentation

The [report skeleton](docs/final_project_report.tex), [skeleton PDF](docs/final_project_report.pdf), [project requirements](docs/report_requirements.pdf), and [original paper](docs/original_paper.pdf) are in `docs/`. The skeleton has empty sections; the completed report must be four pages including references.

## Attribution and contributions

Based on [Chowers and Weiss's code](https://github.com/RheaChowers/CNNs-First-Layer) for their [ICML 2023 paper](https://proceedings.mlr.press/v202/chowers23a.html).

We added a momentum option, training-time measurement, and final-only test evaluation to `train_model` in `utils.py`, and wrote the experiment notebook. the additions in the original files are marked with comments that clearly show what we added and where.

**Markus Massina (GitHub: Markus112105):** project planning, tests, local setup, notebook sections 1, 2, and 5, and the Colab runs.

**Michael Truong (GitHub: Michael-Truong111):** project planning, Colab results analysis, and notebook sections 3, 4, 6, and 7.