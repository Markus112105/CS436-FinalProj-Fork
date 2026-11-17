# CS 436 Final Project

**Question:** Does adding momentum to SGD change VGG11-BN first-layer energy profiles, CIFAR-10 accuracy, and training time?

**Hypothesis 1:** Momentum 0.9 raises test accuracy and leaves the first-layer energy profile and training time close to plain SGD.

**Hypothesis 2:** Plain SGD with learning rate 0.1 matches SGD with momentum 0.9 at learning rate 0.01 in test accuracy and first-layer energy profile.

## Experiment

1. Train VGG11-BN with SGD (momentum 0) and SGD with momentum 0.9 for 30 epochs across three matched seeds: six runs total.
2. Compare first-layer energy profiles, training-loss curves, test accuracy, and training time.
3. Train plain SGD again on the same three seeds with learning rate 0.1 instead of 0.01 and compare it with the momentum runs: three more runs.

## Results

Momentum raised mean test accuracy from 54.54% to 78.56% with the same training time

Plain SGD with learning rate 0.1 matched the energy profile and training loss of momentum (0.99 profile correlation) but its mean test accuracy was 72.81%, which is 5.75 points lower, and it changed by up to 19.08 points when we repeated the same seed

Hypothesis 1 is supported and hypothesis 2 is partly supported

The experiment is in the [notebook](CNNs-First-Layer/optimizer_experiment.ipynb). We trained the first six runs in one Google Colab session on a Tesla T4 GPU and the three learning rate 0.1 runs in later T4 sessions. The measurements are in `results/` and the figures are [docs/energy_and_loss.pdf](docs/energy_and_loss.pdf) and [docs/energy_and_loss_large_lr.pdf](docs/energy_and_loss_large_lr.pdf).

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
if u want to redo the analysis without training, open the notebook and run sections 1, 2, 5, 6, and 9

## Documentation

The [report source](docs/final_project_report.tex), [report PDF](docs/final_project_report.pdf), [project requirements](docs/report_requirements.pdf), and [original paper](docs/original_paper.pdf) are in `docs/`.

## Attribution and contributions

Based on [Chowers and Weiss's code](https://github.com/RheaChowers/CNNs-First-Layer) for their [ICML 2023 paper](https://proceedings.mlr.press/v202/chowers23a.html).

We added a momentum option, training-time measurement, and final-only test evaluation to `train_model` in `utils.py`, and wrote the experiment notebook. the additions in the original files are marked with comments that clearly show what we added and where.

**Markus Massina (GitHub: Markus112105):** project planning, tests, local setup, notebook sections 1, 2, 5, and 8 to 10, and the Colab runs.

**Michael Truong (GitHub: Michael-Truong111):** project planning, Colab results analysis, and notebook sections 3, 4, 6, and 7.