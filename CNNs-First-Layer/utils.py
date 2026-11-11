"""Local compatibility fixes for the original CNN first-layer experiments."""

from pathlib import Path
from random import shuffle
import time  # markus add

import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent.parent
# Pin model definitions and cache downloads outside the tracked source files.
MODEL_REPO = 'chenyaofo/pytorch-cifar-models:786c16252c0fc58ee9adac063f8337cc4a7a497a'


def get_model(model_name):
    torch.hub.set_dir(str(PROJECT_DIR / '.cache' / 'torch' / 'hub'))
    return torch.hub.load(MODEL_REPO, 'cifar10_' + model_name,
                          pretrained=False, trust_repo=True)


def get_first_layer_weights(model):
    if hasattr(model, 'features'):  # VGG stores convolutional layers in features.
        return model.features[0].weight.detach()
    if hasattr(model, 'conv1'):  # ResNet names its first convolution conv1.
        return model.conv1.weight.detach()
    if hasattr(model, 'transformer'):
        return model.transformer.embeddings.patch_embeddings.weight.detach()
    raise ValueError('Getting the first layer of this model is not implemented.')


def fit_formula_to_model(model_energy_profile, lambdas, lambda_squared, num_images):
    # Keep the original gradient-step search, but avoid subtracting nearly equal
    # floats in 1 - (1 - a)^t. The limit at a=0 is t.
    model_energy_profile = model_energy_profile.detach().cpu().double()
    lambdas = lambdas.detach().cpu().double()
    a = 1e-7 * lambda_squared.detach().cpu().double() / (2 * num_images)
    best_profile = None
    max_correlation = -float('inf')
    for gd_step in tqdm(range(10000, 5000000, 250), desc='Formula fit'):
        numerator = -torch.expm1(gd_step * torch.log1p(-a))
        vec = numerator / torch.where(a == 0, torch.ones_like(a), a)
        vec = torch.where(a == 0, torch.full_like(a, gd_step), vec)
        solution = (vec * lambdas).abs()
        if solution.max() == 0:
            continue
        solution /= solution.max()
        corr = np.corrcoef(solution.numpy(), model_energy_profile.numpy())[0, 1]
        if np.isfinite(corr) and corr > max_correlation:
            max_correlation, best_profile = float(corr), solution
    if best_profile is None:
        raise ValueError('Formula fitting requires nonconstant, nonzero energy profiles.')
    return best_profile, max_correlation


def calc_energy_profile(first_layer, model_initialization, components,
                        subtract_init=True, normalize=True):
    # Analysis stays on CPU because Apple MPS does not support float64.
    first_layer = first_layer.detach().cpu().double()
    if subtract_init:
        first_layer = first_layer - model_initialization.detach().cpu().double()
    weights_in_pca_basis = first_layer.flatten(1) @ components.detach().cpu().double()
    # Preserve the upstream scale; its constant cancels after normalization.
    energy_profile = weights_in_pca_basis.square().sum(0).sqrt() / first_layer.shape[0]
    if normalize and energy_profile.max() > 0:
        energy_profile /= energy_profile.max()
    return energy_profile


def get_dataloader(use_random_labels, train=True, batch_size=256, num_workers=0):
    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_DIR / 'data'), train=train, transform=transforms, download=True,
    )
    if use_random_labels:
        shuffle(dataset.targets)
    # Zero workers works in macOS notebooks without multiprocessing setup.
    # Keep partial batches so evaluation covers every image.
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=train, drop_last=False, pin_memory=torch.cuda.is_available(),
    )


def get_patch_pca(train_loader, num_images=10000):
    """Return patch PCA directions, singular values, and actual image count."""
    # Accumulate sums and a 27-by-27 scatter matrix one batch at a time.
    # Centering it gives X.T @ X without storing all 9 million image patches.
    patch_sum = torch.zeros(27, dtype=torch.float64)
    scatter = torch.zeros(27, 27, dtype=torch.float64)
    image_count = patch_count = 0
    for images, _ in train_loader:
        images = images[:num_images - image_count].cpu().double()
        patches = images.unfold(2, 3, 1).unfold(3, 3, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 27)
        patch_sum += patches.sum(0)
        scatter += patches.T @ patches
        patch_count += len(patches)
        image_count += len(images)
        if image_count >= num_images:
            break
    if patch_count < 2:
        raise ValueError('PCA requires at least two image patches.')
    scatter -= torch.outer(patch_sum, patch_sum) / patch_count
    eigenvalues, components = torch.linalg.eigh(scatter)
    # eigh sorts ascending; match the original PCA's descending singular values.
    singular_values = eigenvalues.flip(0).clamp_min(0).sqrt()
    return components.flip(1), singular_values, image_count


# ---------- added code ----------
# momentum and eval_every_epoch are new arguments.
# Their defaults keep the original notebook calls working unchanged.
def train_model(model, train_loader, test_loader, num_epochs=100,
                use_cuda=None, lr=0.01, device=None, momentum=0.0,
                eval_every_epoch=True):
# ---------- end added code ----------
    # An explicit device wins. The old flag still supports forcing CPU/CUDA;
    # otherwise prefer CUDA, then Apple MPS, then CPU.
    if device is None:
        if use_cuda is not None:
            device = 'cuda' if use_cuda else 'cpu'
        elif torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)
    model.to(device)
    # ---------- added code ----------
    # Pass the selected momentum into the existing SGD optimizer.
    # Zero preserves the original SGD behavior; 0.9 accumulates past gradients.
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        momentum=momentum,
    )
    # ---------- end added code ----------
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    pbar = tqdm(range(num_epochs), desc=f'Training ({device})')
    for epoch in pbar:
        model.train()
        train_loss = train_correct = train_count = 0
        # ---------- added code ----------
        # Time the training work only, excluding test evaluation.
        # GPU operations are asynchronous, so wait before starting the timer.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        start_time = time.perf_counter()
        # ---------- end added code ----------
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            # Weight batch means by sample count so partial batches are correct.
            train_loss += loss.item() * len(labels)
            train_correct += (logits.detach().argmax(1) == labels).sum().item()
            train_count += len(labels)

        # ---------- added code ----------
        # Wait for the final training update before stopping the epoch timer.
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize()
        train_seconds = time.perf_counter() - start_time

        # The original notebooks evaluate every epoch; the experiment only
        # evaluates the final model so its test set is not used during training.
        test_loss_value = test_accuracy = None
        test_loss = test_correct = test_count = 0
        if eval_every_epoch or epoch == num_epochs - 1:
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    test_loss += criterion(logits, labels).item() * len(labels)
                    test_correct += (logits.argmax(1) == labels).sum().item()
                    test_count += len(labels)
            test_loss_value = test_loss / test_count
            test_accuracy = test_correct / test_count
        # ---------- end added code ----------
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss / train_count,
            'train_accuracy': train_correct / train_count,
            'test_loss': test_loss_value,
            'test_accuracy': test_accuracy,
            'train_seconds': train_seconds,  # markus add
        })
        pbar.set_postfix(history[-1])
    return history
