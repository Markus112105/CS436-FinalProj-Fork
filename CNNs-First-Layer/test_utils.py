"""Small regression checks for the imported training and analysis utilities."""

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import calc_energy_profile, fit_formula_to_model, get_dataloader, train_model


class BaselineTests(unittest.TestCase):
    # Experiment tests start: verify the new momentum option on tiny synthetic data.
    # These checks do not train VGG11 or run any of the six CIFAR-10 experiments.
    def test_momentum_matches_two_known_updates(self):
        # Two single-image batches reveal the accumulated gradient on step two.
        loader = DataLoader(
            TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)),
            batch_size=1,
        )
        for momentum in (0.0, 0.9):
            with self.subTest(momentum=momentum):
                model = torch.nn.Linear(1, 2, bias=False)
                with torch.no_grad():
                    model.weight.zero_()
                train_model(model, loader, loader, num_epochs=1,
                            use_cuda=False, lr=0.1, momentum=momentum)

                # The first gradient is [-0.5, 0.5], giving weights [0.05, -0.05].
                # Step two adds its gradient plus momentum times that first gradient.
                second_gradient = torch.sigmoid(torch.tensor(-0.1)).item()
                positive_weight = 0.05 + 0.1 * (second_gradient + momentum * 0.5)
                expected = torch.tensor([[positive_weight], [-positive_weight]])
                torch.testing.assert_close(model.weight, expected)

    def test_default_momentum_preserves_baseline(self):
        loader = DataLoader(
            TensorDataset(torch.ones(2, 1), torch.zeros(2, dtype=torch.long)),
            batch_size=1,
        )
        default_model = torch.nn.Linear(1, 2, bias=False)
        zero_model = torch.nn.Linear(1, 2, bias=False)
        zero_model.load_state_dict(default_model.state_dict())

        # Omitting momentum must produce the same updates and metrics as zero.
        default_history = train_model(default_model, loader, loader,
                                      num_epochs=1, use_cuda=False, lr=0.1)
        zero_history = train_model(zero_model, loader, loader,
                                   num_epochs=1, use_cuda=False, lr=0.1, momentum=0.0)
        torch.testing.assert_close(default_model.weight, zero_model.weight)
        self.assertEqual(default_history, zero_history)

    # new code end
    def test_cpu_metrics_include_partial_batches_and_separate_losses(self):
        # Zero inputs keep predictions fixed, so accuracy and loss are exact.
        model = torch.nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.copy_(torch.tensor([2.0, 0.0]))
        images = torch.zeros(3, 2)
        train = DataLoader(TensorDataset(images, torch.zeros(3, dtype=torch.long)), batch_size=2)
        test = DataLoader(TensorDataset(images, torch.tensor([0, 1, 1])), batch_size=2)
        history = train_model(model, train, test, num_epochs=1, use_cuda=False, lr=0.0)
        self.assertEqual(next(model.parameters()).device.type, 'cpu')
        self.assertEqual(history[0]['train_accuracy'], 1.0)
        self.assertAlmostEqual(history[0]['test_accuracy'], 1 / 3)
        expected = torch.nn.functional.cross_entropy(model(images), torch.zeros(3, dtype=torch.long))
        self.assertAlmostEqual(history[0]['train_loss'], expected.item(), places=6)

    def test_test_loader_downloads_and_keeps_every_image(self):
        dataset = TensorDataset(torch.zeros(5, 3, 32, 32), torch.zeros(5, dtype=torch.long))
        with patch('utils.torchvision.datasets.CIFAR10', return_value=dataset) as cifar:
            loader = get_dataloader(False, train=False, batch_size=2)
        self.assertFalse(cifar.call_args.kwargs['train'])
        self.assertTrue(cifar.call_args.kwargs['download'])
        self.assertEqual(loader.num_workers, 0)
        self.assertEqual(sum(len(labels) for _, labels in loader), 5)
        self.assertIsInstance(loader.sampler, torch.utils.data.SequentialSampler)

    def test_frozen_layer_has_finite_zero_change_profile(self):
        weights = torch.ones(4, 3, 3, 3)
        profile = calc_energy_profile(weights, weights.clone(), torch.eye(27))
        self.assertTrue(torch.isfinite(profile).all())
        self.assertEqual(profile.count_nonzero().item(), 0)

    def test_patch_pca_matches_centered_svd(self):
        from utils import get_patch_pca

        # The streamed 27-by-27 scatter matrix must match PCA of all patches.
        images = torch.randn(3, 3, 5, 5, generator=torch.Generator().manual_seed(436))
        loader = DataLoader(TensorDataset(images, torch.zeros(3)), batch_size=2)
        components, singular_values, count = get_patch_pca(loader, num_images=3)
        patches = images.double().unfold(2, 3, 1).unfold(3, 3, 1)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, 27)
        patches -= patches.mean(0)
        expected = torch.linalg.svdvals(patches)
        self.assertEqual(count, 3)
        torch.testing.assert_close(singular_values, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(components.T @ components, torch.eye(27, dtype=torch.float64))

    def test_formula_fit_handles_zero_singular_values(self):
        # A rank-deficient patch matrix has a=0; the analytical limit is finite.
        singular_values = torch.tensor([5.0, 3.0, 1.0, 0.0], dtype=torch.float64)
        profile, correlation = fit_formula_to_model(
            torch.tensor([1.0, 0.7, 0.2, 0.0]), singular_values,
            singular_values.square(), num_images=100,
        )
        self.assertTrue(torch.isfinite(profile).all())
        self.assertTrue(-1 <= correlation <= 1)


if __name__ == '__main__':
    unittest.main()
