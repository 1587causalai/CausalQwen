"""
Unit tests for the causal language model.

This module contains tests for the various components of the causal language model.
"""

import unittest
import torch
import numpy as np

from src.models.feature_network import MockFeatureNetwork
from src.models.abduction_network import AbductionNetwork
from src.models.action_network import ActionNetwork, ClassificationHead, RegressionHead
from src.models.causal_lm import CausalLanguageModel
from src.utils.distributions import cauchy_pdf, cauchy_cdf, cauchy_sample, cauchy_sample_reparameterized
from src.utils.losses import OvRClassificationLoss, GatedRegressionLoss, CausalLMLoss
from src.data.tokenizer import MockTokenizer
from src.data.dataset import SyntheticDataset


class TestDistributions(unittest.TestCase):
    """
    Tests for the distribution utilities.
    """
    
    def test_cauchy_pdf(self):
        """
        Test the Cauchy PDF function.
        """
        # Test with scalar inputs
        x = 0.0
        loc = 0.0
        scale = 1.0
        pdf = cauchy_pdf(x, loc, scale)
        self.assertAlmostEqual(pdf, 1.0 / np.pi, places=6)
        
        # Test with tensor inputs
        x = torch.tensor([0.0, 1.0, 2.0])
        loc = torch.tensor([0.0, 0.0, 0.0])
        scale = torch.tensor([1.0, 1.0, 1.0])
        pdf = cauchy_pdf(x, loc, scale)
        expected = torch.tensor([1.0 / np.pi, 0.5 / np.pi, 0.2 / np.pi])
        self.assertTrue(torch.allclose(pdf, expected, atol=1e-6))
    
    def test_cauchy_cdf(self):
        """
        Test the Cauchy CDF function.
        """
        # Test with scalar inputs
        x = 0.0
        loc = 0.0
        scale = 1.0
        cdf = cauchy_cdf(x, loc, scale)
        self.assertAlmostEqual(cdf, 0.5, places=6)
        
        # Test with tensor inputs
        x = torch.tensor([0.0, 1.0, 2.0])
        loc = torch.tensor([0.0, 0.0, 0.0])
        scale = torch.tensor([1.0, 1.0, 1.0])
        cdf = cauchy_cdf(x, loc, scale)
        expected = torch.tensor([0.5, 0.75, 0.8524])
        self.assertTrue(torch.allclose(cdf, expected, atol=1e-4))
    
    def test_cauchy_sample(self):
        """
        Test the Cauchy sampling function.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Test with scalar inputs
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        sample = cauchy_sample(loc, scale)
        self.assertTrue(isinstance(sample, torch.Tensor))
        
        # Test with batch inputs
        loc = torch.tensor([0.0, 1.0, 2.0])
        scale = torch.tensor([1.0, 2.0, 3.0])
        samples = cauchy_sample(loc, scale, sample_shape=(10,))
        self.assertEqual(samples.shape, (10, 3))
    
    def test_cauchy_sample_reparameterized(self):
        """
        Test the reparameterized Cauchy sampling function.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Test with scalar inputs
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        epsilon = torch.tensor(0.75)  # 0.75 corresponds to quantile 0.75
        sample = cauchy_sample_reparameterized(loc, scale, epsilon)
        self.assertAlmostEqual(sample.item(), 1.0, places=5)  # tan(pi * (0.75 - 0.5)) = tan(pi/4) = 1.0
        
        # Test with batch inputs
        loc = torch.tensor([0.0, 1.0, 2.0])
        scale = torch.tensor([1.0, 2.0, 3.0])
        epsilon = torch.tensor([0.75, 0.75, 0.75])
        samples = cauchy_sample_reparameterized(loc, scale, epsilon)
        expected = torch.tensor([1.0, 3.0, 5.0])
        self.assertTrue(torch.allclose(samples, expected, atol=1e-5))


class TestFeatureNetwork(unittest.TestCase):
    """
    Tests for the feature network.
    """
    
    def test_mock_feature_network(self):
        """
        Test the mock feature network.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create network
        hidden_size = 1024
        network = MockFeatureNetwork(hidden_size=hidden_size)
        
        # Create input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        # Forward pass
        features = network(input_ids)
        
        # Check output shape
        self.assertEqual(features.shape, (batch_size, hidden_size))


class TestAbductionNetwork(unittest.TestCase):
    """
    Tests for the abduction network.
    """
    
    def test_abduction_network(self):
        """
        Test the abduction network.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create network
        input_size = 1024
        causal_dim = 64
        network = AbductionNetwork(input_size=input_size, causal_dim=causal_dim)
        
        # Create input
        batch_size = 2
        features = torch.randn(batch_size, input_size)
        
        # Forward pass
        loc, scale = network(features)
        
        # Check output shapes
        self.assertEqual(loc.shape, (batch_size, causal_dim))
        self.assertEqual(scale.shape, (batch_size, causal_dim))
        
        # Check that scale is positive
        self.assertTrue(torch.all(scale > 0))


class TestActionNetwork(unittest.TestCase):
    """
    Tests for the action network.
    """
    
    def test_classification_head(self):
        """
        Test the classification head.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create network
        causal_dim = 64
        num_classes = 1000
        network = ClassificationHead(causal_dim=causal_dim, num_classes=num_classes)
        
        # Create input
        batch_size = 2
        causal_loc = torch.randn(batch_size, causal_dim)
        causal_scale = torch.exp(torch.randn(batch_size, causal_dim))
        
        # Forward pass
        score_loc, score_scale = network(causal_loc, causal_scale)
        
        # Check output shapes
        self.assertEqual(score_loc.shape, (batch_size, num_classes))
        self.assertEqual(score_scale.shape, (batch_size, num_classes))
        
        # Check that scale is positive
        self.assertTrue(torch.all(score_scale > 0))
        
        # Test probability computation
        probs = network.compute_probabilities(score_loc, score_scale)
        self.assertEqual(probs.shape, (batch_size, num_classes))
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))
        
        # Test prediction
        predictions = network.predict(score_loc, score_scale)
        self.assertEqual(predictions.shape, (batch_size,))
    
    def test_regression_head(self):
        """
        Test the regression head.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create network
        causal_dim = 64
        network = RegressionHead(causal_dim=causal_dim)
        
        # Create input
        batch_size = 2
        causal_loc = torch.randn(batch_size, causal_dim)
        causal_scale = torch.exp(torch.randn(batch_size, causal_dim))
        
        # Forward pass
        value_loc, value_scale = network(causal_loc, causal_scale)
        
        # Check output shapes
        self.assertEqual(value_loc.shape, (batch_size,))
        self.assertEqual(value_scale.shape, (batch_size,))
        
        # Check that scale is positive
        self.assertTrue(torch.all(value_scale > 0))
        
        # Test prediction
        predictions = network.predict(value_loc, value_scale)
        self.assertEqual(predictions.shape, (batch_size,))
    
    def test_action_network(self):
        """
        Test the action network.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create network
        causal_dim = 64
        num_classes = 1000
        num_token_id = 2
        network = ActionNetwork(causal_dim=causal_dim, num_classes=num_classes, num_token_id=num_token_id)
        
        # Create input
        batch_size = 2
        causal_loc = torch.randn(batch_size, causal_dim)
        causal_scale = torch.exp(torch.randn(batch_size, causal_dim))
        
        # Forward pass
        outputs = network(causal_loc, causal_scale)
        
        # Check output shapes
        self.assertEqual(outputs["cls_loc"].shape, (batch_size, num_classes))
        self.assertEqual(outputs["cls_scale"].shape, (batch_size, num_classes))
        self.assertEqual(outputs["reg_loc"].shape, (batch_size,))
        self.assertEqual(outputs["reg_scale"].shape, (batch_size,))
        self.assertEqual(outputs["cls_probs"].shape, (batch_size, num_classes))
        
        # Test prediction
        predictions = network.predict(causal_loc, causal_scale)
        self.assertEqual(predictions["cls_pred"].shape, (batch_size,))
        self.assertEqual(predictions["reg_pred"].shape, (batch_size,))
        self.assertEqual(predictions["num_prob"].shape, (batch_size,))


class TestCausalLanguageModel(unittest.TestCase):
    """
    Tests for the complete causal language model.
    """
    
    def test_causal_language_model(self):
        """
        Test the causal language model.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create model
        vocab_size = 1000
        num_token_id = 2
        hidden_size = 1024
        causal_dim = 64
        model = CausalLanguageModel(
            vocab_size=vocab_size,
            num_token_id=num_token_id,
            hidden_size=hidden_size,
            causal_dim=causal_dim,
            use_mock_feature_network=True
        )
        
        # Create input
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        numerical_values = torch.randn(batch_size, seq_len)
        
        # Forward pass
        outputs = model(input_ids, numerical_values)
        
        # Check output shapes
        self.assertEqual(outputs["features"].shape, (batch_size, hidden_size))
        self.assertEqual(outputs["causal_loc"].shape, (batch_size, causal_dim))
        self.assertEqual(outputs["causal_scale"].shape, (batch_size, causal_dim))
        self.assertEqual(outputs["cls_loc"].shape, (batch_size, vocab_size))
        self.assertEqual(outputs["cls_scale"].shape, (batch_size, vocab_size))
        self.assertEqual(outputs["reg_loc"].shape, (batch_size,))
        self.assertEqual(outputs["reg_scale"].shape, (batch_size,))
        self.assertEqual(outputs["cls_probs"].shape, (batch_size, vocab_size))
        
        # Test prediction
        predictions = model.predict(input_ids, numerical_values)
        self.assertEqual(predictions["cls_pred"].shape, (batch_size,))
        self.assertEqual(predictions["reg_pred"].shape, (batch_size,))
        self.assertEqual(predictions["num_prob"].shape, (batch_size,))
        
        # Test sampling
        sampled_predictions = model.sample_and_predict(input_ids, numerical_values)
        self.assertEqual(sampled_predictions["cls_pred"].shape, (batch_size,))
        self.assertEqual(sampled_predictions["reg_pred"].shape, (batch_size,))
        self.assertEqual(sampled_predictions["causal_sample"].shape, (batch_size, causal_dim))


class TestLosses(unittest.TestCase):
    """
    Tests for the loss functions.
    """
    
    def test_ovr_classification_loss(self):
        """
        Test the OvR classification loss.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create loss function
        num_classes = 10
        loss_fn = OvRClassificationLoss(num_classes=num_classes)
        
        # Create input
        batch_size = 2
        loc = torch.randn(batch_size, num_classes)
        scale = torch.exp(torch.randn(batch_size, num_classes))
        targets = torch.randint(0, num_classes, (batch_size,))
        
        # Compute loss
        loss = loss_fn(loc, scale, targets)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive
        self.assertTrue(loss > 0)
    
    def test_gated_regression_loss(self):
        """
        Test the gated regression loss.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create loss function
        num_token_id = 2
        loss_fn = GatedRegressionLoss(num_token_id=num_token_id)
        
        # Create input
        batch_size = 4
        reg_loc = torch.randn(batch_size)
        reg_scale = torch.exp(torch.randn(batch_size))
        num_prob = torch.rand(batch_size)
        targets = torch.tensor([0, 1, 2, 3])  # Only one is <NUM>
        target_values = torch.randn(batch_size)
        
        # Compute loss
        loss = loss_fn(reg_loc, reg_scale, num_prob, targets, target_values)
        
        # Check that loss is a scalar
        self.assertEqual(loss.shape, ())
        
        # Check that loss is non-negative
        self.assertTrue(loss >= 0)
    
    def test_causal_lm_loss(self):
        """
        Test the combined causal LM loss.
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create loss function
        num_classes = 10
        num_token_id = 2
        loss_fn = CausalLMLoss(num_classes=num_classes, num_token_id=num_token_id)
        
        # Create input
        batch_size = 4
        cls_loc = torch.randn(batch_size, num_classes)
        cls_scale = torch.exp(torch.randn(batch_size, num_classes))
        reg_loc = torch.randn(batch_size)
        reg_scale = torch.exp(torch.randn(batch_size))
        targets = torch.tensor([0, 1, 2, 2])  # Two are <NUM>
        target_values = torch.randn(batch_size)
        
        # Compute loss
        loss_dict = loss_fn(cls_loc, cls_scale, reg_loc, reg_scale, targets, target_values)
        
        # Check that loss is a dictionary
        self.assertTrue(isinstance(loss_dict, dict))
        
        # Check that loss contains expected keys
        self.assertTrue("loss" in loss_dict)
        self.assertTrue("cls_loss" in loss_dict)
        self.assertTrue("reg_loss" in loss_dict)
        
        # Check that losses are scalars
        self.assertEqual(loss_dict["loss"].shape, ())
        self.assertEqual(loss_dict["cls_loss"].shape, ())
        self.assertEqual(loss_dict["reg_loss"].shape, ())
        
        # Check that losses are non-negative
        self.assertTrue(loss_dict["loss"] >= 0)
        self.assertTrue(loss_dict["cls_loss"] >= 0)
        self.assertTrue(loss_dict["reg_loss"] >= 0)
        
        # Check that total loss is sum of components
        self.assertAlmostEqual(
            loss_dict["loss"].item(),
            loss_dict["cls_loss"].item() + loss_dict["reg_loss"].item(),
            places=5
        )


class TestTokenizer(unittest.TestCase):
    """
    Tests for the tokenizer.
    """
    
    def test_mock_tokenizer(self):
        """
        Test the mock tokenizer.
        """
        # Create tokenizer
        vocab_size = 1000
        tokenizer = MockTokenizer(vocab_size=vocab_size)
        
        # Test tokenization
        text = "The price is 99.9 dollars."
        tokens, numerical_values = tokenizer.tokenize(text)
        
        # Check that <NUM> token is present
        self.assertTrue(tokenizer.num_token in tokens)
        
        # Check that numerical value is correct
        num_index = tokens.index(tokenizer.num_token)
        self.assertAlmostEqual(numerical_values[num_index], 99.9)
        
        # Test encoding
        encoded = tokenizer.encode(text)
        
        # Check that encoded output contains expected keys
        self.assertTrue("input_ids" in encoded)
        self.assertTrue("attention_mask" in encoded)
        self.assertTrue("numerical_values" in encoded)
        
        # Test batch encoding
        texts = ["The price is 99.9 dollars.", "The temperature is 25.5 degrees."]
        batch_encoded = tokenizer.batch_encode_plus(texts, padding=True, return_tensors="pt")
        
        # Check that batch encoded output contains expected keys
        self.assertTrue("input_ids" in batch_encoded)
        self.assertTrue("attention_mask" in batch_encoded)
        self.assertTrue("numerical_values" in batch_encoded)
        
        # Check that tensors have correct shape
        self.assertEqual(batch_encoded["input_ids"].shape[0], len(texts))
        self.assertEqual(batch_encoded["attention_mask"].shape[0], len(texts))
        self.assertEqual(batch_encoded["numerical_values"].shape[0], len(texts))


class TestDataset(unittest.TestCase):
    """
    Tests for the dataset.
    """
    
    def test_synthetic_dataset(self):
        """
        Test the synthetic dataset.
        """
        # Create dataset
        num_samples = 100
        vocab_size = 1000
        hidden_size = 1024
        dataset = SyntheticDataset(
            num_samples=num_samples,
            vocab_size=vocab_size,
            hidden_size=hidden_size
        )
        
        # Check dataset length
        self.assertEqual(len(dataset), num_samples)
        
        # Check sample
        sample = dataset[0]
        
        # Check that sample contains expected keys
        self.assertTrue("feature" in sample)
        self.assertTrue("target_token" in sample)
        self.assertTrue("target_value" in sample)
        
        # Check that tensors have correct shape
        self.assertEqual(sample["feature"].shape, (hidden_size,))
        self.assertEqual(sample["target_token"].shape, ())
        self.assertEqual(sample["target_value"].shape, ())


if __name__ == "__main__":
    unittest.main()

