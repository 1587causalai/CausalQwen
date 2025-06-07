"""
Example script for using the causal language model.

This script demonstrates how to use the causal language model
for both classification and regression tasks.
"""

# 添加路径 - 最直接的解决方案
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from src.models.causal_lm import CausalLanguageModel
from src.data.tokenizer import MockTokenizer
from src.data.dataset import TextWithNumbersGenerator
from src.utils.distributions import cauchy_sample_reparameterized




def basic_example():
    """
    Basic example of using the causal language model.
    """
    print("Running basic example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set hyperparameters
    vocab_size = 1000
    num_token_id = 2
    hidden_size = 1024
    causal_dim = 64
    
    # Create model
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        num_token_id=num_token_id,
        hidden_size=hidden_size,
        causal_dim=causal_dim,
        use_mock_feature_network=True
    )
    
    # Create tokenizer
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    
    # Create input
    text = "The price is 99.9 dollars."
    
    # Tokenize input
    encoded = tokenizer.encode(text, return_tensors="pt")
    input_ids = encoded["input_ids"].unsqueeze(0)  # Add batch dimension
    numerical_values = encoded["numerical_values"].unsqueeze(0)  # Add batch dimension
    
    print(f"Input text: {text}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    print(f"Numerical values: {numerical_values[0].tolist()}")
    
    # Forward pass
    outputs = model(input_ids, numerical_values)
    
    # Get predictions
    predictions = model.predict(input_ids, numerical_values)
    
    # Print predictions
    print("\nPredictions:")
    print(f"  Predicted token: {tokenizer.convert_ids_to_tokens([predictions['cls_pred'].item()])[0]}")
    print(f"  Predicted value: {predictions['reg_pred'].item():.4f}")
    print(f"  <NUM> probability: {predictions['num_prob'].item():.4f}")
    
    # Sample from causal state distribution
    print("\nSampling from causal state distribution:")
    for i in range(3):
        sampled_predictions = model.sample_and_predict(input_ids, numerical_values)
        print(f"  Sample {i+1}:")
        print(f"    Predicted token: {tokenizer.convert_ids_to_tokens([sampled_predictions['cls_pred'].item()])[0]}")
        print(f"    Predicted value: {sampled_predictions['reg_pred'].item():.4f}")


def numerical_example():
    """
    Example of using the causal language model with numerical data.
    """
    print("\nRunning numerical example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set hyperparameters
    vocab_size = 1000
    num_token_id = 2
    hidden_size = 1024
    causal_dim = 64
    
    # Create model
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        num_token_id=num_token_id,
        hidden_size=hidden_size,
        causal_dim=causal_dim,
        use_mock_feature_network=True
    )
    
    # Create tokenizer
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    
    # Create text generator
    generator = TextWithNumbersGenerator(seed=42)
    
    # Generate texts with numerical values
    texts, values = generator.generate_text(num_samples=5)
    
    print("Generated texts with numerical values:")
    for i, (text, value) in enumerate(zip(texts, values)):
        print(f"  {i+1}. {text} (value: {value:.4f})")
    
    # Process the first text
    text = texts[0]
    
    # Tokenize input
    encoded = tokenizer.encode(text, return_tensors="pt")
    input_ids = encoded["input_ids"].unsqueeze(0)  # Add batch dimension
    numerical_values = encoded["numerical_values"].unsqueeze(0)  # Add batch dimension
    
    print(f"\nProcessing: {text}")
    print(f"Input tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    print(f"Numerical values: {numerical_values[0].tolist()}")
    
    # Forward pass
    outputs = model(input_ids, numerical_values)
    
    # Get predictions
    predictions = model.predict(input_ids, numerical_values)
    
    # Print predictions
    print("\nPredictions:")
    print(f"  Predicted token: {tokenizer.convert_ids_to_tokens([predictions['cls_pred'].item()])[0]}")
    print(f"  Predicted value: {predictions['reg_pred'].item():.4f}")
    print(f"  <NUM> probability: {predictions['num_prob'].item():.4f}")
    
    # Generate question-answer pairs
    contexts, questions, answers = generator.generate_qa_pairs(num_samples=3)
    
    print("\nGenerated question-answer pairs:")
    for i, (context, question, answer) in enumerate(zip(contexts, questions, answers)):
        print(f"  {i+1}. Context: {context}")
        print(f"     Question: {question}")
        print(f"     Answer: {answer:.4f}")


def visualization_example():
    """
    Example of visualizing the causal state distribution.
    """
    print("\nRunning visualization example...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set hyperparameters
    vocab_size = 1000
    num_token_id = 2
    hidden_size = 1024
    causal_dim = 2  # Use 2D causal state for visualization
    
    # Create model
    model = CausalLanguageModel(
        vocab_size=vocab_size,
        num_token_id=num_token_id,
        hidden_size=hidden_size,
        causal_dim=causal_dim,
        use_mock_feature_network=True
    )
    
    # Create tokenizer
    tokenizer = MockTokenizer(vocab_size=vocab_size)
    
    # Create text generator
    generator = TextWithNumbersGenerator(seed=42)
    
    # Generate texts with numerical values
    texts, values = generator.generate_text(num_samples=1)
    text = texts[0]
    
    # Tokenize input
    encoded = tokenizer.encode(text, return_tensors="pt")
    input_ids = encoded["input_ids"].unsqueeze(0)  # Add batch dimension
    numerical_values = encoded["numerical_values"].unsqueeze(0)  # Add batch dimension
    
    print(f"Input text: {text}")
    
    # Forward pass
    outputs = model(input_ids, numerical_values)
    
    # Get causal state distribution parameters
    causal_loc = outputs["causal_loc"][0].detach().numpy()
    causal_scale = outputs["causal_scale"][0].detach().numpy()
    
    print(f"Causal state location: [{causal_loc[0]:.4f}, {causal_loc[1]:.4f}]")
    print(f"Causal state scale: [{causal_scale[0]:.4f}, {causal_scale[1]:.4f}]")
    
    # Sample from causal state distribution
    num_samples = 1000
    samples = []
    for _ in range(num_samples):
        epsilon = torch.rand(2)
        sample = cauchy_sample_reparameterized(
            torch.tensor(causal_loc), torch.tensor(causal_scale), epsilon
        ).numpy()
        samples.append(sample)
    
    samples = np.array(samples)
    
    # Plot samples
    plt.figure(figsize=(10, 8))
    
    # Plot samples
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, label="Samples")
    
    # Plot location parameter
    plt.scatter([causal_loc[0]], [causal_loc[1]], color="red", s=100, label="Location")
    
    # Add contour lines for scale
    x = np.linspace(causal_loc[0] - 5 * causal_scale[0], causal_loc[0] + 5 * causal_scale[0], 100)
    y = np.linspace(causal_loc[1] - 5 * causal_scale[1], causal_loc[1] + 5 * causal_scale[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = 1 / (np.pi * causal_scale[0] * causal_scale[1] * (1 + ((X - causal_loc[0]) / causal_scale[0])**2) * (1 + ((Y - causal_loc[1]) / causal_scale[1])**2))
    
    plt.contour(X, Y, Z, levels=5, colors="green", alpha=0.5)
    
    plt.title("Causal State Distribution")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig("causal_state_distribution.png")
    print("Saved visualization to causal_state_distribution.png")


def main():
    """
    Main function.
    """
    basic_example()
    numerical_example()
    visualization_example()


if __name__ == "__main__":
    main()

