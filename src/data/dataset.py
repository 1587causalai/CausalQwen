"""
Dataset module.

This module implements dataset classes for the causal language model,
including a synthetic dataset generator for testing.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from .tokenizer import MockTokenizer


class CausalLMDataset(Dataset):
    """
    Dataset for the causal language model.
    
    This dataset handles both text and numerical data.
    """
    
    def __init__(self, texts, target_texts=None, tokenizer=None):
        """
        Initialize the dataset.
        
        Args:
            texts (List[str]): List of input texts
            target_texts (List[str], optional): List of target texts. If None, targets are the same as inputs.
            tokenizer (optional): Tokenizer to use. If None, a MockTokenizer is created.
        """
        self.texts = texts
        self.target_texts = target_texts if target_texts is not None else texts
        self.tokenizer = tokenizer if tokenizer is not None else MockTokenizer()
        
        # Encode all texts
        self.encoded_inputs = self.tokenizer.batch_encode_plus(
            self.texts, padding=True, return_tensors="pt"
        )
        
        # Encode all target texts
        self.encoded_targets = self.tokenizer.batch_encode_plus(
            self.target_texts, padding=True, return_tensors="pt"
        )
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.texts)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Sample containing input and target tensors
        """
        # Get input tensors
        input_ids = self.encoded_inputs["input_ids"][idx]
        attention_mask = self.encoded_inputs["attention_mask"][idx]
        numerical_values = self.encoded_inputs["numerical_values"][idx]
        
        # Get target tensors
        target_ids = self.encoded_targets["input_ids"][idx]
        target_numerical_values = self.encoded_targets["numerical_values"][idx]
        
        # Create sample
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numerical_values": numerical_values,
            "target_ids": target_ids,
            "target_numerical_values": target_numerical_values
        }
        
        return sample


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing the causal language model.
    
    This dataset generates random data with controlled properties,
    allowing for focused testing of specific model components.
    """
    
    def __init__(
        self,
        num_samples=1000,
        vocab_size=1000,
        seq_length=20,
        num_token_id=2,
        num_probability=0.2,
        hidden_size=1024,
        seed=42
    ):
        """
        Initialize the synthetic dataset.
        
        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to 1000.
            vocab_size (int, optional): Size of the vocabulary. Defaults to 1000.
            seq_length (int, optional): Length of each sequence. Defaults to 20.
            num_token_id (int, optional): Token ID for the <NUM> token. Defaults to 2.
            num_probability (float, optional): Probability of generating a <NUM> token. Defaults to 0.2.
            hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_token_id = num_token_id
        self.num_probability = num_probability
        self.hidden_size = hidden_size
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate synthetic data
        self.generate_data()
    
    def generate_data(self):
        """
        Generate synthetic data.
        
        This method creates random input features, target tokens, and target values.
        """
        # Generate random input features
        self.features = torch.randn(self.num_samples, self.hidden_size)
        
        # Generate random target tokens
        self.target_tokens = torch.randint(
            0, self.vocab_size, (self.num_samples,)
        )
        
        # Randomly select some samples to have <NUM> token as target
        num_mask = torch.rand(self.num_samples) < self.num_probability
        self.target_tokens[num_mask] = self.num_token_id
        
        # Generate random target values for all samples
        # (only relevant for samples with <NUM> token)
        self.target_values = torch.randn(self.num_samples)
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Sample containing feature, target token, and target value
        """
        return {
            "feature": self.features[idx],
            "target_token": self.target_tokens[idx],
            "target_value": self.target_values[idx]
        }


def create_synthetic_dataloader(
    num_samples=1000,
    vocab_size=1000,
    batch_size=32,
    num_token_id=2,
    num_probability=0.2,
    hidden_size=1024,
    seed=42
):
    """
    Create a DataLoader for synthetic data.
    
    Args:
        num_samples (int, optional): Number of samples to generate. Defaults to 1000.
        vocab_size (int, optional): Size of the vocabulary. Defaults to 1000.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_token_id (int, optional): Token ID for the <NUM> token. Defaults to 2.
        num_probability (float, optional): Probability of generating a <NUM> token. Defaults to 0.2.
        hidden_size (int, optional): Size of the hidden representation. Defaults to 1024.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        torch.utils.data.DataLoader: DataLoader for synthetic data
    """
    # Create synthetic dataset
    dataset = SyntheticDataset(
        num_samples=num_samples,
        vocab_size=vocab_size,
        num_token_id=num_token_id,
        num_probability=num_probability,
        hidden_size=hidden_size,
        seed=seed
    )
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    return dataloader


class TextWithNumbersGenerator:
    """
    Generator for text with embedded numerical values.
    
    This class generates synthetic text data with controlled numerical content,
    useful for testing the model's ability to handle mixed text and numerical data.
    """
    
    def __init__(self, seed=42):
        """
        Initialize the generator.
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.seed = seed
        np.random.seed(seed)
        
        # Templates for generating text
        self.templates = [
            "The price is {value} dollars.",
            "The temperature is {value} degrees.",
            "The distance is {value} kilometers.",
            "The weight is {value} kilograms.",
            "The height is {value} meters.",
            "The time is {value} hours.",
            "The speed is {value} km/h.",
            "The age is {value} years.",
            "The score is {value} points.",
            "The rate is {value} percent."
        ]
        
        # Templates for questions
        self.question_templates = [
            "What is the price in dollars?",
            "What is the temperature in degrees?",
            "What is the distance in kilometers?",
            "What is the weight in kilograms?",
            "What is the height in meters?",
            "What is the time in hours?",
            "What is the speed in km/h?",
            "What is the age in years?",
            "What is the score in points?",
            "What is the rate in percent?"
        ]
    
    def generate_value(self, min_val=0, max_val=100, decimal=True):
        """
        Generate a random numerical value.
        
        Args:
            min_val (int, optional): Minimum value. Defaults to 0.
            max_val (int, optional): Maximum value. Defaults to 100.
            decimal (bool, optional): Whether to generate decimal values. Defaults to True.
        
        Returns:
            float: Random numerical value
        """
        value = np.random.uniform(min_val, max_val)
        if not decimal:
            value = int(value)
        return value
    
    def generate_text(self, num_samples=100):
        """
        Generate synthetic text data with numerical values.
        
        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to 100.
        
        Returns:
            Tuple[List[str], List[float]]: List of texts and corresponding values
        """
        texts = []
        values = []
        
        for _ in range(num_samples):
            # Select a random template
            template = np.random.choice(self.templates)
            
            # Generate a random value
            value = self.generate_value()
            
            # Format the template with the value
            text = template.format(value=value)
            
            texts.append(text)
            values.append(value)
        
        return texts, values
    
    def generate_qa_pairs(self, num_samples=100):
        """
        Generate question-answer pairs with numerical answers.
        
        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to 100.
        
        Returns:
            Tuple[List[str], List[str], List[float]]: Lists of contexts, questions, and answers
        """
        contexts = []
        questions = []
        answers = []
        
        for _ in range(num_samples):
            # Select a random template
            template_idx = np.random.randint(0, len(self.templates))
            template = self.templates[template_idx]
            question = self.question_templates[template_idx]
            
            # Generate a random value
            value = self.generate_value()
            
            # Format the template with the value
            context = template.format(value=value)
            
            contexts.append(context)
            questions.append(question)
            answers.append(value)
        
        return contexts, questions, answers

