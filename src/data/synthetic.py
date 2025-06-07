"""
Synthetic data generation module.

This module provides classes for generating synthetic data for testing and validating
the causal language model. It includes generators for basic text-number pairs,
question-answer pairs, multi-number texts, noisy data, and extreme value data.
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Union


class TextWithNumbersGenerator:
    """
    A flexible generator for creating texts containing numbers.
    This generator can be configured for various ranges, templates, and formats.
    """
    
    def __init__(self, seed: int = 42, number_range: Tuple[float, float] = (0, 100), templates: Optional[List[str]] = None):
        """
        Initialize the generator.
        
        Args:
            seed (int): Random seed for reproducibility.
            number_range (Tuple[float, float]): A tuple (min, max) for the range of generated numbers.
            templates (Optional[List[str]]): A list of string templates to use. If None, default templates are used.
        """
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.number_range = number_range
        
        if templates is None:
            self.templates = [
                "The price is {num} dollars.",
                "The temperature is {num} degrees Celsius.",
                "The result of the measurement is {num}.",
                "Value: {num}",
                "An observation of {num} was recorded.",
            ]
        else:
            self.templates = templates

    def generate_text(self, num_samples: int = 100) -> Tuple[List[str], List[float]]:
        """
        Generate a list of texts with embedded numbers.
        
        Args:
            num_samples (int): The number of samples to generate.
            
        Returns:
            Tuple[List[str], List[float]]: A tuple containing the list of texts and the list of corresponding values.
        """
        texts = []
        values = []
        
        for _ in range(num_samples):
            value = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            text = template.format(num=value)
            
            texts.append(text)
            values.append(value)
        
        return texts, values

    def generate_qa_pairs(self, num_samples: int = 100) -> Tuple[List[str], List[str], List[float]]:
        """
        Generate context-question-answer triplets.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            contexts: List of context texts
            questions: List of questions
            answers: List of numerical answers
        """
        contexts = []
        questions = []
        answers = []
        
        for _ in range(num_samples):
            value = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            
            context = template.format(num=value)
            question = "What is the value of the number in the text?"
            
            contexts.append(context)
            questions.append(question)
            answers.append(value)
        
        return contexts, questions, answers

    def generate_multi_number_text(self, num_samples: int = 100) -> Tuple[List[str], List[Tuple[float, float]]]:
        """
        Generate texts with multiple numerical values.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            texts: List of generated texts
            value_pairs: List of value pairs (value1, value2)
        """
        texts = []
        value_pairs = []
        
        for _ in range(num_samples):
            value1 = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            value2 = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            text = template.format(num=value1)
            
            texts.append(text)
            value_pairs.append((value1, value2))
        
        return texts, value_pairs

    def generate_noisy_text(self, num_samples: int = 100, noise_level: float = 0.1) -> Tuple[List[str], List[float], List[float]]:
        """
        Generate noisy data.
        
        Args:
            num_samples: Number of samples to generate
            noise_level: Level of noise to add (as a fraction of the value)
            
        Returns:
            texts: List of generated texts (with noisy values)
            true_values: List of true values (without noise)
            noisy_values: List of noisy values (as they appear in the texts)
        """
        texts = []
        true_values = []
        noisy_values = []
        
        for _ in range(num_samples):
            value = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            
            # Add noise
            noise = self.rng.uniform(-noise_level * value, noise_level * value)
            noisy_value = value + noise
            
            text = template.format(num=noisy_value)
            
            texts.append(text)
            true_values.append(value)
            noisy_values.append(noisy_value)
        
        return texts, true_values, noisy_values

    def generate_extreme_text(self, num_samples: int = 100) -> Tuple[List[str], List[float]]:
        """
        Generate data with extreme values.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            texts: List of generated texts
            values: List of corresponding numerical values
        """
        texts = []
        values = []
        
        # Generate normal values (80% of samples)
        for _ in range(int(num_samples * 0.8)):
            value = round(self.rng.uniform(self.number_range[0], self.number_range[1]), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            text = template.format(num=value)
            
            texts.append(text)
            values.append(value)
        
        # Generate extreme values (20% of samples)
        for _ in range(int(num_samples * 0.2)):
            # Use power law distribution to generate extreme values
            value = round(100 * (1 / (1 - self.rng.random())), self.rng.randint(0, 2))
            template = self.rng.choice(self.templates)
            text = template.format(num=value)
            
            texts.append(text)
            values.append(value)
        
        # Shuffle data
        combined = list(zip(texts, values))
        self.rng.shuffle(combined)
        texts, values = zip(*combined)
        
        return list(texts), list(values)

    def generate_mixed_dataset(self, num_samples: int = 1000) -> Dict[str, Union[List[str], List[float]]]:
        """
        Generate a mixed dataset with various types of data.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            dataset: Dictionary containing various types of data
        """
        # Allocate samples to different generators
        basic_samples = int(num_samples * 0.4)
        qa_samples = int(num_samples * 0.2)
        multi_samples = int(num_samples * 0.2)
        noisy_samples = int(num_samples * 0.1)
        extreme_samples = int(num_samples * 0.1)
        
        # Generate data
        basic_texts, basic_values = self.generate_text(basic_samples)
        contexts, questions, answers = self.generate_qa_pairs(qa_samples)
        multi_texts, multi_values = self.generate_multi_number_text(multi_samples)
        noisy_texts, true_values, noisy_values = self.generate_noisy_text(noisy_samples)
        extreme_texts, extreme_values = self.generate_extreme_text(extreme_samples)
        
        # Combine data
        all_texts = basic_texts + contexts + questions + multi_texts + noisy_texts + extreme_texts
        all_values = basic_values + answers + [v[0] for v in multi_values] + true_values + extreme_values
        
        # Create dataset
        dataset = {
            'texts': all_texts,
            'values': all_values,
            'basic_texts': basic_texts,
            'basic_values': basic_values,
            'contexts': contexts,
            'questions': questions,
            'answers': answers,
            'multi_texts': multi_texts,
            'multi_values': multi_values,
            'noisy_texts': noisy_texts,
            'true_values': true_values,
            'noisy_values': noisy_values,
            'extreme_texts': extreme_texts,
            'extreme_values': extreme_values
        }
        
        return dataset


class SyntheticDataset:
    """
    Synthetic dataset for training and evaluating the causal language model.
    
    This class provides a PyTorch-compatible dataset interface for the synthetic data.
    """
    
    def __init__(self, num_samples: int = 1000, vocab_size: int = 1000, hidden_size: int = 1024, seed: int = 42):
        """
        Initialize the synthetic dataset.
        
        Args:
            num_samples: Number of samples in the dataset
            vocab_size: Size of the vocabulary
            hidden_size: Dimension of the feature vectors
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seed = seed
        
        # Initialize random generators
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        
        # Generate synthetic data
        self.features = self.np_rng.randn(num_samples, hidden_size)
        self.target_tokens = self.np_rng.randint(0, vocab_size, num_samples)
        self.target_values = self.np_rng.uniform(0, 100, num_samples)
    
    def __len__(self) -> int:
        """
        Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Sample dictionary with feature, target_token, and target_value
        """
        return {
            'feature': self.features[idx],
            'target_token': self.target_tokens[idx],
            'target_value': self.target_values[idx]
        }


if __name__ == "__main__":
    # Test the generators
    generator = TextWithNumbersGenerator(seed=42)
    
    # Test basic text generation
    texts, values = generator.generate_text(num_samples=5)
    print("Basic text-number pairs:")
    for text, value in zip(texts, values):
        print(f"  {text} (value: {value})")
    
    # Test QA pair generation
    contexts, questions, answers = generator.generate_qa_pairs(num_samples=5)
    print("\nQuestion-answer pairs:")
    for context, question, answer in zip(contexts, questions, answers):
        print(f"  Context: {context}")
        print(f"  Question: {question}")
        print(f"  Answer: {answer}")
    
    # Test multi-number text generation
    texts, value_pairs = generator.generate_multi_number_text(num_samples=5)
    print("\nMulti-number texts:")
    for text, (value1, value2) in zip(texts, value_pairs):
        print(f"  {text} (values: {value1}, {value2})")
    
    # Test noisy text generation
    texts, true_values, noisy_values = generator.generate_noisy_text(num_samples=5, noise_level=0.2)
    print("\nNoisy texts:")
    for text, true_value, noisy_value in zip(texts, true_values, noisy_values):
        print(f"  {text} (true: {true_value}, noisy: {noisy_value:.2f})")
    
    # Test extreme value text generation
    texts, values = generator.generate_extreme_text(num_samples=5)
    print("\nExtreme value texts:")
    for text, value in zip(texts, values):
        print(f"  {text} (value: {value})")
    
    # Test synthetic dataset
    dataset = SyntheticDataset(num_samples=5)
    print("\nSynthetic dataset:")
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"  Sample {i+1}:")
        print(f"    Feature shape: {sample['feature'].shape}")
        print(f"    Target token: {sample['target_token']}")
        print(f"    Target value: {sample['target_value']:.2f}")

