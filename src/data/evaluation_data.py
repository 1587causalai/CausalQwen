"""
Generates standardized evaluation datasets.
"""

import torch
from torch.utils.data import TensorDataset
from .synthetic import TextWithNumbersGenerator

def _create_dataset_from_generator(generator, tokenizer, num_samples):
    """Helper function to create a TensorDataset from a text generator."""
    texts, true_values = generator.generate_text(num_samples=num_samples)
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    numerical_values = inputs['numerical_values']
    
    targets = torch.full((num_samples,), tokenizer.num_token_id, dtype=torch.long)
    target_values = torch.tensor(true_values, dtype=torch.float32)
    
    return TensorDataset(input_ids, attention_mask, numerical_values, targets, target_values)

def get_all_evaluation_datasets(tokenizer, num_samples_per_set=200):
    """
    Get a dictionary of all standard evaluation datasets.
    
    Args:
        tokenizer: The tokenizer to use for creating datasets.
        num_samples_per_set (int): Number of samples for each dataset type.
        
    Returns:
        dict: A dictionary mapping dataset names to TensorDataset objects.
    """
    
    datasets = {}
    
    # 1. Basic text-number data
    basic_gen = TextWithNumbersGenerator(seed=101)
    datasets['basic'] = _create_dataset_from_generator(basic_gen, tokenizer, num_samples_per_set)
    
    # 2. Question-answering data
    qa_gen = TextWithNumbersGenerator(
        seed=102,
        templates=[
            "Q: What is the value? A: It is {num}.",
            "The answer to the question is {num}.",
        ]
    )
    datasets['question_answering'] = _create_dataset_from_generator(qa_gen, tokenizer, num_samples_per_set)
    
    # 3. Extreme value data
    extreme_gen = TextWithNumbersGenerator(
        seed=103,
        number_range=(1e-5, 1e5)
    )
    datasets['extreme_values'] = _create_dataset_from_generator(extreme_gen, tokenizer, num_samples_per_set)

    # 4. Boundary value data (e.g., 0 and 1)
    boundary_gen = TextWithNumbersGenerator(
        seed=104,
        number_range=(0.0, 1.0)
    )
    datasets['boundary_values'] = _create_dataset_from_generator(boundary_gen, tokenizer, num_samples_per_set)
    
    return datasets 