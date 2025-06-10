"""
Generates standardized evaluation datasets.
"""

import torch
from torch.utils.data import TensorDataset
from .synthetic import TextWithNumbersGenerator

def _create_dataset_from_generator(generator, tokenizer, num_samples):
    """Helper function to create a TensorDataset from a text generator."""
    texts, true_values = generator.generate_text(num_samples=num_samples)
    
    # Use batch_encode_plus for consistent processing
    inputs = tokenizer.batch_encode_plus(texts, padding=True, truncation=True, return_tensors='pt')
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    numerical_values = inputs['numerical_values']
    
    # --- Create sequence-to-sequence labels and targets ---
    # Labels: each position predicts the next token
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding positions
    labels[:, :-1] = labels[:, 1:].clone()  # Shift left: predict next token
    labels[:, -1] = -100  # Last position has no next token to predict
    
    # Target values: numerical values for positions where label is <NUM>
    target_values = torch.full_like(numerical_values, float('nan'))
    shifted_numerical_values = numerical_values.clone()
    shifted_numerical_values[:, :-1] = numerical_values[:, 1:].clone()  # Shift left 
    shifted_numerical_values[:, -1] = 0.0  # Last position default
    
    # Only set target values where the label is <NUM> 
    num_mask = (labels == tokenizer.num_token_id)
    target_values[num_mask] = shifted_numerical_values[num_mask]
    
    return TensorDataset(input_ids, attention_mask, numerical_values, labels, target_values)

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