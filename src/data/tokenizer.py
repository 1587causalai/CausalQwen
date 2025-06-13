"""
Tokenizer module.

This module implements the tokenizer for the causal language model,
with special handling for the <NUM> token.
"""

import torch
import re
from typing import List, Tuple, Dict, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizerFast


class QwenTokenizerWrapper:
    """
    A wrapper for the Qwen tokenizer to add special handling for numerical values.

    This class uses the original Qwen tokenizer and extends its functionality
    to recognize, extract, and represent numerical values within text as a special
    <NUM> token, while storing the actual float value in a separate tensor.
    """
    
    def __init__(self, model_path: str, use_real_tokenizer: bool = True):
        """
        Initializes the QwenTokenizerWrapper.

        Args:
            model_path (str): The path to the pretrained Qwen model.
            use_real_tokenizer (bool): Flag to use the real tokenizer. Currently only supports True.
        """
        if not use_real_tokenizer:
            raise NotImplementedError("MockTokenizer is no longer supported. Please use the real tokenizer.")

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.num_token = "<NUM>"
        
        # Add the <NUM> token if it's not already there.
        if self.num_token not in self.tokenizer.vocab:
            # The official Qwen tokenizer has 271 reserved tokens for expansion.
            # We use the first one for our <NUM> token.
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.num_token]})
        
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(self.num_token)
        
        # Regex to find numbers (integers, floats, negative numbers)
        # It uses negative lookbehind and lookahead to avoid matching numbers that are part of other words (e.g. 'Qwen2.5')
        self.number_pattern = re.compile(r'(?<![a-zA-Z0-9_])(-?\d+(\.\d+)?)(?![a-zA-Z0-9_])')
        
        # Use a unique placeholder that is unlikely to be in the vocabulary
        self.placeholder = "_NUM_HOLDER_"
        self.tokenizer.add_tokens([self.placeholder], special_tokens=True)
        self.placeholder_id = self.tokenizer.convert_tokens_to_ids(self.placeholder)


    def __call__(self, *args, **kwargs):
        """Alias for batch_encode_plus."""
        return self.batch_encode_plus(*args, **kwargs)

    def _preprocess_texts(self, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Replaces numbers with a placeholder and extracts them.
        """
        processed_texts = []
        all_numerical_values = []
        for text in texts:
            numerical_values = [float(m.group(1)) for m in self.number_pattern.finditer(text)]
            all_numerical_values.append(numerical_values)
            
            processed_text = self.number_pattern.sub(f" {self.placeholder} ", text)
            processed_texts.append(processed_text)
            
        return processed_texts, all_numerical_values

    def batch_encode_plus(
        self,
        texts: List[str],
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Encodes a batch of texts, handling numerical values.

        The process is as follows:
        1. Pre-process texts to replace numbers with a placeholder and extract their values.
        2. Use the standard tokenizer to encode the processed texts.
        3. Post-process the encoded inputs to replace placeholder IDs with <NUM> token IDs
           and create the `numerical_values` tensor.
        """
        processed_texts, all_numerical_values = self._preprocess_texts(texts)

        # Standard encoding using the Qwen tokenizer
        inputs = self.tokenizer(
            processed_texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors="pt" # Always return PT tensors for internal processing
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Initialize numerical_values tensor with zeros
        numerical_values_tensor = torch.zeros_like(input_ids, dtype=torch.float32)

        # Post-processing to handle the placeholders
        for i in range(input_ids.size(0)):
            # Find all placeholder locations for the current sample
            placeholder_indices = (input_ids[i] == self.placeholder_id).nonzero(as_tuple=True)[0]
            
            # Ensure we found the same number of placeholders as extracted numbers
            if len(placeholder_indices) != len(all_numerical_values[i]):
                print(f"Warning: Mismatch in sample {i}. Found {len(placeholder_indices)} placeholders but extracted {len(all_numerical_values[i])} numbers. This can happen with complex tokenization.")
                # We will only fill the values for which we found a placeholder
                num_to_fill = min(len(placeholder_indices), len(all_numerical_values[i]))
            else:
                num_to_fill = len(placeholder_indices)


            # Replace placeholder ID with <NUM> ID and populate the numerical_values tensor
            if num_to_fill > 0:
                input_ids[i, placeholder_indices[:num_to_fill]] = self.num_token_id
                numerical_values_tensor[i, placeholder_indices[:num_to_fill]] = torch.tensor(all_numerical_values[i][:num_to_fill], dtype=torch.float32)

        final_output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numerical_values": numerical_values_tensor,
        }
        
        # If the user did not want tensors, convert back.
        if return_tensors is None:
            return {k: v.tolist() for k, v in final_output.items()}
        
        return final_output

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back to a string.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Converts a list of token IDs to a list of tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def vocab_size_info(self) -> dict:
        """
        Provides information about the tokenizer's vocabulary size.
        """
        base_vocab_size = self.tokenizer.vocab_size
        if self.num_token in self.tokenizer.get_added_vocab():
             base_vocab_size -= 1
        
        return {
            "qwen_vocab_original": self.tokenizer.vocab_size,
            "causalqwen_vocab": len(self.tokenizer),
            "num_token_id": self.num_token_id
        }

