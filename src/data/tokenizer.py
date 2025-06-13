"""
Tokenizer module.

This module implements the tokenizer for the causal language model,
with special handling for the <NUM> token.
"""

import torch
import re
from typing import List, Tuple, Dict, Optional, Union
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoConfig


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
        
        # --- Vocab Size Calculation ---
        # 0. Capture the full, configured vocabulary size of the base Qwen model. This includes all reserved slots.
        self._qwen_config_vocab_size = self.tokenizer.vocab_size

        self.num_token = "<NUM>"
        
        # 1. Get the base Qwen vocabulary size before any additions. This might be different from capacity.
        self._qwen_base_vocab_size = len(self.tokenizer)

        # 2. Add the <NUM> token.
        if self.num_token not in self.tokenizer.vocab:
            self.tokenizer.add_special_tokens({'additional_special_tokens': [self.num_token]})
        
        # 3. This is the "official" vocabulary size for the CausalQwen model.
        self._causalqwen_vocab_size = len(self.tokenizer)
        
        self.num_token_id = self.tokenizer.convert_tokens_to_ids(self.num_token)
        
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        # 4. A regex to find floats, supporting scientific notation.
        self.float_pattern = re.compile(r"[-+]?\d*\.\d+|\d+\.?\d*e[-+]?\d+|[-+]?\d+")
        
        # 5. The final length of the tokenizer object, for internal use.
        self._tokenizer_internal_len = len(self.tokenizer)

        # Regex to find numbers (integers, floats, negative numbers)
        self.number_pattern = re.compile(r'(?<![a-zA-Z0-9_])(-?\d+(\.\d+)?)(?![a-zA-Z0-9_])')
        
        # 4. Add an internal placeholder token for processing. This is not part of the public model vocab.
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
        Provides clear and accurate information about the tokenizer's vocabulary sizes.
        
        - qwen_base_vocab: The original vocabulary size of the Qwen model.
        - causalqwen_vocab: The base size + 1 (for the <NUM> token). This is the official model vocab size.
        - config_capacity: The full capacity of the original Qwen model
        - tokenizer_internal_len: The CausalQwen size + 1 (for the internal placeholder). Used by the tokenizer only.
        """
        return {
            "qwen_base_vocab": self._qwen_base_vocab_size,
            "causalqwen_vocab": self._causalqwen_vocab_size,
            "config_capacity": self._qwen_config_vocab_size, # The full capacity of the original Qwen model
            "tokenizer_internal_len": self._tokenizer_internal_len,
            "num_token_id": self.num_token_id,
        }

    @property
    def vocab_size(self) -> int:
        return self._causalqwen_vocab_size

