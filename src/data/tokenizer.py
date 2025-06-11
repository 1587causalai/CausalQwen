"""
Tokenizer module.

This module implements the tokenizer for the causal language model,
with special handling for the <NUM> token.
"""

import torch
import re
from typing import List, Tuple, Dict, Optional, Union


class MockTokenizer:
    """
    Mock tokenizer for the causal language model.
    
    This is a simplified tokenizer that handles basic tokenization
    and special treatment for numerical values.
    """
    
    def __init__(self, vocab_size=10000):
        """
        Initialize the mock tokenizer.
        
        Args:
            vocab_size (int, optional): Size of the vocabulary. Defaults to 10000.
        """
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.num_token = "<NUM>"
        
        # Token IDs
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.num_token_id = 2
        
        # Simple vocabulary (for demonstration)
        self.vocab = {
            self.pad_token: self.pad_token_id,
            self.unk_token: self.unk_token_id,
            self.num_token: self.num_token_id,
        }
        
        # Add some dummy words to the vocabulary
        for i in range(3, min(1000, vocab_size)):
            word = f"word_{i}"
            self.vocab[word] = i
        
        # Reverse mapping from ID to token
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Regular expression for identifying numbers
        self.number_pattern = re.compile(r'\b\d+(\.\d+)?\b')
    
    def __call__(self, *args, **kwargs):
        """Alias for batch_encode_plus."""
        return self.batch_encode_plus(*args, **kwargs)

    def tokenize(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Tokenize text into tokens, with special handling for numbers.
        
        Args:
            text (str): Input text
        
        Returns:
            Tuple[List[str], List[float]]: Tokens and corresponding numerical values
        """
        # Initialize lists for tokens and numerical values
        tokens = []
        numerical_values = []
        
        # Find all numbers in the text
        number_matches = list(self.number_pattern.finditer(text))
        
        # If no numbers, tokenize normally
        if not number_matches:
            # Simple whitespace tokenization for demonstration
            tokens = text.split()
            # Add new words to vocab
            for token in tokens:
                if token not in self.vocab:
                    new_id = len(self.vocab)
                    if new_id < self.vocab_size:
                        self.vocab[token] = new_id
                        self.id_to_token[new_id] = token
            numerical_values = [0.0] * len(tokens)
            return tokens, numerical_values
        
        # Process text with numbers
        last_end = 0
        for match in number_matches:
            # Add tokens before the number
            if match.start() > last_end:
                prefix_text = text[last_end:match.start()]
                prefix_tokens = prefix_text.split()
                tokens.extend(prefix_tokens)
                numerical_values.extend([0.0] * len(prefix_tokens))
            
            # Add the number as a special token
            tokens.append(self.num_token)
            numerical_values.append(float(match.group()))
            
            last_end = match.end()
        
        # Add tokens after the last number
        if last_end < len(text):
            suffix_text = text[last_end:]
            suffix_tokens = suffix_text.split()
            tokens.extend(suffix_tokens)
            numerical_values.extend([0.0] * len(suffix_tokens))
        
        return tokens, numerical_values
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to token IDs.
        
        Args:
            tokens (List[str]): List of tokens
        
        Returns:
            List[int]: List of token IDs
        """
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert token IDs to tokens.
        
        Args:
            ids (List[int]): List of token IDs
        
        Returns:
            List[str]: List of tokens
        """
        return [self.id_to_token.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, return_tensors: Optional[str] = None) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Encode text into token IDs and numerical values.
        
        Args:
            text (str): Input text
            return_tensors (Optional[str], optional): If "pt", return PyTorch tensors. Defaults to None.
        
        Returns:
            Dict[str, Union[List, torch.Tensor]]: Dictionary containing token IDs and numerical values
        """
        # Tokenize text
        tokens, numerical_values = self.tokenize(text)
        
        # Convert tokens to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Create attention mask (all 1s for simplicity)
        attention_mask = [1] * len(input_ids)
        
        # Prepare output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numerical_values": numerical_values
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            output["input_ids"] = torch.tensor(input_ids)
            output["attention_mask"] = torch.tensor(attention_mask)
            output["numerical_values"] = torch.tensor(numerical_values)
        
        return output
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.
        
        Returns:
            str: Decoded text
        """
        # Convert IDs to tokens
        tokens = self.convert_ids_to_tokens(token_ids)
        
        # Skip special tokens if requested
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in [self.pad_token, self.unk_token]]
        
        # Join tokens with spaces
        text = " ".join(tokens)
        
        return text
    
    def batch_encode_plus(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Encode a batch of texts.
        
        Args:
            texts (List[str]): List of input texts
            max_length (Optional[int], optional): Maximum sequence length. Defaults to None.
            padding (bool, optional): Whether to pad sequences. Defaults to False.
            truncation (bool, optional): Whether to truncate sequences. Defaults to False.
            return_tensors (Optional[str], optional): If "pt", return PyTorch tensors. Defaults to None.
        
        Returns:
            Dict[str, Union[List, torch.Tensor]]: Dictionary containing batched inputs
        """
        # Encode each text
        encoded_texts = [self.encode(text) for text in texts]
        
        # Get maximum sequence length
        if max_length is None and padding:
            max_length = max(len(encoded["input_ids"]) for encoded in encoded_texts)
        
        # Initialize batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "numerical_values": []
        }
        
        # Process each encoded text
        for encoded in encoded_texts:
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            numerical_values = encoded["numerical_values"]
            
            # Truncate if needed
            if truncation and max_length is not None and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
                numerical_values = numerical_values[:max_length]
            
            # Pad if needed
            if padding and max_length is not None:
                padding_length = max_length - len(input_ids)
                if padding_length > 0:
                    input_ids = input_ids + [self.pad_token_id] * padding_length
                    attention_mask = attention_mask + [0] * padding_length
                    numerical_values = numerical_values + [0.0] * padding_length
            
            # Add to batch
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["numerical_values"].append(numerical_values)
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            batch["input_ids"] = torch.tensor(batch["input_ids"])
            batch["attention_mask"] = torch.tensor(batch["attention_mask"])
            batch["numerical_values"] = torch.tensor(batch["numerical_values"])
        
        return batch


class QwenTokenizerWrapper:
    """
    Wrapper for Qwen tokenizer to make it compatible with our data pipeline.
    
    This wrapper adds support for the <NUM> token and numerical value handling.
    """
    
    def __init__(self, model_path="~/models/Qwen2.5-0.5B", use_real_tokenizer=True):
        """
        Initialize the Qwen tokenizer wrapper.
        
        Args:
            model_path (str): Path to the Qwen model directory
            use_real_tokenizer (bool): Whether to use the real Qwen tokenizer or mock implementation
        """
        self.model_path = model_path
        self.use_real_tokenizer = use_real_tokenizer
        
        # Special tokens
        self.num_token = "<NUM>"
        
        if use_real_tokenizer:
            try:
                from transformers import AutoTokenizer
                import os
                
                # Expand the tilde in the path
                expanded_path = os.path.expanduser(model_path)
                
                print(f"Loading Qwen tokenizer from {expanded_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
                
                # Record original vocab size
                original_vocab_size = len(self.tokenizer)
                print(f"Original Qwen vocab size: {original_vocab_size}")
                
                # Add the <NUM> token ONLY if it doesn't exist
                if self.num_token not in self.tokenizer.get_vocab():
                    # Add exactly one new token
                    num_new_tokens = self.tokenizer.add_tokens([self.num_token])
                    print(f"Added {num_new_tokens} new token(s): {self.num_token}")
                else:
                    print(f"{self.num_token} token already exists in vocabulary")
                
                # Get final vocab size and token ID
                self.vocab_size = len(self.tokenizer)
                self.num_token_id = self.tokenizer.convert_tokens_to_ids(self.num_token)
                
                # Verify the addition was successful
                expected_vocab_size = original_vocab_size + (1 if self.num_token not in self.tokenizer.get_vocab() else 0)
                print(f"Final vocab size: {self.vocab_size}")
                print(f"Expected vocab size: {original_vocab_size} + 1 = {original_vocab_size + 1}")
                print(f"<NUM> token ID: {self.num_token_id}")
                
                if self.vocab_size == original_vocab_size + 1:
                    print("✅ Vocabulary size correctly increased by 1")
                elif self.vocab_size == original_vocab_size:
                    print("⚠️ Vocabulary size unchanged (token already existed)")
                else:
                    print(f"❌ Unexpected vocab size change: {original_vocab_size} → {self.vocab_size}")
                
                # Expose pad_token_id for convenience
                self.pad_token_id = self.tokenizer.pad_token_id
                
                print(f"Successfully loaded Qwen tokenizer with vocab size {self.vocab_size}")
                
            except Exception as e:
                print(f"Failed to load Qwen tokenizer: {e}")
                print("Falling back to mock implementation")
                self.use_real_tokenizer = False
                self._init_mock()
        else:
            self._init_mock()
    
    def _init_mock(self):
        """Initialize mock tokenizer as fallback."""
        self.mock_tokenizer = MockTokenizer()
        self.num_token_id = self.mock_tokenizer.num_token_id
        self.vocab_size = self.mock_tokenizer.vocab_size
    
    def __call__(self, *args, **kwargs):
        """Alias for batch_encode_plus."""
        return self.batch_encode_plus(*args, **kwargs)
    
    def tokenize_with_numbers(self, text: str):
        """
        Tokenize text with special handling for numbers.
        
        Args:
            text (str): Input text
        
        Returns:
            Tuple[List[str], List[float]]: Tokens and corresponding numerical values
        """
        if not self.use_real_tokenizer:
            return self.mock_tokenizer.tokenize(text)
        
        # Use regex to find numbers and replace with <NUM> token
        import re
        number_pattern = re.compile(r'\b\d+(\.\d+)?\b')
        
        # Find all numbers and their values
        numbers = []
        number_positions = []
        
        for match in number_pattern.finditer(text):
            numbers.append(float(match.group()))
            number_positions.append((match.start(), match.end()))
        
        # Replace numbers with <NUM> token
        modified_text = text
        offset = 0
        for i, (start, end) in enumerate(number_positions):
            # Adjust positions due to previous replacements
            adjusted_start = start + offset
            adjusted_end = end + offset
            
            # Calculate the offset change
            original_length = end - start
            new_length = len(self.num_token)
            offset += new_length - original_length
            
            # Replace the number with <NUM> token
            modified_text = modified_text[:adjusted_start] + self.num_token + modified_text[adjusted_end:]
        
        # Tokenize the modified text
        tokens = self.tokenizer.tokenize(modified_text)
        
        # Create numerical values array
        numerical_values = []
        number_idx = 0
        
        for token in tokens:
            if token == self.num_token and number_idx < len(numbers):
                numerical_values.append(numbers[number_idx])
                number_idx += 1
            else:
                numerical_values.append(0.0)
        
        return tokens, numerical_values
    
    def encode(self, text: str, return_tensors: Optional[str] = None, max_length: Optional[int] = None):
        """
        Encode text into token IDs and numerical values.
        
        Args:
            text (str): Input text
            return_tensors (Optional[str], optional): If "pt", return PyTorch tensors. Defaults to None.
            max_length (Optional[int], optional): Maximum sequence length
        
        Returns:
            Dict[str, Union[List, torch.Tensor]]: Dictionary containing token IDs and numerical values
        """
        if not self.use_real_tokenizer:
            return self.mock_tokenizer.encode(text, return_tensors)
        
        # Tokenize with number handling
        tokens, numerical_values = self.tokenize_with_numbers(text)
        
        # Convert tokens to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Handle truncation
        if max_length is not None and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            numerical_values = numerical_values[:max_length]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Prepare output
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "numerical_values": numerical_values
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            output["input_ids"] = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
            output["attention_mask"] = torch.tensor(attention_mask).unsqueeze(0)
            output["numerical_values"] = torch.tensor(numerical_values).unsqueeze(0)
        
        return output
    
    def batch_encode_plus(
        self, 
        texts: List[str], 
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Union[List, torch.Tensor]]:
        """
        Batch encode multiple texts.
        
        Args:
            texts (List[str]): List of input texts
            max_length (Optional[int], optional): Maximum sequence length
            padding (bool, optional): Whether to pad sequences. Defaults to False.
            truncation (bool, optional): Whether to truncate sequences. Defaults to False.
            return_tensors (Optional[str], optional): Return format. Defaults to None.
        
        Returns:
            Dict[str, Union[List, torch.Tensor]]: Batch of encoded sequences
        """
        if not self.use_real_tokenizer:
            return self.mock_tokenizer.batch_encode_plus(texts, max_length, padding, truncation, return_tensors)
        
        # Encode each text individually
        batch_input_ids = []
        batch_attention_mask = []
        batch_numerical_values = []
        
        for text in texts:
            encoded = self.encode(text, max_length=max_length if truncation else None)
            batch_input_ids.append(encoded["input_ids"])
            batch_attention_mask.append(encoded["attention_mask"])
            batch_numerical_values.append(encoded["numerical_values"])
        
        # Handle padding
        if padding:
            # If max_length is not specified, use the maximum length in the batch
            if max_length is None:
                max_length = max(len(seq) for seq in batch_input_ids)
            
            # Pad sequences to max_length
            for i in range(len(batch_input_ids)):
                current_length = len(batch_input_ids[i])
                if current_length < max_length:
                    pad_length = max_length - current_length
                    batch_input_ids[i].extend([self.tokenizer.pad_token_id] * pad_length)
                    batch_attention_mask[i].extend([0] * pad_length)
                    batch_numerical_values[i].extend([0.0] * pad_length)
                elif current_length > max_length and truncation:
                    # Truncate if necessary
                    batch_input_ids[i] = batch_input_ids[i][:max_length]
                    batch_attention_mask[i] = batch_attention_mask[i][:max_length]
                    batch_numerical_values[i] = batch_numerical_values[i][:max_length]
        
        # Prepare output
        output = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "numerical_values": batch_numerical_values
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            output["input_ids"] = torch.tensor(batch_input_ids)
            output["attention_mask"] = torch.tensor(batch_attention_mask)
            output["numerical_values"] = torch.tensor(batch_numerical_values)
        
        return output
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids (List[int]): List of token IDs
            skip_special_tokens (bool, optional): Whether to skip special tokens. Defaults to True.
        
        Returns:
            str: Decoded text
        """
        # We need to filter out the <NUM> token manually before decoding
        # as the base tokenizer doesn't know about it.
        filtered_ids = [tid for tid in token_ids if tid != self.num_token_id]
        
        # Use the real tokenizer to decode
        return self.tokenizer.decode(filtered_ids, skip_special_tokens=skip_special_tokens)

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Converts a sequence of ids in a list of tokens.
        This is a convenience method to expose the underlying tokenizer's functionality.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

