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
    A wrapper around the Qwen tokenizer that adds special handling for numerical values.
    
    This wrapper adds a special <NUM> token to represent numerical values in text,
    allowing the model to handle mixed text-numerical data in a unified way.
    
    重要设计决策：
    - 使用 Qwen 的完整配置容量（151,936）而非实际使用的词汇（151,665）
    - <NUM> token 占用第一个预留位置（ID: 151,665）
    - 保留剩余 270 个预留位置供未来扩展
    """
    
    def __init__(self, model_path: str = 'Qwen/Qwen2.5-0.5B', use_real_tokenizer: bool = True):
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
                
                # 使用 Qwen 的完整配置容量，而非仅实际使用的词汇
                # 这确保了与 Qwen 架构的完全兼容性
                self.vocab_size = 151936  # Qwen 的 config.vocab_size
                self.num_token_id = 151665  # 第一个预留位置
                
                # 添加必要的属性以兼容 HuggingFace 接口
                self.pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
                self.eos_token_id = self.tokenizer.eos_token_id
                self.bos_token_id = self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None
                self.unk_token_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, 'unk_token_id') else None
                
                print(f"QwenTokenizerWrapper initialized:")
                print(f"  - Qwen 配置容量: 151,936")
                print(f"  - Qwen 实际使用: 151,665") 
                print(f"  - <NUM> token ID: {self.num_token_id} (第一个预留位置)")
                print(f"  - 剩余预留: 270 个位置")
                print(f"✅ Successfully initialized with full vocabulary capacity")
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
    
    def _extract_and_replace_numbers(self, text):
        """
        Extract numerical values from text and replace them with <NUM> tokens.
        
        这个方法需要修复，确保能正确提取数值。
        """
        import re
        
        # 更全面的数字正则表达式
        # 匹配整数、小数、负数、百分数等
        number_pattern = r'-?\d+\.?\d*%?'
        
        numerical_values = []
        modified_text = text
        
        # 查找所有数字
        matches = list(re.finditer(number_pattern, text))
        
        # 从后往前替换，避免位置偏移
        for match in reversed(matches):
            number_str = match.group()
            start, end = match.span()
            
            # 解析数值
            try:
                # 处理百分号
                if number_str.endswith('%'):
                    value = float(number_str[:-1]) / 100.0
                else:
                    value = float(number_str)
                
                # 替换为 <NUM>
                modified_text = modified_text[:start] + '<NUM>' + modified_text[end:]
                # 记录数值（注意要倒序插入）
                numerical_values.insert(0, value)
            except ValueError:
                # 如果解析失败，跳过
                continue
        
        return modified_text, numerical_values
    
    def encode_plus(self, text, return_tensors=None, padding=False, truncation=False, 
                   max_length=None, return_attention_mask=True, **kwargs):
        """
        增强的编码方法，确保正确处理数值。
        """
        # 提取和替换数字
        modified_text, numerical_values = self._extract_and_replace_numbers(text)
        
        # 调试输出
        if numerical_values:
            print(f"[DEBUG] 原文: '{text}'")
            print(f"[DEBUG] 修改后: '{modified_text}'")
            print(f"[DEBUG] 提取的数值: {numerical_values}")
        
        # 使用修改后的文本进行分词
        encoding = self.tokenizer.encode_plus(
            modified_text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            **kwargs
        )
        
        # 创建数值向量
        if return_tensors == 'pt':
            seq_len = encoding['input_ids'].shape[-1]
            numerical_values_tensor = torch.zeros(seq_len, dtype=torch.float)
            
            # 找到 <NUM> token 的位置并填充数值
            input_ids = encoding['input_ids'].squeeze(0)  # 假设 batch_size=1
            num_idx = 0
            for i, token_id in enumerate(input_ids):
                if token_id == self.num_token_id and num_idx < len(numerical_values):
                    numerical_values_tensor[i] = numerical_values[num_idx]
                    num_idx += 1
            
            encoding['numerical_values'] = numerical_values_tensor.unsqueeze(0)  # 添加 batch 维度
        else:
            # 非张量返回
            encoding['numerical_values'] = [0.0] * len(encoding['input_ids'])
            # 填充数值
            num_idx = 0
            for i, token_id in enumerate(encoding['input_ids']):
                if token_id == self.num_token_id and num_idx < len(numerical_values):
                    encoding['numerical_values'][i] = numerical_values[num_idx]
                    num_idx += 1
        
        return encoding
    
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
    
    def vocab_size_info(self) -> dict:
        """返回词汇表大小的详细信息"""
        if self.use_real_tokenizer:
            return {
                'config_capacity': 151936,      # Qwen 配置容量
                'qwen_used': 151665,           # Qwen 实际使用
                'causalqwen_vocab': self.vocab_size,    # CausalQwen 使用完整容量
                'num_token_id': self.num_token_id,
                'reserved_slots': 271,         # 总预留槽位
                'reserved_used': 1,            # 已使用的预留（<NUM>）
                'reserved_remaining': 270      # 剩余预留
            }
        else:
            # Mock tokenizer 的情况
            return {
                'config_capacity': self.vocab_size,
                'qwen_used': self.vocab_size - 1,
                'causalqwen_vocab': self.vocab_size,
                'num_token_id': self.num_token_id,
                'reserved_slots': 1,
                'reserved_used': 1,
                'reserved_remaining': 0
            }

