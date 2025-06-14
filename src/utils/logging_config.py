"""统一的日志配置"""

import logging
import warnings
import os

def setup_logging(verbose: bool = False):
    """设置统一的日志配置
    
    Args:
        verbose: 是否显示详细日志
    """
    # 设置基础日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s',  # 简化格式，只显示消息
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 抑制特定库的警告
    if not verbose:
        # 抑制 transformers 的冗余日志
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        
        # 抑制 PyTorch 分布式警告
        warnings.filterwarnings("ignore", message="NOTE: Redirects are currently not supported")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
        
        # 抑制 torch warnings
        import torch
        torch.set_warn_always(False)
        
        # 抑制 tokenizers 警告
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # 抑制 wandb 的一些信息
        logging.getLogger("wandb").setLevel(logging.WARNING)
        os.environ['WANDB_SILENT'] = 'true'
        
        # 抑制 transformers 的特殊 token 警告
        warnings.filterwarnings("ignore", message="Special tokens have been added in the vocabulary")
        
        # 抑制 past_key_values 警告
        warnings.filterwarnings("ignore", message="We detected that you are passing `past_key_values` as a tuple")
