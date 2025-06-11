#!/usr/bin/env python3
"""
探索Qwen模型中271个预留token的行为

这个脚本将深入分析这271个预留token在训练和推理中的行为，
包括它们的权重初始化、梯度更新、以及实际使用情况。
"""

import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

def explore_reserved_tokens():
    """探索预留token的详细行为"""
    
    print("🔍 探索Qwen模型中271个预留token的行为")
    print("=" * 60)
    
    # 加载模型和分词器
    model_path = "~/models/Qwen2.5-0.5B"
    expanded_path = os.path.expanduser(model_path)
    
    print(f"📁 加载模型: {expanded_path}")
    
    try:
        # 加载分词器和模型
        tokenizer = AutoTokenizer.from_pretrained(expanded_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            expanded_path, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None
        )
        
        # 基本信息
        config = model.config
        lm_head = model.lm_head
        
        config_vocab_size = config.vocab_size      # 151,936
        actual_vocab_size = len(tokenizer)         # 151,665
        reserved_count = config_vocab_size - actual_vocab_size  # 271
        
        print(f"✅ 基本信息:")
        print(f"   配置词汇表大小: {config_vocab_size}")
        print(f"   实际词汇表大小: {actual_vocab_size}")
        print(f"   预留token数量: {reserved_count}")
        
        # 分析lm_head权重
        lm_head_weight = lm_head.weight.data  # [151936, 896]
        
        # 分离实际使用的权重和预留的权重
        used_weights = lm_head_weight[:actual_vocab_size, :]      # [151665, 896]
        reserved_weights = lm_head_weight[actual_vocab_size:, :]  # [271, 896]
        
        print(f"\n📊 权重分析:")
        print(f"   实际使用权重形状: {used_weights.shape}")
        print(f"   预留权重形状: {reserved_weights.shape}")
        
        # 统计分析
        print(f"\n📈 权重统计对比:")
        print(f"   实际使用权重:")
        print(f"     均值: {used_weights.mean().item():.6f}")
        print(f"     标准差: {used_weights.std().item():.6f}")
        print(f"     最小值: {used_weights.min().item():.6f}")
        print(f"     最大值: {used_weights.max().item():.6f}")
        
        print(f"   预留权重:")
        print(f"     均值: {reserved_weights.mean().item():.6f}")
        print(f"     标准差: {reserved_weights.std().item():.6f}")
        print(f"     最小值: {reserved_weights.min().item():.6f}")
        print(f"     最大值: {reserved_weights.max().item():.6f}")
        
        # 检查预留权重是否为零
        is_zero = torch.all(reserved_weights == 0)
        print(f"   预留权重是否全为零: {'是' if is_zero else '否'}")
        
        if not is_zero:
            # 计算非零元素比例
            non_zero_count = torch.sum(reserved_weights != 0).item()
            total_elements = reserved_weights.numel()
            non_zero_ratio = non_zero_count / total_elements
            print(f"   预留权重非零元素比例: {non_zero_ratio:.4f} ({non_zero_count}/{total_elements})")
        
        # 测试预留token的输出
        print(f"\n🧪 测试预留token的输出:")
        
        # 创建包含预留token ID的测试输入
        test_reserved_ids = list(range(actual_vocab_size, config_vocab_size))[:5]  # 取前5个预留token
        print(f"   测试的预留token IDs: {test_reserved_ids}")
        
        # 测试是否可以直接使用预留token ID
        try:
            test_input = torch.tensor([test_reserved_ids])  # [1, 5]
            with torch.no_grad():
                outputs = model(test_input)
                logits = outputs.logits  # [1, 5, 151936]
                
            print(f"   ✅ 预留token可以正常前向传播")
            print(f"   输出logits形状: {logits.shape}")
            
            # 分析预留token位置的logits输出
            reserved_logits = logits[0, -1, actual_vocab_size:].cpu()  # 最后一个位置的预留token logits
            print(f"   预留token位置logits统计:")
            print(f"     均值: {reserved_logits.mean().item():.6f}")
            print(f"     标准差: {reserved_logits.std().item():.6f}")
            
        except Exception as e:
            print(f"   ❌ 预留token前向传播失败: {e}")
        
        # 分析embedding层是否也有预留
        print(f"\n📝 检查embedding层:")
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            # 对于GPT类模型
            embedding = model.transformer.wte
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            # 对于Llama类模型
            embedding = model.model.embed_tokens
        else:
            # 寻找embedding层
            embedding = None
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    if module.weight.shape[0] == config_vocab_size:
                        embedding = module
                        print(f"   找到embedding层: {name}")
                        break
        
        if embedding is not None:
            embed_weight = embedding.weight.data
            print(f"   embedding权重形状: {embed_weight.shape}")
            
            # 分析embedding的预留部分
            used_embeds = embed_weight[:actual_vocab_size, :]
            reserved_embeds = embed_weight[actual_vocab_size:, :]
            
            print(f"   实际使用embedding统计:")
            print(f"     均值: {used_embeds.mean().item():.6f}")
            print(f"     标准差: {used_embeds.std().item():.6f}")
            
            print(f"   预留embedding统计:")
            print(f"     均值: {reserved_embeds.mean().item():.6f}")
            print(f"     标准差: {reserved_embeds.std().item():.6f}")
            
            embed_is_zero = torch.all(reserved_embeds == 0)
            print(f"   预留embedding是否全为零: {'是' if embed_is_zero else '否'}")
        
        # 分析预留token的梯度行为
        print(f"\n🔄 分析预留token的梯度行为:")
        
        # 创建一个简单的损失来测试梯度
        model.train()
        test_input = torch.tensor([[1, 2, 3]])  # 正常token
        test_target = torch.tensor([[2, 3, actual_vocab_size]])  # 目标包含一个预留token
        
        try:
            # 前向传播
            outputs = model(test_input, labels=test_target)
            loss = outputs.loss
            
            # 清零梯度
            model.zero_grad()
            
            # 反向传播
            loss.backward()
            
            # 检查预留token位置的梯度
            lm_head_grad = lm_head.weight.grad
            if lm_head_grad is not None:
                used_grad = lm_head_grad[:actual_vocab_size, :]
                reserved_grad = lm_head_grad[actual_vocab_size:, :]
                
                print(f"   ✅ 梯度计算成功")
                print(f"   实际使用权重梯度统计:")
                print(f"     非零元素数: {torch.sum(used_grad != 0).item()}")
                print(f"     均值: {used_grad.mean().item():.8f}")
                print(f"     标准差: {used_grad.std().item():.8f}")
                
                print(f"   预留权重梯度统计:")
                print(f"     非零元素数: {torch.sum(reserved_grad != 0).item()}")
                print(f"     均值: {reserved_grad.mean().item():.8f}")
                print(f"     标准差: {reserved_grad.std().item():.8f}")
                
                grad_is_zero = torch.all(reserved_grad == 0)
                print(f"   预留权重梯度是否全为零: {'是' if grad_is_zero else '否'}")
            else:
                print(f"   ❌ 没有计算出梯度")
                
        except Exception as e:
            print(f"   ❌ 梯度测试失败: {e}")
        
        # 生成总结报告
        print(f"\n📋 总结报告:")
        print(f"   🎯 预留token的设计目的:")
        print(f"     - 为未来扩展词汇表预留空间")
        print(f"     - 保持模型架构的灵活性")
        print(f"     - 避免因词汇表增长而重新训练整个模型")
        
        print(f"   🔧 预留token的技术特点:")
        print(f"     - 权重矩阵支持完整的{config_vocab_size}维输出")
        print(f"     - 前{actual_vocab_size}个位置对应实际词汇")
        print(f"     - 后{reserved_count}个位置保持预留状态")
        
        print(f"   💡 对CausalQwen的启示:")
        print(f"     - 理论设计应基于配置容量K={config_vocab_size}")
        print(f"     - 实际实现使用有效大小K={actual_vocab_size}")
        print(f"     - 我们的K+1={actual_vocab_size+1}设计是正确的")
        
        return {
            'config_vocab_size': config_vocab_size,
            'actual_vocab_size': actual_vocab_size,
            'reserved_count': reserved_count,
            'reserved_weights_stats': {
                'mean': reserved_weights.mean().item(),
                'std': reserved_weights.std().item(),
                'is_zero': is_zero.item()
            }
        }
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None

if __name__ == "__main__":
    explore_reserved_tokens() 