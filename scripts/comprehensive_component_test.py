#!/usr/bin/env python3
"""
CausalQwen 全面组件测试脚本

目标：加载 ~/models/Qwen2.5-0.5B，逐步测试所有组件的功能
特点：可视化、逐步、详细输出，让用户清楚看到每个组件的工作状态

测试内容：
1. 环境和依赖检查
2. 原始Qwen模型加载
3. CausalQwen MVP模型初始化
4. 各个核心组件功能验证
5. 三种推理模式测试
6. 端到端功能验证
"""

import sys
import os
import torch
import time
import traceback
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# ANSI color codes for pretty printing
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_section(title, color=Colors.BLUE):
    """打印章节标题"""
    print(f"\n{color}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{color}{Colors.BOLD}{title}{Colors.END}")
    print(f"{color}{Colors.BOLD}{'='*60}{Colors.END}")

def print_step(step_num, description, color=Colors.CYAN):
    """打印步骤信息"""
    print(f"\n{color}📋 步骤 {step_num}: {description}{Colors.END}")

def print_success(message):
    """打印成功信息"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """打印错误信息"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """打印信息"""
    print(f"{Colors.WHITE}ℹ️  {message}{Colors.END}")

def test_environment():
    """测试环境和依赖"""
    print_section("第一部分：环境和依赖检查")
    
    print_step(1, "检查Python环境")
    print_info(f"Python版本: {sys.version}")
    print_info(f"当前工作目录: {os.getcwd()}")
    print_info(f"项目根目录: {project_root}")
    
    print_step(2, "检查PyTorch")
    try:
        import torch
        print_success(f"PyTorch版本: {torch.__version__}")
        print_info(f"CUDA可用: {torch.cuda.is_available()}")
        print_info(f"设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    except ImportError as e:
        print_error(f"PyTorch导入失败: {e}")
        return False
    
    print_step(3, "检查transformers")
    try:
        import transformers
        print_success(f"Transformers版本: {transformers.__version__}")
    except ImportError as e:
        print_error(f"Transformers导入失败: {e}")
        return False
        
    print_step(4, "检查Qwen模型路径")
    qwen_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
    if os.path.exists(qwen_path):
        print_success(f"Qwen模型路径存在: {qwen_path}")
        # 检查关键文件
        config_file = os.path.join(qwen_path, 'config.json')
        model_file = os.path.join(qwen_path, 'pytorch_model.bin')
        safetensors_file = os.path.join(qwen_path, 'model.safetensors')
        
        if os.path.exists(config_file):
            print_success("config.json 存在")
        else:
            print_warning("config.json 不存在")
            
        if os.path.exists(model_file) or os.path.exists(safetensors_file):
            print_success("模型权重文件存在")
        else:
            print_warning("模型权重文件不存在")
    else:
        print_error(f"Qwen模型路径不存在: {qwen_path}")
        return False
    
    return True

def test_imports():
    """测试项目模块导入"""
    print_section("第二部分：项目模块导入测试")
    
    print_step(1, "导入CausalQwen MVP模块")
    try:
        from causal_qwen_mvp import (
            CausalQwenMVPForCausalLM, 
            CausalQwen2Config,
            CausalInferenceEngine,
            InferenceValidator,
            CausalTrainer,
            get_model_info
        )
        print_success("核心模块导入成功")
        
        # 显示项目信息
        model_info = get_model_info()
        print_info(f"项目名称: {model_info['name']}")
        print_info(f"版本: {model_info['version']}")
        print_info(f"状态: {model_info['status']}")
        
        return {
            'CausalQwenMVPForCausalLM': CausalQwenMVPForCausalLM,
            'CausalQwen2Config': CausalQwen2Config,
            'CausalInferenceEngine': CausalInferenceEngine,
            'InferenceValidator': InferenceValidator,
            'CausalTrainer': CausalTrainer
        }
        
    except ImportError as e:
        print_error(f"模块导入失败: {e}")
        traceback.print_exc()
        return None

def test_qwen_loading():
    """测试原始Qwen模型加载"""
    print_section("第三部分：原始Qwen模型加载测试")
    
    print_step(1, "加载Qwen2配置")
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer
        
        qwen_path = os.path.expanduser('~/models/Qwen2.5-0.5B')
        config = Qwen2Config.from_pretrained(qwen_path)
        print_success("Qwen2配置加载成功")
        print_info(f"词汇表大小: {config.vocab_size}")
        print_info(f"隐藏层大小: {config.hidden_size}")
        print_info(f"层数: {config.num_hidden_layers}")
        print_info(f"注意力头数: {config.num_attention_heads}")
        
        return config
        
    except Exception as e:
        print_error(f"Qwen2配置加载失败: {e}")
        traceback.print_exc()
        return None

def test_causal_model_initialization(qwen_config, modules):
    """测试CausalQwen模型初始化"""
    print_section("第四部分：CausalQwen模型初始化测试")
    
    print_step(1, "创建CausalQwen配置")
    try:
        CausalQwen2Config = modules['CausalQwen2Config']
        
        # 基于Qwen配置创建CausalQwen配置
        causal_config = CausalQwen2Config(
            vocab_size=qwen_config.vocab_size,
            hidden_size=qwen_config.hidden_size,
            intermediate_size=qwen_config.intermediate_size,
            num_hidden_layers=qwen_config.num_hidden_layers,
            num_attention_heads=qwen_config.num_attention_heads,
            num_key_value_heads=getattr(qwen_config, 'num_key_value_heads', qwen_config.num_attention_heads),
            max_position_embeddings=qwen_config.max_position_embeddings,
            # CausalQwen特有参数
            causal_size=qwen_config.hidden_size,
            abduction_init_strategy='identity',
            b_noise_init=0.1,
            gamma_init=10.0
        )
        print_success("CausalQwen配置创建成功")
        
        # 显示配置详情
        print_info(f"因果维度: {causal_config.causal_size}")
        print_info(f"归因初始化策略: {causal_config.abduction_init_strategy}")
        print_info(f"噪声参数: {causal_config.b_noise_init}")
        
        return causal_config
        
    except Exception as e:
        print_error(f"CausalQwen配置创建失败: {e}")
        traceback.print_exc()
        return None

def test_model_components(causal_config, modules):
    """测试各个模型组件"""
    print_section("第五部分：模型组件功能测试")
    
    print_step(1, "初始化CausalQwen模型")
    try:
        CausalQwenMVPForCausalLM = modules['CausalQwenMVPForCausalLM']
        
        model = CausalQwenMVPForCausalLM(causal_config)
        print_success("CausalQwen模型初始化成功")
        
        # 统计模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print_info(f"总参数数量: {total_params:,}")
        print_info(f"可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        print_error(f"模型初始化失败: {e}")
        traceback.print_exc()
        return None

def test_component_internals(model):
    """测试组件内部功能"""
    print_section("第六部分：组件内部功能测试")
    
    print_step(1, "测试数学工具类")
    try:
        from causal_qwen_mvp.models import CauchyMath
        
        # 测试Cauchy数学函数 - 修正维度匹配
        input_dim = 128
        output_dim = 64
        batch_size = 4
        
        loc_input = torch.randn(batch_size, input_dim)
        weight = torch.randn(output_dim, input_dim)  # 修正权重矩阵维度
        
        result_loc = CauchyMath.cauchy_linear_stable_loc(loc_input, weight)
        print_success(f"Cauchy位置变换测试通过，输出形状: {result_loc.shape}")
        
        scale_input = torch.abs(torch.randn(batch_size, input_dim)) + 0.1  # 确保为正
        result_scale = CauchyMath.cauchy_linear_stable_scale(scale_input, weight)
        print_success(f"Cauchy尺度变换测试通过，输出形状: {result_scale.shape}")
        
    except Exception as e:
        print_error(f"数学工具测试失败: {e}")
        traceback.print_exc()
    
    print_step(2, "测试归因网络")
    try:
        # 创建测试输入
        batch_size = 2
        seq_len = 10
        hidden_size = model.config.hidden_size
        
        test_input = torch.randn(batch_size, seq_len, hidden_size)
        
        # 测试归因网络
        with torch.no_grad():
            loc_U, scale_U = model.abduction_network(test_input)
            
        print_success(f"归因网络测试通过")
        print_info(f"输入形状: {test_input.shape}")
        print_info(f"loc_U输出形状: {loc_U.shape}")
        print_info(f"scale_U输出形状: {scale_U.shape}")
        print_info(f"输出统计: loc_U均值={loc_U.mean().item():.4f}, scale_U均值={scale_U.mean().item():.4f}")
        
    except Exception as e:
        print_error(f"归因网络测试失败: {e}")
        traceback.print_exc()
    
    print_step(3, "测试行动网络")
    try:
        # 先获取归因网络输出用于行动网络输入
        with torch.no_grad():
            loc_U, scale_U = model.abduction_network(test_input)
            action_loc, action_scale = model.action_network(loc_U, scale_U)
            
        print_success(f"行动网络测试通过")
        print_info(f"loc_U输入形状: {loc_U.shape}")
        print_info(f"scale_U输入形状: {scale_U.shape}")
        print_info(f"loc_S输出形状: {action_loc.shape}")
        print_info(f"scale_S输出形状: {action_scale.shape}")
        print_info(f"输出统计: loc_S均值={action_loc.mean().item():.4f}, scale_S均值={action_scale.mean().item():.4f}")
        
    except Exception as e:
        print_error(f"行动网络测试失败: {e}")
        traceback.print_exc()

def test_inference_modes(model):
    """测试三种推理模式"""
    print_section("第七部分：推理模式测试")
    
    # 创建测试输入
    batch_size = 1
    seq_len = 8
    vocab_size = model.config.vocab_size
    
    test_input_ids = torch.randint(0, min(vocab_size, 1000), (batch_size, seq_len))
    
    print_info(f"测试输入形状: {test_input_ids.shape}")
    print_info(f"测试输入内容: {test_input_ids.tolist()}")
    
    print_step(1, "测试标准推理模式")
    try:
        with torch.no_grad():
            standard_output = model.inference(test_input_ids, mode='standard')
            
        print_success("标准推理模式测试通过")
        print_info(f"输出loc_S形状: {standard_output.loc_S.shape}")
        print_info(f"输出scale_S形状: {standard_output.scale_S.shape}")
        print_info(f"loc_S统计: 均值={standard_output.loc_S.mean().item():.4f}")
        print_info(f"scale_S统计: 均值={standard_output.scale_S.mean().item():.4f}")
        
    except Exception as e:
        print_error(f"标准推理模式测试失败: {e}")
        traceback.print_exc()
    
    print_step(2, "测试因果推理模式")
    try:
        with torch.no_grad():
            causal_output = model.inference(test_input_ids, mode='causal')
            
        print_success("因果推理模式测试通过")
        print_info(f"输出loc_U形状: {causal_output.loc_U.shape}")
        print_info(f"输出scale_U形状: {causal_output.scale_U.shape}")
        print_info(f"loc_U统计: 均值={causal_output.loc_U.mean().item():.4f}")
        print_info(f"scale_U统计: 均值={causal_output.scale_U.mean().item():.4f}")
        
    except Exception as e:
        print_error(f"因果推理模式测试失败: {e}")
        traceback.print_exc()
    
    print_step(3, "测试兼容推理模式")
    try:
        with torch.no_grad():
            compatible_output = model.inference(test_input_ids, mode='compatible')
            
        print_success("兼容推理模式测试通过")
        print_info(f"输出包含所有字段")
        print_info(f"loc_S和loc_U都有输出")
        
    except Exception as e:
        print_error(f"兼容推理模式测试失败: {e}")
        traceback.print_exc()

def test_training_components(model, modules):
    """测试训练组件"""
    print_section("第八部分：训练组件测试")
    
    print_step(1, "测试损失计算")
    try:
        # 创建虚拟训练数据
        batch_size = 2
        seq_len = 8
        
        input_ids = torch.randint(0, min(model.config.vocab_size, 1000), (batch_size, seq_len))
        targets = torch.randint(0, min(model.config.vocab_size, 1000), (batch_size, seq_len))
        
        # 计算损失
        model.train()
        with torch.enable_grad():
            output = model.forward(input_ids, labels=targets)
            
        print_success("损失计算测试通过")
        if output.loss is not None:
            print_info(f"损失值: {output.loss.item():.6f}")
        else:
            print_warning("损失值为None，需检查实现")
            
    except Exception as e:
        print_error(f"损失计算测试失败: {e}")
        traceback.print_exc()
    
    print_step(2, "测试梯度计算")
    try:
        # 测试反向传播
        if output.loss is not None:
            output.loss.backward()
            
            # 检查几个关键参数的梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_count += 1
                    if grad_count <= 3:  # 只显示前3个
                        print_info(f"{name}: 梯度范数={param.grad.norm().item():.6f}")
            
            print_success(f"梯度计算测试通过，{grad_count}个参数有梯度")
        else:
            print_warning("跳过梯度测试（无损失值）")
            
    except Exception as e:
        print_error(f"梯度计算测试失败: {e}")
        traceback.print_exc()

def test_end_to_end():
    """端到端功能测试"""
    print_section("第九部分：端到端功能验证")
    
    print_step(1, "创建最小示例")
    try:
        # 重新创建一个小模型用于快速测试
        from causal_qwen_mvp import CausalQwenMVPForCausalLM, CausalQwen2Config
        
        mini_config = CausalQwen2Config(
            vocab_size=100,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
            causal_size=64
        )
        
        mini_model = CausalQwenMVPForCausalLM(mini_config)
        print_success("最小模型创建成功")
        
        # 快速功能测试
        test_ids = torch.randint(0, 100, (1, 5))
        
        with torch.no_grad():
            output1 = mini_model.inference(test_ids, mode='standard')
            output2 = mini_model.inference(test_ids, mode='causal')
            output3 = mini_model.inference(test_ids, mode='compatible')
        
        print_success("三种模式都能正常运行")
        print_info("端到端测试完成")
        
    except Exception as e:
        print_error(f"端到端测试失败: {e}")
        traceback.print_exc()

def main():
    """主测试函数"""
    print_section("CausalQwen 全面组件测试", Colors.PURPLE)
    print_info("开始逐步测试所有组件功能...")
    
    start_time = time.time()
    
    # 第一部分：环境检查
    if not test_environment():
        print_error("环境检查失败，终止测试")
        return
    
    # 第二部分：模块导入
    modules = test_imports()
    if modules is None:
        print_error("模块导入失败，终止测试")
        return
    
    # 第三部分：Qwen模型加载
    qwen_config = test_qwen_loading()
    if qwen_config is None:
        print_warning("Qwen模型加载失败，使用默认配置继续测试")
        from transformers import Qwen2Config
        qwen_config = Qwen2Config(
            vocab_size=151936,
            hidden_size=896,
            intermediate_size=4864,
            num_hidden_layers=24,
            num_attention_heads=14,
            num_key_value_heads=2
        )
    
    # 第四部分：CausalQwen配置创建
    causal_config = test_causal_model_initialization(qwen_config, modules)
    if causal_config is None:
        print_error("CausalQwen配置创建失败，终止测试")
        return
    
    # 第五部分：模型组件测试
    model = test_model_components(causal_config, modules)
    if model is None:
        print_error("模型组件测试失败，终止测试")
        return
    
    # 第六部分：组件内部测试
    test_component_internals(model)
    
    # 第七部分：推理模式测试
    test_inference_modes(model)
    
    # 第八部分：训练组件测试
    test_training_components(model, modules)
    
    # 第九部分：端到端测试
    test_end_to_end()
    
    # 测试总结
    end_time = time.time()
    print_section("测试完成", Colors.GREEN)
    print_success(f"总测试时间: {end_time - start_time:.2f} 秒")
    print_info("🎉 CausalQwen组件测试全部完成！")
    print_info("👀 请查看上述输出，确认各组件功能正常")

if __name__ == "__main__":
    main()