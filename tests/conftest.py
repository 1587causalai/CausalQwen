"""
pytest配置文件和共享fixtures

为CausalQwen测试提供可重用的测试资源，包括：
- 模型配置
- 测试模型实例
- 测试数据
- 工具函数
"""

import pytest
import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


# ========== 模型配置 Fixtures ==========

@pytest.fixture
def small_model_config():
    """创建一个小型测试用的模型配置"""
    from causal_qwen_mvp import CausalQwen2Config
    
    return CausalQwen2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        causal_size=64,
        b_noise_init=0.1,
        gamma_init=1.0,
        ovr_threshold_init=0.0
    )


@pytest.fixture
def medium_model_config():
    """创建一个中型测试用的模型配置"""
    from causal_qwen_mvp import CausalQwen2Config
    
    return CausalQwen2Config(
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=512,
        causal_size=128,
        b_noise_init=0.1,
        gamma_init=1.0
    )


# ========== 模型实例 Fixtures ==========

@pytest.fixture
def test_model(small_model_config):
    """创建一个测试用的CausalQwen模型"""
    from causal_qwen_mvp import CausalQwenMVPForCausalLM
    
    model = CausalQwenMVPForCausalLM(small_model_config)
    model.eval()  # 设置为评估模式
    return model


@pytest.fixture
def action_network(small_model_config):
    """创建单独的ActionNetwork用于测试"""
    from causal_qwen_mvp.models import ActionNetwork
    
    return ActionNetwork(small_model_config)


@pytest.fixture
def abduction_network(small_model_config):
    """创建单独的AbductionNetwork用于测试"""
    from causal_qwen_mvp.models import AbductionNetwork
    
    return AbductionNetwork(small_model_config)


# ========== 测试数据 Fixtures ==========

@pytest.fixture
def sample_input_ids():
    """创建标准的测试输入ID"""
    batch_size, seq_len = 2, 8
    return torch.randint(0, 100, (batch_size, seq_len))


@pytest.fixture
def sample_loc_U():
    """创建测试用的个体位置参数"""
    batch_size, seq_len, causal_size = 2, 5, 64
    return torch.randn(batch_size, seq_len, causal_size)


@pytest.fixture
def sample_scale_U():
    """创建测试用的个体尺度参数"""
    batch_size, seq_len, causal_size = 2, 5, 64
    return torch.abs(torch.randn(batch_size, seq_len, causal_size)) + 1.0


@pytest.fixture
def sample_hidden_states():
    """创建测试用的隐藏状态"""
    batch_size, seq_len, hidden_size = 2, 8, 64
    return torch.randn(batch_size, seq_len, hidden_size)


# ========== 工具函数 ==========

@pytest.fixture
def tolerance():
    """数值比较的容差设置"""
    return {
        'atol': 1e-6,      # 绝对容差
        'rtol': 1e-5,      # 相对容差
        'logits_atol': 1e-4  # logits比较的容差
    }


@pytest.fixture
def set_random_seed():
    """设置随机种子的函数"""
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    return _set_seed


# ========== 模型加载相关 ==========

@pytest.fixture(scope="session")
def qwen_model_path():
    """Qwen预训练模型路径"""
    return Path.home() / "models" / "Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def qwen_model_available(qwen_model_path):
    """检查Qwen预训练模型是否可用"""
    return qwen_model_path.exists()


# ========== 测试辅助函数 ==========

def assert_cauchy_linear_stability(loc_input, scale_input, weight, bias, 
                                  loc_output, scale_output, tolerance):
    """验证柯西分布的线性稳定性"""
    # 验证位置参数：loc_output = loc_input @ weight.T + bias
    expected_loc = loc_input @ weight.T + bias
    torch.testing.assert_close(loc_output, expected_loc, **tolerance)
    
    # 验证尺度参数：scale_output = scale_input @ |weight|.T
    expected_scale = scale_input @ torch.abs(weight).T
    torch.testing.assert_close(scale_output, expected_scale, **tolerance)


def assert_v2_mode_difference(loc_S_det, scale_S_det, loc_S_samp, scale_S_samp):
    """验证双模式的差异"""
    # 位置参数应该有差异（采样模式扰动了位置）
    loc_diff = torch.abs(loc_S_samp - loc_S_det).mean().item()
    assert loc_diff > 1e-6, f"位置参数差异过小: {loc_diff}"
    
    # 尺度参数的差异取决于具体实现
    scale_diff = torch.abs(scale_S_samp - scale_S_det).mean().item()
    return loc_diff, scale_diff


# ========== pytest配置 ==========

def pytest_configure(config):
    """pytest配置钩子"""
    config.addinivalue_line(
        "markers", "slow: 标记运行较慢的测试"
    )
    config.addinivalue_line(
        "markers", "requires_qwen: 需要Qwen预训练模型的测试"
    )


# ========== 测试报告增强 ==========

def pytest_collection_modifyitems(config, items):
    """自动为测试添加标记"""
    for item in items:
        # 自动标记需要Qwen模型的测试
        if "qwen_model" in item.fixturenames:
            item.add_marker(pytest.mark.requires_qwen)
        
        # 自动标记慢速测试
        if "slow" in item.nodeid or "comparison" in item.nodeid:
            item.add_marker(pytest.mark.slow) 