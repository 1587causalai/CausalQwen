"""
CausalEngine消融版本实现
使用完整CausalEngine架构，但损失计算时仅使用位置参数(loc)
这样可以公平对比因果机制的贡献
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 尝试导入真实的CausalEngine
try:
    from causal_engine import CausalEngine
    print("✅ 使用真实的CausalEngine实现")
    CAUSAL_ENGINE_AVAILABLE = True
except ImportError:
    print("⚠️  使用模拟的CausalEngine实现")
    CAUSAL_ENGINE_AVAILABLE = False
    
    class CausalEngine:
        def __init__(self, **kwargs):
            self.config = kwargs
            # 添加一些虚拟参数用于测试
            self.dummy_param = torch.nn.Parameter(torch.randn(10, 10))
            
        def __call__(self, **inputs):
            # 返回模拟输出
            batch_size = inputs.get('input_ids').shape[0]
            seq_len = inputs.get('input_ids').shape[1]
            vocab_size = self.config.get('causal_vocab_size', 10)
            
            decision_scores = torch.randn(batch_size, seq_len, vocab_size, 2)
            return {
                'decision_scores': decision_scores,
                'loss': torch.tensor(0.0),
                'causal_loss': torch.tensor(0.0)
            }
        
        def train(self):
            pass
            
        def eval(self):
            pass
            
        def parameters(self):
            return [self.dummy_param]
            
        def state_dict(self):
            return {'dummy_param': self.dummy_param}
            
        def load_state_dict(self, state_dict):
            pass
            
        def to(self, device):
            return self


class AblationCausalEngineWrapper:
    """
    CausalEngine的消融包装器
    使用完整的CausalEngine进行前向传播，但损失计算时仅使用loc参数
    """
    
    def __init__(
        self,
        causal_engine: CausalEngine,
        task_type: str = 'classification',  # 'classification' or 'regression'
        num_classes: Optional[int] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化消融包装器
        
        参数:
            causal_engine: 完整的CausalEngine实例
            task_type: 任务类型 ('classification' 或 'regression')
            num_classes: 分类任务的类别数
            device: 计算设备
        """
        self.engine = causal_engine
        self.task_type = task_type
        self.num_classes = num_classes
        self.device = torch.device(device)
        
        # 损失函数
        if task_type == 'classification':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def compute_loss_ablation(
        self, 
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        消融版本的损失计算：仅使用loc参数
        
        参数:
            hidden_states: 输入的隐藏状态张量
            targets: 目标值
            
        返回:
            损失值
        """
        if CAUSAL_ENGINE_AVAILABLE:
            # 使用真实CausalEngine API
            # hidden_states = inputs.get('values')  # [batch_size, input_dim]
            
            # 如果输入是2D，需要添加seq_len维度
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)  # [batch_size, 1, input_dim]
            
            # 运行CausalEngine
            outputs = self.engine(
                hidden_states=hidden_states,
                do_sample=False,
                temperature=1.0,
                return_dict=True,
                apply_activation=False  # 不使用激活头，直接使用决策分布
            )
            
            # 提取决策分布参数
            loc_S = outputs['loc_S']    # [batch_size, seq_len, vocab_size]
            scale_S = outputs['scale_S'] # [batch_size, seq_len, vocab_size]
            
            # 获取最后一个时间步的输出
            loc = loc_S[:, -1, :]  # [batch_size, vocab_size]
            
        else:
            # 使用模拟版本的API
            outputs = self.engine(**inputs)
            decision_scores = outputs['decision_scores']  # [batch_size, seq_len, vocab_size, 2]
            final_scores = decision_scores[:, -1, :, :]  # [batch_size, vocab_size, 2]
            loc = final_scores[:, :, 0]  # [batch_size, vocab_size]
        
        if self.task_type == 'classification':
            # 分类任务：使用softmax + 交叉熵
            # 只取前num_classes个输出
            logits = loc[:, :self.num_classes]
            loss = self.loss_fn(logits, targets)
        else:
            # 回归任务：使用MSE
            # 假设只有一个输出维度
            predictions = loc[:, 0]  # [batch_size]
            loss = self.loss_fn(predictions, targets)
        
        return loss
    
    def compute_loss_full(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        完整版本的损失计算：使用CausalEngine的完整因果机制
        
        参数:
            hidden_states: 输入的隐藏状态张量
            targets: 目标值
            
        返回:
            损失值
        """
        if CAUSAL_ENGINE_AVAILABLE:
            # 使用真实CausalEngine API进行完整因果推理
            # hidden_states = inputs.get('values')  # [batch_size, input_dim]
            
            # 如果输入是2D，需要添加seq_len维度
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)  # [batch_size, 1, input_dim]
            
            # 运行CausalEngine with activation
            outputs = self.engine(
                hidden_states=hidden_states,
                do_sample=False,
                temperature=1.0,
                return_dict=True,
                apply_activation=True  # 使用激活头进行完整的因果推理
            )
            
            # 从激活输出中获取最终结果
            final_output = outputs['output']  # [batch_size, seq_len, output_dim]
            final_output = final_output[:, -1, :]  # [batch_size, output_dim]
            
            if self.task_type == 'classification':
                # 分类任务：使用激活头的输出
                loss = self.loss_fn(final_output, targets)
            else:
                # 回归任务：使用激活头的输出
                predictions = final_output.squeeze(-1) if final_output.dim() > 1 else final_output
                loss = self.loss_fn(predictions, targets)
            
        else:
            # 使用模拟版本的API
            # 准备因果目标
            if self.task_type == 'classification':
                # 将类别标签转换为one-hot编码
                causal_targets = F.one_hot(targets, num_classes=self.num_classes).float()
            else:
                # 回归目标
                causal_targets = targets.unsqueeze(-1) if targets.dim() == 1 else targets
            
            # 将因果目标添加到输入中
            inputs['causal_targets'] = causal_targets
            
            # 运行CausalEngine，它会自动计算因果损失
            outputs = self.engine(**inputs)
            
            # 返回CausalEngine计算的损失
            loss = outputs.get('loss', outputs.get('causal_loss', torch.tensor(0.0)))
        
        return loss


def create_ablation_experiment(
    input_dim: int,
    hidden_dim: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    task_type: str = 'classification',
    num_classes: Optional[int] = None,
    output_dim: int = 1,
    dropout: float = 0.1,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[CausalEngine, AblationCausalEngineWrapper]:
    """
    创建消融实验所需的组件
    
    返回:
        engine: CausalEngine实例
        wrapper: 消融包装器
    """
    # 确定词汇表大小
    if task_type == 'classification':
        vocab_size = num_classes
    else:
        vocab_size = output_dim
    
    # 创建CausalEngine（使用真实的API）
    if CAUSAL_ENGINE_AVAILABLE:
        engine = CausalEngine(
            hidden_size=input_dim,
            vocab_size=vocab_size,
            causal_size=hidden_dim,
            activation_modes="classification" if task_type == 'classification' else "regression"
        ).to(device)
    else:
        # 使用模拟版本
        engine = CausalEngine(
            vocab_size=input_dim,
            causal_vocab_size=vocab_size,
            embed_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=hidden_dim * 4,
            dropout=dropout,
            freeze_embeddings=False,
            device=device
        )
    
    # 创建消融包装器
    wrapper = AblationCausalEngineWrapper(
        causal_engine=engine,
        task_type=task_type,
        num_classes=num_classes,
        device=device
    )
    
    return engine, wrapper


class AblationTrainer:
    """
    消融实验训练器
    同时训练和评估消融版本和完整版本
    """
    
    def __init__(
        self,
        engine: CausalEngine,
        wrapper: AblationCausalEngineWrapper,
        lr: float = 1e-4,
        weight_decay: float = 0.01
    ):
        """
        初始化训练器
        
        参数:
            engine: CausalEngine实例
            wrapper: 消融包装器
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.engine = engine
        self.wrapper = wrapper
        
        # 两个版本使用相同的优化器（因为是同一个网络）
        self.optimizer = torch.optim.AdamW(
            engine.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    def train_step_ablation(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        消融版本的训练步骤
        """
        self.engine.train()
        self.optimizer.zero_grad()
        
        # 计算消融损失
        loss = self.wrapper.compute_loss_ablation(hidden_states, targets)
        loss.backward()
        self.optimizer.step()
        
        # 计算指标
        metrics = {'loss': loss.item()}
        
        if self.wrapper.task_type == 'classification':
            with torch.no_grad():
                if CAUSAL_ENGINE_AVAILABLE:
                    # 使用真实CausalEngine API
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(1)
                    
                    outputs = self.engine(
                        hidden_states=hidden_states,
                        do_sample=False,
                        temperature=1.0,
                        return_dict=True,
                        apply_activation=False
                    )
                    loc = outputs['loc_S'][:, -1, :]  # [batch_size, vocab_size]
                    preds = torch.argmax(loc[:, :self.wrapper.num_classes], dim=1)
                else:
                    # 使用模拟版本
                    inputs = {'values': hidden_states}
                    outputs = self.engine(**inputs)
                    decision_scores = outputs['decision_scores'][:, -1, :, :]
                    loc = decision_scores[:, :self.wrapper.num_classes, 0]
                    preds = torch.argmax(loc, dim=1)
                
                acc = (preds == targets).float().mean().item()
                metrics['accuracy'] = acc
        
        return metrics
    
    def train_step_full(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        完整版本的训练步骤
        """
        self.engine.train()
        self.optimizer.zero_grad()
        
        # 计算完整损失
        loss = self.wrapper.compute_loss_full(hidden_states, targets)
        loss.backward()
        self.optimizer.step()
        
        # 计算指标
        metrics = {'loss': loss.item()}
        
        if self.wrapper.task_type == 'classification':
            with torch.no_grad():
                if CAUSAL_ENGINE_AVAILABLE:
                    # 使用真实CausalEngine API
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(1)
                    
                    outputs = self.engine(
                        hidden_states=hidden_states,
                        do_sample=False,
                        temperature=1.0,
                        return_dict=True,
                        apply_activation=True  # 完整版本使用激活头
                    )
                    
                    # 使用激活头的输出进行预测
                    final_output = outputs['output'][:, -1, :]  # [batch_size, output_dim]
                    preds = torch.argmax(final_output, dim=1)
                else:
                    # 使用模拟版本
                    inputs = {'values': hidden_states}
                    outputs = self.engine(**inputs)
                    # 使用因果概率进行预测
                    if 'class_probs' in outputs:
                        probs = outputs['class_probs']
                        preds = torch.argmax(probs, dim=1)
                    else:
                        # 备选方案：使用决策得分
                        decision_scores = outputs['decision_scores'][:, -1, :, :]
                        loc = decision_scores[:, :self.wrapper.num_classes, 0]
                        preds = torch.argmax(loc, dim=1)
                
                acc = (preds == targets).float().mean().item()
                metrics['accuracy'] = acc
        
        return metrics
    
    def eval_step(
        self,
        hidden_states: torch.Tensor,
        targets: torch.Tensor,
        use_ablation: bool = True
    ) -> Dict[str, float]:
        """
        评估步骤
        
        参数:
            use_ablation: 是否使用消融版本
        """
        self.engine.eval()
        
        with torch.no_grad():
            if use_ablation:
                loss = self.wrapper.compute_loss_ablation(hidden_states, targets)
            else:
                loss = self.wrapper.compute_loss_full(hidden_states, targets)
            
            metrics = {'loss': loss.item()}
            
            if self.wrapper.task_type == 'classification':
                if CAUSAL_ENGINE_AVAILABLE:
                    # 使用真实CausalEngine API
                    if hidden_states.dim() == 2:
                        hidden_states = hidden_states.unsqueeze(1)
                    
                    outputs = self.engine(
                        hidden_states=hidden_states,
                        do_sample=False,
                        temperature=1.0,
                        return_dict=True,
                        apply_activation=not use_ablation  # 消融版本不用激活头，完整版本用
                    )
                    
                    if use_ablation:
                        # 使用loc进行预测
                        loc = outputs['loc_S'][:, -1, :]  # [batch_size, vocab_size]
                        preds = torch.argmax(loc[:, :self.wrapper.num_classes], dim=1)
                    else:
                        # 使用激活头输出进行预测
                        final_output = outputs['output'][:, -1, :]  # [batch_size, output_dim]
                        preds = torch.argmax(final_output, dim=1)
                else:
                    # 使用模拟版本
                    inputs = {'values': hidden_states}
                    outputs = self.engine(**inputs)
                    if use_ablation or 'class_probs' not in outputs:
                        # 使用loc进行预测
                        decision_scores = outputs['decision_scores'][:, -1, :, :]
                        loc = decision_scores[:, :self.wrapper.num_classes, 0]
                        preds = torch.argmax(loc, dim=1)
                    else:
                        # 使用因果概率
                        probs = outputs['class_probs']
                        preds = torch.argmax(probs, dim=1)
                
                acc = (preds == targets).float().mean().item()
                metrics['accuracy'] = acc
        
        return metrics


# ============================================================================
# 兼容层：为旧API提供支持
# ============================================================================

class CompatibilityWrapper:
    """为旧API提供兼容性包装器"""
    
    def __init__(self, engine, wrapper, task_type, device='cuda'):
        self.engine = engine
        self.wrapper = wrapper
        self.task_type = task_type
        self.device = device
    
    def to(self, device):
        self.device = device
        self.engine = self.engine.to(device)
        return self
    
    def train(self):
        self.engine.train()
    
    def eval(self):
        self.engine.eval()
    
    def parameters(self):
        return self.engine.parameters()
    
    def state_dict(self):
        return self.engine.state_dict()
    
    def load_state_dict(self, state_dict):
        self.engine.load_state_dict(state_dict)
    
    def __call__(self, x, temperature=1.0, do_sample=False):
        if CAUSAL_ENGINE_AVAILABLE:
            # 使用真实CausalEngine API
            hidden_states = x
            if hidden_states.dim() == 2:
                hidden_states = hidden_states.unsqueeze(1)  # [batch_size, 1, input_dim]
            
            # 前向传播
            outputs = self.engine(
                hidden_states=hidden_states,
                do_sample=do_sample,
                temperature=temperature,
                return_dict=True,
                apply_activation=False  # 兼容层只返回决策分布
            )
            
            # 提取决策分布参数
            loc = outputs['loc_S'][:, -1, :]  # [batch_size, vocab_size]
            
            if self.task_type == 'classification':
                # 只取前num_classes个输出
                return loc[:, :self.wrapper.num_classes]
            else:
                # 回归任务返回第一个输出
                return loc[:, 0:1]
        else:
            # 使用模拟版本API
            batch_size = x.shape[0]
            seq_len = x.shape[1] if len(x.shape) > 1 else x.shape[0]
            
            # 创建输入ID
            input_ids = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            inputs = {
                'input_ids': input_ids,
                'values': x,
                'temperature': temperature,
                'mode': 'causal' if temperature == 0 else 'standard'
            }
            
            # 前向传播
            outputs = self.engine(**inputs)
            
            # 提取决策得分
            decision_scores = outputs['decision_scores'][:, -1, :, :]
            loc = decision_scores[:, :, 0]
            
            if self.task_type == 'classification':
                # 只取前num_classes个输出
                return loc[:, :self.wrapper.num_classes]
            else:
                # 回归任务返回第一个输出
                return loc[:, 0:1]
    
    def predict(self, x, temperature=1.0, do_sample=False):
        outputs = self(x, temperature, do_sample)
        if self.task_type == 'classification':
            return torch.argmax(outputs, dim=-1)
        else:
            return outputs.squeeze(-1)
    
    def predict_proba(self, x, temperature=1.0, do_sample=False):
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        outputs = self(x, temperature, do_sample)
        return torch.softmax(outputs, dim=-1)


# 旧API兼容函数
def create_ablated_classifier(input_size, num_classes, causal_size=128, **kwargs):
    """创建消融版本的分类器（兼容旧API）"""
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    engine, wrapper = create_ablation_experiment(
        input_dim=input_size,
        hidden_dim=causal_size,
        num_layers=4,
        num_heads=8,
        task_type='classification',
        num_classes=num_classes,
        device=device
    )
    return CompatibilityWrapper(engine, wrapper, 'classification', device)


def create_full_causal_classifier(input_size, num_classes, causal_size=128, **kwargs):
    """创建完整版本的分类器（兼容旧API）"""
    # 对于兼容性，完整版本和消融版本使用相同的网络
    # 区别在训练时的损失函数
    return create_ablated_classifier(input_size, num_classes, causal_size, **kwargs)


def create_ablated_regressor(input_size, output_size=1, causal_size=128, **kwargs):
    """创建消融版本的回归器（兼容旧API）"""
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    engine, wrapper = create_ablation_experiment(
        input_dim=input_size,
        hidden_dim=causal_size,
        num_layers=4,
        num_heads=8,
        task_type='regression',
        output_dim=output_size,
        device=device
    )
    return CompatibilityWrapper(engine, wrapper, 'regression', device)


def create_full_causal_regressor(input_size, output_size=1, causal_size=128, **kwargs):
    """创建完整版本的回归器（兼容旧API）"""
    # 对于兼容性，完整版本和消融版本使用相同的网络
    # 区别在训练时的损失函数
    return create_ablated_regressor(input_size, output_size, causal_size, **kwargs)