import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
import math
import os
from typing import Optional, Tuple, List

# 确保模型路径正确，处理 "~" 符号
QWEN_MODEL_PATH = os.path.expanduser("~/models/Qwen2.5-0.5B")

class CausalQwen(Qwen2ForCausalLM):
    """
    CausalQwen 的极简化 PyTorch 实现。
    继承自 Qwen2ForCausalLM，并扩展了因果推断和数值处理能力。
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 获取模型核心维度
        H = config.hidden_size
        C = H  # 设计决策: 因果表征维度 C 与隐藏维度 H 相等
        V_full = config.vocab_size + 1 # Qwen 词汇表 + 1个 <NUM> 词元
        
        # === 1. 定义新增模块的参数 ===
        
        # 1.1 数值感知嵌入模块
        # 方向向量 e, 初始化为随机方向并归一化
        numeric_direction = torch.randn(H)
        self.numeric_direction = nn.Parameter(numeric_direction / torch.norm(numeric_direction))

        # 1.2 归因网络 (Abduction Network)
        # 将 z [B,S,H] 映射到 loc_U [B,S,C] 和 scale_U [B,S,C]
        self.abduction_loc_layer = nn.Linear(H, C)
        self.abduction_scale_layer = nn.Linear(H, C)

        # 1.3 行动网络 (Action Network)
        # 分类部分直接使用 Qwen 的 lm_head
        self.action_cls_layer = self.lm_head
        
        # 回归部分是新增的
        self.action_reg_loc_layer = nn.Linear(C, 1)
        self.action_reg_scale_layer = nn.Linear(C, 1)

        # 1.4 OvR 阈值
        self.ovr_threshold = nn.Parameter(torch.tensor(100.0))

        # === 2. 执行知识迁移初始化策略 ===
        self.initialize_weights()

    def initialize_weights(self):
        """
        应用文档中描述的精确初始化策略，确保 CausalQwen 初始行为与 Qwen 一致。
        """
        H = self.config.hidden_size
        
        # 步骤2: 归因推断网络 -> 恒等映射 + 高斯不确定性
        # loc_U = z (近似恒等)
        self.abduction_loc_layer.weight.data.copy_(torch.eye(H))
        nn.init.zeros_(self.abduction_loc_layer.bias)
        
        # scale_U = gamma (大常数)
        gamma = 10.0
        nn.init.zeros_(self.abduction_scale_layer.weight)
        self.abduction_scale_layer.bias.data.fill_(math.log(gamma))

        # 步骤3: 行动网络(分类) -> 已通过共享 lm_head 权重实现
        # self.action_cls_layer 与 self.lm_head 是同一对象，权重已继承

        # 步骤4: 行动网络(回归) -> 常规初始化 (PyTorch 默认的 Kaiming He 初始化)
        # 此处无需额外操作，保持默认初始化即可实现小权重效果
        
        print("CausalQwen's custom weights have been initialized.")

    def forward(
        self,
        input_ids: torch.LongTensor,
        numeric_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        numeric_labels: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        
        # B, S = input_ids.shape
        
        # --- 模块一: 数值感知嵌入 ---
        # 1. 词元嵌入
        base_embeddings = self.model.embed_tokens(input_ids)
        
        # 2. 数值编码与融合
        # numeric_values 如果没有提供，则视为全零
        if numeric_values is None:
            numeric_values = torch.zeros_like(input_ids, dtype=torch.float32)
            
        # φ(v) = sign(v) * ln(1 + |v|) * e_vec
        # numeric_values [B, S] -> [B, S, 1] 以便广播
        phi_v = torch.sign(numeric_values) * torch.log1p(torch.abs(numeric_values))
        numeric_encoding = phi_v.unsqueeze(-1) * self.numeric_direction
        
        enhanced_embeddings = base_embeddings + numeric_encoding
        
        # --- 模块二: 特征提取网络 (Qwen主干) ---
        # 直接调用父类的 transformer 模型部分
        transformer_outputs = self.model(
            inputs_embeds=enhanced_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        z = transformer_outputs[0] # 上下文特征 z, 形状 [B, S, H]
        
        # --- 模块三: 归因推断网络 ---
        loc_U = self.abduction_loc_layer(z)
        # 尺度参数需要经过 exp 转换 (因为网络输出的是 log(scale))
        scale_U = torch.exp(self.abduction_scale_layer(z))
        
        # --- 模块四: 行动决策网络 ---
        # 注意：为简化，此处我们直接用 loc_U 作为下一层的输入。
        # 严格的柯西线性变换需要同时作用于 loc 和 scale，但此近似在实践中效果良好。
        
        # 4.1 分类决策
        loc_S = self.action_cls_layer(loc_U)
        # 根据线性变换性质计算 scale_S
        scale_S = torch.matmul(torch.abs(self.action_cls_layer.weight), scale_U.unsqueeze(-1)).squeeze(-1)

        # 4.2 回归决策
        loc_Y = self.action_reg_loc_layer(loc_U).squeeze(-1) # [B, S, 1] -> [B, S]
        scale_Y = torch.abs(self.action_reg_scale_layer(loc_U)).squeeze(-1) # 同样，简化处理

        # --- 模块五: 损失计算 ---
        total_loss = None
        if labels is not None:
            # 1. 计算 OvR 分类损失
            # P_k,i = 1/2 + 1/pi * arctan((loc_S - C_k) / scale_S)
            scores = (loc_S - self.ovr_threshold) / scale_S
            probs = 0.5 + torch.arctan(scores) / math.pi
            
            # 使用 one_hot 标签计算二元交叉熵
            cls_labels = F.one_hot(labels, num_classes=self.config.vocab_size).float()
            # 此处 V_full 包括了 <NUM>，需要对齐
            # 简化起见，我们假设 labels 中不包含 <NUM>_ID，所以 one_hot 维度正确
            loss_cls = F.binary_cross_entropy(probs, cls_labels)
            
            # 2. 计算门控回归损失
            loss_reg_gated = torch.tensor(0.0, device=self.device)
            if numeric_labels is not None:
                # 获取 P_<NUM>
                # 假设 <NUM>_ID 是词汇表的最后一个
                num_token_id = self.config.vocab_size -1 
                p_num = probs[:, :, num_token_id]
                
                # 获取掩码 m_i
                mask = (numeric_labels != 0).float()
                
                # alpha 默认为 0
                gate = mask * p_num
                
                # 柯西负对数似然
                cauchy_nll = torch.log(math.pi * scale_Y) + torch.log1p(((numeric_labels - loc_Y) / scale_Y).pow(2))
                
                loss_reg_gated = (gate * cauchy_nll).mean()

            # 3. 合并总损失
            lambda_reg = 0.5 # 回归损失的权重
            total_loss = loss_cls + lambda_reg * loss_reg_gated

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=loc_S, # 将 loc_S 作为传统 logits 输出以便兼容
            # 可以自定义输出更多内容，如 loc_Y, scale_Y 等
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == '__main__':
    # === 使用示例 ===
    
    # 1. 加载预训练的 Qwen 模型和分词器
    # 确保你已经下载了模型到指定路径
    if not os.path.exists(QWEN_MODEL_PATH):
        print(f"Model path not found: {QWEN_MODEL_PATH}")
        print("Please download Qwen2.5-0.5B model first.")
    else:
        base_model = Qwen2ForCausalLM.from_pretrained(QWEN_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH)
        
        # 添加 <NUM> 词元
        if '<NUM>' not in tokenizer.get_vocab():
            tokenizer.add_tokens(['<NUM>'])
            base_model.resize_token_embeddings(len(tokenizer))
            print("Added <NUM> token.")

        # 2. 实例化 CausalQwen
        causal_model = CausalQwen(base_model.config)
        # 加载预训练权重 (除新增部分外)
        causal_model.load_state_dict(base_model.state_dict(), strict=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        causal_model.to(device)
        causal_model.eval()

        # 3. 准备输入数据
        text = "这家餐厅的评分是4.5分，价格是99.8元。"
        
        # 手动实现分词与数值识别逻辑 (简化版)
        tokens = []
        numerics = []
        import re
        parts = re.split(r'(\d+\.?\d*)', text)
        for part in parts:
            if re.match(r'\d+\.?\d*', part) and part:
                tokens.append('<NUM>')
                numerics.append(float(part))
            elif part:
                toks = tokenizer.tokenize(part)
                tokens.extend(toks)
                numerics.extend([0.0] * len(toks))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], device=device)
        numeric_values = torch.tensor([numerics], device=device, dtype=torch.float32)

        print(f"Input Text: {text}")
        print(f"Tokenized input_ids: {input_ids.shape}\n{input_ids}")
        print(f"Numeric values: {numeric_values.shape}\n{numeric_values}")
        
        # 4. 执行前向传播
        with torch.no_grad():
            outputs = causal_model(input_ids=input_ids, numeric_values=numeric_values)

        # 5. 查看输出
        print("\n--- Model Output ---")
        # loc_S 作为传统 logits
        logits = outputs.logits
        print(f"Output logits shape: {logits.shape}") # [B, S, V_full]
        
        # 打印最后一个 token 的预测
        next_token_logits = logits[0, -1, :]
        predicted_token_id = torch.argmax(next_token_logits).item()
        predicted_token = tokenizer.decode(predicted_token_id)
        
        print(f"Predicted next token: '{predicted_token}' (ID: {predicted_token_id})")