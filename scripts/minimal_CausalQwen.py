import torch
import torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, Union, List

# --- 1. 定义模型超参数 ---
# 假设的超参数，可以放入模型 config 中
CAUSAL_REPRESENTATION_DIM = 128 # 个体因果表征 U 的维度
OVR_THRESHOLD = 100.0           # OvR 分类的决策阈值 C_k
REG_LOSS_WEIGHT = 0.5           # 回归损失的权重 lambda
GATE_ALPHA = 0.1                # 门控损失中的平滑系数 alpha
NUM_TOKEN_ID = 151646           # Qwen2-0.5B 中 '<|extra_0|>' 的 ID, 我们用它作为 <NUM>

class CausalQwenForCausalLM(Qwen2ForCausalLM):
    """
    CausalQwen 模型，通过继承 Qwen2ForCausalLM 实现。
    增加了因果推断和双通道决策（分类+回归）的能力。
    """
    def __init__(self, config):
        super().__init__(config)

        # --- 2. 定义新增的网络层 ---
        hidden_size = config.hidden_size

        # 模块三：归因推断网络 (Abduction Network)
        # 将 Qwen 的输出特征 z 映射到因果表征 U 的分布参数
        self.abduction_loc = nn.Linear(hidden_size, CAUSAL_REPRESENTATION_DIM)
        self.abduction_scale = nn.Linear(hidden_size, CAUSAL_REPRESENTATION_DIM)

        # 模块四：行动决策网络 (Action Network) - 回归部分
        # 将因果表征 U 映射到回归决策 Y 的分布参数
        # 分类部分直接复用基座模型的 self.lm_head
        self.action_reg_loc = nn.Linear(CAUSAL_REPRESENTATION_DIM, 1)
        self.action_reg_scale = nn.Linear(CAUSAL_REPRESENTATION_DIM, 1)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        numerical_values: Optional[torch.FloatTensor] = None, # CausalQwen 新增输入
        labels: Optional[torch.LongTensor] = None,
        numerical_labels: Optional[torch.FloatTensor] = None, # CausalQwen 新增标签
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # --- 模块一 & 二：数值感知嵌入 + 特征提取 ---
        # 在这个极简版中，我们假设数值嵌入已在输入端处理完毕。
        # 我们直接调用基座模型的 transformer 部分来获取上下文特征 z。
        transformer_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 上下文特征 z, 形状: [batch_size, seq_len, hidden_size]
        z = transformer_outputs[0]

        # --- 模块三：归因推断网络 (Abduction Network) ---
        # 计算因果表征 U ~ Cauchy(loc_U, scale_U) 的分布参数
        loc_U = self.abduction_loc(z)
        # 使用 exp 确保尺度 > 0
        scale_U = torch.exp(self.abduction_scale(z))

        # --- 模块四：行动决策网络 (Action Network) ---
        # 核心洞察：利用柯西分布的线性稳定性，直接变换分布参数
        
        # 1. 分类决策 S
        # loc_S 直接由基座的 lm_head 变换 loc_U 得到
        loc_S = self.lm_head(loc_U) # 形状: [B, S, V_full]

        # scale_S 的变换较为复杂 (涉及权重矩阵的绝对值)，此处简化
        # 假设 lm_head 的权重为 W_cls，则 scale_S = |W_cls| @ scale_U
        # 为了简化，我们用一个独立的线性层来近似这个变换
        # (在实际实现中，这部分会有更精巧的处理)
        # 为保持极简，我们在此处假设一个固定的尺度
        scale_S = torch.ones_like(loc_S) * 1.0

        # 2. 回归决策 Y
        loc_Y = self.action_reg_loc(loc_U).squeeze(-1) # 形状: [B, S]
        scale_Y = torch.exp(self.action_reg_scale(loc_U)).squeeze(-1) # 形状: [B, S]

        # --- 模块五：损失计算 (Loss Calculation) ---
        total_loss = None
        if labels is not None:
            # 将 labels 移动到与 loc_S 相同的设备
            labels = labels.to(loc_S.device)
            loss = 0.0

            # 1. OvR 分类损失
            # 计算柯西分布的累积分布函数 (CDF) 来得到概率
            # P(S_k > C_k), C_k 是决策阈值
            probs_S = 0.5 + torch.atan((loc_S - OVR_THRESHOLD) / scale_S) / torch.pi
            
            # 使用二元交叉熵 (BCE)
            # 将 labels 转换为 one-hot 编码
            y_one_hot = F.one_hot(labels, num_classes=self.config.vocab_size).float()
            cls_loss = F.binary_cross_entropy_with_logits(loc_S, y_one_hot, reduction='none').mean()
            loss += cls_loss

            # 2. 门控回归损失
            if numerical_labels is not None:
                numerical_labels = numerical_labels.to(loc_Y.device)
                
                # a. 计算基础柯西负对数似然 (NLL)
                cauchy_nll = torch.log(torch.pi * scale_Y) + \
                             torch.log(1 + ((numerical_labels - loc_Y) / scale_Y).pow(2))

                # b. 计算门控权重
                # 掩码 m_i: 只在真实标签是 <NUM> 的位置计算回归损失
                mask_m = (labels == NUM_TOKEN_ID).float()
                
                # 获取 P_<NUM>,i 概率
                prob_num = probs_S[:, :, NUM_TOKEN_ID]
                
                # 计算门控
                gate = mask_m * (GATE_ALPHA + (1 - GATE_ALPHA) * prob_num)
                
                # c. 应用门控
                reg_loss_gated = (gate * cauchy_nll).mean()
                loss += REG_LOSS_WEIGHT * reg_loss_gated
            
            total_loss = loss

        # 返回 Hugging Face 风格的输出对象
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=loc_S,  # loc_S 可以作为传统 logits 使用
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == '__main__':
    import torch.nn.functional as F

    # --- 使用示例 ---
    # 注意: 请将 '~/models/Qwen2.5-0.5B' 替换为你的实际模型路径
    model_path = 'Qwen/Qwen2.5-0.5B' # 如果已登录 huggingface-cli,可以直接使用
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = CausalQwenForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"模型加载失败，请检查路径 '{model_path}' 是否正确。错误: {e}")
        print("将使用随机初始化的模型进行演示。")
        from transformers import Qwen2Config
        config = Qwen2Config.from_pretrained("~/models/Qwen2.5-0.5B")
        model = CausalQwenForCausalLM(config)


    print("模型加载成功！")
    model.eval()

    # --- 构造伪数据 ---
    text = "这件衣服的价格是 <|extra_0|> 元，那件是 200 元。"
    # 在这个例子中，我们用 <|extra_0|> (ID: 151646) 作为 <NUM> 的代理
    
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs.input_ids
    
    # 构造标签 (向右移动一位)
    labels = input_ids.clone()[:, 1:].contiguous()
    input_ids = input_ids[:, :-1].contiguous()
    
    # 构造对应的数值标签
    # 序列: "这件", "衣服", "的", "价格", "是", " <|extra_0|>", " 元", "，", "那件", "是", " 200"
    # 假设分词后，<|extra_0|> 在第 5 个位置 (index=5)
    # 真实数值为 99.9
    numerical_labels = torch.zeros_like(labels, dtype=torch.float)
    if labels.shape[1] > 5:
      numerical_labels[0, 5] = 99.9 

    print(f"输入 input_ids 形状: {input_ids.shape}")
    print(f"分类 labels 形状: {labels.shape}")
    print(f"数值 numerical_labels 形状: {numerical_labels.shape}")
    print("-" * 20)

    # --- 前向传播与损失计算 ---
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            numerical_labels=numerical_labels
        )
    
    print(f"计算出的总损失 (Total Loss): {outputs.loss}")
    # loc_S 可以被视为传统的 logits
    print(f"输出 logits (loc_S) 形状: {outputs.logits.shape}")

    # --- 检查回归输出 ---
    # 在 forward 方法中，我们可以抽取出 loc_Y 和 scale_Y 进行检查
    # (为保持 main 函数简洁，此处不重复调用 forward)
    # 假设我们能拿到 loc_Y, 它的形状应该是 [batch_size, seq_len]
    print(f"回归值位置参数 (loc_Y) 的期望形状: [{input_ids.shape[0]}, {input_ids.shape[1]}]")