"""CausalQwen 快速对话测试脚本"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass


@dataclass
class QuickConfig:
    """快速测试配置"""
    num_vocab: int = 32000  # Qwen tokenizer大小
    hidden_dim: int = 512
    num_token_id: int = 32000  # <NUM> token (第一个保留词汇)
    eos_token_id: int = 2


class MockCausalQwen:
    """Mock版本的CausalQwen，用于演示"""
    
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        self.qwen_model = None  # 存储原始Qwen模型用于参考
        
    def set_tokenizer(self, tokenizer):
        """设置tokenizer"""
        self.tokenizer = tokenizer
        
    def set_qwen_model(self, model):
        """设置Qwen模型（用于知识迁移）"""
        self.qwen_model = model
        
    def chat(self, messages, stream=False, **kwargs):
        """兼容Qwen的chat接口
        
        Args:
            messages: 对话历史
            stream: 是否流式输出
            
        Returns:
            生成的回复文本或生成器
        """
        # 如果有Qwen模型，使用它来生成（但不用chat方法）
        if self.qwen_model is not None and self.tokenizer is not None:
            # 构建prompt
            prompt = self._build_chat_prompt(messages)
            
            # 使用generate方法
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    inputs.input_ids,
                    max_new_tokens=kwargs.get('max_new_tokens', 100),
                    temperature=kwargs.get('temperature', 0.8),
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            assistant_reply = self._extract_assistant_reply(full_response, prompt)
            
            if stream:
                # 模拟流式输出
                for i in range(0, len(assistant_reply), 3):
                    yield assistant_reply[i:i+3]
            else:
                return assistant_reply
        
        # 否则使用简单的模拟
        prompt = self._build_chat_prompt(messages)
        response = self.generate_text(
            prompt,
            max_new_tokens=kwargs.get('max_new_tokens', 50),
            temperature=kwargs.get('temperature', 0.8),
            top_k=kwargs.get('top_k', 40),
            top_p=kwargs.get('top_p', 0.9),
            sampling_mode=kwargs.get('sampling_mode', 'causal')
        )
        
        assistant_reply = self._extract_assistant_reply(response, prompt)
        
        if stream:
            # 正确处理流式输出
            for i in range(0, len(assistant_reply), 3):
                yield assistant_reply[i:i+3]
        else:
            return assistant_reply
            
    def _build_chat_prompt(self, messages):
        """构建对话prompt（简化版）"""
        prompt = ""
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        return prompt
        
    def _extract_assistant_reply(self, full_response, prompt):
        """从完整回复中提取助手回复"""
        # 简单地去掉prompt部分
        if full_response.startswith(prompt):
            return full_response[len(prompt):].strip()
        return full_response.strip()
        
    def generate_text(self, prompt, **kwargs):
        """文本生成（Mock实现）"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set")
            
        # 模拟生成
        responses = {
            "你好呀！": "你好！很高兴见到你。有什么我可以帮助你的吗？",
            "今天天气怎么样？": "作为AI，我无法直接感知天气，但我建议你查看天气预报或看看窗外。希望今天是个好天气！",
            "请告诉我一个笑话": "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25！（八进制31等于十进制25）",
            "Hello, how are you?": "Hello! I'm doing well, thank you for asking. How can I assist you today?"
        }
        
        # 简单匹配返回
        for key, value in responses.items():
            if key in prompt:
                return prompt + value
                
        # 默认回复
        return prompt + "这是一个有趣的问题！让我想想..."
        
    def generate(self, input_ids, **kwargs):
        """兼容transformers的generate接口"""
        # 这里应该实现真正的生成逻辑
        # 现在只是返回mock结果
        batch_size = input_ids.shape[0]
        max_length = input_ids.shape[1] + kwargs.get('max_new_tokens', 50)
        
        # 模拟生成一些随机tokens
        new_tokens = torch.randint(100, 30000, (batch_size, kwargs.get('max_new_tokens', 50)))
        return torch.cat([input_ids, new_tokens], dim=1)


def create_mock_model():
    """创建一个简化的模型用于演示"""
    config = QuickConfig()
    model = MockCausalQwen(config)
    
    # 加载Qwen tokenizer
    model_path = os.path.expanduser("~/models/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.set_tokenizer(tokenizer)
    
    # 可选：加载Qwen模型（用于知识迁移演示）
    try:
        print("📚 加载Qwen模型用于知识迁移...")
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu"
        )
        model.set_qwen_model(qwen_model)
        print("✅ Qwen模型加载成功！")
    except Exception as e:
        print(f"⚠️ Qwen模型加载失败，使用简单模拟: {e}")
    
    return model


def main():
    """主函数"""
    print("🤖 CausalQwen 快速对话测试")
    print("=" * 50)
    
    # 创建模型
    print("📊 加载模型...")
    model = create_mock_model()
    
    print("✅ 模型加载完成！")
    print("\n💡 输入 'quit' 退出")
    print("-" * 50)
    
    # 对话历史
    messages = []
    
    while True:
        # 获取用户输入
        try:
            user_input = input("\n👤 你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出', 'q']:
                print("👋 再见！")
                break
                
            if not user_input:
                continue
                
            # 添加到对话历史
            messages.append({"role": "user", "content": user_input})
            
            # 生成回复
            print("🤖 CausalQwen: ", end="", flush=True)
            
            try:
                # 使用chat接口
                response = model.chat(
                    messages,
                    stream=True,  # 使用流式输出
                    max_new_tokens=100,
                    temperature=0.8,
                    sampling_mode="causal"
                )
                
                # 正确处理流式输出
                full_response = ""
                if hasattr(response, '__iter__'):
                    for chunk in response:
                        print(chunk, end="", flush=True)
                        full_response += chunk
                else:
                    print(response)
                    full_response = response
                
                print()  # 换行
                
                # 添加回复到历史
                messages.append({"role": "assistant", "content": full_response})
                
                # 保持对话历史在合理长度
                if len(messages) > 10:
                    messages = messages[-10:]
                
            except Exception as e:
                print(f"❌ 生成出错: {e}")
                import traceback
                traceback.print_exc()
                
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")


if __name__ == "__main__":
    main()
