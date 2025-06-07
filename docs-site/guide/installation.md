# 安装指南

本指南将帮助你安装和配置CausalQwen-0.5B项目。

## 系统要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.3+（如果使用GPU）
- 4GB+ RAM
- 10GB+ 磁盘空间

## 安装步骤

### 方法1：使用pip安装

```bash
# 从GitHub克隆仓库
git clone https://github.com/yourusername/causal-lm-project.git
cd causal-lm-project

# 安装项目及其依赖
pip install -e .
```

### 方法2：使用conda安装

```bash
# 从GitHub克隆仓库
git clone https://github.com/yourusername/causal-lm-project.git
cd causal-lm-project

# 创建conda环境
conda create -n causal-lm python=3.8
conda activate causal-lm

# 安装PyTorch（根据你的CUDA版本选择合适的命令）
# 对于CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装项目及其依赖
pip install -e .
```

### 方法3：使用Docker安装

```bash
# 从GitHub克隆仓库
git clone https://github.com/yourusername/causal-lm-project.git
cd causal-lm-project

# 构建Docker镜像
docker build -t causal-lm .

# 运行Docker容器
docker run -it --gpus all causal-lm
```

## 验证安装

安装完成后，可以运行以下命令验证安装是否成功：

```bash
# 运行测试
pytest tests/

# 运行示例
python examples/basic_example.py
```

如果测试和示例运行成功，说明安装已经完成。

## 安装Qwen-0.5B模型（可选）

如果你想使用Qwen-0.5B模型，需要额外安装：

```bash
# 安装transformers库
pip install transformers

# 下载Qwen-0.5B模型
python -c "from transformers import AutoModel, AutoTokenizer; model = AutoModel.from_pretrained('Qwen/Qwen-0.5B'); tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen-0.5B')"
```

## 常见问题

### 1. CUDA相关错误

如果遇到CUDA相关错误，请确保：

- 已安装兼容的CUDA版本
- PyTorch版本与CUDA版本匹配
- 显卡驱动已正确安装

可以使用以下命令检查PyTorch是否能够访问GPU：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
```

### 2. 依赖冲突

如果遇到依赖冲突，建议使用虚拟环境（如conda或venv）进行安装，以避免与系统其他包冲突。

### 3. 内存不足

如果运行时遇到内存不足的问题，可以：

- 减小批处理大小（batch size）
- 使用梯度累积
- 使用混合精度训练

## 下一步

安装完成后，你可以：

- 阅读[快速开始](/guide/quickstart.md)指南，了解基本用法
- 查看[API参考](/guide/api_reference.md)，了解详细的API文档
- 探索[示例](/guide/examples.md)，学习如何使用项目的各种功能

如果在安装过程中遇到任何问题，请查看[常见问题](/faq.md)或在GitHub上提交Issue。

