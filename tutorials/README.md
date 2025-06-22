# CausalEngine 教程与评估框架 (2024更新版)

欢迎来到 CausalEngine 的综合教程和评估框架！基于最新的基准测试协议和数学理论，本框架提供了完整的学习路径和科学验证体系。

## 🌟 2024年核心更新

### 📐 基于最新理论
- **数学基础**: 基于 `MATHEMATICAL_FOUNDATIONS_CN.md` 的完整三阶段架构
- **基准协议**: 遵循 `benchmark_strategy.md` 的标准化实验设计
- **四种推理模式**: 因果、标准、采样、兼容模式的统一框架
- **三种激活机制**: 词元、数值、有序分类的通用支持

### 🧪 科学实验设计
- **固定vs自适应噪声**: 通过 `b_noise.requires_grad` 的优雅控制
- **标准化配置**: AdamW优化器，学习率1e-4，早停机制
- **可复现性**: 统一随机种子，完整实验记录
- **统计验证**: 多轮实验，显著性检验，效应量分析

### 📊 评估的数据集

#### 分类数据集 (二元分类)
| 数据集 | 样本数 | 特征数 | 任务类型 | 领域 | 目标变量 (取值) |
|--------|--------|--------|----------|------|--------------------|
| Adult Income | 48,842 | 14 | 收入预测 | 人口统计 | `income` (2 classes: <=50K, >50K) |
| Bank Marketing | 41,188 | 20 | 营销响应 | 金融 | `y` (2 classes: no, yes) |
| Credit Default | 30,000 | 23 | 违约检测 | 风控 | `default payment next month` (2 classes: 0, 1) |
| Mushroom Safety | 8,124 | 22 | 安全分类 | 生物 | `class` (2 classes: edible, poisonous) |

#### 回归数据集 (Continuous Target)
| 数据集 | 样本数 | 特征数 | 任务类型 | 领域 | 目标变量 (含义) |
|--------|--------|--------|----------|------|--------------------|
| Bike Sharing | 17,379 | 16 | 需求预测 | 交通 | `cnt` (自行车日租量) |
| Wine Quality | 6,497 | 11 | 质量评分 | 食品 | `quality` (葡萄酒质量评分: 3-9) |
| Ames Housing | 2,919 | 79 | 房价预测 | 房地产 | `SalePrice` (房屋售价) |
| California Housing | 20,640 | 8* | 价值估计 | 房地产 | `MedHouseVal` (房屋价值中位数) |

*通过特征工程扩展至10+特征

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖 (基于基准协议)
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn

# 运行快速测试
python tutorials/00_getting_started/basic_usage.py
python tutorials/00_getting_started/benchmark_protocol_intro.py
```

### 基于基准协议的标准配置
```python
from causal_engine import CausalEngine
import torch
import torch.optim as optim

# 1. 创建引擎 (标准配置)
engine = CausalEngine(
    hidden_size=128,
    vocab_size=10,
    causal_size=128,
    activation_modes="classification"
)

# 2. 基准协议训练配置
optimizer = optim.AdamW(engine.parameters(), lr=1e-4, weight_decay=0.01)
# 学习率调度: 前10%线性warm-up，然后cosine decay

# 3. 四种推理模式
modes = {
    'causal': {'temperature': 0, 'do_sample': False},      # 纯因果推理
    'standard': {'temperature': 1.0, 'do_sample': False},  # 标准推理
    'sampling': {'temperature': 0.8, 'do_sample': True},   # 采样推理
    'compatible': {'temperature': 1.0, 'do_sample': False} # 兼容模式
}

# 4. 固定vs自适应噪声控制
engine.action.b_noise.requires_grad = True   # 自适应噪声学习
# engine.action.b_noise.requires_grad = False  # 固定噪声实验
```

## 📖 学习路径

### 初学者路径
1. **基础概念**: `00_getting_started/` - 理解因果vs相关
2. **简单任务**: `01_classification/adult_income_prediction.py` - 第一个分类项目
3. **回归实战**: `02_regression/bike_sharing_demand.py` - 第一个回归项目
4. **消融分析**: `03_ablation_studies/loc_only_vs_full_engine.py` - 理解优势

### 研究者路径
1. **理论基础**: 阅读 `causal_engine/MATHEMATICAL_FOUNDATIONS.md`
2. **全面评估**: 运行 `03_ablation_studies/comprehensive_comparison.py`
3. **高级分析**: 探索 `04_advanced_topics/` 目录
4. **自定义实验**: 基于框架开发新的评估

## 🔬 消融实验说明

### 实验设计原理
CausalEngine 的核心假设是：**仅使用位置输出(loc)时，它等价于传统神经网络**。通过这个消融实验，我们可以量化因果推理（位置+尺度）相比传统方法的提升。

### 三种算法对比
```python
# 1. 传统神经网络 (Baseline)
class TraditionalMLP:
    def forward(self, x):
        return self.layers(x)  # 直接映射

# 2. CausalEngine消融版本 (仅位置输出)
class CausalEngineAblated:
    def forward(self, x):
        return self.abduction_network.loc_net(x)  # 仅使用loc

# 3. 完整CausalEngine (位置+尺度)
class CausalEngineFull:
    def forward(self, x):
        return self.causal_reasoning(x)  # U ~ Cauchy(loc, scale)
```

### 评估指标
- **分类**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **回归**: MAE, MSE, RMSE, R², MAPE
- **不确定性**: 预测置信度分析
- **统计检验**: t-test, Wilcoxon signed-rank test

## 🎯 预期结果

基于CausalEngine的理论基础，我们预期：

1. **传统神经网络 ≈ CausalEngine消融版本**: 验证理论一致性
2. **完整CausalEngine > CausalEngine消融版本**: 证明因果推理优势
3. **温度效应**: 不同温度下性能的变化规律
4. **不确定性量化**: 更准确的预测置信度

## 📁 目录结构

```
tutorials/
├── 00_getting_started/     # 快速入门教程
├── 01_classification/      # 分类任务实战
├── 02_regression/          # 回归任务实战
├── 03_ablation_studies/    # 消融实验核心
├── 04_advanced_topics/     # 高级主题探索
└── utils/                  # 通用工具函数
```

## 🤝 贡献指南

欢迎贡献新的数据集、实验设计或分析方法！请遵循以下步骤：

1. Fork 项目并创建新分支
2. 添加你的实验或教程
3. 确保代码符合项目风格
4. 提交 Pull Request

## 📄 引用

如果你在研究中使用了这个框架，请引用：
```bibtex
@software{causal_engine_tutorials,
  title={CausalEngine Tutorial and Evaluation Framework},
  author={CausalQwen Team},
  year={2024},
  url={https://github.com/causalqwen/causal-engine}
}
```

---

🚀 **开始你的因果推理之旅，见证AI从模式匹配到因果理解的革命性转变！**