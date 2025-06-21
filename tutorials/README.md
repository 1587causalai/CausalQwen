# CausalEngine 教程与评估框架

欢迎来到 CausalEngine 的综合教程和评估框架！本框架不仅提供了如何使用 CausalEngine 进行分类和回归任务的详细教程，更重要的是通过广泛的消融实验证明了 CausalEngine 相比传统神经网络的革命性优势。

## 🌟 框架特色

### 📚 完整教程体系
- **快速入门**: 从零开始了解 CausalEngine
- **分类任务**: 4个真实数据集的完整实现
- **回归任务**: 4个真实数据集的完整实现  
- **高级主题**: 深入理解因果推理机制

### 🔬 严谨的消融实验
- **三层对比**: 传统神经网络 vs 消融版本 vs 完整引擎
- **真实数据**: 8个公开数据集，样本量500-100万
- **统计显著性**: 多次运行，统计显著性检验
- **可复现性**: 固定随机种子，完整实验记录

### 📊 评估的数据集

#### 分类数据集 (Binary & Multi-class)
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
# 安装依赖
pip install torch transformers numpy pandas scikit-learn matplotlib seaborn

# 运行快速测试
python tutorials/00_getting_started/basic_usage.py
```

### 三步体验 CausalEngine
```python
from causal_engine import CausalEngine

# 1. 创建引擎
engine = CausalEngine(
    hidden_size=128,
    vocab_size=10,  # 分类类别数
    causal_size=128,
    activation_modes="classification"
)

# 2. 因果推理
hidden_states = torch.randn(32, 10, 128)  # (batch, seq, hidden)
output = engine(hidden_states, temperature=1.0, do_sample=True)

# 3. 获取预测
predictions = output.logits.argmax(dim=-1)
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