# CausalQwen 用户教程 - 快速开始

🎉 **恭喜！您已经成功创建了面向用户的 CausalQwen 教程体系！**

## 🚀 立即开始

### 方法一：交互式启动器（推荐）
```bash
python user_tutorials/run_user_tutorials.py
```

### 方法二：直接运行教程
```bash
# 环境检查
python user_tutorials/01_quick_start/installation.py

# 第一个完整示例
python user_tutorials/01_quick_start/first_example.py

# 分类任务实战
python user_tutorials/02_classification/synthetic_data.py
python user_tutorials/02_classification/iris_dataset.py

# 回归任务实战  
python user_tutorials/03_regression/synthetic_data.py
python user_tutorials/03_regression/boston_housing.py
```

### 方法三：非交互模式（用于自动化）
```bash
# 自动选择分类任务
python user_tutorials/01_quick_start/first_example.py 1

# 自动选择回归任务
python user_tutorials/01_quick_start/first_example.py 2

# 自动选择第一个场景
python user_tutorials/02_classification/synthetic_data.py 1
```

## 🧪 验证安装

```bash
# 验证所有功能是否正常
python user_tutorials/test_imports.py
python user_tutorials/validate_tutorials.py
```

## 📁 教程结构

```
user_tutorials/
├── README.md                    # 详细的用户指南
├── QUICK_START.md              # 这个文件
├── run_user_tutorials.py       # 交互式启动器
├── test_imports.py             # 导入测试
├── validate_tutorials.py       # 完整验证
│
├── 01_quick_start/             # 5分钟快速体验
│   ├── installation.py        # 环境检查
│   ├── first_example.py       # 第一个完整示例
│   └── test_basic_demo.py      # 基础功能测试
│
├── 02_classification/          # 分类任务实战
│   ├── synthetic_data.py       # 合成数据分类
│   └── iris_dataset.py         # 鸢尾花真实数据
│
├── 03_regression/              # 回归任务实战
│   ├── synthetic_data.py       # 合成数据回归
│   └── boston_housing.py       # 房价预测
│
└── utils/                      # 工具函数库
    ├── __init__.py            # 包初始化
    ├── simple_models.py       # 用户友好的模型
    └── data_helpers.py        # 数据处理工具
```

## 🎯 教程特色

### ✨ 零门槛设计
- 无需深入数学背景
- 详细中文注释
- 预配置最佳参数

### 📈 渐进式学习
- 从5分钟快速体验开始
- 覆盖分类和回归任务
- 包含真实数据集实战

### 🔧 实用工具
- 简化的模型接口
- 自动数据预处理
- 结果可视化

### 🌟 独特优势
- **因果推理**：理解真正的因果关系
- **不确定性量化**：知道预测的可信度
- **多种推理模式**：确定性、标准、探索性
- **可解释性**：清楚了解预测原因

## 🆚 与开发者教程的区别

| 特性 | 开发者教程 (`tutorials/`) | 用户教程 (`user_tutorials/`) |
|------|-------------------------|----------------------------|
| 目标用户 | 研究者、算法开发者 | 业务用户、数据分析师 |
| 技术深度 | 深入算法、消融实验 | 专注应用和效果 |
| 数学要求 | 需要理论基础 | 零数学门槛 |
| 使用难度 | 较高学习曲线 | 即学即用 |

## 💡 最佳实践

1. **初次使用**：从 `first_example.py` 开始
2. **学习分类**：先运行合成数据，再尝试真实数据
3. **学习回归**：了解不确定性量化的价值
4. **实际应用**：使用 `utils/` 中的工具构建自己的项目

## 🔗 获取帮助

- 查看教程中的详细注释
- 运行 `installation.py` 检查环境
- 参考 `utils/` 中的工具文档
- 对比不同推理模式的效果

---

🎉 **开始您的因果推理之旅！体验从相关性到因果性的AI革命！**