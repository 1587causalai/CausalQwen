# CausalQwen 项目结构

> 🎉 **项目状态**: CausalEngine已达到预期性能，回归和分类任务均表现优秀！

## 📦 核心目录结构

```
CausalQwen/
├── 🧠 causal_engine/                 # 核心CausalEngine算法
│   ├── engine.py                    # 主引擎：三阶段因果推理
│   ├── networks.py                  # AbductionNetwork & ActionNetwork
│   ├── heads.py                     # ActivationHead多任务支持
│   └── sklearn/                     # sklearn风格接口
│       ├── classifier.py            # MLPCausalClassifier
│       ├── regressor.py             # MLPCausalRegressor
│       └── base.py                  # 共享基类
│
├── 🧪 quick_test_causal_engine.py    # 主要测试脚本（当前最优）
│
├── 👥 user_tutorials/               # 用户友好教程
│   ├── 01_quick_start/             # 5分钟入门
│   ├── 02_classification/          # 分类教程
│   ├── 03_regression/              # 回归教程
│   └── 04_real_world_examples/     # 真实数据集基准测试
│
├── 🔬 tutorials/                    # 开发者高级教程
│   ├── 00_getting_started/         # 理论基础
│   ├── 01_classification/          # 深度分类案例
│   ├── 02_regression/              # 深度回归案例
│   ├── 03_ablation_studies/        # 消融研究
│   └── 04_advanced_topics/         # 高级主题
│
├── 🏗️ src/causal_qwen_mvp/          # CausalQwen应用层
│   ├── models.py                   # 与Qwen模型集成
│   ├── inference.py                # 推理引擎
│   └── config.py                   # 配置管理
│
├── ⚙️ tests/                        # 单元测试套件
├── 📊 results/                     # 基准测试结果
├── 📚 docs/                        # 技术文档
├── 📝 blog/                        # 技术博客文章
└── 📋 README.md                    # 项目主页
```

## 🎯 当前项目状态

### ✅ 已完成并验证
- **CausalEngine核心算法**: 三阶段因果推理架构完整
- **sklearn接口**: MLPCausalClassifier & MLPCausalRegressor性能优秀
- **数学基础**: 移除非必要复杂逻辑，基于柯西分布的优雅实现
- **性能验证**: Standard模式在分类和回归任务上均超越基线

### 🧹 已清理优化
- **移除梯度裁剪**: 基于柯西分布天然梯度饱和特性
- **移除scale_S正则化**: 让模型自由学习最优尺度参数
- **简化损失函数**: 移除复杂条件判断，提高代码可读性
- **统一优化器**: 使用Adam确保公平对比

### 📈 性能亮点
- **分类任务**: Standard模式达到~80%准确率，超越sklearn和PyTorch基线
- **回归任务**: 在多个数据集上表现优秀
- **训练效率**: 收敛速度显著快于传统方法
- **代码质量**: 简洁、数学上合理、易于维护

## 🚀 使用入口

### 快速测试
```bash
python quick_test_causal_engine.py
```

### 用户教程
```bash
python user_tutorials/run_user_tutorials.py
```

### 开发者教程  
```bash
python run_tutorials.py
```

## 📝 下一步计划

1. **项目整理**: 清理历史遗留的调试文件
2. **文档完善**: 更新README反映当前优秀状态
3. **发布准备**: 准备CausalEngine的正式发布版本

---

> 🎉 **里程碑**: CausalEngine经过深度优化，已达到预期的优秀性能！