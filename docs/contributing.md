# 贡献指南

感谢你对CausalQwen-0.5B项目的关注！我们非常欢迎社区成员参与项目的开发和改进。本指南将帮助你了解如何为项目做出贡献。

## 行为准则

参与本项目的所有贡献者都应遵循以下行为准则：

- 尊重所有参与者，不论背景和经验水平
- 接受建设性的批评和反馈
- 关注项目的最佳利益
- 对其他社区成员表示同理心

## 如何贡献

### 报告问题

如果你发现了bug或有新的功能建议，请通过GitHub Issues报告：

1. 在提交新issue之前，请先搜索现有issues，避免重复
2. 使用清晰的标题和详细的描述
3. 对于bug报告，请包含：
   - 问题的详细描述
   - 复现步骤
   - 预期行为和实际行为
   - 环境信息（操作系统、Python版本、PyTorch版本等）
   - 如果可能，提供最小复现代码
4. 对于功能请求，请包含：
   - 功能的详细描述
   - 使用场景和动机
   - 可能的实现方案（如果有）

### 提交代码

如果你想为项目贡献代码，请按照以下步骤操作：

1. Fork项目仓库
2. 创建一个新的分支：
   ```bash
   git checkout -b feature/your-feature-name
   ```
   或
   ```bash
   git checkout -b fix/your-bug-fix
   ```
3. 在本地进行开发和测试
4. 确保代码符合项目的代码风格和质量标准
5. 提交你的更改：
   ```bash
   git commit -m "描述你的更改"
   ```
6. 推送到你的Fork：
   ```bash
   git push origin feature/your-feature-name
   ```
7. 创建一个Pull Request（PR）

### Pull Request流程

1. 确保PR有一个清晰的标题和详细的描述
2. 链接相关的issue（如果有）
3. 确保所有测试都通过
4. 等待代码审查
5. 根据反馈进行修改（如果需要）
6. 一旦PR被批准，它将被合并到主分支

## 开发指南

### 环境设置

1. 克隆仓库：
   ```bash
   git clone https://github.com/yourusername/causal-lm-project.git
   cd causal-lm-project
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate
   ```

3. 安装开发依赖：
   ```bash
   pip install -e ".[dev]"
   ```

### 代码风格

我们使用以下工具来保持代码质量和一致性：

- **Black**：用于代码格式化
- **isort**：用于导入排序
- **flake8**：用于代码风格检查
- **mypy**：用于类型检查

在提交代码之前，请运行以下命令：

```bash
# 格式化代码
black src tests examples

# 排序导入
isort src tests examples

# 代码风格检查
flake8 src tests examples

# 类型检查
mypy src
```

### 测试

我们使用pytest进行测试。在提交代码之前，请确保所有测试都通过：

```bash
pytest tests/
```

如果你添加了新功能，请同时添加相应的测试。

### 文档

如果你的更改影响了API或添加了新功能，请更新相应的文档：

- 更新函数和类的文档字符串
- 更新README.md（如果需要）
- 更新相关的文档页面

## 分支策略

- `main`：主分支，包含稳定的代码
- `develop`：开发分支，包含最新的开发代码
- `feature/*`：功能分支，用于开发新功能
- `fix/*`：修复分支，用于修复bug
- `release/*`：发布分支，用于准备新版本发布

## 版本控制

我们使用[语义化版本控制](https://semver.org/)：

- **主版本号**：当进行不兼容的API更改时增加
- **次版本号**：当添加向后兼容的功能时增加
- **修订号**：当进行向后兼容的bug修复时增加

## 发布流程

1. 更新版本号（在`setup.py`和`src/__init__.py`中）
2. 更新CHANGELOG.md
3. 创建一个新的发布分支：`release/vX.Y.Z`
4. 进行最终测试和文档更新
5. 合并到`main`分支
6. 创建一个新的标签：`vX.Y.Z`
7. 发布到PyPI（如果适用）

## 许可证

通过贡献代码，你同意你的贡献将在项目的许可证（MIT）下发布。

## 联系方式

如果你有任何问题或需要帮助，可以：

- 在GitHub上提交Issue
- 联系项目维护者：your.email@example.com

感谢你的贡献！

