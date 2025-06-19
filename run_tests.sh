#!/bin/bash
# CausalQwen 测试运行脚本

echo "🧪 CausalQwen 测试套件"
echo "======================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 测试选项
if [ "$1" == "all" ]; then
    echo "运行所有测试（包括需要Qwen模型的测试）..."
    pytest tests/ -v
elif [ "$1" == "quick" ]; then
    echo "运行快速测试（排除慢速和需要Qwen模型的测试）..."
    pytest tests/ -v -m "not requires_qwen and not slow"
elif [ "$1" == "math" ]; then
    echo "运行数学框架测试..."
    pytest tests/test_math_framework.py -v
elif [ "$1" == "compatibility" ]; then
    echo "运行兼容性测试..."
    pytest tests/test_compatibility.py -v
elif [ "$1" == "generation" ]; then
    echo "运行生成功能测试..."
    pytest tests/test_generation.py -v
elif [ "$1" == "comparison" ]; then
    echo "运行与Qwen对比测试（需要Qwen模型）..."
    pytest tests/test_comparison.py -v
elif [ "$1" == "coverage" ]; then
    echo "运行测试并生成覆盖率报告..."
    pytest tests/ -v -m "not requires_qwen" --cov=causal_qwen_mvp --cov-report=html --cov-report=term
    echo -e "\n${GREEN}覆盖率报告已生成到 htmlcov/index.html${NC}"
else
    echo "使用方法："
    echo "  ./run_tests.sh          # 运行默认测试（不需要Qwen模型）"
    echo "  ./run_tests.sh quick    # 快速测试"
    echo "  ./run_tests.sh all      # 所有测试"
    echo "  ./run_tests.sh math     # 数学框架测试"
    echo "  ./run_tests.sh compatibility  # 兼容性测试"
    echo "  ./run_tests.sh generation     # 生成功能测试"
    echo "  ./run_tests.sh comparison     # 与Qwen对比测试"
    echo "  ./run_tests.sh coverage       # 生成覆盖率报告"
    echo ""
    echo "运行默认测试（不需要Qwen模型）..."
    pytest tests/ -v -m "not requires_qwen"
fi

# 显示测试结果统计
echo ""
echo -e "${YELLOW}测试完成！${NC}" 