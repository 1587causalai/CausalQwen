#!/usr/bin/env python3
"""
CausalEngine 教程运行脚本
一键运行所有教程和实验的便捷入口

使用方法:
python run_tutorials.py --help
python run_tutorials.py --demo basic
python run_tutorials.py --demo classification
python run_tutorials.py --demo regression
python run_tutorials.py --demo ablation
python run_tutorials.py --demo all
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


class TutorialRunner:
    """CausalEngine教程运行器"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.tutorials_dir = self.base_dir / "tutorials"
        
        print("🌟 CausalEngine 教程运行器")
        print("=" * 50)
        print(f"基础目录: {self.base_dir}")
        print(f"教程目录: {self.tutorials_dir}")
        
    def run_basic_demo(self):
        """运行基础使用演示"""
        print("\n🚀 运行基础使用演示...")
        script_path = self.tutorials_dir / "00_getting_started" / "basic_usage.py"
        return self._run_script(script_path, "基础使用演示")
    
    def run_classification_demo(self):
        """运行分类任务演示"""
        print("\n🎯 运行分类任务演示...")
        script_path = self.tutorials_dir / "01_classification" / "adult_income_prediction.py"
        return self._run_script(script_path, "分类任务演示")
    
    def run_regression_demo(self):
        """运行回归任务演示"""
        print("\n📈 运行回归任务演示...")
        script_path = self.tutorials_dir / "02_regression" / "bike_sharing_demand.py"
        return self._run_script(script_path, "回归任务演示")
    
    def run_ablation_demo(self):
        """运行消融实验演示"""
        print("\n🔬 运行消融实验演示...")
        script_path = self.tutorials_dir / "03_ablation_studies" / "comprehensive_comparison.py"
        
        # 运行快速版本的消融实验（两个数据集，1轮）
        cmd = [
            sys.executable, str(script_path),
            "--datasets", "adult", "bike_sharing",
            "--num_runs", "1",
            "--output_dir", "results/quick_demo"
        ]
        
        return self._run_command(cmd, "消融实验演示")
    
    def run_comprehensive_experiments(self):
        """运行完整的综合实验"""
        print("\n🎯 运行完整综合实验...")
        print("⚠️  警告：这将运行所有8个数据集的完整实验，可能需要2-4小时")
        
        confirm = input("是否继续？(y/N): ").strip().lower()
        if confirm != 'y':
            print("已取消完整实验")
            return True
        
        script_path = self.tutorials_dir / "03_ablation_studies" / "comprehensive_comparison.py"
        
        cmd = [
            sys.executable, str(script_path),
            "--datasets", "all",
            "--num_runs", "3",
            "--output_dir", "results/comprehensive_evaluation"
        ]
        
        return self._run_command(cmd, "完整综合实验")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("\n🎉 运行所有演示...")
        
        demos = [
            ("基础使用", self.run_basic_demo),
            ("分类任务", self.run_classification_demo),
            ("回归任务", self.run_regression_demo),
            ("消融实验", self.run_ablation_demo)
        ]
        
        results = []
        for name, demo_func in demos:
            print(f"\n{'='*20} {name} {'='*20}")
            success = demo_func()
            results.append((name, success))
            
            if not success:
                print(f"❌ {name} 演示失败")
                break
            else:
                print(f"✅ {name} 演示完成")
        
        # 总结
        print(f"\n📊 演示结果总结:")
        for name, success in results:
            status = "✅ 成功" if success else "❌ 失败"
            print(f"   {name}: {status}")
        
        return all(success for _, success in results)
    
    def _run_script(self, script_path: Path, demo_name: str) -> bool:
        """运行单个Python脚本"""
        if not script_path.exists():
            print(f"❌ 脚本不存在: {script_path}")
            return False
        
        cmd = [sys.executable, str(script_path)]
        return self._run_command(cmd, demo_name)
    
    def _run_command(self, cmd: list, demo_name: str) -> bool:
        """运行命令并处理结果"""
        try:
            print(f"执行命令: {' '.join(cmd)}")
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1小时超时
            )
            
            elapsed_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ {demo_name} 成功完成 ({elapsed_time:.1f}s)")
                if result.stdout:
                    print("输出:")
                    print(result.stdout[-1000:])  # 显示最后1000字符
                return True
            else:
                print(f"❌ {demo_name} 失败 (返回码: {result.returncode})")
                if result.stderr:
                    print("错误信息:")
                    print(result.stderr[-1000:])  # 显示最后1000字符
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {demo_name} 超时")
            return False
        except Exception as e:
            print(f"❌ {demo_name} 运行出错: {str(e)}")
            return False
    
    def check_environment(self):
        """检查运行环境"""
        print("\n🔍 检查运行环境...")
        
        # 检查Python版本
        python_version = sys.version_info
        print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("⚠️  推荐使用Python 3.8+")
        
        # 检查关键依赖
        required_packages = [
            'torch', 'numpy', 'pandas', 'sklearn', 
            'matplotlib', 'seaborn', 'scipy'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (缺失)")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  缺失依赖包: {', '.join(missing_packages)}")
            print("请运行: pip install torch numpy pandas scikit-learn matplotlib seaborn scipy")
            return False
        
        # 检查目录结构
        required_dirs = [
            self.tutorials_dir,
            self.tutorials_dir / "00_getting_started",
            self.tutorials_dir / "01_classification", 
            self.tutorials_dir / "02_regression",
            self.tutorials_dir / "03_ablation_studies",
            self.tutorials_dir / "utils"
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✅ {dir_path.name}/")
            else:
                print(f"❌ {dir_path.name}/ (缺失)")
                return False
        
        print("✅ 环境检查通过")
        return True
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
🌟 CausalEngine 教程运行器 - 帮助信息

📖 可用演示:
  basic          - 基础使用演示 (5-10分钟)
  classification - 分类任务演示 (10-15分钟)  
  regression     - 回归任务演示 (10-15分钟)
  ablation       - 消融实验演示 (15-30分钟)
  all           - 运行所有演示 (30-60分钟)
  comprehensive - 完整综合实验 (2-4小时)

🚀 使用示例:
  python run_tutorials.py --demo basic
  python run_tutorials.py --demo classification
  python run_tutorials.py --demo all
  python run_tutorials.py --check-env

📁 输出文件:
  • 图表: tutorials/*/**.png
  • 报告: tutorials/*/**_report.md
  • 数据: data/
  • 结果: results/

🛠️ 故障排除:
  1. 运行环境检查: python run_tutorials.py --check-env
  2. 查看错误日志中的具体信息
  3. 确保所有依赖包已安装
  4. 检查磁盘空间是否充足 (推荐10GB+)

💡 提示:
  • 首次运行会自动下载数据集
  • 使用GPU可以显著加速训练
  • 可以通过设置CUDA_VISIBLE_DEVICES控制GPU使用
        """
        print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CausalEngine教程运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--demo',
        choices=['basic', 'classification', 'regression', 'ablation', 'all', 'comprehensive'],
        help='要运行的演示类型'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='检查运行环境'
    )
    
    parser.add_argument(
        '--help-detailed',
        action='store_true', 
        help='显示详细帮助信息'
    )
    
    args = parser.parse_args()
    
    runner = TutorialRunner()
    
    # 显示详细帮助
    if args.help_detailed:
        runner.show_help()
        return
    
    # 检查环境
    if args.check_env:
        success = runner.check_environment()
        sys.exit(0 if success else 1)
    
    # 如果没有指定演示类型，显示帮助
    if not args.demo:
        parser.print_help()
        print("\n💡 提示: 使用 --help-detailed 查看详细帮助")
        print("💡 提示: 使用 --check-env 检查运行环境")
        return
    
    # 检查环境（快速检查）
    if not runner.check_environment():
        print("❌ 环境检查失败，请先解决依赖问题")
        sys.exit(1)
    
    # 运行指定的演示
    try:
        if args.demo == 'basic':
            success = runner.run_basic_demo()
        elif args.demo == 'classification':
            success = runner.run_classification_demo()
        elif args.demo == 'regression':
            success = runner.run_regression_demo()
        elif args.demo == 'ablation':
            success = runner.run_ablation_demo()
        elif args.demo == 'all':
            success = runner.run_all_demos()
        elif args.demo == 'comprehensive':
            success = runner.run_comprehensive_experiments()
        else:
            print(f"❌ 未知的演示类型: {args.demo}")
            success = False
        
        if success:
            print(f"\n🎉 {args.demo} 演示成功完成！")
            print("\n📚 下一步建议:")
            if args.demo == 'basic':
                print("   • 尝试分类演示: python run_tutorials.py --demo classification")
            elif args.demo == 'classification':
                print("   • 尝试回归演示: python run_tutorials.py --demo regression")
            elif args.demo == 'regression':
                print("   • 尝试消融实验: python run_tutorials.py --demo ablation")
            elif args.demo == 'ablation':
                print("   • 运行完整实验: python run_tutorials.py --demo comprehensive")
            elif args.demo == 'all':
                print("   • 运行完整实验: python run_tutorials.py --demo comprehensive")
                print("   • 深入学习: 阅读 causal_engine/MATHEMATICAL_FOUNDATIONS.md")
        else:
            print(f"\n❌ {args.demo} 演示失败")
            print("💡 请检查错误信息并解决问题后重试")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  用户中断演示")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 演示运行出错: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()