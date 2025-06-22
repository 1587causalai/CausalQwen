#!/usr/bin/env python3
"""
CausalEngine 教程运行脚本 - 2024更新版
一键运行所有教程和实验的便捷入口，基于最新基准测试协议

使用方法:
python run_tutorials.py --help
python run_tutorials.py --demo basic
python run_tutorials.py --demo benchmark  
python run_tutorials.py --demo classification
python run_tutorials.py --demo regression
python run_tutorials.py --demo ablation
python run_tutorials.py --demo advanced
python run_tutorials.py --demo comprehensive
python run_tutorials.py --demo all
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path


class TutorialRunner2024:
    """CausalEngine教程运行器 - 2024版基于基准测试协议"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.tutorials_dir = self.base_dir / "tutorials"
        
        print("🌟 CausalEngine 教程运行器 (2024更新版)")
        print("基于最新基准测试协议和四种推理模式框架")
        print("=" * 60)
        print(f"基础目录: {self.base_dir}")
        print(f"教程目录: {self.tutorials_dir}")
        
    def run_basic_demo(self):
        """运行基础使用演示 (标准化配置)"""
        print("\n🚀 运行基础使用演示 (标准化配置)...")
        print("   演示内容: 因果推理vs传统方法, 四种推理模式对比")
        script_path = self.tutorials_dir / "00_getting_started" / "basic_usage.py"
        return self._run_script(script_path, "基础使用演示")
    
    def run_benchmark_protocol_demo(self):
        """运行基准测试协议演示 (2024新增)"""
        print("\n🧪 运行基准测试协议演示 (2024核心更新)...")
        print("   演示内容: 标准化配置, 固定vs自适应噪声, 实验设计")
        script_path = self.tutorials_dir / "00_getting_started" / "benchmark_protocol_intro.py"
        return self._run_script(script_path, "基准协议演示")
    
    def run_theoretical_foundations(self):
        """运行理论基础演示"""
        print("\n📐 运行理论基础演示 (最新三阶段架构)...")
        print("   演示内容: 归因→行动→激活, 柯西分布数学原理")
        script_path = self.tutorials_dir / "00_getting_started" / "theoretical_foundations.py"
        return self._run_script(script_path, "理论基础演示")
    
    def run_classification_demo(self):
        """运行分类任务演示 (基准配置)"""
        print("\n🎯 运行分类任务演示 (基准配置)...")
        print("   数据集: Adult Income, 配置: AdamW lr=1e-4")
        
        # 运行新的基准分类演示
        benchmark_script = self.tutorials_dir / "01_classification" / "benchmark_classification_demo.py"
        if benchmark_script.exists():
            print("   使用基准分类演示...")
            return self._run_script(benchmark_script, "基准分类演示")
        else:
            # 回退到原有演示
            print("   使用传统分类演示...")
            script_path = self.tutorials_dir / "01_classification" / "adult_income_prediction.py"
            return self._run_script(script_path, "分类任务演示")
    
    def run_regression_demo(self):
        """运行回归任务演示 (基准配置)"""
        print("\n📈 运行回归任务演示 (基准配置)...")
        print("   数据集: Bike Sharing, 配置: AdamW lr=1e-4")
        
        # 运行新的基准回归演示
        benchmark_script = self.tutorials_dir / "02_regression" / "benchmark_regression_demo.py"
        if benchmark_script.exists():
            print("   使用基准回归演示...")
            return self._run_script(benchmark_script, "基准回归演示")
        else:
            # 回退到原有演示
            print("   使用传统回归演示...")
            script_path = self.tutorials_dir / "02_regression" / "bike_sharing_demand.py"
            return self._run_script(script_path, "回归任务演示")
    
    def run_ablation_demo(self):
        """运行消融实验演示 (双重消融设计)"""
        print("\n🔬 运行消融实验演示 (双重消融设计)...")
        print("   实验一: 经典三层消融 (MLP vs CausalEngine-loc vs CausalEngine-full)")
        print("   实验二: 固定vs自适应噪声 (b_noise.requires_grad 布尔控制)")
        
        # 优先运行新的固定vs自适应噪声实验
        core_ablation_script = self.tutorials_dir / "03_ablation_studies" / "fixed_vs_adaptive_noise_study.py"
        comprehensive_script = self.tutorials_dir / "03_ablation_studies" / "comprehensive_comparison.py"
        
        if core_ablation_script.exists():
            print("   使用固定vs自适应噪声核心实验 (实验二)...")
            return self._run_script(core_ablation_script, "固定vs自适应噪声实验")
        elif comprehensive_script.exists():
            print("   使用经典三层消融实验 (实验一)...")
            # 运行快速版本的消融实验
            cmd = [
                sys.executable, str(comprehensive_script),
                "--datasets", "adult", "bike_sharing",
                "--num_runs", "1",
                "--output_dir", "results/quick_demo"
            ]
            
            return self._run_command(cmd, "经典三层消融实验")
        else:
            print("   ❌ 消融实验文件不存在")
            return False
    
    def run_advanced_topics_demo(self):
        """运行高级主题演示 (2024新增)"""
        print("\n🚀 运行高级主题演示 (四种推理模式深度分析)...")
        print("   内容: 因果/标准/采样/兼容模式, 任务激活机制")
        
        demos = [
            ("四种推理模式深度解析", "04_advanced_topics/four_inference_modes_deep_dive.py"),
            ("任务激活机制详解", "04_advanced_topics/task_activation_mechanisms.py")
        ]
        
        results = []
        for name, script_path in demos:
            full_path = self.tutorials_dir / script_path
            if full_path.exists():
                print(f"\n   运行: {name}")
                success = self._run_script(full_path, name)
                results.append((name, success))
                
                if not success:
                    print(f"   ❌ {name} 失败")
                    return False
                else:
                    print(f"   ✅ {name} 完成")
        
        return all(success for _, success in results)
    
    def run_comprehensive_experiments(self):
        """运行完整的综合实验 (经典三层消融)"""
        print("\n🎯 运行完整综合实验 (经典三层消融)...")
        print("实验设计: 传统MLP vs CausalEngine(仅loc) vs CausalEngine(完整)")
        print("⚠️  警告：这将运行所有数据集的完整实验，可能需要2-4小时")
        
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
        
        return self._run_command(cmd, "完整三层消融实验")
    
    def run_all_demos(self):
        """运行所有演示 (2024完整版)"""
        print("\n🎉 运行所有演示 (2024基准协议版)...")
        
        demos = [
            ("理论基础", self.run_theoretical_foundations),
            ("基准协议", self.run_benchmark_protocol_demo),
            ("基础使用", self.run_basic_demo),
            ("分类任务", self.run_classification_demo),
            ("回归任务", self.run_regression_demo),
            ("消融实验", self.run_ablation_demo),
            ("高级主题", self.run_advanced_topics_demo)
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
                time.sleep(2)  # 短暂暂停
        
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
                    # 显示关键输出信息
                    output_lines = result.stdout.split('\n')
                    important_lines = [
                        line for line in output_lines 
                        if any(keyword in line.lower() for keyword in [
                            '准确率', 'accuracy', 'r²', 'mae', 'rmse', 
                            '✅', '完成', 'complete', '结果', 'result'
                        ])
                    ]
                    
                    if important_lines:
                        print("关键结果:")
                        for line in important_lines[-5:]:  # 显示最后5个重要结果
                            print(f"   {line.strip()}")
                
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
        """检查运行环境 (2024版)"""
        print("\n🔍 检查运行环境 (2024基准协议要求)...")
        
        # 检查Python版本
        python_version = sys.version_info
        print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        if python_version < (3, 8):
            print("⚠️  推荐使用Python 3.8+")
        
        # 检查关键依赖 (基准协议需求)
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
        
        # 检查2024目录结构
        required_dirs = [
            self.tutorials_dir,
            self.tutorials_dir / "00_getting_started",
            self.tutorials_dir / "01_classification", 
            self.tutorials_dir / "02_regression",
            self.tutorials_dir / "03_ablation_studies",
            self.tutorials_dir / "04_advanced_topics",  # 2024新增
            self.tutorials_dir / "utils"
        ]
        
        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✅ {dir_path.name}/")
            else:
                print(f"❌ {dir_path.name}/ (缺失)")
                return False
        
        # 检查关键2024文件
        key_2024_files = [
            "00_getting_started/benchmark_protocol_intro.py",
            "00_getting_started/theoretical_foundations.py", 
            "03_ablation_studies/fixed_vs_adaptive_noise_study.py",
            "04_advanced_topics/four_inference_modes_deep_dive.py",
            "04_advanced_topics/task_activation_mechanisms.py"
        ]
        
        print("\n🔍 检查2024核心文件:")
        for file_path in key_2024_files:
            full_path = self.tutorials_dir / file_path
            if full_path.exists():
                print(f"✅ {file_path}")
            else:
                print(f"⚠️  {file_path} (可选)")
        
        print("✅ 环境检查通过")
        return True
    
    def show_help(self):
        """显示帮助信息 (2024版)"""
        help_text = """
🌟 CausalEngine 教程运行器 (2024版) - 帮助信息

📖 可用演示 (基于基准测试协议):
  basic          - 基础使用演示 (5-10分钟)
                   内容: 因果推理vs传统方法, 四种推理模式对比
                   
  benchmark      - 基准协议演示 (10分钟) [2024核心]
                   内容: 标准化配置, 固定vs自适应噪声设计
                   
  theoretical    - 理论基础演示 (10分钟) [2024更新]
                   内容: 三阶段架构, 柯西分布数学原理
                   
  classification - 分类任务演示 (10-15分钟) [基准配置]
                   数据集: Adult Income, 配置: AdamW lr=1e-4
                   
  regression     - 回归任务演示 (10-15分钟) [基准配置]  
                   数据集: Bike Sharing, 配置: AdamW lr=1e-4
                   
  ablation       - 消融实验演示 (15-30分钟) [双重消融设计]
                   实验一: 经典三层消融 (MLP vs CausalEngine-loc vs CausalEngine-full)
                   实验二: 固定vs自适应噪声 (b_noise.requires_grad)
                   
  advanced       - 高级主题演示 (20-30分钟) [2024新增]
                   内容: 四种推理模式深度分析, 任务激活机制
                   
  all           - 运行所有演示 (60-90分钟) [完整2024体验]
                   推荐: 完整学习路径体验
                   
  comprehensive - 完整综合实验 (2-4小时) [研究级经典三层消融]
                   传统MLP vs CausalEngine(仅loc) vs CausalEngine(完整)
                   所有数据集的完整科学验证

🚀 使用示例 (2024基准协议):
  # 快速体验 (推荐新用户)
  python run_tutorials.py --demo benchmark
  python run_tutorials.py --demo basic
  
  # 理论学习
  python run_tutorials.py --demo theoretical
  
  # 实际应用 (基准配置)
  python run_tutorials.py --demo classification
  python run_tutorials.py --demo regression
  
  # 科学实验 (双重消融设计)
  python run_tutorials.py --demo ablation
  
  # 高级主题 (四种推理模式)
  python run_tutorials.py --demo advanced
  
  # 完整体验
  python run_tutorials.py --demo all
  
  # 环境检查
  python run_tutorials.py --check-env

📁 输出文件 (2024标准):
  • 图表: tutorials/*/**.png
  • 报告: tutorials/*/**_report.md  
  • 实验结果: results/
  • 基准数据: data/

🧪 2024核心创新:
  ✨ 基准测试协议: 标准化AdamW配置
  ✨ 固定vs自适应噪声: 布尔开关实验设计
  ✨ 四种推理模式: 因果/标准/采样/兼容
  ✨ 三种激活机制: 分类/回归/有序分类

🛠️ 故障排除:
  1. 环境检查: python run_tutorials.py --check-env
  2. 查看错误日志的具体信息
  3. 确保所有依赖包已安装
  4. 检查磁盘空间 (推荐10GB+)
  5. 如果遇到CUDA问题，设置 CUDA_VISIBLE_DEVICES

💡 学习路径建议:
  初学者: benchmark → basic → classification → regression
  研究者: theoretical → ablation → advanced → comprehensive  
  工程师: basic → classification → regression → advanced

📚 深入学习资源:
  • 数学理论: causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md
  • 基准协议: causal_engine/misc/benchmark_strategy.md
  • 项目总结: tutorials/SUMMARY.md
  • 高级主题: tutorials/04_advanced_topics/README.md
        """
        print(help_text)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CausalEngine教程运行器 (2024版) - 基于基准测试协议",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--demo',
        choices=[
            'basic', 'benchmark', 'theoretical', 
            'classification', 'regression', 'ablation', 'advanced',
            'all', 'comprehensive'
        ],
        help='要运行的演示类型 (基于2024基准协议)'
    )
    
    parser.add_argument(
        '--check-env',
        action='store_true',
        help='检查运行环境 (包括2024新增要求)'
    )
    
    parser.add_argument(
        '--help-detailed',
        action='store_true', 
        help='显示详细帮助信息 (2024版功能)'
    )
    
    args = parser.parse_args()
    
    runner = TutorialRunner2024()
    
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
        print("\n💡 提示: 使用 --help-detailed 查看详细帮助 (2024版)")
        print("💡 提示: 使用 --check-env 检查运行环境")
        print("💡 推荐: 新用户从 --demo benchmark 开始")
        return
    
    # 检查环境（快速检查）
    if not runner.check_environment():
        print("❌ 环境检查失败，请先解决依赖问题")
        sys.exit(1)
    
    # 运行指定的演示
    try:
        if args.demo == 'basic':
            success = runner.run_basic_demo()
        elif args.demo == 'benchmark':
            success = runner.run_benchmark_protocol_demo()
        elif args.demo == 'theoretical':
            success = runner.run_theoretical_foundations()
        elif args.demo == 'classification':
            success = runner.run_classification_demo()
        elif args.demo == 'regression':
            success = runner.run_regression_demo()
        elif args.demo == 'ablation':
            success = runner.run_ablation_demo()
        elif args.demo == 'advanced':
            success = runner.run_advanced_topics_demo()
        elif args.demo == 'all':
            success = runner.run_all_demos()
        elif args.demo == 'comprehensive':
            success = runner.run_comprehensive_experiments()
        else:
            print(f"❌ 未知的演示类型: {args.demo}")
            success = False
        
        if success:
            print(f"\n🎉 {args.demo} 演示成功完成！")
            print("\n📚 下一步建议 (2024学习路径):")
            if args.demo == 'benchmark':
                print("   • 理论深入: python run_tutorials.py --demo theoretical")
                print("   • 基础实践: python run_tutorials.py --demo basic")
            elif args.demo == 'basic':
                print("   • 分类应用: python run_tutorials.py --demo classification")
                print("   • 理论学习: python run_tutorials.py --demo theoretical")
            elif args.demo == 'classification':
                print("   • 回归应用: python run_tutorials.py --demo regression")
            elif args.demo == 'regression':
                print("   • 核心实验: python run_tutorials.py --demo ablation")
            elif args.demo == 'ablation':
                print("   • 高级主题: python run_tutorials.py --demo advanced")
            elif args.demo == 'advanced':
                print("   • 完整体验: python run_tutorials.py --demo all")
            elif args.demo == 'all':
                print("   • 完整实验: python run_tutorials.py --demo comprehensive")
                print("   • 深入学习: 阅读 causal_engine/MATHEMATICAL_FOUNDATIONS_CN.md")
                print("   • 基准协议: 阅读 causal_engine/misc/benchmark_strategy.md")
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