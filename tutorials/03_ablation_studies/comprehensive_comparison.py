"""
综合消融实验对比脚本
系统性地比较三种模型在所有数据集上的性能

运行方式:
python tutorials/03_ablation_studies/comprehensive_comparison.py --datasets all --output_dir results/
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from tutorials.utils.data_loaders import load_dataset, DATASET_LOADERS
from tutorials.utils.baseline_networks import (
    BaselineMLPClassifier, 
    BaselineMLPRegressor, 
    BaselineTrainer
)
from tutorials.utils.ablation_networks import (
    create_ablation_experiment,
    AblationTrainer
)
from tutorials.utils.evaluation_metrics import (
    calculate_classification_metrics, 
    calculate_regression_metrics,
    compare_model_performance, 
    plot_model_comparison,
    generate_evaluation_report, 
    statistical_significance_test
)


class ComprehensiveAblationStudy:
    """
    综合消融实验研究类
    
    执行三层对比实验：
    1. 传统神经网络基准
    2. CausalEngine消融版本（仅使用loc损失）
    3. 完整CausalEngine（使用完整因果损失）
    """
    
    def __init__(
        self,
        output_dir: str = "results",
        device: str = "auto",
        random_seed: int = 42,
        num_runs: int = 3
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.random_seed = random_seed
        self.num_runs = num_runs
        
        # 设置随机种子
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # 结果存储
        self.all_results = {}
        self.summary_results = {}
        
        print(f"🚀 初始化综合消融实验")
        print(f"   设备: {self.device}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   随机种子: {self.random_seed}")
        print(f"   实验轮数: {self.num_runs}")
    
    def run_single_experiment(
        self,
        dataset_name: str,
        run_id: int = 0
    ) -> Dict:
        """
        在单个数据集上运行一次完整的三模型对比实验
        
        Args:
            dataset_name: 数据集名称
            run_id: 实验轮次ID
            
        Returns:
            results: 实验结果字典
        """
        print(f"\n🔬 开始实验: {dataset_name} (第 {run_id + 1} 轮)")
        
        # 加载数据
        print("📊 加载数据...")
        data_dict = load_dataset(
            dataset_name,
            random_state=self.random_seed + run_id  # 每轮使用不同的随机种子
        )
        
        task_type = data_dict['task_type']
        input_size = data_dict['input_size']
        output_size = data_dict['output_size']
        
        print(f"   数据集: {dataset_name}")
        print(f"   任务类型: {task_type}")
        print(f"   输入维度: {input_size}")
        print(f"   输出维度: {output_size}")
        print(f"   训练样本: {data_dict['train_size']}")
        print(f"   验证样本: {data_dict['val_size']}")
        print(f"   测试样本: {data_dict['test_size']}")
        
        results = {
            'dataset': dataset_name,
            'task_type': task_type,
            'run_id': run_id,
            'data_info': {
                'input_size': input_size,
                'output_size': output_size,
                'train_size': data_dict['train_size'],
                'val_size': data_dict['val_size'],
                'test_size': data_dict['test_size']
            },
            'models': {}
        }
        
        # 1. 训练传统神经网络基准
        print("\n🏗️  训练传统神经网络基准...")
        start_time = time.time()
        
        if task_type == 'classification':
            baseline_model = BaselineMLPClassifier(
                input_dim=input_size,
                num_classes=output_size,
                hidden_dims=[512, 256],
                dropout=0.1
            )
        else:
            baseline_model = BaselineMLPRegressor(
                input_dim=input_size,
                output_dim=output_size,
                hidden_dims=[512, 256],
                dropout=0.1
            )
        
        baseline_trainer = BaselineTrainer(baseline_model, device=self.device)
        
        if task_type == 'classification':
            baseline_trainer.train_classification(
                data_dict['train_loader'],
                data_dict['val_loader'],
                num_epochs=50
            )
        else:
            baseline_trainer.train_regression(
                data_dict['train_loader'],
                data_dict['val_loader'],
                num_epochs=50
            )
        
        baseline_metrics = self._evaluate_baseline_model(
            baseline_model, 
            data_dict['test_loader'], 
            task_type
        )
        baseline_metrics['training_time'] = time.time() - start_time
        results['models']['baseline'] = baseline_metrics
        
        print(f"   基准模型训练完成 ({baseline_metrics['training_time']:.2f}s)")
        
        # 2. 创建CausalEngine（用于消融和完整版本）
        print("\n⚡ 创建CausalEngine...")
        
        engine, wrapper = create_ablation_experiment(
            input_dim=input_size,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            task_type=task_type,
            num_classes=output_size if task_type == 'classification' else None,
            output_dim=output_size if task_type == 'regression' else 1,
            dropout=0.1,
            device=self.device
        )
        
        ablation_trainer = AblationTrainer(engine, wrapper)
        
        # 3. 训练CausalEngine消融版本（仅使用loc损失）
        print("\n⚗️  训练CausalEngine消融版本...")
        start_time = time.time()
        
        # 准备输入 - 真实CausalEngine API
        def prepare_causal_inputs(batch_x):
            # 保持字典格式以兼容ablation_networks
            return {'values': batch_x}
        
        # 训练消融版本
        best_val_loss = float('inf')
        for epoch in range(50):
            train_loss = 0
            train_acc = 0 if task_type == 'classification' else None
            num_batches = 0
            
            for batch_x, batch_y in data_dict['train_loader']:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inputs = prepare_causal_inputs(batch_x)
                hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
                metrics = ablation_trainer.train_step_ablation(hidden_states, batch_y)
                
                train_loss += metrics['loss']
                if 'accuracy' in metrics:
                    train_acc += metrics['accuracy']
                num_batches += 1
            
            # 验证
            val_loss = 0
            val_acc = 0 if task_type == 'classification' else None
            val_batches = 0
            
            for batch_x, batch_y in data_dict['val_loader']:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inputs = prepare_causal_inputs(batch_x)
                hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
                metrics = ablation_trainer.eval_step(hidden_states, batch_y, use_ablation=True)
                
                val_loss += metrics['loss']
                if 'accuracy' in metrics:
                    val_acc += metrics['accuracy']
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 保存最佳模型状态
                best_ablation_state = engine.state_dict()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: train_loss={train_loss/num_batches:.4f}, val_loss={avg_val_loss:.4f}")
        
        # 恢复最佳模型
        engine.load_state_dict(best_ablation_state)
        
        ablation_metrics = self._evaluate_causal_model(
            engine, wrapper, data_dict['test_loader'], 
            task_type, use_ablation=True, prepare_fn=prepare_causal_inputs
        )
        ablation_metrics['training_time'] = time.time() - start_time
        results['models']['ablation'] = ablation_metrics
        
        print(f"   消融模型训练完成 ({ablation_metrics['training_time']:.2f}s)")
        
        # 4. 训练完整CausalEngine（使用完整因果损失）
        print("\n🌟 训练完整CausalEngine...")
        start_time = time.time()
        
        # 重新初始化模型参数
        engine, wrapper = create_ablation_experiment(
            input_dim=input_size,
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            task_type=task_type,
            num_classes=output_size if task_type == 'classification' else None,
            output_dim=output_size if task_type == 'regression' else 1,
            dropout=0.1,
            device=self.device
        )
        
        ablation_trainer = AblationTrainer(engine, wrapper)
        
        # 训练完整版本
        best_val_loss = float('inf')
        for epoch in range(50):
            train_loss = 0
            train_acc = 0 if task_type == 'classification' else None
            num_batches = 0
            
            for batch_x, batch_y in data_dict['train_loader']:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inputs = prepare_causal_inputs(batch_x)
                hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
                metrics = ablation_trainer.train_step_full(hidden_states, batch_y)
                
                train_loss += metrics['loss']
                if 'accuracy' in metrics:
                    train_acc += metrics['accuracy']
                num_batches += 1
            
            # 验证
            val_loss = 0
            val_acc = 0 if task_type == 'classification' else None
            val_batches = 0
            
            for batch_x, batch_y in data_dict['val_loader']:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inputs = prepare_causal_inputs(batch_x)
                hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
                metrics = ablation_trainer.eval_step(hidden_states, batch_y, use_ablation=False)
                
                val_loss += metrics['loss']
                if 'accuracy' in metrics:
                    val_acc += metrics['accuracy']
                val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 保存最佳模型状态
                best_full_state = engine.state_dict()
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}: train_loss={train_loss/num_batches:.4f}, val_loss={avg_val_loss:.4f}")
        
        # 恢复最佳模型
        engine.load_state_dict(best_full_state)
        
        full_metrics = self._evaluate_causal_model(
            engine, wrapper, data_dict['test_loader'], 
            task_type, use_ablation=False, prepare_fn=prepare_causal_inputs
        )
        full_metrics['training_time'] = time.time() - start_time
        results['models']['full_causal'] = full_metrics
        
        print(f"   完整模型训练完成 ({full_metrics['training_time']:.2f}s)")
        
        # 5. 计算模型比较
        comparison = compare_model_performance(results['models'], task_type)
        results['comparison'] = comparison
        
        # 6. 打印本轮结果摘要
        self._print_run_summary(results)
        
        return results
    
    def _evaluate_baseline_model(self, model, test_loader, task_type: str) -> Dict:
        """评估传统基准模型"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = model(batch_x)
                
                if task_type == 'classification':
                    preds = torch.argmax(outputs, dim=-1)
                    probs = torch.softmax(outputs, dim=-1)
                    all_probs.extend(probs.cpu().numpy())
                else:
                    preds = outputs.squeeze()
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # 计算指标
        if task_type == 'classification':
            metrics = calculate_classification_metrics(
                np.array(all_targets), 
                np.array(all_preds),
                np.array(all_probs) if all_probs else None
            )
        else:
            metrics = calculate_regression_metrics(
                np.array(all_targets), 
                np.array(all_preds)
            )
        
        return metrics
    
    def _evaluate_causal_model(
        self, engine, wrapper, test_loader, 
        task_type: str, use_ablation: bool, prepare_fn
    ) -> Dict:
        """评估CausalEngine模型"""
        engine.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                inputs = prepare_fn(batch_x)
                
                # 提取hidden_states从输入字典
                hidden_states = inputs['values'] if isinstance(inputs, dict) else inputs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(1)
                
                outputs = engine(
                    hidden_states=hidden_states,
                    do_sample=False,
                    temperature=1.0,
                    return_dict=True,
                    apply_activation=not use_ablation
                )
                
                if use_ablation:
                    # 消融版本：使用loc进行预测
                    loc = outputs['loc_S'][:, -1, :]  # [batch_size, vocab_size]
                    if task_type == 'classification':
                        logits = loc[:, :wrapper.num_classes]
                        preds = torch.argmax(logits, dim=-1)
                        probs = torch.softmax(logits, dim=-1)
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        preds = loc[:, 0]
                else:
                    # 完整版本：使用激活头输出
                    final_output = outputs['output'][:, -1, :]  # [batch_size, output_dim]
                    if task_type == 'classification':
                        preds = torch.argmax(final_output, dim=-1)
                        probs = torch.softmax(final_output, dim=-1)
                        all_probs.extend(probs.cpu().numpy())
                    else:
                        preds = final_output[:, 0]
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        # 计算指标
        if task_type == 'classification':
            metrics = calculate_classification_metrics(
                np.array(all_targets), 
                np.array(all_preds),
                np.array(all_probs) if all_probs else None
            )
        else:
            metrics = calculate_regression_metrics(
                np.array(all_targets), 
                np.array(all_preds)
            )
        
        return metrics
    
    def _print_run_summary(self, results: Dict):
        """打印单轮实验结果摘要"""
        task_type = results['task_type']
        models = results['models']
        
        print(f"\n📋 第 {results['run_id'] + 1} 轮实验结果摘要:")
        
        if task_type == 'classification':
            key_metric = 'accuracy'
            print("   模型                   | 准确率    | F1分数    | 训练时间")
            print("   --------------------- | --------- | --------- | ---------")
        else:
            key_metric = 'r2'
            print("   模型                   | R²        | RMSE      | 训练时间")
            print("   --------------------- | --------- | --------- | ---------")
        
        for model_name, metrics in models.items():
            model_display = {
                'baseline': '传统神经网络',
                'ablation': 'CausalEngine(消融)',
                'full_causal': 'CausalEngine(完整)'
            }.get(model_name, model_name)
            
            if task_type == 'classification':
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1_score', 0)
                time_str = f"{metrics.get('training_time', 0):.1f}s"
                print(f"   {model_display:21} | {acc:.4f}    | {f1:.4f}    | {time_str}")
            else:
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                time_str = f"{metrics.get('training_time', 0):.1f}s"
                print(f"   {model_display:21} | {r2:.4f}    | {rmse:.4f}  | {time_str}")
        
        # 显示最佳模型
        if 'comparison' in results and 'best_models' in results['comparison']:
            best_model = results['comparison']['best_models'].get(key_metric, 'unknown')
            best_display = {
                'baseline': '传统神经网络',
                'ablation': 'CausalEngine(消融)',
                'full_causal': 'CausalEngine(完整)'
            }.get(best_model, best_model)
            print(f"\n   🏆 本轮最佳模型 ({key_metric}): {best_display}")


def main():
    parser = argparse.ArgumentParser(description='运行CausalEngine综合消融实验')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        help='要测试的数据集列表，或使用"all"测试所有数据集')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (cuda/cpu/auto)')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='每个数据集的实验轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 处理数据集参数
    if args.datasets == ['all']:
        dataset_names = None  # 将测试所有数据集
    else:
        dataset_names = args.datasets
    
    # 创建实验实例
    study = ComprehensiveAblationStudy(
        output_dir=args.output_dir,
        device=args.device,
        random_seed=args.seed,
        num_runs=args.num_runs
    )
    
    # 运行实验
    try:
        results = study.run_single_experiment(dataset_names[0] if dataset_names else 'adult', 0)
        print("\n✅ 实验完成！")
        print(f"   结果保存在: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 