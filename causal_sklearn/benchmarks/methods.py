"""
基准测试方法实现模块

包含所有传统机器学习方法的统一实现，支持回归和分类任务。
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Any, Optional

# 核心依赖
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

# 神经网络方法
from sklearn.neural_network import MLPRegressor, MLPClassifier

# 集成方法
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)

# 支持向量机
from sklearn.svm import SVR, SVC

# 线性方法
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Ridge, RidgeClassifier,
    Lasso, LassoCV,
    ElasticNet, ElasticNetCV
)

# K近邻
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

# 决策树
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

# 可选依赖（优雅降级）
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    # XGBoost warning will be shown only when explicitly requested

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    # LightGBM warning will be shown only when explicitly requested

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    # CatBoost warning will be shown only when explicitly requested

warnings.filterwarnings('ignore')


class BaselineMethodFactory:
    """
    基准方法工厂类
    
    负责创建和管理所有传统机器学习基准方法的实例化、训练和评估。
    """
    
    def __init__(self):
        # 导入PyTorch用于实现稳健MLP
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            warnings.warn("PyTorch不可用，稳健MLP方法将被跳过")
        
        self.available_methods = self._get_available_methods()
    
    def _get_available_methods(self) -> Dict[str, bool]:
        """检查各种方法的可用性"""
        return {
            'sklearn_mlp': True,
            'pytorch_mlp': True,  # 在base.py中已实现
            'mlp_huber': self.torch_available,  # 稳健MLP需要PyTorch
            'mlp_pinball_median': self.torch_available,  # 稳健MLP需要PyTorch
            'mlp_cauchy': self.torch_available,  # 稳健MLP需要PyTorch
            'random_forest': True,
            'extra_trees': True,
            'gradient_boosting': True,
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'catboost': CATBOOST_AVAILABLE,
            'svm_rbf': True,
            'svm_linear': True,
            'svm_poly': True,
            'linear_regression': True,
            'ridge': True,
            'lasso': True,
            'elastic_net': True,
            'knn': True,
            'decision_tree': True
        }
    
    def is_method_available(self, method_name: str) -> bool:
        """检查指定方法是否可用"""
        return self.available_methods.get(method_name, False)
    
    def create_model(self, method_name: str, task_type: str, **params) -> Any:
        """
        创建指定的机器学习模型
        
        Args:
            method_name: 方法名称
            task_type: 'regression' 或 'classification'
            **params: 方法特定参数
        
        Returns:
            sklearn兼容的模型实例
        """
        if not self.is_method_available(method_name):
            raise ValueError(f"方法 {method_name} 不可用或未安装相关依赖")
        
        # 移除不相关的参数
        filtered_params = self._filter_params(method_name, params)
        
        # 创建模型
        if method_name == 'sklearn_mlp':
            return self._create_sklearn_mlp(task_type, **filtered_params)
        elif method_name == 'pytorch_mlp':
            return self._create_pytorch_mlp(task_type, **filtered_params)
        elif method_name == 'mlp_huber':
            return self._create_robust_mlp(task_type, 'huber', **filtered_params)
        elif method_name == 'mlp_pinball_median':
            return self._create_robust_mlp(task_type, 'pinball', **filtered_params)
        elif method_name == 'mlp_cauchy':
            return self._create_robust_mlp(task_type, 'cauchy', **filtered_params)
        elif method_name == 'random_forest':
            return self._create_random_forest(task_type, **filtered_params)
        elif method_name == 'extra_trees':
            return self._create_extra_trees(task_type, **filtered_params)
        elif method_name == 'gradient_boosting':
            return self._create_gradient_boosting(task_type, **filtered_params)
        elif method_name == 'xgboost':
            return self._create_xgboost(task_type, **filtered_params)
        elif method_name == 'lightgbm':
            return self._create_lightgbm(task_type, **filtered_params)
        elif method_name == 'catboost':
            return self._create_catboost(task_type, **filtered_params)
        elif method_name.startswith('svm_'):
            return self._create_svm(method_name, task_type, **filtered_params)
        elif method_name == 'linear_regression':
            return self._create_linear(task_type, **filtered_params)
        elif method_name == 'ridge':
            return self._create_ridge(task_type, **filtered_params)
        elif method_name == 'lasso':
            return self._create_lasso(task_type, **filtered_params)
        elif method_name == 'elastic_net':
            return self._create_elastic_net(task_type, **filtered_params)
        elif method_name == 'knn':
            return self._create_knn(task_type, **filtered_params)
        elif method_name == 'decision_tree':
            return self._create_decision_tree(task_type, **filtered_params)
        else:
            raise ValueError(f"未知的方法: {method_name}")
    
    def _filter_params(self, method_name: str, params: Dict) -> Dict:
        """过滤掉不相关的参数"""
        # 通用参数映射
        common_mappings = {
            'learning_rate_init': ['learning_rate_init'],
            'max_iter': ['max_iter', 'n_estimators', 'iterations'],
            'random_state': ['random_state', 'seed']
        }
        
        # 方法特定的参数过滤
        if method_name in ['xgboost', 'lightgbm', 'catboost']:
            # Boosting方法特定处理
            filtered = {}
            for k, v in params.items():
                if k in ['n_estimators', 'learning_rate', 'max_depth', 'random_state', 
                        'min_child_weight', 'subsample', 'colsample_bytree', 'verbosity',
                        'num_leaves', 'min_child_samples', 'iterations', 'depth', 
                        'l2_leaf_reg', 'verbose', 'thread_count', 'n_jobs']:
                    filtered[k] = v
            return filtered
        
        return params
    
    # === 具体模型创建方法 ===
    
    def _create_sklearn_mlp(self, task_type: str, **params) -> Any:
        """创建sklearn MLP模型"""
        if task_type == 'regression':
            return MLPRegressor(**params)
        else:
            return MLPClassifier(**params)
    
    def _create_pytorch_mlp(self, task_type: str, **params) -> Any:
        """创建PyTorch MLP模型"""
        from ..regressor import MLPPytorchRegressor
        if task_type == 'regression':
            return MLPPytorchRegressor(**params)
        else:
            raise NotImplementedError("PyTorch MLP目前只支持回归任务")
    
    def _create_random_forest(self, task_type: str, **params) -> Any:
        """创建随机森林模型"""
        if task_type == 'regression':
            return RandomForestRegressor(**params)
        else:
            return RandomForestClassifier(**params)
    
    def _create_extra_trees(self, task_type: str, **params) -> Any:
        """创建极端随机树模型"""
        if task_type == 'regression':
            return ExtraTreesRegressor(**params)
        else:
            return ExtraTreesClassifier(**params)
    
    def _create_gradient_boosting(self, task_type: str, **params) -> Any:
        """创建梯度提升模型"""
        if task_type == 'regression':
            return GradientBoostingRegressor(**params)
        else:
            return GradientBoostingClassifier(**params)
    
    def _create_xgboost(self, task_type: str, **params) -> Any:
        """创建XGBoost模型"""
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost未安装")
        
        if task_type == 'regression':
            return xgb.XGBRegressor(**params)
        else:
            return xgb.XGBClassifier(**params)
    
    def _create_lightgbm(self, task_type: str, **params) -> Any:
        """创建LightGBM模型"""
        if not LIGHTGBM_AVAILABLE:
            raise RuntimeError("LightGBM未安装")
        
        if task_type == 'regression':
            return lgb.LGBMRegressor(**params)
        else:
            return lgb.LGBMClassifier(**params)
    
    def _create_catboost(self, task_type: str, **params) -> Any:
        """创建CatBoost模型"""
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost未安装")
        
        if task_type == 'regression':
            return cb.CatBoostRegressor(**params)
        else:
            return cb.CatBoostClassifier(**params)
    
    def _create_svm(self, method_name: str, task_type: str, **params) -> Any:
        """创建SVM模型"""
        kernel = method_name.split('_')[1]  # 从 'svm_rbf' 提取 'rbf'
        params['kernel'] = kernel
        
        if task_type == 'regression':
            return SVR(**params)
        else:
            return SVC(**params)
    
    def _create_linear(self, task_type: str, **params) -> Any:
        """创建线性模型"""
        if task_type == 'regression':
            return LinearRegression(**params)
        else:
            return LogisticRegression(**params)
    
    def _create_ridge(self, task_type: str, **params) -> Any:
        """创建Ridge模型"""
        if task_type == 'regression':
            return Ridge(**params)
        else:
            return RidgeClassifier(**params)
    
    def _create_lasso(self, task_type: str, **params) -> Any:
        """创建Lasso模型"""
        if task_type == 'regression':
            return Lasso(**params)
        else:
            # 分类任务使用L1正则化的Logistic回归
            return LogisticRegression(penalty='l1', solver='liblinear', **params)
    
    def _create_elastic_net(self, task_type: str, **params) -> Any:
        """创建Elastic Net模型"""
        if task_type == 'regression':
            return ElasticNet(**params)
        else:
            # 分类任务使用Elastic Net正则化的Logistic回归
            return LogisticRegression(penalty='elasticnet', solver='saga', **params)
    
    def _create_knn(self, task_type: str, **params) -> Any:
        """创建K近邻模型"""
        if task_type == 'regression':
            return KNeighborsRegressor(**params)
        else:
            return KNeighborsClassifier(**params)
    
    def _create_decision_tree(self, task_type: str, **params) -> Any:
        """创建决策树模型"""
        if task_type == 'regression':
            return DecisionTreeRegressor(**params)
        else:
            return DecisionTreeClassifier(**params)
    
    def _create_robust_mlp(self, task_type: str, loss_type: str, **params) -> Any:
        """创建稳健MLP模型"""
        if not self.torch_available:
            raise RuntimeError("PyTorch未安装，无法创建稳健MLP模型")
        
        # 导入新的sklearn兼容稳健MLP类
        from ..regressor import MLPHuberRegressor, MLPPinballRegressor, MLPCauchyRegressor
        
        if task_type != 'regression':
            raise NotImplementedError("稳健MLP目前只支持回归任务")
        
        if loss_type == 'huber':
            return MLPHuberRegressor(**params)
        elif loss_type == 'pinball':
            return MLPPinballRegressor(**params)
        elif loss_type == 'cauchy':
            return MLPCauchyRegressor(**params)
        else:
            raise ValueError(f"不支持的损失类型: {loss_type}")
    
    def train_and_evaluate(self, method_name: str, model: Any, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          task_type: str) -> Dict[str, Dict[str, float]]:
        """
        训练模型并评估性能
        
        Returns:
            包含验证集和测试集性能指标的字典
        """
        # 训练模型 - 支持早停的方法使用验证集
        if supports_early_stopping(method_name):
            model = fit_with_early_stopping(
                method_name, model, X_train, y_train, X_val, y_val, task_type
            )
        else:
            model.fit(X_train, y_train)
        
        # 预测
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        
        # 计算指标
        if task_type == 'regression':
            return {
                'val': {
                    'MAE': mean_absolute_error(y_val, pred_val),
                    'MdAE': median_absolute_error(y_val, pred_val),
                    'RMSE': np.sqrt(mean_squared_error(y_val, pred_val)),
                    'R²': r2_score(y_val, pred_val)
                },
                'test': {
                    'MAE': mean_absolute_error(y_test, pred_test),
                    'MdAE': median_absolute_error(y_test, pred_test),
                    'RMSE': np.sqrt(mean_squared_error(y_test, pred_test)),
                    'R²': r2_score(y_test, pred_test)
                }
            }
        else:
            n_classes = len(np.unique(y_test))
            avg_method = 'binary' if n_classes == 2 else 'macro'
            
            return {
                'val': {
                    'Acc': accuracy_score(y_val, pred_val),
                    'Precision': precision_score(y_val, pred_val, average=avg_method, zero_division=0),
                    'Recall': recall_score(y_val, pred_val, average=avg_method, zero_division=0),
                    'F1': f1_score(y_val, pred_val, average=avg_method, zero_division=0)
                },
                'test': {
                    'Acc': accuracy_score(y_test, pred_test),
                    'Precision': precision_score(y_test, pred_test, average=avg_method, zero_division=0),
                    'Recall': recall_score(y_test, pred_test, average=avg_method, zero_division=0),
                    'F1': f1_score(y_test, pred_test, average=avg_method, zero_division=0)
                }
            }


class MethodDependencyChecker:
    """
    方法依赖检查器
    
    检查和报告各种基准方法的依赖可用性。
    """
    
    @staticmethod
    def check_all_dependencies() -> Dict[str, Dict[str, Any]]:
        """检查所有依赖的可用性"""
        dependencies = {
            'xgboost': {
                'available': XGBOOST_AVAILABLE,
                'install_cmd': 'pip install xgboost',
                'description': 'Gradient Boosting框架'
            },
            'lightgbm': {
                'available': LIGHTGBM_AVAILABLE,
                'install_cmd': 'pip install lightgbm',
                'description': '微软开发的梯度提升框架'
            },
            'catboost': {
                'available': CATBOOST_AVAILABLE,
                'install_cmd': 'pip install catboost',
                'description': 'Yandex开发的梯度提升框架'
            }
        }
        
        return dependencies
    
    @staticmethod
    def print_dependency_status():
        """打印依赖状态报告"""
        deps = MethodDependencyChecker.check_all_dependencies()
        
        print("\n📦 基准方法依赖状态")
        print("=" * 60)
        
        for name, info in deps.items():
            status = "✅ 可用" if info['available'] else "❌ 不可用"
            print(f"{name:<12} {status:<8} - {info['description']}")
            if not info['available']:
                print(f"{'':12} 安装命令: {info['install_cmd']}")
        
        print("=" * 60)
        
        available_count = sum(1 for info in deps.values() if info['available'])
        total_count = len(deps)
        
        print(f"📊 可选依赖可用性: {available_count}/{total_count}")
        
        if available_count < total_count:
            print("💡 安装缺失的依赖以获得完整的基准测试能力")


def get_default_method_selection(task_type: str, selection_type: str = 'basic') -> list:
    """
    获取默认的方法选择
    
    Args:
        task_type: 'regression' 或 'classification'
        selection_type: 'basic', 'comprehensive', 'competitive', 'lightweight'
    
    Returns:
        方法名列表
    """
    from .method_configs import get_task_recommendations, get_method_group
    
    if selection_type in ['basic', 'comprehensive', 'competitive', 'lightweight']:
        return get_method_group(selection_type)
    else:
        return get_task_recommendations(task_type, selection_type)


def filter_available_methods(method_list: list) -> Tuple[list, list]:
    """
    过滤可用的方法
    
    Returns:
        (可用方法列表, 不可用方法列表)
    """
    factory = BaselineMethodFactory()
    available = []
    unavailable = []
    
    for method in method_list:
        if factory.is_method_available(method):
            available.append(method)
        else:
            unavailable.append(method)
    
    return available, unavailable


# === 稳健MLP实现已迁移到 robust_regressors.py ===


# === 早停支持方法 ===

def supports_early_stopping(method_name: str) -> bool:
    """检查方法是否支持早停"""
    early_stopping_methods = {
        'xgboost', 'lightgbm', 'catboost', 'gradient_boosting'
    }
    return method_name in early_stopping_methods


def fit_with_early_stopping(method_name: str, model: Any, 
                           X_train, y_train, X_val, y_val, task_type: str):
    """为支持早停的方法实现验证集早停"""
    
    if method_name == 'xgboost':
        return _fit_xgboost_with_early_stopping(
            model, X_train, y_train, X_val, y_val, task_type
        )
    elif method_name == 'lightgbm':
        return _fit_lightgbm_with_early_stopping(
            model, X_train, y_train, X_val, y_val, task_type
        )
    elif method_name == 'catboost':
        return _fit_catboost_with_early_stopping(
            model, X_train, y_train, X_val, y_val, task_type
        )
    elif method_name == 'gradient_boosting':
        return _fit_sklearn_gb_with_early_stopping(
            model, X_train, y_train, X_val, y_val, task_type
        )
    else:
        # 默认训练
        model.fit(X_train, y_train)
        return model


def _fit_xgboost_with_early_stopping(model, X_train, y_train, X_val, y_val, task_type):
    """XGBoost早停训练"""
    if not XGBOOST_AVAILABLE:
        model.fit(X_train, y_train)
        return model
    
    try:
        # 尝试新版本XGBoost API
        eval_set = [(X_val, y_val)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
    except TypeError:
        try:
            # 尝试使用callbacks (新版本)
            import xgboost as xgb
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[xgb.callback.EarlyStopping(rounds=50)]
            )
        except:
            # 如果都失败，直接训练
            model.fit(X_train, y_train)
    
    return model


def _fit_lightgbm_with_early_stopping(model, X_train, y_train, X_val, y_val, task_type):
    """LightGBM早停训练"""
    if not LIGHTGBM_AVAILABLE:
        model.fit(X_train, y_train)
        return model
    
    # LightGBM使用callbacks参数进行早停
    import lightgbm as lgb
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    return model


def _fit_catboost_with_early_stopping(model, X_train, y_train, X_val, y_val, task_type):
    """CatBoost早停训练"""
    if not CATBOOST_AVAILABLE:
        model.fit(X_train, y_train)
        return model
    
    eval_set = (X_val, y_val)
    
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=50,
        verbose=False
    )
    return model


def _fit_sklearn_gb_with_early_stopping(model, X_train, y_train, X_val, y_val, task_type):
    """sklearn Gradient Boosting早停训练"""
    from sklearn.metrics import mean_squared_error, log_loss
    
    # sklearn的GradientBoosting不直接支持早停，需要手动实现
    n_estimators = model.n_estimators
    best_score = float('inf')
    best_n_estimators = n_estimators
    patience_counter = 0
    patience = 50
    
    # 逐步增加估计器数量并在验证集上评估
    for n_est in range(10, n_estimators + 1, 10):
        temp_model = model.__class__(**model.get_params())
        temp_model.n_estimators = n_est
        temp_model.fit(X_train, y_train)
        
        val_pred = temp_model.predict(X_val)
        
        if task_type == 'regression':
            val_score = mean_squared_error(y_val, val_pred)
        else:
            try:
                val_proba = temp_model.predict_proba(X_val)
                val_score = log_loss(y_val, val_proba)
            except:
                val_score = 1.0 - temp_model.score(X_val, y_val)
        
        if val_score < best_score:
            best_score = val_score
            best_n_estimators = n_est
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience // 10:  # 调整patience以适应步长
            break
    
    # 使用最佳估计器数量重新训练
    model.n_estimators = best_n_estimators
    model.fit(X_train, y_train)
    return model