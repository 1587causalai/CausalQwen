import numpy as np
from typing import Tuple

def inject_shuffle_noise(
    y: np.ndarray,
    noise_ratio: float,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对一部分样本的标签进行全局随机洗牌，以切断其与特征的因果关系。
    这是项目MVP阶段唯一的因果标签异常注入逻辑。

    该方法基于一个理论上更优的策略：
    1. 创建一个全局洗牌后的标签副本 y_shuffled。
    2. 随机选择一部分索引。
    3. 将这些索引位置的原始标签替换为 y_shuffled 中对应位置的标签。
    这确保了被污染的标签与原始特征完全无关。

    Args:
        y: 原始标签数组。
        noise_ratio: 需要注入噪声的样本比例 (0.0 到 1.0)。
        random_state: 随机种子，用于复现。

    Returns:
        A tuple containing:
        - y_noisy (np.ndarray): 注入了洗牌噪声的标签数组。
        - noise_indices (np.ndarray): 被注入噪声的样本的原始索引。
    """
    if random_state is not None:
        np.random.seed(random_state)

    if not (0.0 <= noise_ratio <= 1.0):
        raise ValueError("noise_ratio 必须在 0.0 和 1.0 之间")

    n_samples = len(y)
    if noise_ratio == 0 or n_samples == 0:
        return y.copy(), np.array([], dtype=int)

    # 1. 创建一个全局洗牌后的标签副本 y'
    y_shuffled = y.copy()
    np.random.shuffle(y_shuffled)

    # 2. 随机选择要污染的索引
    n_noisy = int(n_samples * noise_ratio)
    noise_indices = np.random.choice(n_samples, size=n_noisy, replace=False)

    # 3. 创建一个新的 y_noisy 向量
    y_noisy = y.copy()
    
    # 4. 在选定的索引处，用 y' 的值替换 y 的值
    y_noisy[noise_indices] = y_shuffled[noise_indices]
    
    return y_noisy, noise_indices 