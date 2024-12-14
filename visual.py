import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 准备数据
# 简化模型名称以便显示
model_names = [
    'Qwen-1.5B-MATH',
    'Qwen-7B',
    'Qwen-14B',
    'Llama-70B',
    'Llama-8B',
    'Qwen-72B'
]

# 提取性能指标
metrics = ['ACCURACY', 'PRECISION', 'RECALL', 'F1']
data = [
    [61.52, 36.30, 0.75, 1.48],
    [66.01, 56.19, 50.68, 53.29],
    [69.83, 57.49, 81.15, 67.30],
    [69.50, 56.89, 83.74, 67.75],
    [61.59, 49.29, 13.39, 21.05],
    [70.75, 57.80, 87.31, 69.55]
]

# 转换为numpy数组
data = np.array(data)

# 设置雷达图的角度
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)

# 闭合雷达图
angles = np.concatenate((angles, [angles[0]]))

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

# 设置色彩方案
colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#FF99CC', '#99CCFF']

# 绘制每个模型的数据
for i, model in enumerate(model_names):
    values = data[i]
    values = np.concatenate((values, [values[0]]))  # 闭合数据
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

# 设置角度刻度
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)

# 设置网格和刻度
ax.set_ylim(0, 100)
ax.grid(True)

# 添加标题和图例
plt.title('Legal Judgement Prediction', pad=20, size=15)
plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

# 调整布局
plt.tight_layout()

# Save
plt.savefig('./visual/radar_chart.png')