import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 9

# 模型名称
models = [
    'Linear Regression', 'Ridge Regression', 'Lasso Regression',
    'Bayesian Ridge Regression', 'Elastic Net Regression',
    'Decision Tree', 'Support Vector Regression', 'K Neighbors Regressor',
    'Random Forest', 'Gradient Boosting', 'Extra Trees',
    'AdaBoost', 'Bagging', 'MLP Regressor',
    'XGB', 'LGBM', 'CNN', 'RNN'
]

# 四个指标数据
rmse = [1.1796, 1.1602, 1.5013, 1.1574, 1.5013, 1.2712, 1.3537, 1.2505,
        0.9130, 1.1601, 0.8885, 1.3373, 0.9647, 1.0755, 0.9735, 0.9978, 1.4625, 1.4539]
pcc  = [0.6416, 0.6347, np.nan, 0.6370, np.nan, 0.6453, 0.4412, 0.5711,
        0.7975, 0.6517, 0.8066, 0.4862, 0.7665, 0.7223, 0.7614, 0.7520, 0.2484, 0.2694]
mae  = [0.8759, 0.8833, 1.1777, 0.8808, 1.1777, 0.8335, 1.0382, 0.9325,
        0.6420, 0.8906, 0.6014, 1.0734, 0.6826, 0.7933, 0.7013, 0.7410, 1.1447, 1.1373]
r2   = [max(0, v) for v in [0.3825, 0.4025, -0.0003, 0.4054, -0.0003, 0.2822, 0.1867, 0.3059,
        0.6300, 0.4026, 0.6495, 0.2059, 0.5869, 0.4865, 0.5790, 0.5580, 0.0506, 0.0617]]

# 找到最优模型索引（RMSE/MAE越小越好，PCC/R²越大越好）
best_rmse_idx = np.argmin(rmse)
best_pcc_idx  = np.nanargmax(pcc)
best_mae_idx  = np.argmin(mae)
best_r2_idx   = np.argmax(r2)

print(f"Best RMSE: {models[best_rmse_idx]} ({rmse[best_rmse_idx]})")
print(f"Best PCC:  {models[best_pcc_idx]} ({pcc[best_pcc_idx]})")
print(f"Best MAE:  {models[best_mae_idx]} ({mae[best_mae_idx]})")
print(f"Best R平方: {models[best_r2_idx]} ({r2[best_r2_idx]})")

# 颜色设置
blue = '#4472C4'
orange = '#ED7D31'

def get_colors(best_idx, n):
    colors = [blue] * n
    colors[best_idx] = orange
    return colors

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

y_pos = np.arange(len(models))
bar_height = 0.6

# a) RMSE (越小越好)
ax = axes[0, 0]
ax.barh(y_pos, rmse, bar_height, color=get_colors(best_rmse_idx, len(models)))
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('RMSE', fontsize=12, fontweight='bold')
ax.set_title('a', fontsize=14, fontweight='bold', loc='left')
for i, v in enumerate(rmse):
    ax.text(v + 0.02, i, f'{v:.2f}', va='center', fontsize=8)
ax.set_xlim(0, 1.6)

# b) PCC (越大越好)
ax = axes[0, 1]
pcc_display = [x if not np.isnan(x) else 0.01 for x in pcc]
ax.barh(y_pos, pcc_display, bar_height, color=get_colors(best_pcc_idx, len(models)))
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('PCC', fontsize=12, fontweight='bold')
ax.set_title('b', fontsize=14, fontweight='bold', loc='left')
for i, v in enumerate(pcc):
    label = f'{v:.2f}' if not np.isnan(v) else 'nan'
    ax.text(v + 0.01 if not np.isnan(v) else 0.02, i, label, va='center', fontsize=8)
ax.set_xlim(0, 0.9)

# c) MAE (越小越好)
ax = axes[1, 0]
ax.barh(y_pos, mae, bar_height, color=get_colors(best_mae_idx, len(models)))
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('c', fontsize=14, fontweight='bold', loc='left')
for i, v in enumerate(mae):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
ax.set_xlim(0, 1.3)

# d) R² (越大越好)
ax = axes[1, 1]
ax.barh(y_pos, r2, bar_height, color=get_colors(best_r2_idx, len(models)))
ax.set_yticks(y_pos)
ax.set_yticklabels(models, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(r'$R^2$', fontsize=12, fontweight='bold')
ax.set_title('d', fontsize=14, fontweight='bold', loc='left')
for i, v in enumerate(r2):
    ax.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=8)
ax.set_xlim(0, 0.75)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("\n图片已保存为 model_comparison.png")
