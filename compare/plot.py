import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings

# 忽略警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ================= 文件路径与样式 =================
OUTPUT_DIR = './benchmark_output'
CSV_PATH = os.path.join(OUTPUT_DIR, 'model_performance_10fold.csv')

COLOR_DLK = '#009c95'
COLOR_UNI = '#8ecae6'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# ================= 读取数据 =================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"未找到结果文件 {CSV_PATH}，请确认之前的代码已运行并生成该文件！")
df = pd.read_csv(CSV_PATH)

# ================= 初始化画布 =================
fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
fig.patch.set_facecolor('white')

# -------------------- 图 a: R² Boxplot --------------------
metrics_a = []
for fold in range(1, 11):
    fold_df = df[df['Fold'] == fold]
    true = fold_df['True_Value']
    r2_d = r2_score(true, fold_df['DLKcat_Predicted'])
    metrics_a.append({'Fold': fold, 'Model': 'DLKcat', 'R2': r2_d})
    r2_u = r2_score(true, fold_df['UniKP_Predicted'])
    metrics_a.append({'Fold': fold, 'Model': 'UniKP', 'R2': r2_u})
df_metrics_a = pd.DataFrame(metrics_a)

sns.boxplot(ax=axes[0, 0], data=df_metrics_a, x='Model', y='R2', hue='Model', palette=[COLOR_DLK, COLOR_UNI],
            legend=False)
axes[0, 0].set_title('a', loc='left', fontweight='bold', fontsize=16)
axes[0, 0].set_ylabel('$R^2$ on test set', fontsize=14)
axes[0, 0].set_xlabel('')

# -------------------- 图 b: RMSE Barplot (已调细柱子) --------------------
metrics_b = []
for fold in range(1, 11):
    fold_df = df[df['Fold'] == fold]
    true = fold_df['True_Value']
    rmse_d = np.sqrt(mean_squared_error(true, fold_df['DLKcat_Predicted']))
    metrics_b.append({'Model': 'DLKcat', 'RMSE': rmse_d})
    rmse_u = np.sqrt(mean_squared_error(true, fold_df['UniKP_Predicted']))
    metrics_b.append({'Model': 'UniKP', 'RMSE': rmse_u})

df_metrics_b = pd.DataFrame(metrics_b)
df_plot_b = df_metrics_b.groupby('Model').agg(Mean=('RMSE', 'mean'), Std=('RMSE', 'std')).reset_index()

# 重点修改：添加了 width=0.4
sns.barplot(ax=axes[0, 1], data=df_plot_b, x='Model', y='Mean', hue='Model', palette=[COLOR_DLK, COLOR_UNI],
            errorbar=None, legend=False, width=0.2)
axes[0, 1].errorbar(x=range(len(df_plot_b)), y=df_plot_b['Mean'], yerr=df_plot_b['Std'], fmt='none', ecolor='black',
                    capsize=5)
axes[0, 1].set_title('b', loc='left', fontweight='bold', fontsize=16)
axes[0, 1].set_ylabel('Test Set RMSE', fontsize=14)
axes[0, 1].set_xlabel('')

# -------------------- 图 c: Density Scatter Plot (复刻原图 c) --------------------
from scipy.stats import gaussian_kde

x = df['True_Value']
y = df['DLKcat_Predicted']
pcc, _ = pearsonr(x, y)
ax_c = axes[0, 2]

# 计算二维密度
xy = np.vstack([x, y])
z = gaussian_kde(xy)(xy)

# 根据密度排序，让密度低的点在底层，密度高的点在最上层
idx = z.argsort()
x_sorted, y_sorted, z_sorted = x.values[idx], y.values[idx], z[idx]

sc = ax_c.scatter(x_sorted, y_sorted, c=z_sorted, s=15, cmap='Spectral_r', edgecolor='none')
ax_c.plot([x.min(), x.max()], [y.min(), y.max()], 'r--', lw=2)

ax_c.set_title('c', loc='left', fontweight='bold', fontsize=16)
ax_c.set_xlabel('$log_{10}$experimental $k_{cat}$ value ($s^{-1}$)', fontsize=14)
ax_c.set_ylabel('$log_{10}$predicted $k_{cat}$ value ($s^{-1}$)', fontsize=14)
ax_c.text(0.05, 0.95, f'PCC = {pcc:.2f}\nN = {len(x)}', transform=ax_c.transAxes, fontsize=14, va='top')

cb = plt.colorbar(sc, ax=ax_c)
cb.set_label('Density', fontsize=12)

# -------------------- 图 d: RMSE by kcat range --------------------
bins = [-np.inf, 0, 1, 2, 3, 4, 5, np.inf]
labels = ['<0', '1-0', '2-1', '3-2', '4-3', '5-4', '>5']
df['True_Bin'] = pd.cut(df['True_Value'], bins=bins, labels=labels)

rmse_bin = df.groupby('True_Bin', observed=False).apply(lambda g: pd.Series({
    'DLKcat': np.sqrt(mean_squared_error(g['True_Value'], g['DLKcat_Predicted'])),
    'UniKP': np.sqrt(mean_squared_error(g['True_Value'], g['UniKP_Predicted']))
}), include_groups=False).reset_index().melt(id_vars='True_Bin', var_name='Model', value_name='RMSE')

order = ['>5', '5-4', '4-3', '3-2', '2-1', '1-0', '<0']
sns.barplot(ax=axes[1, 0], data=rmse_bin, x='True_Bin', y='RMSE', hue='Model', palette=[COLOR_DLK, COLOR_UNI],
            order=order)
axes[1, 0].set_title('d', loc='left', fontweight='bold', fontsize=16)
axes[1, 0].set_xlabel('$log_{10}$experimental $k_{cat}$ value ($s^{-1}$)', fontsize=14)
axes[1, 0].legend(frameon=False)

# -------------------- 图 e: Boxplot by Type --------------------
axes[1, 1].set_title('e', loc='left', fontweight='bold', fontsize=16)

df_type_valid = df.dropna(subset=['Type'])

if len(df_type_valid) > 0:
    type_counts = df_type_valid['Type'].value_counts()
    if len(type_counts) > 2:
        top_types = type_counts.nlargest(2).index.tolist()
    else:
        top_types = type_counts.index.tolist()

    df_plot_e = df_type_valid[df_type_valid['Type'].isin(top_types)].copy()

    sns.boxplot(ax=axes[1, 1], data=df_plot_e, x='Type', y='DLKcat_Predicted',
                palette=['#3b75af', '#f49e4c'], width=0.5)

    if len(top_types) == 2:
        group1 = df_plot_e[df_plot_e['Type'] == top_types[0]]['DLKcat_Predicted']
        group2 = df_plot_e[df_plot_e['Type'] == top_types[1]]['DLKcat_Predicted']
        if len(group1) >= 2 and len(group2) >= 2:
            stat, p_val = ttest_ind(group1, group2)
            axes[1, 1].text(0.5, 0.95, f'p-value = {p_val:.2e}', ha='center', va='top', transform=axes[1, 1].transAxes,
                            fontsize=14)

    print(f"\n图 e 实际绘制的 Type 类别有: {top_types}")
else:
    axes[1, 1].text(0.5, 0.5, "Type 列中没有有效的数据", ha='center', va='center', fontsize=12, color='red')

axes[1, 1].set_xlabel('')
axes[1, 1].set_ylabel('$log_{10}$predicted $k_{cat}$ value ($s^{-1}$)', fontsize=14)

# # -------------------- 图 f: SHAP Feature Importance Bar Plot --------------------
# axes[1, 2].set_title('f', loc='left', fontweight='bold', fontsize=16)
#
# shap_csv_path = os.path.join(OUTPUT_DIR, 'SHAP_Top20_Features_Global.csv')
#
# try:
#     shap_df = pd.read_csv(shap_csv_path)
#     if shap_df.empty:
#         raise ValueError("SHAP CSV 为空")
#
#     shap_df = shap_df.sort_values(by='Mean_Absolute_SHAP', ascending=False).head(20)
#
#     sns.barplot(
#         ax=axes[1, 2],
#         data=shap_df,
#         x='Mean_Absolute_SHAP',
#         y='Feature_Name',
#         hue='Feature_Name',
#         palette='viridis',
#         legend=False,
#         dodge=False,
#         edgecolor='black',
#         linewidth=0.5
#     )
#
#     axes[1, 2].set_xlabel('SHAP value (impact on model output)', fontsize=14)
#     axes[1, 2].set_ylabel('')
#
# except Exception as e:
#     axes[1, 2].text(0.5, 0.5, f"读取 SHAP CSV 失败\n{str(e)}", ha='center', va='center', fontsize=12, color='red')

# -------------------- 图 f: 直接嵌入原生的 SHAP 蜂群图 --------------------
axes[1, 2].set_title('f', loc='left', fontweight='bold', fontsize=16)

# 指向你之前生成的 SHAP 原图
shap_image_path = os.path.join(OUTPUT_DIR, 'SHAP_summary_Global_Final.png')

if os.path.exists(shap_image_path):
    try:
        # 读取图片
        img = plt.imread(shap_image_path)
        # 贴到子图上，aspect='auto' 会自动适应画布比例，避免变形
        axes[1, 2].imshow(img, aspect='auto')
        # 隐藏原图的边框和刻度，让它看起来像是直接画上去的
        axes[1, 2].axis('off')
    except Exception as e:
        axes[1, 2].text(0.5, 0.5, f"读取图片时发生错误:\n{str(e)}", ha='center', va='center', fontsize=12, color='red')
else:
    axes[1, 2].text(0.5, 0.5, f"找不到图片文件:\n{shap_image_path}\n请确认原代码已生成该图。", ha='center', va='center', fontsize=12, color='red')

# ================= 保存最终合成图 =================
plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Paper_Figure_Only_Testset.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'Full_Paper_Figure_Only_Testset.pdf'), bbox_inches='tight')
print(f"修复版结果图已保存至: {OUTPUT_DIR}/Full_Paper_Figure_Only_Testset.png")
plt.show()