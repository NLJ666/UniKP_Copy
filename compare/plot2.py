import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

# ================= 配置路径与颜色 =================
OUTPUT_DIR = './benchmark_output'
CSV_PATH = os.path.join(OUTPUT_DIR, 'model_performance_10fold.csv')

COLOR_DLK = '#009c95'  # 深青色
COLOR_UNI = '#8ecae6'  # 浅蓝色
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

# ================= 读取数据 =================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"未找到结果文件 {CSV_PATH}，请确认文件路径无误！")
df = pd.read_csv(CSV_PATH)

# 排除没有 Type 的数据
df = df.dropna(subset=['Type'])

# ================= 初始化画布 (1行3列) =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
fig.patch.set_facecolor('white')

# -------------------- 图 a: Wild-type 散点图 --------------------
data_wt = df[df['Type'] == 'wildtype']
x_wt, y_wt = data_wt['True_Value'], data_wt['DLKcat_Predicted']
pcc_wt, _ = pearsonr(x_wt, y_wt)
print(f"Wild-type: N = {len(x_wt)}, PCC = {pcc_wt:.2f}")

xy_wt = np.vstack([x_wt, y_wt])
z_wt = gaussian_kde(xy_wt)(xy_wt)
idx_wt = z_wt.argsort()
sc_wt = axes[0].scatter(x_wt.values[idx_wt], y_wt.values[idx_wt], c=z_wt[idx_wt], s=15, cmap='Spectral_r', edgecolor='none')
axes[0].plot([x_wt.min(), x_wt.max()], [y_wt.min(), y_wt.max()], 'r--', lw=1.5)

axes[0].set_title(f'a         PCC = {pcc_wt:.2f}\n         N = {len(x_wt)}', loc='left', fontweight='bold', fontsize=14)
axes[0].set_xlabel(r'$log_{10}$experimental $k_{cat}$ value ($s^{-1}$)', fontsize=12)
axes[0].set_ylabel(r'$log_{10}$predicted $k_{cat}$ value ($s^{-1}$)', fontsize=12)
plt.colorbar(sc_wt, ax=axes[0]).set_label('Density', fontsize=11)
axes[0].text(0.85, 0.95, 'Wild-type', transform=axes[0].transAxes, ha='center', va='top', fontsize=13)

# -------------------- 图 b: Mutant 散点图 --------------------
data_mt = df[df['Type'] == 'mutant']
x_mt, y_mt = data_mt['True_Value'], data_mt['DLKcat_Predicted']
pcc_mt, _ = pearsonr(x_mt, y_mt)
print(f"Mutant:    N = {len(x_mt)}, PCC = {pcc_mt:.2f}")

xy_mt = np.vstack([x_mt, y_mt])
z_mt = gaussian_kde(xy_mt)(xy_mt)
idx_mt = z_mt.argsort()
sc_mt = axes[1].scatter(x_mt.values[idx_mt], y_mt.values[idx_mt], c=z_mt[idx_mt], s=15, cmap='Spectral_r', edgecolor='none')
axes[1].plot([x_mt.min(), x_mt.max()], [y_mt.min(), y_mt.max()], 'r--', lw=1.5)

axes[1].set_title(f'b         PCC = {pcc_mt:.2f}\n         N = {len(x_mt)}', loc='left', fontweight='bold', fontsize=14)
axes[1].set_xlabel(r'$log_{10}$experimental $k_{cat}$ value ($s^{-1}$)', fontsize=12)
axes[1].set_ylabel(r'$log_{10}$predicted $k_{cat}$ value ($s^{-1}$)', fontsize=12)
plt.colorbar(sc_mt, ax=axes[1]).set_label('Density', fontsize=11)
axes[1].text(0.85, 0.95, 'Mutant', transform=axes[1].transAxes, ha='center', va='top', fontsize=13)

# -------------------- 图 c: PCC 分组对比柱状图 (图例已移至左上角) --------------------
metrics_c = []
for model in ['DLKcat_Predicted', 'UniKP_Predicted']:
    for type_name in ['wildtype', 'mutant']:
        tmp = df[df['Type'] == type_name]
        pcc_val, _ = pearsonr(tmp['True_Value'], tmp[model])
        metrics_c.append({'Type': type_name.capitalize(), 'Model': 'DLKcat' if 'DLKcat' in model else 'UniKP', 'PCC': pcc_val})

df_metrics_c = pd.DataFrame(metrics_c)

sns.barplot(ax=axes[2], data=df_metrics_c, x='Type', y='PCC', hue='Model',
            palette=[COLOR_DLK, COLOR_UNI], width=0.4, legend=True)
axes[2].set_title('c', loc='left', fontweight='bold', fontsize=16)
axes[2].set_ylabel('PCC', fontsize=14)
axes[2].set_xlabel('')
axes[2].set_ylim(0, 1.0)

# ================= 关键修改：图例放到左上角 =================
# 注意：删掉了 bbox_to_anchor=(1, 1)，改为左上角对齐，完全不挡图
sns.move_legend(axes[2], "upper left", title=None)

# ================= 保存最终图 =================
plt.savefig(os.path.join(OUTPUT_DIR, 'New_Figure_Mutant_Wildtype.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'New_Figure_Mutant_Wildtype.pdf'), bbox_inches='tight')
print(f"\n大兄弟，新图已保存至: {OUTPUT_DIR}/New_Figure_Mutant_Wildtype.png")
plt.show()