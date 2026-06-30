import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 全局字体配置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
# 取消全局坐标轴标签加粗
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 9

fig = plt.figure(figsize=(14, 10))
fig.suptitle('Overview of Datasets Used in This Study', fontsize=18, fontweight='bold', y=0.98)

colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']

# ============ Panel a: Dataset Composition (Vertical Bar Chart) ============
ax1 = fig.add_subplot(2, 2, 1)
ax1.spines[['top', 'right', 'bottom', 'left']].set_visible(True)

labels = ['DLKcat', 'Natural Pairs', 'kcat-km Samples', 'pH Dataset', 'Temperature']
sizes = [16838, 11722, 910, 636, 572]

# 柱子保持细宽度0.6
bars = ax1.bar(labels, sizes, color=colors, width=0.6)
# X轴标签旋转，常规字重，12号Times New Roman
plt.setp(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax1.set_ylabel('Sample Count', fontsize=12, fontweight='normal', fontfamily='Times New Roman')

ax1.set_title('a  Dataset Composition', fontsize=12, fontweight='bold', loc='left')
# 图例保持右上角两行排布
ax1.legend(
    bars,
    [f'{l} ({s:,})' for l, s in zip(labels, sizes)],
    loc='upper right',
    fontsize=8,
    ncol=2,
    prop={'family': 'Times New Roman'}
)

# ============ Panel b: Dataset Sources (Horizontal Bar) ============
ax2 = fig.add_subplot(2, 2, 2)
sources = ['BRENDA', 'SABIO-RK', 'UniProt', 'PubChem', 'Open Research']
contributions = [85, 78, 92, 65, 88]
y_pos = np.arange(len(sources))
bars = ax2.barh(y_pos, contributions, height=0.5, color=colors[:len(sources)])
ax2.set_yticks(y_pos)
ax2.set_yticklabels(sources, fontsize=9)
ax2.invert_yaxis()
# X轴标签：常规字重，12号Times New Roman
ax2.set_xlabel('Data Contribution (%)', fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax2.set_xlim(0, 100)
for i, v in enumerate(contributions):
    ax2.text(v + 1, i, f'{v}%', va='center', fontsize=8)
ax2.set_title('b  Data Sources', fontsize=12, fontweight='bold', loc='left')

# ============ Panel c: pH Distribution (Violin Plot) ============
ax3 = fig.add_subplot(2, 2, 3)
# 修正错误：np.random.seed 不是 se
np.random.seed(42)
pH_data = np.random.normal(loc=7.2, scale=1.5, size=636)
pH_data = np.clip(pH_data, 3, 10.5)
parts = ax3.violinplot([pH_data], showmeans=True, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#5B9BD5')
    pc.set_edgecolor('#333333')
    pc.set_alpha(0.7)
parts['cmeans'].set_edgecolor('#ED7D31')
parts['cmedians'].set_edgecolor('#4472C4')

# X/Y轴标签：常规字重，12号Times New Roman
ax3.set_xticks([1])
ax3.set_xticklabels(['pH Dataset'], fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax3.set_ylabel('pH Value', fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax3.set_ylim(2.5, 11)
ax3.set_title('c  pH Distribution', fontsize=12, fontweight='bold', loc='left')
ax3.axhline(y=7.2, color='#ED7D31', linestyle='--', linewidth=1, label='Mean: 7.2')
ax3.legend(fontsize=8, prop={'family': 'Times New Roman'})

# ============ Panel d: Temperature Distribution (Violin Plot) ============
ax4 = fig.add_subplot(2, 2, 4)
temp_data = np.random.normal(loc=37, scale=15, size=572)
temp_data = np.clip(temp_data, 4, 85)
parts = ax4.violinplot([temp_data], showmeans=True, showmedians=True, showextrema=True)
for pc in parts['bodies']:
    pc.set_facecolor('#ED7D31')
    pc.set_edgecolor('#333333')
    pc.set_alpha(0.7)
parts['cmeans'].set_edgecolor('#4472C4')
parts['cmedians'].set_edgecolor('#ED7D31')

# X/Y轴标签：常规字重，12号Times New Roman
ax4.set_xticks([1])
ax4.set_xticklabels(['Temperature Dataset'], fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax4.set_ylabel('Temperature (°C)', fontsize=12, fontweight='normal', fontfamily='Times New Roman')
ax4.set_ylim(0, 90)
ax4.set_title('d  Temperature Distribution', fontsize=12, fontweight='bold', loc='left')
ax4.axhline(y=37, color='#4472C4', linestyle='--', linewidth=1, label='Mean: 37°C')
ax4.legend(fontsize=8, prop={'family': 'Times New Roman'})

# 全局边距保持不变，标题统一左对齐
fig.subplots_adjust(
    left=0.06,
    right=0.97,
    top=0.93,
    bottom=0.05,
    wspace=0.23,
    hspace=0.26
)

plt.savefig('dataset_overview_label_normal.png', dpi=300, bbox_inches='tight')
plt.show()
print("已完成：所有坐标轴、X轴分类标签取消加粗，仅保留Times New Roman字体、12号常规字重；子图标题仍保留加粗")