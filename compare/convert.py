# convert.py
import pickle
import numpy as np
import os

# 您代码中定义的 DLKcat 特征文件夹路径
DLKCAT_FEAT_DIR = '../Feature_For_DLKcat/'
files = ['compounds.pkl', 'adjacencies.pkl', 'proteins.pkl']

print("正在将旧的 .pkl 转换为永久的 .npy 文件...")
for f in files:
    src = os.path.join(DLKCAT_FEAT_DIR, f)
    print(f"正在读取: {f} ...")
    with open(src, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
        # 关键修复：将数据转为 dtype=object 的数组，完美解决形状不均匀报错！
        data_np = np.asarray(data, dtype=object)
        dst = src.replace('.pkl', '.npy')
        np.save(dst, data_np, allow_pickle=True)
        print(f"✅ 已成功保存: {dst}")
print("\n【成功】所有特征文件已转换为 .npy 格式！")