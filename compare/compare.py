import json
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import gc
import os
import warnings
# ================= 新增导入 =================
from tqdm import tqdm
# ==========================================

warnings.filterwarnings('ignore')

# ==================== 全局输出路径设置 ====================
OUTPUT_DIR = './benchmark_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"已创建输出目录: {OUTPUT_DIR}")


# ==================== 1. 模型定义区 (DLKcat网络结构) ====================
class KcatPrediction(nn.Module):
    def __init__(self, dim, layer_gnn, window, layer_cnn, layer_output, n_fingerprint, n_word):
        super(KcatPrediction, self).__init__()
        self.dim = dim
        self.layer_gnn = layer_gnn
        self.window = window
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output

        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * window + 1, stride=1, padding=window) for _ in
             range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2 * dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)
        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):
        fingerprints, adjacency, words = inputs
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, self.layer_gnn)
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector, word_vectors, self.layer_cnn)
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return interaction

    def __call__(self, data, train=True):
        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        if train:
            if correct_interaction.dim() == 1:
                correct_interaction = correct_interaction.unsqueeze(1)
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy().flatten()
            predicted_values = predicted_interaction.to('cpu').data.numpy().flatten()
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy().flatten()
            predicted_values = predicted_interaction.to('cpu').data.numpy().flatten()
            return correct_values, predicted_values


class TrainerDLKcat(object):
    def __init__(self, model, device, lr=0.001, weight_decay=0.0):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        loss_total = 0
        trainTrue, trainPred = [], []
        for data in dataset:
            inputs, correct_interaction = data[:-1], data[-1]
            inputs_gpu = (inputs[0].to(self.device), inputs[1].to(self.device), inputs[2].to(self.device))
            correct_gpu = correct_interaction.to(self.device)
            data_gpu = (inputs_gpu[0], inputs_gpu[1], inputs_gpu[2], correct_gpu)

            loss, correct_values, predicted_values = self.model(data_gpu)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.item()

            trainTrue.extend([math.log10(math.pow(2, x)) for x in correct_values])
            trainPred.extend([math.log10(math.pow(2, x)) for x in predicted_values])
        return loss_total, np.array(trainTrue), np.array(trainPred)


def evaluate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    return r2, rmse, mae, pcc


# ==================== 2. 数据处理与特征加载 ====================
def load_and_filter_data(json_path):
    with open(json_path, 'r') as f:
        datasets = json.load(f)
    filtered_data = []
    valid_indices = []
    for idx, data in enumerate(datasets):
        smi = data['Smiles']
        val = float(data['Value'])
        if '.' not in smi and val != 0.0:
            filtered_data.append(data)
            valid_indices.append(idx)
    print(f"原始样本数: {len(datasets)}, 过滤后样本数: {len(filtered_data)}")
    return filtered_data, valid_indices


def load_pickle_file(file_path):
    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith('numpy._core'):
                module = module.replace('numpy._core', 'numpy.core')
            return super().find_class(module, name)
    with open(file_path, 'rb') as f:
        return NumpyCompatUnpickler(f).load()


def prepare_dlkcat_features(valid_indices, filtered_n, compound_dir):
    compounds = load_pickle_file(os.path.join(compound_dir, "compounds.pkl"))
    adjacencies = load_pickle_file(os.path.join(compound_dir, "adjacencies.pkl"))
    proteins = load_pickle_file(os.path.join(compound_dir, "proteins.pkl"))

    if len(compounds) == filtered_n:
        return compounds, adjacencies, proteins

    return [compounds[i] for i in valid_indices], \
           [adjacencies[i] for i in valid_indices], \
           [proteins[i] for i in valid_indices]


def load_unikp_features(smiles_pkl_path, protein_pkl_path, valid_indices, filtered_n):
    smiles_feat = load_pickle_file(smiles_pkl_path)
    protein_feat = load_pickle_file(protein_pkl_path)

    features = np.concatenate((smiles_feat, protein_feat), axis=1)

    if features.shape[0] == filtered_n:
        return features, smiles_feat.shape[1], protein_feat.shape[1]

    features = features[valid_indices]
    return features, smiles_feat.shape[1], protein_feat.shape[1]


# ==================== 3. 核心实验循环 ====================
def run_experiment():
    JSON_PATH = '../datasets/Kcat_combination_0918_wildtype_mutant.json'
    DLKCAT_FEAT_DIR = '../Feature_For_DLKcat/'
    UNIKP_SMILES_PKL = './PreKcat_new/smiles_feature.pkl'
    UNIKP_PROTEIN_PKL = './PreKcat_new/protein_feature.pkl'

    DIM, LAYER_GNN, WINDOW, LAYER_CNN, LAYER_OUTPUT = 128, 3, 3, 3, 3
    LR, WEIGHT_DECAY = 0.001, 0.0
    EPOCHS = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用设备: {device}")

    filtered_data, valid_indices = load_and_filter_data(JSON_PATH)
    n = len(filtered_data)

    sequences = [d['Sequence'] for d in filtered_data]
    smiles_list = [d['Smiles'] for d in filtered_data]
    ec_numbers = [d['ECNumber'] for d in filtered_data]
    organisms = [d['Organism'] for d in filtered_data]
    substrates = [d['Substrate'] for d in filtered_data]
    types = [d['Type'] for d in filtered_data]

    y_true_log10 = np.array([math.log(float(d['Value']), 10) for d in filtered_data])

    compounds, adjacencies, proteins = prepare_dlkcat_features(valid_indices, n, DLKCAT_FEAT_DIR)
    n_fingerprint, n_word = 3958, 8542

    unikp_features, smi_dim, pro_dim = load_unikp_features(UNIKP_SMILES_PKL, UNIKP_PROTEIN_PKL, valid_indices, n)
    assert len(unikp_features) == n, "UniKP特征长度与数据不匹配！"

    all_results = []
    SEEDS = [i for i in range(10)]

    # ---------- 2. 10次 9:1 随机划分 ----------
    for fold, seed in enumerate(SEEDS):
        print(f"\n========== 开始第 {fold + 1} / 10 折训练 ==========")
        idx_train, idx_test = train_test_split(np.arange(n), test_size=0.1, random_state=seed)

        # ---------- A. DLKcat 训练和预测 ----------
        print("训练 DLKcat 模型...")
        train_dataset = [
            (
                torch.tensor(compounds[i]).long(),
                torch.tensor(adjacencies[i]).float(),
                torch.tensor(proteins[i]).long(),
                torch.tensor(y_true_log10[i] / math.log10(2)).float()
            )
            for i in idx_train
        ]
        test_dataset = [
            (
                torch.tensor(compounds[i]).long(),
                torch.tensor(adjacencies[i]).float(),
                torch.tensor(proteins[i]).long(),
                torch.tensor(y_true_log10[i] / math.log10(2)).float()
            )
            for i in idx_test
        ]

        model_dlkcat = KcatPrediction(DIM, LAYER_GNN, WINDOW, LAYER_CNN, LAYER_OUTPUT, n_fingerprint, n_word).to(device)
        trainer = TrainerDLKcat(model_dlkcat, device=device, lr=LR, weight_decay=WEIGHT_DECAY)

        # ================= 核心修改：添加 tqdm 进度条 =================
        for epoch in tqdm(range(EPOCHS), desc=f"Fold {fold+1} DLKcat Epochs"):
            loss, _, _ = trainer.train(train_dataset)
        # =============================================================

        model_dlkcat.eval()
        y_pred_dlkcat_train, y_pred_dlkcat_test = [], []

        for data in train_dataset:
            inputs = data[:-1]
            inputs_gpu = (inputs[0].to(device), inputs[1].to(device), inputs[2].to(device))
            _, pred_log2 = model_dlkcat((inputs_gpu[0], inputs_gpu[1], inputs_gpu[2], torch.tensor(0.0).to(device)),
                                        train=False)
            y_pred_dlkcat_train.extend([math.log10(math.pow(2, p)) for p in pred_log2])

        for data in test_dataset:
            inputs = data[:-1]
            inputs_gpu = (inputs[0].to(device), inputs[1].to(device), inputs[2].to(device))
            _, pred_log2 = model_dlkcat((inputs_gpu[0], inputs_gpu[1], inputs_gpu[2], torch.tensor(0.0).to(device)),
                                        train=False)
            y_pred_dlkcat_test.extend([math.log10(math.pow(2, p)) for p in pred_log2])

        y_pred_dlkcat_train = np.array(y_pred_dlkcat_train)
        y_pred_dlkcat_test = np.array(y_pred_dlkcat_test)

        r2_tr_d, rmse_tr_d, mae_tr_d, pcc_tr_d = evaluate_metrics(y_true_log10[idx_train], y_pred_dlkcat_train)
        r2_te_d, rmse_te_d, mae_te_d, pcc_te_d = evaluate_metrics(y_true_log10[idx_test], y_pred_dlkcat_test)

        # ---------- B. UniKP 训练和预测 ----------
        print("训练 UniKP 模型...")
        X_train, X_test = unikp_features[idx_train], unikp_features[idx_test]
        y_train, y_test = y_true_log10[idx_train], y_true_log10[idx_test]

        model_unikp = ExtraTreesRegressor(n_estimators=100, random_state=seed)
        model_unikp.fit(X_train, y_train)

        y_pred_unikp_train = model_unikp.predict(X_train)
        y_pred_unikp_test = model_unikp.predict(X_test)

        r2_tr_u, rmse_tr_u, mae_tr_u, pcc_tr_u = evaluate_metrics(y_train, y_pred_unikp_train)
        r2_te_u, rmse_te_u, mae_te_u, pcc_te_u = evaluate_metrics(y_test, y_pred_unikp_test)

        # ---------- C. 存入表格 ----------
        dlkcat_pred_dict = dict(zip(idx_test, y_pred_dlkcat_test))
        unikp_pred_dict = dict(zip(idx_test, y_pred_unikp_test))

        for i in idx_test:
            all_results.append({
                'Fold': fold + 1,
                'Sequence': sequences[i],
                'Smiles': smiles_list[i],
                'ECNumber': ec_numbers[i],
                'Organism': organisms[i],
                'Substrate': substrates[i],
                'Type': types[i],
                'True_Value': y_true_log10[i],
                'DLKcat_Predicted': dlkcat_pred_dict[i],
                'UniKP_Predicted': unikp_pred_dict[i]
            })

        print(
            f"Fold {fold + 1} DLKcat -> Test R2: {r2_te_d:.4f}, RMSE: {rmse_te_d:.4f}, MAE: {mae_te_d:.4f}, PCC: {pcc_te_d:.4f}")
        print(
            f"Fold {fold + 1} UniKP -> Test R2: {r2_te_u:.4f}, RMSE: {rmse_te_u:.4f}, MAE: {mae_te_u:.4f}, PCC: {pcc_te_u:.4f}")

        gc.collect()
        torch.cuda.empty_cache()

    # ==================== 3. 保存最终表格 ====================
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, 'model_performance_10fold.csv'), index=False)
    print(f"\n预测结果详细表格已保存至: {OUTPUT_DIR}/model_performance_10fold.csv")

    # ==================== 4. 计算10次平均值及箱线图 ====================
    final_metrics = []
    for fold in range(1, 11):
        fold_data = df_res[df_res['Fold'] == fold]
        y_true = fold_data['True_Value'].values
        y_pred_d = fold_data['DLKcat_Predicted'].values
        y_pred_u = fold_data['UniKP_Predicted'].values

        r2_d, rmse_d, mae_d, pcc_d = evaluate_metrics(y_true, y_pred_d)
        r2_u, rmse_u, mae_u, pcc_u = evaluate_metrics(y_true, y_pred_u)

        final_metrics.append({'Fold': fold, 'Model': 'DLKcat', 'R2': r2_d, 'RMSE': rmse_d, 'MAE': mae_d, 'PCC': pcc_d})
        final_metrics.append({'Fold': fold, 'Model': 'UniKP', 'R2': r2_u, 'RMSE': rmse_u, 'MAE': mae_u, 'PCC': pcc_u})

    metrics_df = pd.DataFrame(final_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_10fold_summary.csv'), index=False)

    avg_dlkcat = metrics_df[metrics_df['Model'] == 'DLKcat'][['R2', 'RMSE', 'MAE', 'PCC']].mean()
    avg_unikp = metrics_df[metrics_df['Model'] == 'UniKP'][['R2', 'RMSE', 'MAE', 'PCC']].mean()
    print("\n======= 10次交叉验证平均结果 =======")
    print("DLKcat:", avg_dlkcat.to_dict())
    print("UniKP :", avg_unikp.to_dict())

    # 图a & 图b
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=metrics_df, x='Model', y='R2', palette=["#00A896", "#9AC9F5"])
    plt.title('R² on test set')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Boxplot_R2_Test.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=metrics_df, x='Model', y='RMSE', palette=["#00A896", "#9AC9F5"])
    plt.title('RMSE on test set')
    plt.savefig(os.path.join(OUTPUT_DIR, 'Boxplot_RMSE_Test.png'), dpi=300)
    plt.close()

    # 图c - 10 折聚合散点图
    plt.figure(figsize=(7, 6))
    min_val = min(df_res['True_Value'].min(), df_res['DLKcat_Predicted'].min(), df_res['UniKP_Predicted'].min())
    max_val = max(df_res['True_Value'].max(), df_res['DLKcat_Predicted'].max(), df_res['UniKP_Predicted'].max())

    plt.scatter(df_res['True_Value'], df_res['UniKP_Predicted'], c='blue', alpha=0.3, s=8, label='UniKP')
    plt.scatter(df_res['True_Value'], df_res['DLKcat_Predicted'], c='green', alpha=0.3, s=8, label='DLKcat')
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')

    plt.xlabel('log10(experimental kcat value (s−1))')
    plt.ylabel('log10(predicted kcat value (s−1))')
    plt.title('10-Fold Aggregated Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Scatter_Pred_vs_True_10Fold.png'), dpi=300)
    plt.close()
    print(f"10折聚合散点图已保存至: {OUTPUT_DIR}/Scatter_Pred_vs_True_10Fold.png")

    # ==================== 5. 全量数据重训生成“平均”SHAP分析 ====================
    print("\n========== 正在训练最终全局模型并生成平均 SHAP 解释 ===========")
    model_unikp_final = ExtraTreesRegressor(n_estimators=100, random_state=42)
    model_unikp_final.fit(unikp_features, y_true_log10)

    print("正在计算 SHAP 值，这可能需要点时间...")
    explainer = shap.TreeExplainer(model_unikp_final)
    sample_size = min(1000, len(unikp_features))
    shap_values = explainer.shap_values(unikp_features[:sample_size])

    feature_names = [f'Substrate_{i}' for i in range(smi_dim)] + [f'Enzyme_{i}' for i in range(pro_dim)]

    shap_mean = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(shap_mean)[-20:]
    top_shap_values = shap_values[:, top_indices]
    top_feature_names = [feature_names[i] for i in top_indices]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(top_shap_values, unikp_features[:sample_size, top_indices], feature_names=top_feature_names,
                      show=False)
    plt.title('Global SHAP Analysis (Full Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'SHAP_summary_Global_Final.png'), dpi=300)
    plt.close()
    print(f"SHAP 图已保存至: {OUTPUT_DIR}/SHAP_summary_Global_Final.png")

    shap_df = pd.DataFrame({
        'Feature_Index': top_indices,
        'Feature_Name': top_feature_names,
        'Mean_Absolute_SHAP': shap_mean[top_indices]
    }).sort_values('Mean_Absolute_SHAP', ascending=False)
    shap_df.to_csv(os.path.join(OUTPUT_DIR, 'SHAP_Top20_Features_Global.csv'), index=False)
    print(f"SHAP 详细特征表格已保存至: {OUTPUT_DIR}/SHAP_Top20_Features_Global.csv")

    print("\n所有流程跑通，所有图表及 CSV 文件均已存入 'benchmark_output' 文件夹中。")


if __name__ == '__main__':
    run_experiment()