# logP 分子表征与预测项目（Transformer Seq2Seq + RandomForest）
输入一个分子的 `IUPAC` 名称，模型先生成对应的 `SELFIES`，再基于编码器表征预测该分子的 `logP`。

## 1. 关键指标解释 
- `epoch`：训练轮次编号。  
- `train_loss`：训练集 token 级交叉熵损失。  
- `train_property_loss`：训练集 `logP` 回归损失。  
- `train_joint_loss`：训练集联合损失。  
- `train_token_acc`：训练集 token 准确率。  
- `valid_loss`：验证集 token 级交叉熵损失。  
- `valid_property_loss`：验证集 `logP` 回归损失。  
- `valid_joint_loss`：验证集联合损失。  
- `valid_token_acc`：验证集 token 准确率。  
- `lr`：当前学习率。  
- `exact_match`：验证集整条 SELFIES 完全匹配率。  
- `avg_edit_distance`：验证集平均编辑距离，越低越好，表示若要修改为正确的selfies需要进行修改的步数，每修改一次就会惩罚数就会＋1。 
- `normalized_edit_distance`：验证集归一化编辑距离，越低越好，表示若要修改为正确的selfies需要进行修改的步数，每修改一次就会惩罚数就会＋1。   
- `RMSE / MAE / R2`：下游 `logP` 回归评估指标，其中 `RMSE/MAE` 越低越好，`R2` 越高越好。  

## 2. 数据集处理流程

`logp_project/data.py`：

- 读取 Excel 数据文件。  
- 清洗分子记录：校验 `smiles / selfies / iupac / logP` 四列是否存在并过滤缺失值。  
- 将 `IUPAC` 按字符级切分为源序列，将 `SELFIES` 按 token 切分为目标序列。  
- 基于训练集构建 `src_vocab` 与 `tgt_vocab`。  
- 按随机种子切分为 train / valid / test。  
- 保存 `results/dataset/data_splits.csv` 供后续训练与复现使用。  

### 2.1 数据划分：

1. 读取并标准化数据列  
- 从 `dataset/rawdata.xlsx` 读入数据。  
- 要求至少包含 `smiles`、`selfies`、`iupac`、`logP` 四列。  

2. 单条记录预处理  
- 去掉换行与首尾空格。  
- 若任一关键字段为空，则该条记录跳过。  
- 为每条可用分子记录分配连续 `row_id`。  

3. token 化  
- `IUPAC`：按字符级切分，用作 Seq2Seq 源序列。  
  例如 `ethanol` 会被拆成 `e / t / h / a / n / o / l`。这样做的好处是实现简单，不依赖额外分词词典，而且对括号、数字、连字符这类 IUPAC 命名中的局部模式也能直接学习。  
- `SELFIES`：优先按 `[...]` 规则切分，用作目标序列。  
  例如 `[C][=O][O]` 会被拆成 `[C] / [=O] / [O]`。项目里会优先用正则把这些完整块提取出来，而不是按单个字符去拆，这样能保留 SELFIES 天然的语义边界，更适合生成任务学习。  

4. 构建词表  
- 仅基于训练集构建词表。  
- 特殊符号固定包含 `<pad> / <bos> / <eos> / <unk>`。  

5. 划分训练/验证/测试集  
- 先基于 `seed=42` 打乱。  
- 配置目标比例：  
- `train = 0.8`  
- `valid = 0.1`  
- `test = 0.1`  
- 当前 `results/dataset/data_splits.csv` 中的实际分子数为：  
- `train = 24534` 个分子  
- `valid = 3067` 个分子  
- `test = 3067` 个分子  
- 总计 `30668` 个可用分子  

即按 **8:1:1** 划分。  

## 3. 超参数设置（学习率 / AdamW / 预热 / Transformer）

- 优化器：`AdamW(lr=2.5e-4, weight_decay=1e-2)`  
- 学习率调度：`LambdaLR + warmup + cosine decay`  
- warmup 比例：`0.05`  
- 梯度裁剪：`clip_grad_norm_(..., 1.0)`  
- 最大训练轮数：`epochs=120`  
- 实际训练轮数：`85`（Early Stopping 提前停止）  
- 批大小：`batch_size=64`  
- 标签平滑：`label_smoothing=0.05`  
- 属性损失权重：`property_loss_weight=0.5`  
- Early Stopping：`patience=15`, `min_delta=1e-4`  
- 模型参数：`d_model=384`, `nhead=8`, `encoder_layers=5`, `decoder_layers=5`, `dim_feedforward=1536`, `dropout=0.15`  
- Morgan 指纹：`radius=2`, `bits=2048`  
- RandomForest：`n_estimators=300`  

联合损失定义：

`JointLoss = SeqCrossEntropy + 0.5 * MSE(pred_logP, target_logP)`  

它和训练日志中的几个字段对应关系如下：

- `train_loss`：就是上式里的 `SeqCrossEntropy`。  
- `train_property_loss`：就是上式里的 `MSE(pred_logP, target_logP)`。  
- `train_joint_loss`：就是完整的 `JointLoss = train_loss + 0.5 * train_property_loss`。  

验证集同理：

- `valid_loss`：验证集上的 `SeqCrossEntropy`。  
- `valid_property_loss`：验证集上的 `MSE(pred_logP, target_logP)`。  
- `valid_joint_loss`：验证集上的 `valid_loss + 0.5 * valid_property_loss`。  

## 4. `logp_project` 文件介绍

- `cli.py`：项目主入口，负责训练、测试、特征提取、下游回归评估与单条预测。  
- `data.py`：数据读取、token 化、词表构建、数据集封装、DataLoader 与数据切分。  
- `model.py`：  
  - `encode()`：把输入的 `IUPAC` 序列编码成 Transformer 的上下文表示，是整个模型理解分子命名信息的核心步骤。  
  - `decode()`：根据编码结果自回归生成 `SELFIES`，输出每个位置对目标词表的预测 logits，是序列生成的核心步骤。  
  - `forward_multitask()`：训练时最关键的前向函数，一次同时完成两件事：生成 `SELFIES`，以及预测分子的 `logP`。这也是本项目“序列生成 + 性质预测”联合训练的核心。  
  - `greedy_decode()`：推理阶段逐步生成 `SELFIES` 的函数。它每一步都选当前概率最大的 token，直到结束，用于 `predict_one.py` 的实际预测。  
  - `extract_encoder_embeddings()`：提取编码器学到的分子表示，后续 `RandomForest` 回归、特征对比和 PCA/t-SNE/UMAP 可视化都依赖它导出的 embedding。  
- `training.py`：训练主流程、验证、编辑距离评估、日志保存和曲线绘制。  
- `evaluation.py`：对多种分子特征做 `logP` 回归评估，并生成散点图和汇总图。  
- `features.py`：  
  - 生成 `Morgan fingerprint`、`MACCS keys`、`RDKit 2D descriptors`。  
  - 做 PCA / t-SNE / UMAP 可视化。  
  - 针对 `RDKit 2D` 做公平性过滤，移除 `MolLogP` 及 `SlogP_VSA*`。  
- `inference.py`：加载 checkpoint，执行单个 `IUPAC` 的 SELFIES 生成与 `logP` 预测。  
- `paths.py`：统一维护 `results/` 下各类输出目录。  
- `constants.py`：项目常量，当前性质名为 `logP`。  

## 5. `train.py` 和 `predict_one.py` 介绍

- `main/train.py`  
  - 训练主入口。  
  - 实际调用 `logp_project.cli.main()`。  
  - 完成数据切分、Seq2Seq 训练、测试集评估、编码器 embedding 导出、传统特征计算、RandomForest 回归比较和可视化生成。  

- `main/predict_one.py`  
  - 单分子推理入口。  
  - 无参数运行时会交互式要求输入 `IUPAC` 名称。  
  - 也可命令行传入 `--predict_iupac` 与可选 `--checkpoint`。  
  - 当前单条预测仅支持 `predict_feature_set=encoder_memory`。  

## 6. `results` 收录的结果

### Seq2Seq 测试评估
- `results/test_evaluation/seq2seq_transformer_test_metrics.json`：测试集上 Seq2Seq 与属性头的整体评估指标。  

### 特征对比评估
- `results/test_evaluation/feature_comparison_results/property_prediction_results.csv`：不同特征下 `logP` 回归结果对比。  
- `results/test_evaluation/feature_comparison_results/best_property_results.csv`：最佳特征方案。  
- `results/test_evaluation/feature_comparison_results/评估依据.txt`：最佳方案判定说明。  

### 单样本预测结果
- `results/test_evaluation/feature_prediction_comparison_results/*.csv`：各特征方案在测试集上的逐条预测结果。  

### 可视化图表
- `results/figures/training_curves/seq2seq_loss_curve.png`：训练/验证损失曲线。  
- `results/figures/training_curves/seq2seq_accuracy_curve.png`：token_acc、exact_match、normalized_edit_distance 曲线。  
- `results/figures/training_curves/seq2seq_lr_curve.png`：学习率变化曲线。  
- `results/figures/property_scatter/*.png`：预测值与真实值散点图。  
- `results/figures/property_summary/property_summary_logP.png`：不同特征方案的整体表现对比图。  
- `results/figures/projection_pca/*.png`、`projection_tsne/*.png`、`projection_umap/*.png`：不同特征空间对 `logP` 的投影视图。  

### 日志与模型
- `results/logs/seq2seq_history.csv`：逐 epoch 训练与验证日志。  
- `results/logs/training_summary.json`：最佳 epoch 与最终训练摘要。  
- `results/models/best_model.pt`：最佳 Seq2Seq 模型权重。  
- `results/models/*.pkl`：不同特征方案对应的 RandomForest 回归器。  

### 元数据与表征
- `results/metadata/run_info/config.json`：训练配置。  
- `results/metadata/run_info/environment.json`：运行环境信息。  
- `results/metadata/vocab/*.json`：源/目标词表。  
- `results/metadata/rdkit/*.json`：RDKit 描述符名称与公平性过滤规则。  
- `results/representations/embedding/*.npy`：编码器表征。  
- `results/representations/row_ids/*.npy`：embedding 对应样本的 `row_id`。  

## 7. 结果

**训练信息**（具体训练细节见"logP\results\logs\seq2seq_history.csv"）
- 实际训练轮数：`85`
- 最佳 epoch：`70`
- 最佳验证 `valid_loss`：`0.8399151632125857`
- 最佳验证 `property_loss`：`0.3708254790166165`
- 最佳验证 `valid_joint_loss`：`1.025275290148399`
- 最佳验证 `token_acc`：`0.8504278261975953`
- 最佳验证 `exact_match`：`0.006194978806651451`
- 最佳验证 `avg_edit_distance`：`19.23866970981415`
- 最佳验证 `normalized_edit_distance`：`0.4794387181351431`


**测试集评估**
- `test_loss`: 0.851131889124057,
- `test_property_loss`: 0.4091310195010222,
- `test_joint_loss`: 1.0557794882647176,
- `test_token_acc`: 0.8474630265285156,
- `test_exact_match`: 0.007173133355070101,
- `test_avg_edit_distance`: 19.304858167590478,
- `test_normalized_edit_distance`: 0.48322233253509184


## 8. 评估

```json

### logP 下游回归结果对比
[
  {
    "feature_set": "rdkit_2d",
    "rmse": 0.47768620511665366,
    "mae": 0.3659126186726346,
    "r2": 0.8671209881908377,
    "fairness_note": "rdkit_filtered_removed_13"
  },
  {
    "feature_set": "maccs_keys",
    "rmse": 0.7504148310786194,
    "mae": 0.5851391099158446,
    "r2": 0.6720755431148145,
    "fairness_note": "standard"
  },
  {
    "feature_set": "encoder_memory",
    "rmse": 0.8108111045747769,
    "mae": 0.6183205613333624,
    "r2": 0.6171661187607683,
    "fairness_note": "standard"
  },
  {
    "feature_set": "morgan_fp",
    "rmse": 0.837540274519113,
    "mae": 0.6592986892516493,
    "r2": 0.5915090960618837,
    "fairness_note": "standard"
  }
]
```

结论：根据rmse，下游任务`rdkit_2d` 是表现最好的 `logP` 特征方案，encoder_memory表征方式效果仅位列第三。rmse是评估feature效能的指标，越小越好。(具体评估性能见"logP\results\test_evaluation\feature_prediction_comparison_results")
注释：
`fairness_note`：特征评估时的公平性说明，用来标记该特征是否做了“防信息泄漏”处理。当前 `rdkit_2d` 会移除 `MolLogP` 和 `SlogP_VSA*` 这类与目标 `logP` 强相关的描述符，避免结果虚高；`standard` 表示未额外做这类过滤。
`rdkit_filtered_removed_13`:数据rdkit_2d表征方法中存在与logp相似的指标，为了防止rdkit_2d预测器偷看答案，一共排除13个相关的指标。

## 9. 生成案例
1：
Input IUPAC: N-[1-(4-fluorophenyl)sulfonyl-3,4-dihydro-2H-quinolin-7-yl]quinoline-2-carboxamide
Predicted SELFIES: [O][=C][Branch2][Ring1][O][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][C][C][C][N][Ring1][#Branch1][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][Branch1][C][F][C][=C][Ring1][#Branch1][N][C][C][O][C][C][Ring1][=Branch1]
Predicted logP: 3.623795

答案：
IUPAC 名称：N-[1-(4-fluorophenyl)sulfonyl-3,4-dihydro-2H-quinolin-7-yl]quinoline-2-carboxamide
sefies:[O][=C][Branch2][Ring2][#Branch1][N][C][=C][C][=C][C][=Branch1][Ring2][=C][Ring1][=Branch1][N][Branch2][Ring1][Ring1][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][Branch1][C][F][C][=C][Ring1][#Branch1][C][C][C][Ring1][P][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=N][Ring1][#Branch2]
logP:4.7677

2:
Input IUPAC: N-[1-(4-methylsulfonyl-2-nitrophenyl)piperidin-4-yl]methanesulfonamide
Predicted SELFIES: [C][S][=Branch1][C][=O][=Branch1][C][=O][N][C][C][C][Branch2][Ring1][Ring1][C][N][S][=Branch1][C][=O][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][N+1][=Branch1][C][=O][O-1][C][C][Ring2][Ring1][C]
Predicted logP: 1.275912

答案：
IUPAC 名称：N-[1-(4-methylsulfonyl-2-nitrophenyl)piperidin-4-yl]methanesulfonamide
sefies:[C][S][=Branch1][C][=O][=Branch1][C][=O][N][C][C][C][N][Branch2][Ring1][Branch2][C][=C][C][=C][Branch1][=Branch2][S][Branch1][C][C][=Branch1][C][=O][=O][C][=C][Ring1][#Branch2][N+1][=Branch1][C][=O][O-1][C][C][Ring2][Ring1][Ring1]
logP:0.5163

结论：第一，由于前面的四种分子表征可知，iupac位于第三因此效能不佳，由分子生成案例可知与logP真实值差距较大。第二，由于训练分子数据仅为三万多个，有可能训练集和测试集中的分子结果复杂程度和logp分布差异，也有可能是iupac不擅长logP任务，最终seflies生成结果也不佳。
注释：生成案例采用测试集中的分子
