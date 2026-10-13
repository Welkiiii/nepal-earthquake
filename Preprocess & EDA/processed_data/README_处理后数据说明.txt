尼泊尔地震建筑损毁项目：处理后数据说明
====================================

一、先说明：本文件夹中的 CSV 都不是“完全未处理的原始数据”。
它们均由竞赛原始数据经过检查、特征工程和（部分文件的）训练/验证集切分后得到。
原始数据没有被覆盖或修改。

原始数据的来源及用途：
- train_values.csv：260,601 栋训练建筑的原始特征，包含 building_id 和 38 个原始特征。
- train_labels.csv：与训练建筑一一对应的真实标签，包含 building_id 和 damage_grade。
- test_values.csv：86,868 栋测试建筑的原始特征；没有 damage_grade，供最终预测使用。
- submission_format.csv：最终提交结果的格式模板。

本文件夹中的数据文件说明：

1. train_features_engineered.csv
- 来源：train_values.csv。
- 内容：完整训练特征，保留 building_id，并加入了本项目构建的新特征。
- 主要新增特征：material_count、height_per_floor、families_per_floor、
  has_reinforced_concrete、age_is_995、age_capped_100。
- 用途：当你希望自行重新划分训练/验证集，或在全部训练数据上重训最终模型时使用。

2. test_features_engineered.csv
- 来源：test_values.csv。
- 内容：完整测试特征，使用与训练集完全一致的特征工程规则，也保留 building_id。
- 用途：模型最终训练完成后，用它生成对测试集的 damage_grade 预测。

3. train_labels_preserved.csv
- 来源：train_labels.csv。
- 内容：building_id 与真实 damage_grade，没有对标签进行改动。
- 用途：可按 building_id 与 train_features_engineered.csv 合并，获得完整训练数据。

4. train_validation_split.csv
- 来源：完整训练数据的分层 80/20 切分。
- 内容：记录每个训练样本属于训练集还是验证集。
- 说明：使用 stratified split，保证 damage_grade 1、2、3 的比例在两部分中大致一致。

5. X_train.csv 和 y_train.csv
- 来源：完整训练数据中约 80% 的样本。
- X_train.csv：用于训练模型的输入特征 X。
- y_train.csv：对应的真实损毁等级 y（damage_grade）。
- 用途：直接用于 model.fit(X_train, y_train)。

6. X_validation.csv 和 y_validation.csv
- 来源：完整训练数据中约 20% 的样本，且不参与模型训练。
- X_validation.csv：验证模型时的输入特征。
- y_validation.csv：验证集的真实 damage_grade。
- 用途：用模型预测 X_validation，再与 y_validation 比较，计算 micro F1、macro F1、混淆矩阵等。

7. feature_metadata.json
- 来源：本次预处理过程自动生成的说明文件。
- 内容：记录特征名称、处理规则、构建的新特征和数据划分等元信息。
- 用途：供需要核对变量含义或复现处理流程的成员查看。

二、给建模同学的最简单使用方式
1. 读取 X_train.csv、y_train.csv，训练模型。
2. 使用 X_validation.csv 预测，并与 y_validation.csv 计算评价指标。
3. 确定模型后，可把 train_features_engineered.csv 与 train_labels_preserved.csv 用 building_id 合并，
   在完整训练数据上重新训练。
4. 使用 test_features_engineered.csv 做最终预测，并按 building_id 生成 submission。

三、重要提醒
- building_id 必须保留：它用于合并标签及生成最终 submission。
  但在放入模型训练前，应将 building_id 从输入特征 X 中排除。
- 地理 ID 和其他类别变量尚应在建模 pipeline 中进行 One-Hot Encoding 或适合该模型的编码；
  不应把类别代码误当成有连续大小关系的普通数值。
- 不要把验证集用于反复调参后再当作“完全独立测试集”；课程报告中应清楚说明验证流程。