# 数据处理日志

## 数据来源
- 读取项目内 raw_data 文件夹中的 train_values.csv、train_labels.csv、test_values.csv 和 submission_format.csv。
- 不修改原始 CSV。工程特征、切分数据和图片仅输出到新目录。
- building_id 作为追踪与提交键保留；训练模型时才从特征矩阵中排除。

## 数据质量检查
- train_values：260601 行、39 列、0 个缺失单元格、0 个完全重复行。
- train_labels：260601 行、2 列、0 个缺失单元格、0 个完全重复行。
- test_values：86868 行、39 列、0 个缺失单元格、0 个完全重复行。
- submission_format：86868 行、2 列、0 个缺失单元格、0 个完全重复行。
- 训练特征和标签 ID 集合完全匹配；测试特征和提交模板 ID 集合完全匹配；训练和测试 schema 一致。
- 二值材料及用途变量检查为 0/1；次要用途汇总变量与细项变量没有逻辑冲突。

## 新建特征
1. age_is_995：age=995 取 1，否则取 0。
2. age_capped_100：年龄值上限截断为 100。
3. material_count：主体结构材料指示变量的逐行总和。
4. height_per_floor：height_percentage 除以 count_floors_pre_eq。
5. families_per_floor：count_families 除以 count_floors_pre_eq。
6. has_reinforced_concrete：任一钢筋混凝土指示变量为 1 时取 1。
7. 地理编号转换为字符串类别，避免作为连续数值处理。

## 划分规则
- 使用 random_state=42、80/20、按 damage_grade 分层的训练验证划分。
- 所有编码器、标准化器和重采样工具只能在 X_train 上 fit，防止验证集泄漏。

## EDA 结果
- Damage grade 1、2、3 的比例为 9.64%、56.89%、33.47%。
- age=995 有 1,390 条，占 0.53%，因此保留原值并构建标记和截尾特征。
- 不同地基类别的三级损毁观察比例范围为 2.09% 至 37.85%。
- 这些图形关系为描述性结果；模型价值应由独立验证集指标确认。


## 6 Expanded EDA
Additional figures cover: overall age distribution; overall numeric distributions; categorical frequency profiles; numeric and engineered-feature correlations; material prevalence versus observed Grade 3 damage; and damage composition by floor count. These diagnostics identify skew, rare categories, association patterns, and candidate predictors. They remain descriptive findings to be validated with held-out model results.
