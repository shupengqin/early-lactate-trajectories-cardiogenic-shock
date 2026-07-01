# ICU 心源性休克患者早期乳酸轨迹与院内死亡风险研究方案

## 1. 拟定题目

中文题目：

> ICU 心源性休克患者早期乳酸轨迹与院内死亡风险：基于 MIMIC-IV、eICU-CRD 和 SICdb 的多队列研究

英文题目：

> Early lactate trajectories and in-hospital mortality in ICU patients with cardiogenic shock: a multicohort study using MIMIC-IV, eICU-CRD, and SICdb

可选副标题：

> External validation, machine-learning prediction, and exploratory AKI mediation analysis

## 2. 研究定位

本研究定位为多数据库、回顾性队列研究。MIMIC-IV 作为发现队列和模型开发队列，eICU-CRD 作为主要外部验证队列，SICdb 作为第二外部验证队列或敏感性验证队列。

核心问题不是简单证明乳酸升高与死亡相关，而是回答：

1. ICU 心源性休克患者是否存在不同的早期乳酸轨迹？
2. 不同乳酸轨迹是否与院内死亡风险独立相关？
3. 乳酸轨迹是否比单次乳酸或乳酸清除率提供更强的预测信息？
4. 该发现能否在 eICU-CRD 和 SICdb 中外部验证？
5. AKI 是否可能部分中介异常乳酸轨迹与院内死亡之间的关联？

## 3. 研究假设

H1：持续高乳酸或乳酸升高轨迹患者的院内死亡风险更高。

H2：乳酸轨迹较单次初始乳酸和乳酸清除率能提供额外预后信息。

H3：将乳酸轨迹加入死亡预测模型后，模型的判别能力、校准表现和临床净获益提高。

H4：AKI 可能部分中介异常乳酸轨迹与院内死亡之间的关联。

## 4. 数据库分工

| 数据库 | 角色 | 主要任务 |
|---|---|---|
| MIMIC-IV | 发现队列 / 建模队列 | 队列构建、乳酸轨迹识别、主回归、机器学习训练 |
| eICU-CRD | 主要外部验证队列 | 验证轨迹-死亡关联、验证预测模型性能 |
| SICdb | 第二外部验证 / 敏感性验证 | 若字段完整，验证方向一致性和模型泛化性 |

SICdb 纳入前需要先完成字段可用性审查。最低要求包括：心源性休克诊断、ICU 入科时间、乳酸时间戳和值、院内死亡或 ICU 死亡、肌酐/尿量、基础人口学信息。

## 5. PICO 框架

| 项目 | 定义 |
|---|---|
| Population | 成人 ICU 心源性休克患者 |
| Exposure | ICU 入科后 0-24 h 早期乳酸轨迹 |
| Comparator | 低乳酸稳定组或乳酸下降组 |
| Outcome | 主要结局为院内死亡；次要结局包括 ICU 死亡、AKI、机械通气、ICU 住院时间 |

## 6. 时间轴设计

推荐主分析采用 landmark 设计：

```text
T0：ICU 入科
0-24 h：定义乳酸轨迹和基线协变量
24 h：landmark time
24-72 h：观察新发 AKI
24 h 至出院：观察院内死亡
```

0-48 h 乳酸轨迹可作为敏感性分析。主分析优先使用 0-24 h，以减少早死患者被系统性排除造成的偏倚。

## 7. 纳入与排除标准

### 7.1 纳入标准

1. 年龄 >= 18 岁。
2. 首次 ICU 入住。
3. 诊断为心源性休克。
4. 入 ICU 后 24 h 内至少 2 次乳酸检测。
5. 有明确院内死亡结局。
6. 主 landmark 分析中要求患者存活超过 24 h。

### 7.2 排除标准

1. 年龄 < 18 岁。
2. 非首次 ICU 入住。
3. 缺失 ICU 入科时间或死亡结局。
4. 乳酸检测不足。
5. 中介分析中排除 0-24 h 已发生 AKI 或终末期肾病患者。

## 8. 心源性休克定义

MIMIC-IV 可优先使用 ICD 编码：

| 编码体系 | 编码 | 诊断 |
|---|---|---|
| ICD-9 | 785.51 | Cardiogenic shock |
| ICD-10 | R57.0 | Cardiogenic shock |

eICU-CRD 和 SICdb 可根据诊断字段、入院诊断、问题列表或文本诊断匹配 cardiogenic shock。正式分析中建议设置两种定义：

1. 严格定义：明确心源性休克编码或诊断。
2. 宽松定义：诊断文本包含 cardiogenic shock，并结合血管活性药或低血压支持。

主分析采用严格定义，宽松定义用于敏感性分析。

## 9. 暴露变量：乳酸轨迹

### 9.1 乳酸原始值

提取 ICU 入科后 0-24 h 内所有乳酸值，记录检测时间和值。

单位需统一为 mmol/L。若数据库存在不同单位，统一换算后再分析。

### 9.2 可迁移乳酸特征

为便于跨数据库验证，除复杂轨迹模型外，应同时构建可迁移特征：

| 特征 | 定义 |
|---|---|
| initial_lactate | 0-24 h 内首次乳酸 |
| peak_lactate | 0-24 h 内最高乳酸 |
| minimum_lactate | 0-24 h 内最低乳酸 |
| last_lactate | 0-24 h 内最后一次乳酸 |
| lactate_clearance | (initial_lactate - last_lactate) / initial_lactate * 100% |
| lactate_slope | 乳酸值对检测时间的线性斜率 |
| persistent_high_lactate | 末次乳酸 >= 4 mmol/L 或峰值持续升高 |

### 9.3 轨迹模型

首选方法：

1. Group-based trajectory modeling, GBTM。
2. Latent class mixed model, LCMM。

候选组数为 2-5 组。根据 BIC、AIC、entropy、每组样本量和临床可解释性确定最终组数。

预计轨迹组：

1. Low-stable：低乳酸稳定组。
2. Moderate-decreasing：中度升高后下降组。
3. High-decreasing：高乳酸后下降组。
4. Persistently-high/rising：持续高乳酸或升高组。

## 10. 结局变量

主要结局：

1. 院内死亡。

次要结局：

1. ICU 死亡。
2. 28 天死亡，如数据库可定义。
3. ICU 住院时间。
4. 总住院时间。
5. 新发 AKI。
6. 机械通气。
7. 肾脏替代治疗。

## 11. 协变量

### 11.1 人口学

年龄、性别、种族、体重或 BMI。

### 11.2 合并症

高血压、糖尿病、慢性肾病、冠心病、心力衰竭、房颤、COPD、肝病、恶性肿瘤。

### 11.3 生命体征

心率、平均动脉压、收缩压、舒张压、呼吸频率、体温、SpO2。

### 11.4 实验室指标

肌酐、尿素氮、白细胞、血红蛋白、血小板、钠、钾、氯、碳酸氢根、pH、PaO2、PaCO2、INR、总胆红素。

### 11.5 严重程度评分

MIMIC-IV：SOFA、SAPS II、OASIS。

eICU-CRD：APACHE IV/APS 等可用评分。

SICdb：根据字段可用性决定是否重建 SOFA 或使用数据库内置评分。

### 11.6 治疗和器官支持

机械通气、血管活性药、肾脏替代治疗、IABP、ECMO、液体输入量。

## 12. 三数据库变量映射表

| 变量模块 | MIMIC-IV | eICU-CRD | SICdb | 备注 |
|---|---|---|---|---|
| 患者 ID | subject_id, hadm_id, stay_id | uniquepid, patientunitstayid | 待确认 | 需统一到 ICU stay 层级 |
| ICU 入科时间 | icustays.intime | patient.unitadmittime 或等效字段 | 待确认 | 时间零点 T0 |
| 出 ICU 时间 | icustays.outtime | patient.unitdischargetime | 待确认 | ICU 结局 |
| 院内死亡 | admissions.deathtime/hospital_expire_flag | hospitaldischargestatus | 待确认 | 主结局 |
| 心源性休克 | diagnoses_icd | diagnosis/admissiondx | 待确认 | ICD 或文本匹配 |
| 乳酸 | labevents + d_labitems | lab 表 | 待确认 | 需时间戳和值 |
| 肌酐 | labevents | lab 表 | 待确认 | AKI 定义 |
| 尿量 | outputevents | intakeOutput | 待确认 | AKI 定义，可能缺失较多 |
| 血管活性药 | inputevents | infusionDrug/treatment | 待确认 | 混杂调整 |
| 机械通气 | procedureevents/chartevents | respiratoryCharting/treatment | 待确认 | 混杂调整 |
| SOFA/APACHE | 可重建 SOFA/OASIS | APACHE 变量较完整 | 待确认 | 跨库不完全一致 |

## 13. 主统计分析

### 13.1 描述性分析

按乳酸轨迹组描述基线特征。连续变量使用均值和标准差或中位数和四分位数；分类变量使用频数和百分比。

组间比较：

1. 连续变量：t 检验、ANOVA、Mann-Whitney U 或 Kruskal-Wallis。
2. 分类变量：卡方检验或 Fisher 精确检验。

### 13.2 主要回归模型

主要模型：

```text
in_hospital_mortality ~ lactate_trajectory + demographics + comorbidities
                      + severity_score + vital_signs + labs + organ_support
```

建议设置三个层次：

1. Model 1：年龄、性别、种族。
2. Model 2：Model 1 + 合并症 + 严重程度评分。
3. Model 3：Model 2 + 生命体征 + 实验室指标 + 器官支持治疗。

报告 OR、95% CI 和 P 值。

### 13.3 生存分析

若死亡时间可靠，使用 Cox 回归分析出院前死亡风险。

```text
Survival time = ICU 入科或 landmark time 至死亡/出院
```

报告 HR 和 95% CI。

### 13.4 非线性分析

使用限制性立方样条分析以下变量与死亡风险的非线性关系：

1. 初始乳酸。
2. 峰值乳酸。
3. 乳酸清除率。
4. 乳酸斜率。

## 14. 外部验证

外部验证分为两层：

### 14.1 轨迹可重复性验证

在 eICU-CRD 和 SICdb 中尝试复现 MIMIC-IV 的轨迹组。若复杂轨迹模型不可稳定复现，则使用可迁移特征进行分组：

1. 低乳酸稳定。
2. 乳酸下降。
3. 持续高乳酸。
4. 乳酸升高。

### 14.2 风险关联验证

在外部队列中使用相同协变量框架，检验轨迹组与院内死亡风险的关联方向和效应量是否一致。

外部验证报告：

1. OR/HR 方向是否一致。
2. 95% CI。
3. 模型 AUROC。
4. 校准曲线。
5. Brier score。

## 15. 机器学习分析

### 15.1 核心问题

乳酸轨迹是否能提高院内死亡预测模型性能。

### 15.2 模型分层

| 模型 | 变量 |
|---|---|
| ML-1 | 基础临床变量 |
| ML-2 | 基础变量 + 初始乳酸 |
| ML-3 | 基础变量 + 乳酸清除率 |
| ML-4 | 基础变量 + 乳酸轨迹 |
| ML-5 | 全变量模型 |

### 15.3 算法

1. Logistic regression。
2. LASSO logistic regression。
3. Random Forest。
4. XGBoost 或 LightGBM。
5. CatBoost。

### 15.4 验证策略

MIMIC-IV 内部训练和调参：

1. 训练集/验证集划分，或嵌套交叉验证。
2. 缺失值填补、标准化和变量筛选仅在训练集内拟合。
3. 将同一流程应用到 eICU-CRD 和 SICdb。

### 15.5 评价指标

1. AUROC。
2. AUPRC。
3. Brier score。
4. Calibration slope/intercept。
5. Calibration curve。
6. Decision curve analysis。
7. SHAP 变量重要性。

### 15.6 数据泄漏控制

所有预测变量必须来自 0-24 h 窗口内或窗口前。不得纳入 24 h 后的实验室、治疗或结局相关信息。

## 16. 中介分析

### 16.1 推荐中介变量

首选中介变量：24-72 h 新发 AKI。

因果路径：

```text
异常乳酸轨迹 -> 新发 AKI -> 院内死亡
```

### 16.2 分析人群

1. 存活超过 24 h 的患者。
2. 排除 0-24 h 已发生 AKI 的患者。
3. 排除终末期肾病患者。

### 16.3 模型设置

暴露：乳酸轨迹组，重点比较持续高乳酸/升高组 vs 低乳酸稳定组。

中介：24-72 h 新发 AKI。

结局：院内死亡。

调整变量：年龄、性别、基础肾功能、合并症、SOFA/APACHE、血管活性药、机械通气、初始肌酐、平均动脉压。

输出：

1. Total effect。
2. Natural direct effect。
3. Natural indirect effect。
4. Proportion mediated。

### 16.4 表述限制

中介分析应表述为 exploratory mediation analysis 或 potential mediation pathway。由于回顾性观察研究存在未测量混杂，不宜使用强因果表述。

## 17. 敏感性分析

1. 将乳酸轨迹窗口由 0-24 h 改为 0-48 h。
2. 仅纳入至少 3 次乳酸检测者。
3. 不排除 24 h 内死亡患者，改用早期乳酸特征分析。
4. 使用严格心源性休克定义和宽松定义分别分析。
5. 使用乳酸清除率替代轨迹组。
6. 完整病例分析 vs 多重插补。
7. 排除 ECMO 或 IABP 患者。
8. 按 AMI 相关心源性休克和非 AMI 心源性休克分层。
9. 按是否接受机械通气分层。
10. eICU-CRD 和 SICdb 分别验证，不简单合并。

## 18. 亚组分析

1. AMI 相关心源性休克 vs 非 AMI 心源性休克。
2. 心力衰竭病史有无。
3. 年龄 < 65 vs >= 65 岁。
4. 男性 vs 女性。
5. 是否机械通气。
6. 是否使用血管活性药。
7. 基线肾功能正常 vs 异常。
8. 初始乳酸 < 4 vs >= 4 mmol/L。

## 19. 图表计划

### 19.1 主文图

1. Figure 1：三数据库队列筛选流程图。
2. Figure 2：乳酸轨迹曲线。
3. Figure 3：不同轨迹组院内死亡率。
4. Figure 4：轨迹组与死亡风险的调整后 OR/HR 森林图。
5. Figure 5：机器学习模型 ROC、PR 曲线和校准曲线。
6. Figure 6：SHAP 变量重要性。
7. Figure 7：AKI 中介分析路径图。

### 19.2 主文表

1. Table 1：MIMIC-IV 基线特征。
2. Table 2：不同乳酸轨迹组结局比较。
3. Table 3：多变量回归模型。
4. Table 4：机器学习模型性能。
5. Table 5：eICU-CRD 和 SICdb 外部验证结果。

### 19.3 补充材料

1. 三数据库变量映射表。
2. ICD 和诊断文本定义。
3. 缺失值比例。
4. 敏感性分析。
5. 亚组分析。

## 20. 预期创新点

1. 从单次乳酸转向早期乳酸动态轨迹。
2. 聚焦 ICU 心源性休克专病人群。
3. 使用 MIMIC-IV 建模，并在 eICU-CRD 和 SICdb 做多队列验证。
4. 比较乳酸轨迹、乳酸清除率和单次乳酸的预测价值。
5. 结合机器学习和 SHAP 解释，评估临床预测增益。
6. 探索 AKI 在乳酸轨迹与死亡之间的潜在中介作用。

## 21. 主要风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| SICdb 字段不完整 | 无法正式外部验证 | 改为敏感性验证或移除 SICdb |
| 乳酸检测次数不足 | 轨迹模型不稳定 | 主分析要求 >=2 次，敏感性分析要求 >=3 次 |
| 0-48 h 窗口排除早死患者 | 选择偏倚 | 主分析使用 0-24 h landmark |
| 数据库变量不一致 | 外部验证困难 | 使用可迁移乳酸特征和统一变量字典 |
| 机器学习数据泄漏 | 结果虚高 | 严格限定 0-24 h 变量 |
| 中介分析因果假设强 | 审稿风险 | 明确为探索性分析，避免强因果语言 |

## 22. 推荐写作规范

1. 观察性研究主体遵循 STROBE。
2. 预测模型部分参考 TRIPOD/TRIPOD+AI。
3. 机器学习报告包括数据划分、缺失值处理、调参、外部验证、校准和临床效用。
4. 中介分析明确时间顺序、混杂调整和因果假设限制。

## 23. 下一步执行清单

1. 确认三数据库访问权限。
2. 完成 SICdb 字段字典审查。
3. 写 MIMIC-IV 队列提取 SQL。
4. 写 eICU-CRD 队列提取 SQL。
5. 如果 SICdb 可用，写 SICdb 变量提取脚本。
6. 先跑样本量：心源性休克总例数、乳酸 >=2 次人数、院内死亡数。
7. 根据样本量决定轨迹组数和机器学习复杂度。
8. 输出第一版 Table 1 和队列流程图。

