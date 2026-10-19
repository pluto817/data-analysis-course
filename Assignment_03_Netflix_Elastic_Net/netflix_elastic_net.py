
import pandas as pd
from pathlib import Path

# 获取当前 Python 文件所在目录
data_dir = Path(__file__).parent

# 读取 Netflix 数据集
file_path = data_dir / "netflix_titles.csv"

try:
    data = pd.read_csv(file_path, encoding="utf-8")
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding="latin1")


# 查看数据基本情况
print("数据前5行：")
print(data.head())

print("\n数据集大小：")
print(data.shape)

print("\n数据基本信息：")
data.info()

print("\n各列缺失值数量：")
print(data.isnull().sum())

# 1. 查看电影和电视剧的数量
print("\n电影和电视剧数量：")
print(data["type"].value_counts())

# 2. 只保留电影
movies = data[data["type"] == "Movie"].copy()

# 3. 从 duration 中提取电影时长（分钟）
movies["duration_minutes"] = (
    movies["duration"]
    .str.extract(r"(\d+)")
    .astype(float)
)

# 4. 查看电影数据的基本情况
print("\n电影数据集大小：")
print(movies.shape)

print("\n电影时长统计：")
print(movies["duration_minutes"].describe())

# 5. 查看电影时长缺失值
print("\n电影时长缺失值数量：")
print(movies["duration_minutes"].isnull().sum())

# 6. 查看电影时长示例
print("\n电影时长示例：")
print(movies[["title", "duration", "duration_minutes"]].head(10))

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 第一步：准备建模数据
model_data = movies.copy()

# 从 listed_in 中取第一个类别作为电影类型
model_data["genre"] = (
    model_data["listed_in"]
    .str.split(",")
    .str[0]
    .str.strip()
)

# 从 country 中取第一个国家或地区
model_data["main_country"] = (
    model_data["country"]
    .fillna("Unknown")
    .str.split(",")
    .str[0]
    .str.strip()
)

# 缺失的年龄分级用 Unknown 表示
model_data["rating"] = model_data["rating"].fillna("Unknown")

# 选择输入特征和预测目标
features = [
    "release_year",
    "rating",
    "main_country",
    "genre"
]

X = model_data[features]
y = model_data["duration_minutes"]

# 第二步：划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 第三步：分别处理数值特征和类别特征
numeric_features = ["release_year"]
categorical_features = ["rating", "main_country", "genre"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 第四步：建立并训练 Elastic Net 模型
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000))
])

model.fit(X_train, y_train)

# 第五步：预测测试集
y_pred = model.predict(X_test)

# 第六步：评估模型表现
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== Elastic Net 模型结果 ==========")
print(f"训练集样本数：{len(X_train)}")
print(f"测试集样本数：{len(X_test)}")
print(f"MAE：{mae:.2f} 分钟")
print(f"RMSE：{rmse:.2f} 分钟")
print(f"R²：{r2:.4f}")

from sklearn.base import clone
from sklearn.linear_model import Ridge, Lasso

# 建立三种模型
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1, max_iter=10000),
    "Elastic Net": ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        max_iter=10000
    )
}

results = []

# 使用相同训练集和测试集比较模型
for name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("regressor", regressor)
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    })

# 输出对比结果
results_df = pd.DataFrame(results)

print("\n========== 三种模型对比 ==========")
print(results_df.round(4).to_string(index=False))

from sklearn.model_selection import GridSearchCV

# 使用交叉验证搜索 Elastic Net 的较优参数
elastic_pipeline = Pipeline([
    ("preprocessor", clone(preprocessor)),
    ("regressor", ElasticNet(max_iter=20000))
])

param_grid = {
    "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
    "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}

grid_search = GridSearchCV(
    estimator=elastic_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

# 只用训练集选择参数
grid_search.fit(X_train, y_train)

# 使用最佳模型评估测试集
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test)

print("\n========== Elastic Net 参数调优 ==========")
print("最佳参数：", grid_search.best_params_)
print(f"交叉验证 MAE：{-grid_search.best_score_:.4f}")
print(
    f"测试集 MAE："
    f"{mean_absolute_error(y_test, best_predictions):.4f}"
)
print(
    f"测试集 RMSE："
    f"{np.sqrt(mean_squared_error(y_test, best_predictions)):.4f}"
)
print(f"测试集 R²：{r2_score(y_test, best_predictions):.4f}")

import matplotlib
matplotlib.use("Agg")  # 使用非交互式绘图后端，不需要 Tkinter
import matplotlib.pyplot as plt

# 创建图片保存目录
output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

# 统一使用调优后的 Elastic Net 计算指标
tuned_mae = mean_absolute_error(y_test, best_predictions)
tuned_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
tuned_r2 = r2_score(y_test, best_predictions)

# ==========================================
# 图1：真实电影时长 vs. 预测电影时长
# ==========================================
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.5
)

# 理想预测线：预测值等于真实值
min_value = min(y_test.min(), best_predictions.min())
max_value = max(y_test.max(), best_predictions.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    label="Ideal prediction"
)

plt.xlabel("Actual Duration (minutes)")
plt.ylabel("Predicted Duration (minutes)")
plt.title("Elastic Net: Actual vs Predicted")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "actual_vs_predicted.png",
    dpi=300
)


# ==========================================
# 图2：三种模型的误差对比
# ==========================================
comparison_df = results_df.copy()

# 将原始 Elastic Net 的指标替换为调优后的指标
comparison_df.loc[
    comparison_df["Model"] == "Elastic Net",
    ["MAE", "RMSE", "R2"]
] = [tuned_mae, tuned_rmse, tuned_r2]

x = np.arange(len(comparison_df))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(
    x - width / 2,
    comparison_df["MAE"],
    width,
    label="MAE"
)

ax.bar(
    x + width / 2,
    comparison_df["RMSE"],
    width,
    label="RMSE"
)

ax.set_xticks(x)
ax.set_xticklabels(comparison_df["Model"])
ax.set_ylabel("Error (minutes)")
ax.set_title("Model Performance Comparison")
ax.legend()

plt.tight_layout()
plt.savefig(
    output_dir / "model_comparison.png",
    dpi=300
)


# ==========================================
# 图3：Netflix 电影时长分布
# ==========================================
plt.figure(figsize=(8, 6))

plt.hist(
    movies["duration_minutes"],
    bins=30,
    edgecolor="black"
)

plt.axvline(
    movies["duration_minutes"].mean(),
    linestyle="--",
    label="Mean"
)

plt.axvline(
    movies["duration_minutes"].median(),
    linestyle=":",
    label="Median"
)

plt.xlabel("Movie Duration (minutes)")
plt.ylabel("Number of Movies")
plt.title("Distribution of Netflix Movie Durations")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "duration_distribution.png",
    dpi=300
)


# ==========================================
# 保存最终模型对比结果
# ==========================================
comparison_df.to_csv(
    output_dir / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== 可视化完成 ==========")
print("结果保存位置：", output_dir)
print("\n最终模型对比：")
print(comparison_df.round(4).to_string(index=False))
