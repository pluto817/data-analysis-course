
import pandas as pd

# 读取 Hitters 数据集
data = pd.read_csv("Hitters.csv")

# 查看前5行数据
print("数据前5行：")
print(data.head())

# 查看数据集的行数和列数
print("\n数据集大小：")
print(data.shape)

# 查看每一列的数据类型和缺失情况
print("\n数据基本信息：")
data.info()

# 查看 Salary 列的缺失情况
print("\nSalary 缺失值数量：")
print(data["Salary"].isna().sum())

# 1. 删除 Salary 缺失的记录
data_clean = data.dropna(subset=["Salary"]).copy()

print("\n删除 Salary 缺失值后：")
print("数据集大小：", data_clean.shape)
print("Salary 缺失值数量：", data_clean["Salary"].isna().sum())

# 2. 查看分类变量的取值
print("\nLeague 的取值：")
print(data_clean["League"].value_counts())

print("\nDivision 的取值：")
print(data_clean["Division"].value_counts())

print("\nNewLeague 的取值：")
print(data_clean["NewLeague"].value_counts())

# 3. 将分类变量转换成数值变量
X = data_clean.drop(columns=["Salary"])
X = pd.get_dummies(X, drop_first=True, dtype=int)

# 4. 提取预测目标 Salary
y = data_clean["Salary"]

print("\n处理后的特征数据大小：", X.shape)
print("目标变量大小：", y.shape)
print("\n处理后的特征数据前5行：")
print(X.head())

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# 1. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n训练集大小：", X_train.shape)
print("测试集大小：", X_test.shape)

# 2. 设置候选参数
alphas = np.logspace(-3, 3, 50)

# 3. 定义三种模型及其参数搜索范围
models = {
    "Ridge": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge())
        ]),
        {"model__alpha": alphas}
    ),
    "Lasso": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(max_iter=100000))
        ]),
        {"model__alpha": alphas}
    ),
    "Elastic Net": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(max_iter=100000))
        ]),
        {
            "model__alpha": alphas,
            "model__l1_ratio": [0.1, 0.5, 0.9]
        }
    )
}

# 4. 交叉验证选择参数，并在测试集上评估
results = []

for name, (pipeline, param_grid) in models.items():
    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=1
    )
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 统计模型的非零系数数量
    coefficients = search.best_estimator_.named_steps["model"].coef_
    nonzero = np.sum(np.abs(coefficients) > 1e-8)

    results.append({
        "Model": name,
        "Best Parameters": search.best_params_,
        "Test RMSE": rmse,
        "Test R²": r2,
        "Non-zero Coefficients": nonzero
    })

# 5. 输出比较结果
results_df = pd.DataFrame(results)

print("\n三种回归模型的比较结果：")
print(results_df.to_string(index=False))
