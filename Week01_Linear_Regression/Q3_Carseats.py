"""
第三题：Carseats 多元线性回归

运行前请确保 Carseats.csv 与本文件位于同一文件夹。

任务：
1. 以 Sales 为响应变量；
2. 选取 Price、Income、Advertising 和 ShelveLoc；
3. 建立多元线性回归模型；
4. 指出 ShelveLoc 的基准组；
5. 解读 ShelveLoc[Good] 系数；
6. 计算各解释变量的 VIF，并判断多重共线性风险。
"""

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# 1. 加载 Carseats 数据集
# ============================================================

DATA_FILE = "Carseats.csv"

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("数据基本信息")
print("=" * 70)
print(df.head())
print("\n变量信息：")
print(df.info())


# ============================================================
# 2. 建立多元线性回归模型
# ============================================================
# C(ShelveLoc) 表示 ShelveLoc 是分类变量。
# statsmodels 会自动生成哑变量，并选择一个类别作为基准组。

model = smf.ols(
    "Sales ~ Price + Income + Advertising + C(ShelveLoc)",
    data=df
).fit()

print("\n" + "=" * 70)
print("多元线性回归结果")
print("=" * 70)
print(model.summary())


# ============================================================
# 3. 查看回归系数
# ============================================================

print("\n" + "=" * 70)
print("回归系数")
print("=" * 70)
print(model.params)


# ============================================================
# 4. 判断 ShelveLoc 的基准组
# ============================================================
# 获取 ShelveLoc 的所有类别。
# drop_first=True 后，未出现在哑变量中的类别就是基准组。

shelve_categories = list(df["ShelveLoc"].dropna().unique())

print("\n" + "=" * 70)
print("ShelveLoc 类别")
print("=" * 70)
print(shelve_categories)

# statsmodels 默认使用分类变量的第一个排序类别作为基准组。
# 为了直接确认，可以查看模型的设计矩阵列名。
shelve_columns = [
    col for col in model.model.exog_names
    if "C(ShelveLoc)" in col
]

print("\n模型中的 ShelveLoc 哑变量：")
for col in shelve_columns:
    print(col)

print(
    "\n基准组需要根据上述哑变量名称判断："
    "没有对应系数的 ShelveLoc 类别即为基准组。"
)


# ============================================================
# 5. 解读 ShelveLoc[Good] 系数
# ============================================================

good_term = "C(ShelveLoc)[T.Good]"

print("\n" + "=" * 70)
print("ShelveLoc[Good] 系数")
print("=" * 70)

if good_term in model.params.index:
    good_coef = model.params[good_term]
    print(f"{good_term} = {good_coef:.4f}")

    print(
        "\n解释：在 Price、Income、Advertising 等其他变量保持不变的情况下，"
        "与 ShelveLoc 的基准组相比，Good 货架位置对应的预测 Sales "
        f"平均变化 {good_coef:.4f} 个单位。"
    )
else:
    print("模型中没有找到 C(ShelveLoc)[T.Good]，请检查数据中的类别名称。")


# ============================================================
# 6. 计算 VIF
# ============================================================
# VIF 用来判断解释变量之间是否存在多重共线性。
# 对分类变量先进行哑变量编码。
#
# drop_first=True 可以避免完整哑变量陷阱。
# 注意：这里不把常数项作为重点解释对象。

X = pd.get_dummies(
    df[["Price", "Income", "Advertising", "ShelveLoc"]],
    columns=["ShelveLoc"],
    drop_first=True
)

X = X.astype(float)
X = sm.add_constant(X)

vif = pd.DataFrame()
vif["Variable"] = X.columns
vif["VIF"] = [
    variance_inflation_factor(X.values, i)
    for i in range(X.shape[1])
]

print("\n" + "=" * 70)
print("VIF 结果")
print("=" * 70)
print(vif)


# ============================================================
# 7. 根据常见经验阈值进行提示
# ============================================================

print("\n" + "=" * 70)
print("VIF 简单判断")
print("=" * 70)

for _, row in vif.iterrows():
    variable = row["Variable"]
    value = row["VIF"]

    if variable == "const":
        continue

    if value < 5:
        risk = "通常认为没有明显的多重共线性风险"
    elif value < 10:
        risk = "存在一定的多重共线性，需要关注"
    else:
        risk = "存在较明显的多重共线性风险"

    print(f"{variable}: VIF = {value:.4f} → {risk}")


# ============================================================
# 8. 最终提示
# ============================================================

print("\n" + "=" * 70)
print("作业检查要点")
print("=" * 70)
print("1. 查看回归 summary 中的 R-squared、Adj. R-squared 和系数。")
print("2. 找到 ShelveLoc 的基准组。")
print("3. 记录 C(ShelveLoc)[T.Good] 的系数并进行商业含义解释。")
print("4. 查看 VIF，并根据结果评价多重共线性风险。")
