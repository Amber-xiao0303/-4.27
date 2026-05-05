# -4.27
import pandas as pd
import numpy as np
#M1
# 缺失率 + 异常值统计
df = pd.read_parquet(r"C:\Users\肖乐儿\Desktop\data.parquet")
missing_report = pd.DataFrame({
    "缺失数量": df.isnull().sum(),
    "缺失率(%)": round(df.isnull().sum() / len(df) * 100, 2)
}).sort_values(by="缺失率(%)", ascending=False)

print("\n 缺失值统计：")
print(missing_report)

# 2.2 异常值报告（基于四分位法 IQR）
for col in df.select_dtypes(include=[np.number]).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()
    print(f"{col} | 异常值数量：{outliers}")

# 3. 清洗数据（每一步都写理由！）
# 清洗策略 1：删除缺失率过高的列（>50% 无意义）
# 理由：缺失超过一半，无法有效填充，保留会干扰模型/分析
drop_cols = missing_report[missing_report["缺失率(%)"] > 50].index.tolist()
if len(drop_cols) > 0:
    df = df.drop(columns=drop_cols)
    print(f"删除缺失率过高的列：{drop_cols}")
# 清洗策略 2：数值列用中位数填充（稳健，不受异常值影响）
# 理由：中位数比均值更抗异常值，适合交通/行程类数据
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
# 清洗策略 3：删除完全重复行
# 理由：重复数据会导致统计偏差，无业务价值
before = len(df)
df = df.drop_duplicates()
print(f"删除重复行：{before - len(df)} 行")
# 清洗策略 4：去除极端异常值（保留合理范围）
# 理由：行程数据不可能为负，也不可能过大（明显错误记录）
if "trip_duration" in df.columns:
    df = df[df["trip_duration"] > 0]
    df = df[df["trip_duration"] < df["trip_duration"].quantile(0.99)]
print("数据清洗完成")
# 4. 时间特征提取
# 把毫秒时间戳转为标准日期（你最重要的一步！）
df["pickup_time"] = pd.to_datetime(df["tpep_pickup_datetime"], unit="ms")
df["dropoff_time"] = pd.to_datetime(df["tpep_dropoff_datetime"], unit="ms")

# 5. 提取时间特征
# 提取小时
df["pick_hour"] = df["pickup_time"].dt.hour
# 提取星期（0=周一，6=周日）
df["weekday"] = df["pickup_time"].dt.weekday
# 是否早晚高峰（7-9，17-19）
df["is_peak"] = (df["pick_hour"].apply
    (lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0))
# 自创特征 1：行程时长分段（短/中/长）
# 理由：不同时长的行程行为差异大，适合分析/建模
if "trip_duration" in df.columns:
    bins = [0, 10*60, 30*60, np.inf]
    labels = ["短途", "中途", "长途"]
    df["trip_type"] = pd.cut(df["trip_duration"], bins=bins, labels=labels)
# 自创特征 2：是否周末
# 理由：周末与工作日行程规律完全不同，是强特征
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >= 5 else 0)
# 自创特征 3：时段分类（凌晨/上午/下午/晚上）
# 理由：比单纯小时更有业务解释力
def get_period(h):
    if 5 <= h < 12:
        return "上午"
    elif 12 <= h < 17:
        return "下午"
    elif 17 <= h < 22:
        return "晚上"
    else:
        return "凌晨"

df["time_period"] = df["pick_hour"].apply(get_period)

print("所有特征生成完成！")

# 6. 输出最终结果
print("最终处理后数据预览")
print(df.head())

#M2
import matplotlib.pyplot as plt
import os

df["pick_hour"] = df["pickup_time"].dt.hour
df["weekday"] = df["pickup_time"].dt.weekday
df["is_weekend"] = df["weekday"].apply(lambda x: 1 if x >=5 else 0)
df["trip_duration_sec"] = (df["dropoff_time"] - df["pickup_time"]).dt.total_seconds()
df["is_peak"] = df["pick_hour"].apply(lambda h: 1 if (7<=h<=9) or (17<=h<=19) else 0)

#桌面 outputs 文件夹
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_folder = os.path.join(desktop_path, "outputs")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

#分析1：小时出行需求
plt.figure(figsize=(12,5))
hour_count = df.groupby("pick_hour").size()
plt.plot(hour_count.index, hour_count.values, marker='o', color="#1f77b4")
plt.title("每小时出行需求分布", fontsize=14)
plt.xlabel("小时")
plt.ylabel("订单数量")
plt.xticks(range(0,24))
plt.grid(alpha=0.3)
plt.savefig(os.path.join(output_folder, "1_出行需求时间规律.png"), dpi=300, bbox_inches="tight")
plt.close()

#分析2：区域热度TOP10
if "PULocationID" in df.columns:
    top_loc = df["PULocationID"].value_counts().head(10)
    plt.figure(figsize=(10,5))
    top_loc.plot(kind="bar", color="#ff7f0e")
    plt.title("上车热度 TOP10 区域")
    plt.xlabel("区域ID")
    plt.ylabel("订单量")
    plt.xticks(rotation=0)
    plt.savefig(os.path.join(output_folder, "2_区域热度TOP10.png"), dpi=300, bbox_inches="tight")
    plt.close()

#分析3：车费与距离关系
if "trip_distance" in df.columns and "fare_amount" in df.columns:
    plt.figure(figsize=(10,6))
    plt.scatter(df["trip_distance"], df["fare_amount"], alpha=0.1, color="#2ca02c")
    plt.title("行程距离 vs 车费")
    plt.xlabel("行程距离")
    plt.ylabel("车费")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_folder, "3_车费影响因素.png"), dpi=300, bbox_inches="tight")
    plt.close()

# 分析4：不同时段 平均车费对比
if "fare_amount" in df.columns:
    period_fare = df.groupby("time_period")["fare_amount"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    period_fare.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    plt.title("不同时段平均车费对比", fontsize=14)
    plt.xlabel("时段")
    plt.ylabel("平均车费（元）")
    plt.xticks(rotation=0)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_folder, "4_时段平均车费对比.png"), dpi=300, bbox_inches="tight")
    plt.close()

print("图表已保存到：")
print(output_folder)
