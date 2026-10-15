import pandas as pd
import numpy as np

df = pd.read_csv("Salary_Data.csv")

print(df.shape)
print(df.head())
df.info()
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

df["Education Level"] = df["Education Level"].replace({
    "Bachelor's Degree": "Bachelor's",
    "Master's Degree": "Master's",
    "phD": "PhD"
})

df = df[df["Gender"] != "Other"]
df = df[df["Salary"] >= 10000]

q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)
iqr = q3 - q1
df = df[(df["Salary"] >= q1 - 1.5 * iqr) & (df["Salary"] <= q3 + 1.5 * iqr)]

df["Education Code"] = df["Education Level"].map({"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3})
df["Gender Code"] = df["Gender"].map({"Male": 0, "Female": 1})

print(df.shape)
print(df.describe())
print(df.groupby("Education Level")["Salary"].mean())
print(df[["Age", "Years of Experience", "Education Code", "Gender Code", "Salary"]].corr()["Salary"])

df.to_csv("cleaned_salary_data.csv", index=False)
