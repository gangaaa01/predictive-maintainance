import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("ai4i2020.csv")

# Basic information
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nMachine failure counts:")
print(df["Machine failure"].value_counts())

print("\nFirst 5 rows:")
print(df.head())

print("\nStatistical summary:")
print(df.describe())


# ==========================================
# GRAPH 1: MACHINE FAILURE DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))
sns.countplot(x="Machine failure", data=df)
plt.title("Machine Failure Distribution")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Number of Machines")
plt.tight_layout()
plt.savefig("machine_failure_distribution.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 2: TORQUE VS MACHINE FAILURE
# ==========================================

plt.figure(figsize=(7, 5))
sns.boxplot(x="Machine failure", y="Torque [Nm]", data=df)
plt.title("Torque vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Torque [Nm]")
plt.tight_layout()
plt.savefig("torque_vs_failure.png", dpi=300)
plt.close()


# ==========================================
# GRAPH 3: TOOL WEAR VS MACHINE FAILURE
# ==========================================

plt.figure(figsize=(7, 5))
sns.boxplot(x="Machine failure", y="Tool wear [min]", data=df)
plt.title("Tool Wear vs Machine Failure")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)")
plt.ylabel("Tool Wear [min]")
plt.tight_layout()
plt.savefig("tool_wear_vs_failure.png", dpi=300)
plt.close()


print("\nAll EDA graphs have been saved successfully!")