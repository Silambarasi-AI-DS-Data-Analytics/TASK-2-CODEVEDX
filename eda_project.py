import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- STEP 1: Load Dataset ----
df = pd.read_csv("titanic.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# ---- STEP 2: Missing Values ----
print("\nMissing Values:")
print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)
print("Missing values handled!")

# ---- CHART 1: Age Distribution ----
plt.figure(figsize=(8,5))
df['Age'].hist(bins=30, color='steelblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()
print("Age Distribution saved!")

# ---- CHART 2: Age Box Plot ----
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age'], color='lightgreen')
plt.title('Age Box Plot')
plt.savefig('age_box_plot.png')
plt.close()
print("Age Box Plot saved!")

# ---- CHART 3: Fare Distribution ----
plt.figure(figsize=(8,5))
df['Fare'].hist(bins=30, color='orange', edgecolor='black')
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.savefig('fare_distribution.png')
plt.close()
print("Fare Distribution saved!")

# ---- CHART 4: Male vs Female ----
plt.figure(figsize=(8,5))
sns.countplot(x='Sex', data=df, palette='Set1')
plt.title('Male vs Female Count')
plt.savefig('male_vs_female.png')
plt.close()
print("Male vs Female saved!")

# ---- CHART 5: Survival by Gender ----
plt.figure(figsize=(8,5))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Gender')
plt.savefig('survival_by_gender.png')
plt.close()
print("Survival by Gender saved!")

# ---- CHART 6: Survival Distribution ----
plt.figure(figsize=(8,5))
sns.countplot(x='Survived', data=df, palette='coolwarm')
plt.title('Survival Distribution')
plt.xticks([0,1], ['Not Survived', 'Survived'])
plt.savefig('survival_distribution.png')
plt.close()
print("Survival Distribution saved!")

# ---- CHART 7: Correlation Heatmap ----
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()
print("Correlation Heatmap saved!")

print("\nEDA Complete! All 7 charts saved in your folder!")