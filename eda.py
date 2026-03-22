
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import config


sns.set_style("white")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("EDA")

train = pd.read_csv(config.TRAIN_PATH, index_col='id')
test = pd.read_csv(config.TEST_PATH, index_col='id')

try:
    original = pd.read_csv(config.ORIGINAL_PATH, index_col='User_ID')
    original = original.rename(columns={'Gender': 'Sex'})
    print(f"Train dataset: {train.shape}")
    print(f"Test dataset: {test.shape}")
    print(f"Original dataset: {original.shape}")
except:
    original = None
    print(f"Train dataset: {train.shape}")
    print(f"Test dataset: {test.shape}")
    print("  Original dataset not found")



print("\n Train Data Info:")
print(train.info())

print("\n Statistical Summary:")
print(train.describe())

print("\n Missing Values:")
print(train.isnull().sum())

print("\nTarget Variable (Calories):")
print(f"  Mean: {train['Calories'].mean():.2f}")
print(f"  Median: {train['Calories'].median():.2f}")
print(f"  Std: {train['Calories'].std():.2f}")
print(f"  Min: {train['Calories'].min():.2f}")
print(f"  Max: {train['Calories'].max():.2f}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))


axes[0].hist(train['Calories'], bins=50, color='#667eea', edgecolor='black', alpha=0.7)
axes[0].set_title('Calories Distribution (Original Scale)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Calories')
axes[0].set_ylabel('Frequency')
axes[0].grid(axis='y', alpha=0.3)


axes[1].hist(np.log1p(train['Calories']), bins=50, color='#764ba2', edgecolor='black', alpha=0.7)
axes[1].set_title('Calories Distribution (Log Scale)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(Calories + 1)')
axes[1].set_ylabel('Frequency')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_target_distribution.png', dpi=300, bbox_inches='tight')
print(" Saved eda_target_distribution.png")
plt.close()


numeric_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    axes[idx].hist(train[feature], bins=30, color='#667eea', edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(axis='y', alpha=0.3)


    mean_val = train[feature].mean()
    median_val = train[feature].median()
    axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
    axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
    axes[idx].legend()

plt.tight_layout()
plt.savefig('eda_feature_distributions.png', dpi=300, bbox_inches='tight')
print(" eda_feature_distributions.png")
plt.close()


print("\n Gender Distribution:")
print(train['Sex'].value_counts())
print("\nPercentage:")
print(train['Sex'].value_counts(normalize=True) * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


train['Sex'].value_counts().plot(kind='bar', ax=axes[0], color=['#667eea', '#764ba2'], edgecolor='black')
axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Male', 'Female'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)


train.boxplot(column='Calories', by='Sex', ax=axes[1], patch_artist=True)
axes[1].set_title('Calories by Gender', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Calories')
axes[1].set_xticklabels(['Male', 'Female'])
plt.suptitle('')

plt.tight_layout()
plt.savefig('eda_gender_analysis.png', dpi=300, bbox_inches='tight')
print(" Saved: eda_gender_analysis.png")
plt.close()




X = train.drop('Calories', axis=1)
y = np.log1p(train['Calories'])
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Calculate mutual information
mi_scores = mutual_info_regression(X, y, random_state=config.RANDOM_STATE)
mi_df = pd.DataFrame({
    'Feature': X.columns,
    'Mutual_Information': mi_scores
}).sort_values('Mutual_Information', ascending=False)

print("\n Mutual Information Scores:")
print(mi_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mi_df)))
bars = ax.barh(mi_df['Feature'], mi_df['Mutual_Information'], color=colors, edgecolor='black')
ax.set_xlabel('Mutual Information Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance (Mutual Information)', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add values on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.3f}',
            ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('eda_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: eda_feature_importance.png")
plt.close()




train_encoded = train.copy()
train_encoded['Sex'] = train_encoded['Sex'].map({'male': 0, 'female': 1})

# Calculate correlation
corr_matrix = train_encoded.corr()
print("\n Correlation with Calories:")
print(corr_matrix['Calories'].sort_values(ascending=False))

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

# Plot
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
    ax=ax
)

ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: eda_correlation_matrix.png")
plt.close()


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    axes[idx].scatter(train_encoded[feature], train_encoded['Calories'],
                      alpha=0.3, s=10, color='#667eea', edgecolors='none')
    axes[idx].set_xlabel(feature, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Calories', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{feature} vs Calories', fontsize=12, fontweight='bold')
    axes[idx].grid(alpha=0.3)

    # Add correlation value
    corr = train_encoded[[feature, 'Calories']].corr().iloc[0, 1]
    axes[idx].text(0.05, 0.95, f'Corr: {corr:.3f}',
                   transform=axes[idx].transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=10, fontweight='bold', verticalalignment='top')

plt.tight_layout()
plt.savefig('eda_scatter_plots.png', dpi=300, bbox_inches='tight')
print(" Saved: eda_scatter_plots.png")
plt.close()

if original is not None:
    print("\n9. TRAIN VS ORIGINAL DATA COMPARISON")
    print("-" * 60)

    # Prepare original data
    original_encoded = original.copy()
    original_encoded['Sex'] = original_encoded['Sex'].map({'male': 0, 'female': 1})

    # Calculate correlations for both
    corr_train = train_encoded.corr()
    corr_orig = original_encoded.corr()

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Train heatmap
    mask_train = np.triu(np.ones_like(corr_train, dtype=bool), k=1)
    sns.heatmap(corr_train, mask=mask_train, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True, ax=axes[0],
                cbar_kws={'shrink': 0.7})
    axes[0].set_title('Train Dataset', fontsize=14, fontweight='bold')

    # Original heatmap
    mask_orig = np.triu(np.ones_like(corr_orig, dtype=bool), k=1)
    sns.heatmap(corr_orig, mask=mask_orig, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True, ax=axes[1],
                cbar_kws={'shrink': 0.7})
    axes[1].set_title('Original Dataset', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('eda_train_vs_original.png', dpi=300, bbox_inches='tight')
    print("Saved: eda_train_vs_original.png")
    plt.close()

# ============================================
# 10. SUMMARY REPORT
# ============================================
print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

print(f"""
📊 Dataset Overview:
   - Training samples: {len(train):,}
   - Test samples: {len(test):,}
   - Original samples: {len(original) if original is not None else 0:,}
   - Total features: {len(X.columns)}
   - Target variable: Calories

🎯 Target Variable (Calories):
   - Range: {train['Calories'].min():.0f} - {train['Calories'].max():.0f}
   - Mean: {train['Calories'].mean():.2f}
   - Median: {train['Calories'].median():.2f}
   - Std Dev: {train['Calories'].std():.2f}

🔍 Key Findings:
   1. Duration has the highest correlation with Calories ({corr_matrix.loc['Duration', 'Calories']:.3f})
   2. Body_Temp and Heart_Rate also show strong relationships
   3. Gender shows minimal direct correlation with Calories
   4. Log transformation helps stabilize predictions
   5. No missing values detected

💡 Feature Importance (Top 3):
   1. {mi_df.iloc[0]['Feature']}: {mi_df.iloc[0]['Mutual_Information']:.4f}
   2. {mi_df.iloc[1]['Feature']}: {mi_df.iloc[1]['Mutual_Information']:.4f}
   3. {mi_df.iloc[2]['Feature']}: {mi_df.iloc[2]['Mutual_Information']:.4f}

✅ Recommendations:
   - Use log transformation for target variable
   - Focus on Duration, Body_Temp, and Heart_Rate features
   - Consider ensemble methods for better predictions
   - Combine train and original datasets for improved generalization
""")

print("\n" + "=" * 60)
print("✅ EDA COMPLETE!")
print("=" * 60)
print("\n📁 Generated Visualizations:")
print("   - eda_target_distribution.png")
print("   - eda_feature_distributions.png")
print("   - eda_gender_analysis.png")
print("   - eda_feature_importance.png")
print("   - eda_correlation_matrix.png")
print("   - eda_scatter_plots.png")
if original is not None:
    print("   - eda_train_vs_original.png")
print("\n🎯 Next Steps:")
print("   1. Review the visualizations")
print("   2. Run: python train.py")
print("   3. Run: streamlit run app.py")