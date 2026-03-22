

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold

import config



start_time = time.time()


train = pd.read_csv(config.TRAIN_PATH, index_col='id')
test = pd.read_csv(config.TEST_PATH, index_col='id')

print(f" Training loaded{train.shape}")
print(f"Test loaded {test.shape}")


try:
    original = pd.read_csv(config.ORIGINAL_PATH, index_col='User_ID')
    original = original.rename(columns={'Gender': 'Sex'})
    print(f" Original dataset loaded: {original.shape}")
    use_original = True
except FileNotFoundError:
    print("⚠  Original dataset not found - training without augmentation")
    original = None
    use_original = False

load_time = time.time() - start_time
print(f"\n⏱️  Data loading time: {load_time:.2f}s")



print("   - Encoding gender: male=0, female=1")
train['Sex'] = train['Sex'].map(config.GENDER_MAPPING)
test['Sex'] = test['Sex'].map(config.GENDER_MAPPING)


X_train = train.drop(config.TARGET, axis=1)
y_train = train[config.TARGET]
X_test = test.copy()


y_train_log = np.log1p(y_train)

print(f"   - Original target range: [{y_train.min():.1f}, {y_train.max():.1f}]")
print(f"   - Log-transformed range: [{y_train_log.min():.3f}, {y_train_log.max():.3f}]")
print("   Transformation stabilizes wide range for better predictions")


if use_original:
    print("\n Combining train and original datasets...")
    original['Sex'] = original['Sex'].map(config.GENDER_MAPPING)
    X_original = original.drop(config.TARGET, axis=1)
    y_original_log = np.log1p(original[config.TARGET])

    # Combine datasets
    X_combined = pd.concat([X_train, X_original], axis=0, ignore_index=True)
    y_combined = pd.concat([y_train_log, y_original_log], axis=0, ignore_index=True)


else:
    X_combined = X_train
    y_combined = y_train_log
    print(f"\n   - Training samples: {len(X_combined):,}")



# Initialize cross-validation
cv = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

# Store models and scores
models = {}
cv_scores = {}


print("1. RANDOM FOREST (Bagging Ensemble)")

for key, value in config.RF_PARAMS.items():
    print(f"   - {key}: {value}")

print(f"\n Training with {config.N_FOLDS}-fold cross-validation...")
start_time = time.time()

rf_model = RandomForestRegressor(**config.RF_PARAMS)
rf_model.fit(X_combined, y_combined)

rf_cv_scores = -cross_val_score(
    rf_model, X_combined, y_combined,
    cv=cv, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

rf_time = time.time() - start_time

print(f"\n📊 Cross-Validation Results:")
for fold, score in enumerate(rf_cv_scores, 1):
    print(f"   Fold {fold}: RMSE = {score:.4f}")

print(f"\n📈 Performance Metrics:")
print(f"   - Mean RMSE: {rf_cv_scores.mean():.4f}")
print(f"   - Std Dev:   {rf_cv_scores.std():.4f}")
print(f"   - Training time: {rf_time:.2f}s")

models['rf'] = rf_model
cv_scores['rf'] = rf_cv_scores

# Save model
joblib.dump(rf_model, config.RF_MODEL)
print(f"\n💾 Model saved: {config.RF_MODEL}")

# ===== GRADIENT BOOSTING =====
print("\n" + "-" * 70)
print("2. GRADIENT BOOSTING (Sequential Ensemble)")
print("-" * 70)
print("\n⚙️  Hyperparameters:")
for key, value in config.GB_PARAMS.items():
    print(f"   - {key}: {value}")

print(f"\n🔄 Training with {config.N_FOLDS}-fold cross-validation...")
start_time = time.time()

gb_model = GradientBoostingRegressor(**config.GB_PARAMS)
gb_model.fit(X_combined, y_combined)

# Cross-validation scores
gb_cv_scores = -cross_val_score(
    gb_model, X_combined, y_combined,
    cv=cv, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

gb_time = time.time() - start_time

print(f"\n📊 Cross-Validation Results:")
for fold, score in enumerate(gb_cv_scores, 1):
    print(f"   Fold {fold}: RMSE = {score:.4f}")

print(f"\n📈 Performance Metrics:")
print(f"   - Mean RMSE: {gb_cv_scores.mean():.4f}")
print(f"   - Std Dev:   {gb_cv_scores.std():.4f}")
print(f"   - Training time: {gb_time:.2f}s")

models['gb'] = gb_model
cv_scores['gb'] = gb_cv_scores

# Save model
joblib.dump(gb_model, config.GB_MODEL)
print(f"\n💾 Model saved: {config.GB_MODEL}")

# ===== RIDGE REGRESSION =====
print("\n" + "-" * 70)
print("3. RIDGE REGRESSION (Linear Baseline)")
print("-" * 70)
print("\n⚙️  Hyperparameters:")
for key, value in config.RIDGE_PARAMS.items():
    print(f"   - {key}: {value}")

print(f"\n🔄 Training with {config.N_FOLDS}-fold cross-validation...")
start_time = time.time()

ridge_model = Ridge(**config.RIDGE_PARAMS)
ridge_model.fit(X_combined, y_combined)

# Cross-validation scores
ridge_cv_scores = -cross_val_score(
    ridge_model, X_combined, y_combined,
    cv=cv, scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

ridge_time = time.time() - start_time

print(f"\n📊 Cross-Validation Results:")
for fold, score in enumerate(ridge_cv_scores, 1):
    print(f"   Fold {fold}: RMSE = {score:.4f}")

print(f"\n📈 Performance Metrics:")
print(f"   - Mean RMSE: {ridge_cv_scores.mean():.4f}")
print(f"   - Std Dev:   {ridge_cv_scores.std():.4f}")
print(f"   - Training time: {ridge_time:.2f}s")

models['ridge'] = ridge_model
cv_scores['ridge'] = ridge_cv_scores

# Save model
joblib.dump(ridge_model, config.RIDGE_MODEL)
print(f"\n💾 Model saved: {config.RIDGE_MODEL}")

# ============================================
# 4. ENSEMBLE PREDICTION
# ============================================
print("\n" + "=" * 70)
print("STEP 4: WEIGHTED AVERAGING ENSEMBLE")
print("=" * 70)

print("\n⚖️  Ensemble Configuration:")
print(f"   - Random Forest weight:      {config.WEIGHTS['rf']:.1%}")
print(f"   - Gradient Boosting weight:  {config.WEIGHTS['gb']:.1%}")
print(f"   - Ridge Regression weight:   {config.WEIGHTS['ridge']:.1%}")
print("\n   ✅ Weights optimized via cross-validation performance")

print("\n🔮 Generating predictions...")

# Make predictions on test set (log scale)
rf_pred_log = models['rf'].predict(X_test)
gb_pred_log = models['gb'].predict(X_test)
ridge_pred_log = models['ridge'].predict(X_test)

# Weighted ensemble (log scale)
ensemble_pred_log = (
        config.WEIGHTS['rf'] * rf_pred_log +
        config.WEIGHTS['gb'] * gb_pred_log +
        config.WEIGHTS['ridge'] * ridge_pred_log
)

# Convert back from log scale to original scale
final_predictions = np.expm1(ensemble_pred_log)

print(f"\n📊 Prediction Statistics:")
print(f"   - Predictions generated: {len(final_predictions):,}")
print(f"   - Min predicted calories: {final_predictions.min():.2f}")
print(f"   - Max predicted calories: {final_predictions.max():.2f}")
print(f"   - Mean predicted calories: {final_predictions.mean():.2f}")

# ============================================
# 5. CREATE SUBMISSION FILE
# ============================================
print("\n" + "=" * 70)
print("STEP 5: SUBMISSION FILE GENERATION")
print("=" * 70)

submission = pd.DataFrame({
    'id': X_test.index,
    'Calories': final_predictions
})

submission.to_csv(config.SUBMISSION_PATH, index=False)

print(f"\n💾 Submission file saved: {config.SUBMISSION_PATH}")
print(f"\n📋 Sample predictions:")
print(submission.head(10).to_string(index=False))

# ============================================
# 6. PERFORMANCE SUMMARY
# ============================================
print("\n" + "=" * 70)
print("TRAINING SUMMARY")
print("=" * 70)

print("\n📊 Individual Model Performance (Cross-Validation):")
print(f"   {'Model':<20} {'Mean RMSE':<12} {'Std Dev':<10} {'Training Time':<15}")
print(f"   {'-' * 20} {'-' * 12} {'-' * 10} {'-' * 15}")
print(f"   {'Random Forest':<20} {cv_scores['rf'].mean():<12.4f} {cv_scores['rf'].std():<10.4f} {rf_time:<15.2f}s")
print(f"   {'Gradient Boosting':<20} {cv_scores['gb'].mean():<12.4f} {cv_scores['gb'].std():<10.4f} {gb_time:<15.2f}s")
print(
    f"   {'Ridge Regression':<20} {cv_scores['ridge'].mean():<12.4f} {cv_scores['ridge'].std():<10.4f} {ridge_time:<15.2f}s")

# Estimate ensemble performance (approximate)
ensemble_estimate = (
        config.WEIGHTS['rf'] * cv_scores['rf'].mean() +
        config.WEIGHTS['gb'] * cv_scores['gb'].mean() +
        config.WEIGHTS['ridge'] * cv_scores['ridge'].mean()
)

print(f"\n🎯 Estimated Ensemble Performance:")
print(f"   - Weighted Average RMSE: {ensemble_estimate:.4f}")
print(f"   - Target RMSE: 0.059 (as per project goal)")

print(f"\n📁 Saved Models:")
print(f"   - {config.RF_MODEL}")
print(f"   - {config.GB_MODEL}")
print(f"   - {config.RIDGE_MODEL}")

print(f"\n📤 Output Files:")
print(f"   - {config.SUBMISSION_PATH}")

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)

