import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import json
import os


DATA = 'laptop_data2.csv'

# ── Load & clean ──
df = pd.read_csv(DATA)
df = df.dropna()
df = df[df['generation'] != 0]  # drop unknown generation
df = df[df['price'] < 300000]   # remove extreme outliers

print(f"Training on {len(df)} records")
print(f"Price range: ₹{df['price'].min():,} – ₹{df['price'].max():,}")

# ── Encode categoricals ──
cat_cols = ['brand', 'storage_type', 'os', 'processor_brand', 'processor_series']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ── Features & target ──
FEATURES = ['brand', 'ram_gb', 'storage_gb', 'storage_type',
            'screen_size', 'os', 'processor_brand', 'processor_series', 'generation']

X = df[FEATURES]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train multiple models ──
models = {
    "Random Forest":       RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    "Linear Regression":   LinearRegression()
}

results = {}
best_model = None
best_r2 = -999

print("\n── Model Comparison ──")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    cv   = cross_val_score(model, X, y, cv=5, scoring='r2').mean()

    results[name] = {
        "mae":  round(mae, 2),
        "rmse": round(rmse, 2),
        "r2":   round(r2, 4),
        "cv_r2": round(cv, 4)
    }

    print(f"\n{name}")
    print(f"  MAE:   ₹{mae:,.0f}")
    print(f"  RMSE:  ₹{rmse:,.0f}")
    print(f"  R²:    {r2:.4f}")
    print(f"  CV R²: {cv:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\n✓ Best model: {best_name} (R² = {best_r2:.4f})")

# ── Feature importance ──
if hasattr(best_model, 'feature_importances_'):
    fi = dict(zip(FEATURES, best_model.feature_importances_.round(4).tolist()))
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))
else:
    fi = {}

# ── Save model & encoders ──
joblib.dump(best_model, 'model.pkl')
joblib.dump(encoders,   'encoders.pkl')

# ── Save metadata ──
meta = {
    "best_model":        best_name,
    "r2":                round(best_r2, 4),
    "model_results":     results,
    "feature_importance": fi,
    "features":          FEATURES,
    "cat_cols":          cat_cols,
    "options": {
        "brand":            encoders['brand'].classes_.tolist(),
        "ram_gb":           sorted(pd.read_csv(DATA)['ram_gb'].dropna().unique().tolist()),
        "storage_gb":       sorted(pd.read_csv(DATA)['storage_gb'].dropna().unique().tolist()),
        "storage_type":     encoders['storage_type'].classes_.tolist(),
        "os":               encoders['os'].classes_.tolist(),
        "processor_brand":  encoders['processor_brand'].classes_.tolist(),
        "processor_series": encoders['processor_series'].classes_.tolist(),
        "generation":       sorted(pd.read_csv(DATA)['generation'].dropna().unique().tolist()),
        "screen_size":      sorted(pd.read_csv(DATA)['screen_size'].dropna().unique().tolist()),
        "brand":            encoders['brand'].classes_.tolist(),
    },
    "price_stats": {
        "min":  int(pd.read_csv(DATA)['price'].min()),
        "max":  int(pd.read_csv(DATA)['price'].max()),
        "mean": int(pd.read_csv(DATA)['price'].mean()),
    },
    "total_records": len(df)
}

with open('meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("\n✓ model.pkl saved")
print("✓ encoders.pkl saved")
print("✓ meta.json saved")
print("\nTraining complete!")
