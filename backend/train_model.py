from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Base.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

if not DATA.exists():
    raise FileNotFoundError(
        f"{DATA} not found. Put the dataset's Base.csv at backend/../data/Base.csv."
    )

df = pd.read_csv(DATA)

# Matches Phase-1 notebook cleaning.
df_clean = df.drop(columns=["device_fraud_count", "income_bracket", "no_prev_address"])

low_missing_cols = [
    "current_address_months_count",
    "session_length_in_minutes",
    "credit_risk_score",
    "device_distinct_emails_8w",
]
medians = {}
for col in low_missing_cols:
    median_val = df_clean.loc[df_clean[col] != -1, col].median()
    medians[col] = float(median_val)
    df_clean[col] = df_clean[col].replace(-1, median_val)

categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=["fraud_bool"])
y = df_encoded["fraud_bool"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Matches the final RF + SMOTE path in the notebook.
X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(
    X_train_scaled, y_train
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_smote, y_train_smote)

joblib.dump(model, MODEL_DIR / "rf_smote.joblib")
joblib.dump(scaler, MODEL_DIR / "standard_scaler.joblib")
(MODEL_DIR / "feature_columns.json").write_text(
    json.dumps(list(X_train.columns), indent=2)
)
(MODEL_DIR / "training_medians.json").write_text(
    json.dumps(medians, indent=2)
)

print("MODEL READY")
print("Encoded features:", len(X_train.columns))
print("Saved:", MODEL_DIR)
