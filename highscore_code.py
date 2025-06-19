original_copy = original.copy()
for k in range(6):
    original = pd.concat([original,original_copy],axis=0)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

# 1. Alle verfügbaren Trainingsdaten kombinieren
def prepare_combined_data(original, train, test):
    """
    Kombiniert original, train und test Daten für maximale Trainingsdatenmenge
    """
    # Original Daten vorbereiten (haben bereits Labels)
    original_prepared = original.copy()

    # Falls original andere Spaltennamen hat, anpassen
    if 'Fertilizer Name' not in original_prepared.columns:
        # Prüfe nach ähnlichen Spaltenamen
        fert_cols = [col for col in original_prepared.columns if 'fertilizer' in col.lower() or 'fert' in col.lower()]
        if fert_cols:
            original_prepared = original_prepared.rename(columns={fert_cols[0]: 'Fertilizer Name'})

    # Train Daten sind bereits korrekt formatiert
    train_prepared = train.copy()

    # Alle Daten mit Labels kombinieren (original + train)
    # Stelle sicher, dass beide DataFrames gleiche Spalten haben
    common_cols = list(set(original_prepared.columns) & set(train_prepared.columns))

    if 'id' not in original_prepared.columns:
        # Füge IDs für original hinzu, die nicht mit train/test kollidieren
        original_prepared['id'] = range(len(original_prepared))

    # Kombiniere alle Trainingsdaten
    combined_train = pd.concat([
        original_prepared[common_cols],
        train_prepared[common_cols]
    ], axis=0, ignore_index=True)

    # Duplikate entfernen (falls vorhanden)
    combined_train = combined_train.drop_duplicates()

    return combined_train, test

# Daten kombinieren
combined_train_data, test_data = prepare_combined_data(original, train, test)
print(f"Combined training data shape: {combined_train_data.shape}")

# 2. Feature Engineering
def add_enhanced_features(df):
    """Erweiterte Feature Engineering Funktion"""
    df = df.copy()

    # NPK Features
    df['NPK_total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1e-8)
    df['N_to_K'] = df['Nitrogen'] / (df['Potassium'] + 1e-8)
    df['P_to_K'] = df['Phosphorous'] / (df['Potassium'] + 1e-8)

    # Umwelt Features
    df['Temp_Humidity'] = df['Temparature'] * df['Humidity'] / 100
    df['Moisture_Temp'] = df['Moisture'] * df['Temparature']

    # Kategorische Interaktionen
    if 'Soil Type' in df.columns and 'Crop Type' in df.columns:
        df['Soil_Crop'] = df['Soil Type'].astype(str) + '_' + df['Crop Type'].astype(str)

    return df

# Features zu allen Datensätzen hinzufügen
combined_train_data = add_enhanced_features(combined_train_data)
test_data = add_enhanced_features(test_data)

# 3. Label Encoding
# Zuerst Target Variable enkodieren
le_target = LabelEncoder()
combined_train_data['Fertilizer_enc'] = le_target.fit_transform(combined_train_data['Fertilizer Name'])

# Kategorische Features
categorical_features = ['Soil Type', 'Crop Type']
if 'Soil_Crop' in combined_train_data.columns:
    categorical_features.append('Soil_Crop')

# Label Encoder für kategorische Features
label_encoders = {}
for col in categorical_features:
    if col in combined_train_data.columns:
        le = LabelEncoder()

        # Alle unique Werte aus train und test sammeln
        all_values = list(combined_train_data[col].unique())
        if col in test_data.columns:
            all_values.extend(list(test_data[col].unique()))
        all_values = list(set(all_values))

        # Fit auf alle Werte
        le.fit(all_values)

        # Transform
        combined_train_data[col] = le.transform(combined_train_data[col])
        if col in test_data.columns:
            test_data[col] = le.transform(test_data[col])

        label_encoders[col] = le

# 4. Feature Auswahl
numerical_features = [
    'Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous',
    'NPK_total', 'N_to_P', 'N_to_K', 'P_to_K', 'Temp_Humidity', 'Moisture_Temp'
]

# Prüfe welche Features tatsächlich vorhanden sind
available_categorical = [col for col in categorical_features if col in combined_train_data.columns]
available_numerical = [col for col in numerical_features if col in combined_train_data.columns]

all_features = available_categorical + available_numerical
print(f"Using features: {all_features}")

# 5. Daten vorbereiten
X_combined = combined_train_data[all_features]
y_combined = combined_train_data['Fertilizer_enc']

print(f"Final training data shape: {X_combined.shape}")
print(f"Number of unique classes: {len(np.unique(y_combined))}")

# 6. Train-Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X_combined, y_combined,
    stratify=y_combined,
    test_size=0.15,  # Kleinerer Validation Set da wir mehr Daten haben
    random_state=42
)

# 7. Erweiterte Hyperparameter Suche
param_grid = {
    'n_estimators': [800],
    'max_depth': [12],
    'learning_rate': [0.078, 0.0107],
    'subsample': [0.8],
    'colsample_bytree': [0.5],
    'reg_alpha': [0.3],
    'reg_lambda': [0.03]
}

xgb = XGBClassifier(
    tree_method='hist',
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1
)

# Stratified K-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Starting GridSearch...")
grid = GridSearchCV(
    xgb, param_grid,
    cv=cv,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)
print("Best CV-Score:", grid.best_score_)

# 8. Validation Score
val_score = grid.score(X_val, y_val)
print(f"Validation Score auf Holdout: {val_score}")

# 9. Finales Modell auf allen Daten trainieren
print("Training final model on all data...")
best_xgb = grid.best_estimator_
best_xgb.fit(X_combined, y_combined)

# 10. Feature Importance
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': best_xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
print(feature_importance.head(10))

# 11. Predictions für Test Set
print("Making predictions...")
X_test = test_data[all_features]
probas = best_xgb.predict_proba(X_test)

# Top-3 Predictions
top3_indices = np.argsort(probas, axis=1)[:, -3:][:, ::-1]

# Mapping zurück zu Labels
unique_labels = combined_train_data[['Fertilizer_enc', 'Fertilizer Name']].drop_duplicates()
unique_labels = unique_labels.sort_values('Fertilizer_enc')
label_names = unique_labels['Fertilizer Name'].values

# Top-3 Namen erstellen
top3_names = []
for row in top3_indices:
    names = [label_names[i] for i in row]
    top3_names.append(" ".join(names))

# 12. Submission erstellen
submission = test_data[['id']].copy()
submission['Fertilizer Name'] = top3_names

# Speichern
submission.to_csv("enhanced_submission.csv", index=False)
print("Submission saved as 'enhanced_submission.csv'")
print("\nFirst 5 predictions:")
print(submission.head())

# 13. Modell Performance Summary
print(f"\n=== Model Performance Summary ===")
print(f"Training Data Size: {len(X_combined):,} samples")
print(f"Number of Features: {len(all_features)}")
print(f"Number of Classes: {len(np.unique(y_combined))}")
print(f"Best CV Score: {grid.best_score_:.4f}")
print(f"Validation Score: {val_score:.4f}")
print(f"Best Parameters: {grid.best_params_}")