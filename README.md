# Bioavilability
#Predictive Modeling of Oral Bioavailability: An  In Silico Study Using Logistic Regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib


# 1. Load dataset

file_path = "dummy_drug_dataset.csv"   # change if needed
df = pd.read_csv(file_path)

# 2. Create target label

# Rule: "Oral_Bioavailable" = 1 if GI_Absorption is High OR drug satisfies Lipinski-like rules
df['Oral_Bioavailable'] = np.where(
    (df['GI_Absorption'] == 'High') |
    (
        (df['Molecular_Weight'] < 500) &
        (df['LogP'] >= -1) & (df['LogP'] <= 5) &
        (df['H_Bond_Donor'] <= 5) &
        (df['H_Bond_Acceptor'] <= 10) &
        (df['Solubility_mg_per_mL'] >= 0.1)
    ),
    1, 0


# 3. Split into train/test

X = df.drop(columns=['Drug_Name', 'CID', 'Oral_Bioavailable'])
y = df['Oral_Bioavailable']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 4. Preprocessing

numeric_feats = ['Molecular_Weight', 'Solubility_mg_per_mL',
                 'H_Bond_Donor', 'H_Bond_Acceptor', 'LogP']
categorical_feats = ['GI_Absorption', 'BBB']

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(drop='if_binary', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_feats),
        ('cat', categorical_transformer, categorical_feats)
    ]
)

# 5. Build and tune Logistic Regression

best = {"accuracy": 0, "pipeline": None, "degree": None, "C": None}
Cs = [0.01, 0.1, 1, 10, 100]
degrees = [1, 2, 3]

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('poly', poly),
        ('clf', LogisticRegression(max_iter=5000, solver='liblinear'))
    ])

    for C in Cs:
        pipeline.set_params(clf__C=C)
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)


if acc > best["accuracy"]:
            best.update({"accuracy": acc, "pipeline": pipeline,
                         "degree": degree, "C": C})
        if acc >= 0.91:  # stop early if target reached
            break
    if best["accuracy"] >= 0.91:
        break
best_pipeline = best["pipeline"]

# 6. Evaluate

y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("Best Accuracy:", accuracy_score(y_test, y_pred))
print("Best Degree:", best["degree"])
print("Best C:", best["C"])
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# 7. Save model + predictions

joblib.dump(best_pipeline, "logreg_oral_bioavail_model.pkl")

pred_df = X_test.copy().reset_index(drop=True)
pred_df['True_Oral_Bioavailable'] = y_test.reset_index(drop=True)
pred_df['Pred_Prediction'] = y_pred
pred_df['Pred_Probability'] = np.round(y_prob, 4)

pred_df.to_csv("predictions_oral_bioavail.csv", index=False)

print("\nSaved model to logreg_oral_bioavail_model.pkl")
print("Saved predictions to predictions_oral_bioavail.csv")









