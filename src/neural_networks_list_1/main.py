from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

ROUNDING = 2


def main() -> None:
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features

    vars_df = heart_disease.variables

    features = vars_df[vars_df["role"] == "Feature"]
    numeric_features = features[features["type"] == "Integer"]["name"].tolist()
    categorical_features = features[features["type"] == "Categorical"]["name"].tolist()

    print(f"Numeral features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")

    for feature in numeric_features:
        mean = round(X[feature].mean(), ROUNDING)
        std = round(X[feature].std(), ROUNDING)
        print(f"Feature '{feature}' has a mean of {mean} and a standard deviation of {std}")

    y = heart_disease.data.targets
    print("\nClass distribution:")
    print(y.value_counts())

    print("\nDistribution percentages:")
    print(y.value_counts(normalize=True) * 100)

    y_binary = (y > 0).astype(int)
    print("\nClass distribution after binary normalization:")
    print(y_binary.value_counts())

    print("\nDistribution percentages:")
    print(y_binary.value_counts(normalize=True) * 100)

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    for feature in numeric_features:
        plt.figure(figsize=(8, 5))

        X[feature].dropna().hist(bins=20)

        plt.title(f"Rozkład cechy: {feature}")
        plt.xlabel(feature)
        plt.ylabel("Liczba obserwacji")
        plt.tight_layout()

        plt.savefig(plots_dir / f"{feature}.png")
        plt.close()

    print("\nMissing values in features:")
    print(X.isna().sum())

    print("\nMissing values in target:")
    print(y.isna().sum())

    print("\nCategorical feature distributions:")
    for feature in categorical_features:
        print(f"\nFeature '{feature}':")
        print(X[feature].value_counts(dropna=False))

        print("\nPercentages:")
        print((X[feature].value_counts(normalize=True, dropna=False) * 100).round(ROUNDING))

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    final_y = heart_disease.data.targets.iloc[:, 0]
    final_y = (final_y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        final_y,
        test_size=0.2,
        random_state=42,
        stratify=final_y,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nBaseline results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.3f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
