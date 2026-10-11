from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
from pathlib import Path

ROUNDING = 2


def main() -> None:
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features

    vars_df = heart_disease.variables

    features = vars_df[vars_df["role"] == "Feature"]
    numeral_features = features[features["type"] == "Integer"]["name"].tolist()
    categorical_features = features[features["type"] == "Categorical"]["name"].tolist()

    print(f"Numeral features: {numeral_features}")
    print(f"Categorical features: {categorical_features}")

    for feature in numeral_features:
        mean = round(X[feature].mean(), ROUNDING)
        std = round(X[feature].std(), ROUNDING)
        print(f"Feature '{feature}' has a mean of {mean} and a standard deviation of {std}")

    y = heart_disease.data.targets
    print("\nClass distribution:")
    print(y.value_counts())

    print("\nDistribution percentages:")
    print(y.value_counts(normalize=True) * 100)


    y_binary = (y > 0).astype(int)
    print("\nClass distribution after inary normalization:")
    print(y_binary.value_counts())

    print("\nDistribution percentages:")
    print(y_binary.value_counts(normalize=True) * 100)


    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    for feature in numeral_features:
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
