from ucimlrepo import fetch_ucirepo

ROUNDING = 2


def main() -> None:
    print("Hello from neural-networks-list-1!")

    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features

    vars_df = heart_disease.variables

    # Filtrujemy tylko cechy (bierzemy pod uwagę tylko te, które są w X)
    features_info = vars_df[vars_df["role"] == "Feature"]

    # Rozdzielamy na podstawie oficjalnego typu w bazie UCI
    cechy_liczbowe = features_info[features_info["type"] == "Integer"]["name"].tolist()

    cechy_kategoryczne = features_info[features_info["type"] == "Categorical"]["name"].tolist()

    print("Liczbowe (Continuous):", cechy_liczbowe)
    print("Kategoryczne/Binarne:", cechy_kategoryczne)

    for feature in cechy_liczbowe:
        mean = round(X[feature].mean(), ROUNDING)
        std = round(X[feature].std(), ROUNDING)
        print(f"Feature {feature} has a mean of {mean} and a standard deviation of {std}")

    y = heart_disease.data.targets
    print(y.value_counts())

    y_binary = (y > 0).astype(int)

    # Sprawdzenie balansu klas
    print("Rozkład klas po transformacji binarnej:")
    print(y_binary.value_counts())

    # Wyświetlenie jako procenty
    print("\nRozkład procentowy:")
    print(y_binary.value_counts(normalize=True) * 100)
