from ucimlrepo import fetch_ucirepo

def main() -> None:
    print("Hello from neural-networks-list-1!")

    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # metadata
    print(heart_disease.metadata)

    # variable information
    print(heart_disease.variables)
