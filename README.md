### Sync the dependencies

```bash
uv sync --extra dev
```

### Configure and run pre-commit

```bash
uv run prek install
uv run prek run --all-files
```

### Install the dataset

The dataset must be installed from [here](https://archive.ics.uci.edu/dataset/45/heart+disease) and the unzipped folder must be put into the `heart+disease/` folder in the projects's root.

### Run the project

```bash
uv run main
```
