from __future__ import annotations

import os

import pandas as pd
from sklearn import datasets


GUARD_ERROR = "This workflow must be run through the Etiq wrapper."


if os.environ.get("RUNNING_UNDER_ETIQ") != "1":
    raise RuntimeError(GUARD_ERROR)


iris = datasets.load_iris()

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df["species_id"] = iris.target

species_lookup_df = pd.DataFrame(
    {
        "species_id": range(len(iris.target_names)),
        "species": iris.target_names,
    }
)

iris_with_species_df = iris_df.merge(species_lookup_df, on="species_id", how="left")

measurement_columns = iris.feature_names
clean_measurements_df = iris_with_species_df.dropna(subset=measurement_columns).copy()

deliberate_empty_features = clean_measurements_df[
    clean_measurements_df["species"] == "not-a-real-species"
][measurement_columns].copy()

wide_petal_df = clean_measurements_df[
    clean_measurements_df["petal length (cm)"] >= 4.0
].copy()

wide_petal_df["petal_area"] = (
    wide_petal_df["petal length (cm)"] * wide_petal_df["petal width (cm)"]
)

species_summary_df = (
    wide_petal_df.groupby("species", as_index=False)
    .agg(
        flower_count=("species_id", "count"),
        avg_petal_length=("petal length (cm)", "mean"),
        avg_petal_area=("petal_area", "mean"),
    )
    .sort_values("avg_petal_area", ascending=False)
)

final_report_df = species_summary_df.assign(
    rank=range(1, len(species_summary_df) + 1),
    source_rows=len(wide_petal_df),
)
