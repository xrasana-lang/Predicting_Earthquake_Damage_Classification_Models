"""
duckdb_wrangle.py

Data wrangling module for the Nepal earthquake damage prediction project.
Provides `wrangle_nepal_data()` for loading and preparing the data.

Usage:
    from src.duckdb_wrangle import wrangle_nepal_data

    # All districts
    df = wrangle_nepal_data("./data")

    # Specific district (Gorkha = 4)
    df_gorkha = wrangle_nepal_data("./data", district_id=4)
"""

import duckdb
import pandas as pd


def wrangle_nepal_data(csv_path: str = "./data", district_id: int | None = None) -> pd.DataFrame:
    """
    Retrieve and prepare Nepal earthquake data for binary classification.

    Reads building_structure.csv, building_damage.csv, and id_map.csv
    using DuckDB. Joins on building_id, filters by district (optional),
    creates binary target `severe_damage` (Grade 4-5 = 1), and drops
    damage_grade to prevent target leakage.

    Parameters
    ----------
    csv_path : str
        Path to the data folder (default: "./data")
    district_id : int, optional
        Filter by district (e.g., 4 for Gorkha). If None, returns all districts.

    Returns
    -------
    pd.DataFrame
        Indexed by building_id, with 6 pre-earthquake features,
        district_id, and binary target `severe_damage`.
    """
    structure_path = f"{csv_path}/building_structure.csv"
    damage_path = f"{csv_path}/building_damage.csv"
    id_map_path = f"{csv_path}/id_map.csv"

    where_clause = "WHERE b.damage_grade IS NOT NULL"
    if district_id is not None:
        where_clause += f" AND i.district_id = {district_id}"

    query = f"""
        SELECT DISTINCT
            s.building_id,
            i.district_id,
            s.age_building,
            s.plinth_area_sq_ft,
            s.height_ft_pre_eq,
            s.foundation_type,
            s.ground_floor_type,
            b.damage_grade
        FROM read_csv_auto('{structure_path}') s
        JOIN read_csv_auto('{damage_path}') b ON s.building_id = b.building_id
        JOIN read_csv_auto('{id_map_path}') i ON s.building_id = i.building_id
        {where_clause}
    """

    return (
        duckdb.sql(query)
        .df()
        .set_index("building_id")
        .assign(
            severe_damage=lambda x: (
                x["damage_grade"]
                .str.contains("Grade 4|Grade 5")
                .fillna(False)
                .astype(int)
            )
        )
        .drop(columns=["damage_grade"])
    )
