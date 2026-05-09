"""Clean and feature-engineer the raw RTA dataset.

Input:  data/RTA Dataset.csv
Output: data/data_cleaned.csv

Run from project root:
    uv run python clean-feature-engineer.py
"""

import pandas as pd

INPUT_PATH = "data/RTA Dataset.csv"
OUTPUT_PATH = "data/data_cleaned.csv"

# Columns dropped unconditionally before modeling.
# Defect_of_vehicle: 36% NaN + nonsensical integer codes (5, 7).
# Casualty_severity: direct target leakage.
# Casualty group: post-hoc records with zero predictive signal for Accident_severity
#   (severity distribution is identical across all Casualty_class values).
DROP_COLUMNS = [
    "Defect_of_vehicle",
    "Casualty_severity",
    "Casualty_class",
    "Sex_of_casualty",
    "Age_band_of_casualty",
    "Work_of_casuality",
    "Fitness_of_casuality",
    "Pedestrian_movement",
]

# Ordinal mappings. Unknown / unrecognised values map to -1 (sentinel for
# "not on the scale") so tree models and TabNet can treat them separately.
AGE_BAND_ORDINAL = {"Under 18": 0, "18-30": 1, "31-50": 2, "Over 51": 3}
DRIVING_EXP_ORDINAL = {
    "No Licence": 0, "Below 1yr": 1, "1-2yr": 2,
    "2-5yr": 3, "5-10yr": 4, "Above 10yr": 5,
}
SERVICE_YEAR_ORDINAL = {
    "Below 1yr": 0, "1-2yr": 1, "2-5yrs": 2, "5-10yrs": 3, "Above 10yr": 4,
}
EDU_ORDINAL = {
    "Illiterate": 0, "Writing & reading": 1, "Elementary school": 2,
    "Junior high school": 3, "High school": 4, "Above high school": 5,
}
SEVERITY_CODE = {"Slight Injury": 0, "Serious Injury": 1, "Fatal injury": 2}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw RTA DataFrame.

    Pure function — does not mutate the input.
    """
    out = df.copy()

    # 1. Drop noise, leakage, and zero-signal columns.
    out = out.drop(columns=[c for c in DROP_COLUMNS if c in out.columns])

    # 2. Strip whitespace on every string column.
    #    Collapses ' Church areas' / 'Church areas' and similar duplicates.
    str_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        out[c] = out[c].astype("string").str.strip()

    # 3. Pedestrian_movement corruption: the substring 'na' was globally replaced
    #    with 'Not a Pedestrian', mangling 'stationary' -> 'statioNot a Pedestrianry'.
    #    Column is dropped in step 1 now, but kept here as a guard if it reappears.
    if "Pedestrian_movement" in out.columns:
        out["Pedestrian_movement"] = out["Pedestrian_movement"].str.replace(
            "statioNot a Pedestrianry", "stationary", regex=False
        )

    # 4. Type_of_vehicle encoding fix: '?' is an artifact for '-'.
    if "Type_of_vehicle" in out.columns:
        out["Type_of_vehicle"] = out["Type_of_vehicle"].str.replace("?", "-", regex=False)

    # 5. Fitness_of_casuality typo (column is dropped, kept as guard).
    if "Fitness_of_casuality" in out.columns:
        out["Fitness_of_casuality"] = out["Fitness_of_casuality"].replace(
            {"NormalNormal": "Normal"}
        )

    # 6. Area_accident_occured concatenation glitch (20 rows).
    if "Area_accident_occured" in out.columns:
        out["Area_accident_occured"] = out["Area_accident_occured"].replace(
            {"Rural village areasOffice areas": "Other"}
        )

    # 7. Normalize unknown/missing markers.
    #    'na'  : 4 443 rows across casualty cols (post-hoc, dropped above).
    #    'other': lowercase variant in Lanes_or_Medians.
    for c in str_cols:
        if c in out.columns:
            out[c] = out[c].replace({
                "na":      "Unknown",
                "unknown": "Unknown",
                "Missing": "Unknown",
                "missing": "Unknown",
                "other":   "Other",
            })

    # 8. Age_band_of_casualty invalid value '5' (244 rows).
    if "Age_band_of_casualty" in out.columns:
        out["Age_band_of_casualty"] = out["Age_band_of_casualty"].replace({"5": "Unknown"})

    # 9. Replace remaining true NaN in object columns with 'Unknown'.
    obj_cols = out.select_dtypes(include=["object", "string"]).columns
    out[obj_cols] = out[obj_cols].fillna("Unknown")
    for c in obj_cols:
        out[c] = out[c].astype("object")

    return out


def _time_of_day(hour: int) -> str:
    if hour < 6:
        return "Night"
    if hour < 12:
        return "Morning"
    if hour < 18:
        return "Afternoon"
    return "Evening"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and ordinal encodings. Expects a cleaned DataFrame."""
    out = df.copy()

    # Time -> Hour, TimeOfDay bucket, RushHour flag.
    # Raw Time string (1 074 distinct values) is too high cardinality for any model.
    if "Time" in out.columns:
        hours = pd.to_datetime(out["Time"], format="%H:%M:%S", errors="coerce").dt.hour
        out["Hour"] = hours.fillna(-1).astype(int)
        out["TimeOfDay"] = hours.map(_time_of_day).fillna("Unknown")
        out["RushHour"] = (
            hours.between(7, 9).fillna(False) | hours.between(16, 19).fillna(False)
        ).astype(int)
        out = out.drop(columns=["Time"])

    # Weekend flag.
    if "Day_of_week" in out.columns:
        out["IsWeekend"] = out["Day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    # Ordinal encodings. Unknown -> -1 (sentinel, not on the ordered scale).
    def _ord(col: str, mapping: dict) -> None:
        if col in out.columns:
            out[col + "_ord"] = out[col].map(mapping).fillna(-1).astype(int)

    _ord("Age_band_of_driver", AGE_BAND_ORDINAL)
    _ord("Driving_experience", DRIVING_EXP_ORDINAL)
    _ord("Service_year_of_vehicle", SERVICE_YEAR_ORDINAL)
    _ord("Educational_level", EDU_ORDINAL)

    # Integer target alongside the original string label.
    if "Accident_severity" in out.columns:
        out["Severity_code"] = out["Accident_severity"].map(SEVERITY_CODE).astype(int)

    return out


def main() -> None:
    print(f"Reading {INPUT_PATH} ...")
    raw = pd.read_csv(INPUT_PATH)
    print(f"  raw shape: {raw.shape}")

    cleaned = clean_data(raw)
    print(f"  after cleaning: {cleaned.shape}  "
          f"(dropped {raw.shape[1] - cleaned.shape[1]} cols)")

    final = engineer_features(cleaned)
    print(f"  after feature engineering: {final.shape}  "
          f"(added {final.shape[1] - cleaned.shape[1]} cols)")

    # Confirm no NaN remains.
    n_nan = final.isna().sum().sum()
    assert n_nan == 0, f"unexpected NaN: {n_nan}"

    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nExported to {OUTPUT_PATH}")
    print(f"  rows: {len(final)}")
    print(f"  cols: {list(final.columns)}")
    print(f"\nTarget distribution:")
    print(final["Accident_severity"].value_counts())


if __name__ == "__main__":
    main()
