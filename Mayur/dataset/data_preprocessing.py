"""
Rebuild the cleaned inflation forecasting dataset from raw CPI and market files.

The default behavior now uses the official MoSPI "All India / Combined" CPI
rows for General, Consumer Food Price, and Fuel and Light. A legacy mode is
still available if we ever need to reproduce the older averaged dataset.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUTPUT_PATH = BASE_DIR / "inflation_forecasting_dataset.csv"
CPI_WORKBOOK_PATH = RAW_DIR / "cpi_147.xlsx"
CPI_ARCHIVE_PATH = RAW_DIR / "cpi_147_archive.zip"

CPI_USECOLS = [
    "year",
    "month_code",
    "state",
    "sector",
    "group",
    "subgroup",
    "index",
    "inflation",
]

CPI_SERIES = [
    {
        "group": "Consumer Food Price",
        "subgroup": "Consumer Food Price-Overall",
        "index_col": "Food_Price_Index",
        "inflation_col": "Food_Price_Inflation",
    },
    {
        "group": "Fuel and Light",
        "subgroup": "Fuel and Light-Overall",
        "index_col": "Fuel_Light_Index",
        "inflation_col": "Fuel_Light_Inflation",
    },
    {
        "group": "General",
        "subgroup": "General-Overall",
        "index_col": "CPI_General_Index",
        "inflation_col": "CPI_General_Inflation",
    },
]

MARKET_SERIES = [
    ("USD_INR Historical Data.csv", "USD_INR_Price"),
    ("Brent Oil Futures Historical Data.csv", "Crude_Oil_Price"),
    ("Bloomberg Commodity Historical Data.csv", "Commodity_Price_Index"),
]

INFLATION_COLUMNS = [
    "Food_Price_Inflation",
    "Fuel_Light_Inflation",
    "CPI_General_Inflation",
]

FINAL_COLUMNS = [
    "Year",
    "Month",
    "Food_Price_Index",
    "Fuel_Light_Index",
    "CPI_General_Index",
    "Food_Price_Inflation",
    "Fuel_Light_Inflation",
    "CPI_General_Inflation",
    "USD_INR_Price",
    "Crude_Oil_Price",
    "Commodity_Price_Index",
]

OFFICIAL_CPI_METHOD = "official_combined"
LEGACY_CPI_METHOD = "legacy_mean"


def _read_cpi_workbook() -> pd.DataFrame:
    if CPI_WORKBOOK_PATH.exists():
        return pd.read_excel(CPI_WORKBOOK_PATH, usecols=CPI_USECOLS)

    if CPI_ARCHIVE_PATH.exists():
        with ZipFile(CPI_ARCHIVE_PATH) as archive:
            workbook_name = next(
                (name for name in archive.namelist() if name.lower().endswith("cpi_147.xlsx")),
                None,
            )
            if workbook_name is None:
                raise FileNotFoundError(
                    f"Could not find cpi_147.xlsx inside {CPI_ARCHIVE_PATH}"
                )

            workbook_bytes = BytesIO(archive.read(workbook_name))
            return pd.read_excel(workbook_bytes, usecols=CPI_USECOLS)

    raise FileNotFoundError(
        f"Missing CPI source. Expected either {CPI_WORKBOOK_PATH} or {CPI_ARCHIVE_PATH}."
    )


def _build_cpi_frame(
    cpi_raw: pd.DataFrame,
    cpi_method: str = OFFICIAL_CPI_METHOD,
) -> pd.DataFrame:
    monthly_frames: list[pd.DataFrame] = []

    for config in CPI_SERIES:
        subgroup_rows = cpi_raw.loc[
            (cpi_raw["group"] == config["group"])
            & (cpi_raw["subgroup"] == config["subgroup"])
        ]

        if cpi_method == OFFICIAL_CPI_METHOD:
            subgroup_rows = subgroup_rows.loc[
                (subgroup_rows["state"] == "All India")
                & (subgroup_rows["sector"] == "Combined")
            ]
        elif cpi_method != LEGACY_CPI_METHOD:
            raise ValueError(
                f"Unsupported cpi_method={cpi_method!r}. "
                f"Expected {OFFICIAL_CPI_METHOD!r} or {LEGACY_CPI_METHOD!r}."
            )

        if cpi_method == OFFICIAL_CPI_METHOD:
            # A few official rows are duplicated in the workbook. Prefer the
            # row that carries the inflation value, since it is internally
            # consistent with the published index level for that month.
            monthly_series = (
                subgroup_rows.assign(_has_inflation=subgroup_rows["inflation"].notna())
                .sort_values(
                    ["year", "month_code", "_has_inflation"],
                    ascending=[True, True, False],
                )
                .drop_duplicates(subset=["year", "month_code"], keep="first")
                .loc[:, ["year", "month_code", "index", "inflation"]]
                .rename(
                    columns={
                        "year": "Year",
                        "month_code": "Month",
                        "index": config["index_col"],
                        "inflation": config["inflation_col"],
                    }
                )
            )
        else:
            monthly_series = (
                subgroup_rows.loc[:, ["year", "month_code", "index", "inflation"]]
                .groupby(["year", "month_code"], as_index=False)
                .mean(numeric_only=True)
                .rename(
                    columns={
                        "year": "Year",
                        "month_code": "Month",
                        "index": config["index_col"],
                        "inflation": config["inflation_col"],
                    }
                )
            )
        monthly_frames.append(monthly_series)

    cpi_frame = monthly_frames[0]
    for monthly_series in monthly_frames[1:]:
        cpi_frame = cpi_frame.merge(monthly_series, on=["Year", "Month"], how="inner")

    return cpi_frame


def _clean_market_price(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def _build_market_frame() -> pd.DataFrame:
    merged_market: pd.DataFrame | None = None

    for filename, output_column in MARKET_SERIES:
        source_path = RAW_DIR / filename
        market_df = pd.read_csv(source_path)
        market_df["Date"] = pd.to_datetime(market_df["Date"], dayfirst=True, errors="coerce")
        market_df[output_column] = _clean_market_price(market_df["Price"])
        market_df["Year"] = market_df["Date"].dt.year
        market_df["Month"] = market_df["Date"].dt.month

        monthly_market = (
            market_df.sort_values("Date")
            .dropna(subset=["Date", output_column])
            .drop_duplicates(subset=["Year", "Month"], keep="last")
            [["Year", "Month", output_column]]
        )

        if merged_market is None:
            merged_market = monthly_market
        else:
            merged_market = merged_market.merge(
                monthly_market,
                on=["Year", "Month"],
                how="inner",
            )

    if merged_market is None:
        raise RuntimeError("No market series were loaded.")

    return merged_market


def build_dataset(cpi_method: str = OFFICIAL_CPI_METHOD) -> pd.DataFrame:
    cpi_raw = _read_cpi_workbook()
    cpi_frame = _build_cpi_frame(cpi_raw, cpi_method=cpi_method)
    market_frame = _build_market_frame()

    dataset = (
        cpi_frame.merge(market_frame, on=["Year", "Month"], how="inner")
        .loc[:, FINAL_COLUMNS]
        .sort_values(["Year", "Month"])
        .reset_index(drop=True)
    )

    dataset = dataset[dataset["Year"] > 2011].reset_index(drop=True)
    dataset[INFLATION_COLUMNS] = dataset[INFLATION_COLUMNS].interpolate(method="linear")
    dataset = dataset.dropna().reset_index(drop=True)

    return dataset


def run_preprocessing_pipeline(
    output_path: Path = OUTPUT_PATH,
    cpi_method: str = OFFICIAL_CPI_METHOD,
) -> pd.DataFrame:
    print("=" * 70)
    print(" Rebuilding inflation forecasting dataset from raw source files")
    print("=" * 70)
    print(f"[1/4] Loading CPI source from: {CPI_WORKBOOK_PATH.name} or archive fallback")

    dataset = build_dataset(cpi_method=cpi_method)

    print(f"[2/4] CPI monthly series reconstructed using method: {cpi_method}")
    print("[3/4] Market price series merged from forex, crude, and commodity CSVs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)

    start = f"{int(dataset.iloc[0]['Year'])}-{int(dataset.iloc[0]['Month']):02d}"
    end = f"{int(dataset.iloc[-1]['Year'])}-{int(dataset.iloc[-1]['Month']):02d}"

    print(f"[4/4] Saved cleaned dataset to: {output_path}")
    print(f" Final shape: {dataset.shape[0]} rows x {dataset.shape[1]} columns")
    print(f" Coverage: {start} to {end}")
    print("=" * 70)

    return dataset


if __name__ == "__main__":
    run_preprocessing_pipeline()
