from __future__ import annotations

import argparse
from dataclasses import dataclass
import pandas as pd
from scipy.stats import binom


@dataclass(frozen=True)
class OverallStats:
    p_make: float
    p_miss: float
    p3: float
    p2: float


@dataclass(frozen=True)
class BinomialStats:
    p_3_of_4_makes: float
    p_4_of_5_threes: float


@dataclass(frozen=True)
class ConditionalStats:
    p_make_given_3: float
    p_make_given_2: float
    p_lead_given_3: float
    p_lead_given_2: float
    p3_given_make: float
    p2_given_make: float


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    # Handles TRUE/FALSE strings, 1/0, booleans, etc.
    if pd.api.types.is_bool_dtype(s):
        return s
    s2 = s.astype(str).str.strip().str.upper()
    return s2.map({"TRUE": True, "FALSE": False, "1": True, "0": False, "T": True, "F": False})


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize expected columns
    if "result" in df.columns:
        df["result"] = _coerce_bool_series(df["result"])
    if "lead" in df.columns:
        df["lead"] = _coerce_bool_series(df["lead"])

    # Basic validation
    required = {"result", "shot_type", "lead"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df


def compute_overall(df: pd.DataFrame) -> OverallStats:
    n = len(df)
    p_make = (df["result"] == True).sum() / n
    p_miss = 1 - p_make

    p3 = (df["shot_type"] == 3).sum() / n
    p2 = 1 - p3  # complement for consistency

    return OverallStats(p_make=float(p_make), p_miss=float(p_miss), p3=float(p3), p2=float(p2))


def compute_binomial(overall: OverallStats) -> BinomialStats:
    # P(exactly 3 makes in 4 shots)
    p_3_of_4_makes = float(binom.pmf(k=3, n=4, p=overall.p_make))

    # P(exactly 4 threes in 5 shots)
    p_4_of_5_threes = float(binom.pmf(k=4, n=5, p=overall.p3))

    return BinomialStats(
        p_3_of_4_makes=p_3_of_4_makes,
        p_4_of_5_threes=p_4_of_5_threes,
    )


def compute_conditionals(df: pd.DataFrame, overall: OverallStats) -> ConditionalStats:
    n = len(df)

    # Joint probabilities
    p_make_and_3 = ((df["result"] == True) & (df["shot_type"] == 3)).sum() / n
    p_make_and_2 = ((df["result"] == True) & (df["shot_type"] == 2)).sum() / n

    p_lead_and_3 = ((df["lead"] == True) & (df["shot_type"] == 3)).sum() / n
    p_lead_and_2 = ((df["lead"] == True) & (df["shot_type"] == 2)).sum() / n

    # Future conditionals (Bayes-style)
    p_make_given_3 = p_make_and_3 / overall.p3
    p_make_given_2 = p_make_and_2 / overall.p2

    p_lead_given_3 = p_lead_and_3 / overall.p3
    p_lead_given_2 = p_lead_and_2 / overall.p2

    # Retrospective (given made)
    p3_given_make = p_make_and_3 / overall.p_make
    p2_given_make = 1 - p3_given_make  # complement

    return ConditionalStats(
        p_make_given_3=float(p_make_given_3),
        p_make_given_2=float(p_make_given_2),
        p_lead_given_3=float(p_lead_given_3),
        p_lead_given_2=float(p_lead_given_2),
        p3_given_make=float(p3_given_make),
        p2_given_make=float(p2_given_make),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Steph Curry shot probabilities.")
    parser.add_argument("--csv", required=True, help="Path to Steph Curry shots CSV")
    args = parser.parse_args()

    df = load_data(args.csv)
    overall = compute_overall(df)
    bino = compute_binomial(overall)
    cond = compute_conditionals(df, overall)

    print("=== Overall ===")
    print(overall)
    print("\n=== Binomial ===")
    print(bino)
    print("\n=== Conditional ===")
    print(cond)


if __name__ == "__main__":
    main()
