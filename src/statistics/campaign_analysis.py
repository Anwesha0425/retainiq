from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import (
    chi2_contingency,
    mannwhitneyu,
    norm,
)

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    def multipletests(pvals, alpha=0.05, method="fdr_bh", **kwargs):
        pvals = np.asarray(pvals)
        n = len(pvals)
        if n == 0:
            return np.array([]), np.array([]), None, None
        order = np.argsort(pvals)
        ranked_pvals = pvals[order]
        qvals = np.zeros(n)
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = ranked_pvals[i] * n / rank
            if i == n - 1:
                qvals[i] = min(val, 1.0)
            else:
                qvals[i] = min(val, qvals[i + 1], 1.0)
        adj_pvals = np.empty(n)
        adj_pvals[order] = qvals
        rejected = adj_pvals <= alpha
        return rejected, adj_pvals, None, None


def chi_square_response_test(
    df: pd.DataFrame,
    categorical_column: str,
    target_column: str = "response",
) -> dict:
    """
    Test whether campaign response is associated
    with a categorical customer attribute.
    """

    required_columns = {
        categorical_column,
        target_column,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    contingency_table = pd.crosstab(
        df[categorical_column],
        df[target_column],
    )

    chi2, p_value, degrees_of_freedom, _ = (
        chi2_contingency(contingency_table)
    )

    return {
        "feature": categorical_column,
        "chi2": float(chi2),
        "p_value": float(p_value),
        "degrees_of_freedom": int(
            degrees_of_freedom
        ),
    }


def compare_response_groups(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    target_column: str = "response",
) -> pd.DataFrame:
    """
    Compare continuous/numeric customer features
    between campaign responders and non-responders.

    Mann-Whitney U is used because customer spending
    and behavioral variables are typically skewed.

    Effect size:
        r = |Z| / sqrt(N)
    """

    feature_columns = list(feature_columns)

    required_columns = set(feature_columns) | {
        target_column
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    responders = df[
        df[target_column] == 1
    ]

    non_responders = df[
        df[target_column] == 0
    ]

    results = []

    for feature in feature_columns:

        responder_values = responders[feature].dropna()

        non_responder_values = (
            non_responders[feature].dropna()
        )

        statistic, p_value = mannwhitneyu(
            responder_values,
            non_responder_values,
            alternative="two-sided",
        )

        total_n = (
            len(responder_values)
            + len(non_responder_values)
        )

        # Convert the two-sided p-value to
        # the magnitude of the corresponding Z statistic.
        if p_value == 0:
            z_absolute = float("inf")
        else:
            z_absolute = abs(
                norm.ppf(p_value / 2)
            )

        effect_size = (
            z_absolute / (total_n ** 0.5)
        )

        results.append(
            {
                "feature": feature,
                "responder_median": float(
                    responder_values.median()
                ),
                "non_responder_median": float(
                    non_responder_values.median()
                ),
                "u_statistic": float(statistic),
                "p_value": float(p_value),
                "effect_size": float(effect_size),
            }
        )

    result_df = pd.DataFrame(results)

    if not result_df.empty:

        rejected, adjusted_p_values, _, _ = (
            multipletests(
                result_df["p_value"],
                alpha=0.05,
                method="fdr_bh",
            )
        )

        result_df["adjusted_p_value"] = (
            adjusted_p_values
        )

        result_df["significant"] = rejected

    return result_df


def summarize_significant_features(
    results: pd.DataFrame,
    p_value_column: str = "adjusted_p_value",
) -> pd.DataFrame:
    """
    Return statistically significant features.
    """

    if p_value_column not in results.columns:
        raise ValueError(
            f"Missing column: {p_value_column}"
        )

    return results[
        results[p_value_column] < 0.05
    ].copy()