import numpy as np
import pandas as pd
from scipy import stats

def run_chi_square_test(df, feature='cp', alpha=0.05):
    """
    Perform Chi-Square Test of Independence between categorical feature and Disease status (target).
    Crucial Rule: Categorical variables like Chest Pain (CP) MUST be tested with Chi-Square,
    NOT with t-tests.
    """
    # Feature category mappings
    category_labels = {
        'cp': {
            0: "Typical Angina (0)",
            1: "Atypical Angina (1)",
            2: "Non-anginal Pain (2)",
            3: "Asymptomatic (3)"
        },
        'sex': {
            0: "Female (0)",
            1: "Male (1)"
        },
        'fbs': {
            0: "Fasting Blood Sugar <= 120 (0)",
            1: "Fasting Blood Sugar > 120 (1)"
        },
        'restecg': {
            0: "Normal (0)",
            1: "ST-T Wave Abnormality (1)",
            2: "Left Ventricular Hypertrophy (2)"
        },
        'exang': {
            0: "No Exercise Angina (0)",
            1: "Exercise Angina Present (1)"
        },
        'slope': {
            0: "Upsloping (0)",
            1: "Flat (1)",
            2: "Downsloping (2)"
        },
        'ca': {
            0: "0 Major Vessels (0)",
            1: "1 Major Vessel (1)",
            2: "2 Major Vessels (2)",
            3: "3 Major Vessels (3)",
            4: "4 Major Vessels (4)"
        },
        'thal': {
            0: "Null/Unknown (0)",
            1: "Normal (1)",
            2: "Fixed Defect (2)",
            3: "Reversible Defect (3)"
        }
    }

    feature_names = {
        'cp': 'Chest Pain Type (CP)',
        'sex': 'Biological Sex',
        'fbs': 'Fasting Blood Sugar',
        'restecg': 'Resting ECG Results',
        'exang': 'Exercise-Induced Angina',
        'slope': 'ST Segment Slope',
        'ca': 'Major Fluoroscopy Vessels',
        'thal': 'Thalassemia'
    }
    label = feature_names.get(feature, feature)
    cat_map = category_labels.get(feature, {})

    # Create contingency table
    contingency = pd.crosstab(df[feature], df['target'], margins=False)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Format contingency table for display
    observed_table = []
    for cat_val in contingency.index:
        c_label = cat_map.get(cat_val, f"Category {cat_val}")
        n_no = int(contingency.loc[cat_val, 0]) if 0 in contingency.columns else 0
        n_yes = int(contingency.loc[cat_val, 1]) if 1 in contingency.columns else 0
        total = n_no + n_yes
        prop_disease = (n_yes / total * 100) if total > 0 else 0.0
        observed_table.append({
            "category_code": int(cat_val),
            "category_label": c_label,
            "no_disease_count": n_no,
            "disease_count": n_yes,
            "total": total,
            "disease_rate_pct": round(prop_disease, 1)
        })

    # Expected frequencies table
    expected_table = []
    for idx, cat_val in enumerate(contingency.index):
        c_label = cat_map.get(cat_val, f"Category {cat_val}")
        expected_table.append({
            "category_code": int(cat_val),
            "category_label": c_label,
            "expected_no_disease": round(float(expected[idx, 0]), 2),
            "expected_disease": round(float(expected[idx, 1]), 2)
        })

    # Effect size: Cramér's V
    n_obs = contingency.to_numpy().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2_stat / (n_obs * min_dim)) if min_dim > 0 and n_obs > 0 else 0.0

    is_significant = bool(p_value < alpha)

    if is_significant:
        conclusion = (
            f"Statistically significant association detected between {label} and Disease Status "
            f"(χ² = {chi2_stat:.4f}, df = {dof}, p = {p_value:.4e}, α = {alpha}). "
            f"We reject the null hypothesis of independence. Patients with different categories of {label} "
            f"exhibit significantly distinct disease risks (Cramér's V = {cramers_v:.4f})."
        )
    else:
        conclusion = (
            f"Insufficient evidence to reject independence between {label} and Disease Status "
            f"(χ² = {chi2_stat:.4f}, df = {dof}, p = {p_value:.4f}, α = {alpha}). "
            f"The distribution of disease does not vary significantly across categories of {label}."
        )

    # Methodological reminder distinguishing from t-test and Bayesian probability
    method_note = (
        f"Statistical Rule: {label} is a categorical variable. Applying a Student's or Welch's t-test "
        f"to categorical codes is mathematically invalid because categorical codes lack metric interval properties. "
        f"The Chi-Square Test of Independence is the appropriate frequentist test. Note also that the p-value "
        f"({p_value:.4e}) measures statistical significance against H0, whereas the Bayesian Network computes "
        f"posterior disease risk probability P(Disease | Evidence) for individual patients."
    )

    return {
        "feature": feature,
        "feature_label": label,
        "null_hypothesis": f"H0: {label} and Heart Disease status are independent in the population.",
        "alt_hypothesis": f"H1: {label} and Heart Disease status are significantly dependent.",
        "chi2_statistic": round(float(chi2_stat), 4),
        "degrees_of_freedom": int(dof),
        "p_value": float(p_value),
        "p_value_formatted": "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}",
        "alpha": alpha,
        "cramers_v": round(float(cramers_v), 4),
        "is_significant": is_significant,
        "conclusion": conclusion,
        "method_note": method_note,
        "observed_table": observed_table,
        "expected_table": expected_table
    }


def run_all_chi_square_tests(df, alpha=0.05):
    """Run chi-square tests for all categorical variables."""
    features = ['cp', 'sex', 'exang', 'slope', 'ca', 'thal', 'restecg', 'fbs']
    results = {}
    for f in features:
        if f in df.columns:
            results[f] = run_chi_square_test(df, feature=f, alpha=alpha)
    return results
