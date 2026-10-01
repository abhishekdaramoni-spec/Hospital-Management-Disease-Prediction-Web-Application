import numpy as np
import pandas as pd
from scipy import stats

def run_two_sample_z_test(df, feature='thalach', alpha=0.05):
    """
    Perform a Two-Sample Z-Test comparing feature means between Heart Disease (target=1)
    and No Heart Disease (target=0).
    Assumptions: Large sample size (n1, n2 >= 30), independent observations, CLT applies.
    """
    g1 = df[df['target'] == 1][feature].dropna()
    g0 = df[df['target'] == 0][feature].dropna()

    n1, n2 = len(g1), len(g0)
    mean1, mean2 = float(g1.mean()), float(g0.mean())
    std1, std2 = float(g1.std(ddof=1)), float(g0.std(ddof=1))

    # Standard error of the difference in means
    se_diff = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    diff = mean1 - mean2
    z_stat = diff / se_diff if se_diff > 0 else 0.0

    # Two-tailed p-value
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    # 95% Confidence Interval for the difference in means
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = diff - z_crit * se_diff
    ci_upper = diff + z_crit * se_diff

    # Assumptions validation
    assumptions = {
        "n1_valid": bool(n1 >= 30),
        "n2_valid": bool(n2 >= 30),
        "large_sample_clt": bool(n1 >= 30 and n2 >= 30),
        "independence": True,
        "assumptions_met": bool(n1 >= 30 and n2 >= 30)
    }

    feature_labels = {
        'thalach': 'Maximum Heart Rate Achieved (bpm)',
        'trestbps': 'Resting Blood Pressure (mmHg)',
        'chol': 'Serum Cholesterol (mg/dl)',
        'oldpeak': 'ST Depression (mm)',
        'age': 'Patient Age (years)'
    }
    label = feature_labels.get(feature, feature)

    is_significant = p_value < alpha

    if is_significant:
        conclusion = (
            f"Statistically significant evidence against H0 at α = {alpha} (Z = {z_stat:.4f}, p = {p_value:.4e}). "
            f"There is a demonstrable population difference in {label} between patients with heart disease "
            f"(Mean = {mean1:.2f}) and patients without heart disease (Mean = {mean2:.2f})."
        )
    else:
        conclusion = (
            f"Insufficient evidence to reject H0 at α = {alpha} (Z = {z_stat:.4f}, p = {p_value:.4f}). "
            f"The observed difference in {label} between the two cohorts is consistent with random variation."
        )

    return {
        "test_name": f"Two-Sample Z-Test for Difference in Means ({label})",
        "feature": feature,
        "feature_label": label,
        "null_hypothesis": f"H0: μ(Disease) - μ(No Disease) = 0 (No difference in mean {label})",
        "alt_hypothesis": f"H1: μ(Disease) - μ(No Disease) ≠ 0 (Significant difference in mean {label})",
        "group1_name": "Heart Disease (Target = 1)",
        "group1_n": n1,
        "group1_mean": round(mean1, 2),
        "group1_std": round(std1, 2),
        "group2_name": "No Heart Disease (Target = 0)",
        "group2_n": n2,
        "group2_mean": round(mean2, 2),
        "group2_std": round(std2, 2),
        "mean_difference": round(diff, 2),
        "std_error": round(se_diff, 4),
        "z_statistic": round(z_stat, 4),
        "p_value": p_value,
        "p_value_formatted": "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}",
        "alpha": alpha,
        "ci_95": (round(ci_lower, 2), round(ci_upper, 2)),
        "is_significant": is_significant,
        "conclusion": conclusion,
        "assumptions": assumptions
    }


def run_one_sample_z_test(df, feature='chol', population_mean=200.0, alpha=0.05):
    """
    Perform a One-Sample Z-Test comparing cohort mean against clinical guideline threshold
    (e.g., Clinical borderline cholesterol = 200 mg/dl, Normal BP = 120 mmHg).
    """
    data = df[feature].dropna()
    n = len(data)
    sample_mean = float(data.mean())
    sample_std = float(data.std(ddof=1))

    se = sample_std / np.sqrt(n)
    z_stat = (sample_mean - population_mean) / se
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    feature_labels = {
        'chol': 'Serum Cholesterol (mg/dl)',
        'trestbps': 'Resting Blood Pressure (mmHg)',
        'thalach': 'Maximum Heart Rate (bpm)'
    }
    label = feature_labels.get(feature, feature)
    is_significant = p_value < alpha

    if is_significant:
        conclusion = (
            f"Statistically significant evidence against H0 at α = {alpha} (Z = {z_stat:.4f}, p = {p_value:.4e}). "
            f"The cohort mean {label} ({sample_mean:.2f}) significantly differs from the clinical benchmark ({population_mean:.2f})."
        )
    else:
        conclusion = (
            f"Insufficient evidence to reject H0 at α = {alpha} (Z = {z_stat:.4f}, p = {p_value:.4f}). "
            f"The cohort mean {label} ({sample_mean:.2f}) does not significantly differ from {population_mean:.2f}."
        )

    return {
        "test_name": f"One-Sample Z-Test ({label} vs Clinical Benchmark {population_mean})",
        "feature": feature,
        "feature_label": label,
        "null_hypothesis": f"H0: μ = {population_mean} (Cohort mean equals clinical standard)",
        "alt_hypothesis": f"H1: μ ≠ {population_mean} (Cohort mean differs from clinical standard)",
        "sample_n": n,
        "sample_mean": round(sample_mean, 2),
        "sample_std": round(sample_std, 2),
        "benchmark_mean": population_mean,
        "z_statistic": round(z_stat, 4),
        "p_value": p_value,
        "p_value_formatted": "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}",
        "alpha": alpha,
        "is_significant": is_significant,
        "conclusion": conclusion
    }


def run_two_proportion_z_test(df, feature='exang', alpha=0.05):
    """
    Perform a Two-Proportion Z-Test comparing proportion of binary risk factor between
    Disease and No Disease cohorts.
    """
    g1 = df[df['target'] == 1][feature]
    g0 = df[df['target'] == 0][feature]

    n1, n2 = len(g1), len(g0)
    count1, count2 = int(g1.sum()), int(g0.sum())
    p1 = count1 / n1
    p2 = count2 / n2

    # Pooled proportion
    p_pool = (count1 + count2) / (n1 + n2)
    se_prop = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    z_stat = (p1 - p2) / se_prop if se_prop > 0 else 0.0
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    feature_labels = {
        'exang': 'Exercise-Induced Angina',
        'sex': 'Male Sex Ratio',
        'fbs': 'Fasting Blood Sugar > 120 mg/dl'
    }
    label = feature_labels.get(feature, feature)
    is_significant = p_value < alpha

    return {
        "test_name": f"Two-Proportion Z-Test ({label})",
        "feature": feature,
        "feature_label": label,
        "null_hypothesis": f"H0: p(Disease) = p(No Disease) (Proportion of {label} is equal)",
        "alt_hypothesis": f"H1: p(Disease) ≠ p(No Disease) (Proportion of {label} differs)",
        "group1_n": n1,
        "group1_success": count1,
        "group1_prop": round(p1 * 100, 2),
        "group2_n": n2,
        "group2_success": count2,
        "group2_prop": round(p2 * 100, 2),
        "z_statistic": round(z_stat, 4),
        "p_value": p_value,
        "p_value_formatted": "< 0.0001" if p_value < 0.0001 else f"{p_value:.4f}",
        "alpha": alpha,
        "is_significant": is_significant,
        "conclusion": (
            f"Statistically significant difference in {label} proportions (Z = {z_stat:.4f}, p = {p_value:.4e}). "
            f"Disease cohort: {p1*100:.1f}%, Control cohort: {p2*100:.1f}%."
            if is_significant else
            f"No statistically significant difference in {label} proportions (Z = {z_stat:.4f}, p = {p_value:.4f})."
        )
    }
