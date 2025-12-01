import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load NCHS data
nchs = pd.read_csv('NCHS_-_U.S._and_State_Trends_on_Teen_Births.csv')
nchs = nchs.rename(columns={
    'State': 'state',
    'State Rate': 'teen_birth_rate',
    'Age Group (Years)': 'age_group',
    'Year': 'year'
})

# keep only 2019, ages 15–19
nchs = nchs[(nchs["age_group"] == "15-19 years") & (nchs["year"] == 2019)].copy()
nchs["state"] = nchs["state"].str.strip().str.lower()
nchs["teen_birth_rate"] = pd.to_numeric(nchs["teen_birth_rate"], errors="coerce")

# 2. Load policy data + variation table
policy = pd.read_csv('Sex_Policies.csv')
policy.columns = [c.strip() for c in policy.columns]
policy = policy[policy["Jurisdiction"].str.upper() != "TOTAL"].copy()

policy_cols_raw = [
    "Sex ed mandated",
    "HIV ed mandated",
    "Ed must be medically accurate",
    "Ed must be age-appropriate",
    "Ed must include abstinence",
    "Ed must include contraception",
    "Ed must cover consent",
    "Ed must include sexual orientation and gender identity",
    "Ed must cover healthy relationships",
]

# variation table
variation_dict = {}
for col in policy_cols_raw:
    counts = (
        policy[col]
        .fillna("Missing")
        .astype(str)
        .str.strip()
        .replace("", "Missing")
        .value_counts()
        .sort_index()
    )
    variation_dict[col] = counts

variation_df = pd.DataFrame(variation_dict).T.fillna(0)
variation_df.index.name = "Policy Feature"

# variation plot
plot_df = variation_df.reset_index().melt(
    id_vars="Policy Feature",
    var_name="Category",
    value_name="Count"
)
plot_df = plot_df[plot_df["Count"] > 0]

plt.figure(figsize=(12, 8))
sns.barplot(
    data=plot_df,
    y="Policy Feature",
    x="Count",
    hue="Category",
    palette="tab20"
)
plt.title("Variation in Sex Education Policies Across States")
plt.xlabel("Number of States")
plt.ylabel("Policy Feature")
plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\nSex Ed Policy Variation Table (Counts by Category):\n")
print(variation_df)

# 3. Policy vs teen birth rate (binary columns only)

# rename policy columns to shorter names
policy = policy.rename(columns={
    "Jurisdiction": "state",
    "Sex ed mandated": "sex_mand",
    "HIV ed mandated": "hiv_mand",
    "Ed must be medically accurate": "accurate",
    "Ed must be age-appropriate": "ageapp",
    "Ed must include abstinence": "abstinence",
    "Ed must include contraception": "contraception",
    "Ed must cover consent": "consent",
    "Ed must include sexual orientation and gender identity": "sogi",
    "Ed must cover healthy relationships": "relationships"
})

policy1 = policy[['state', 'sex_mand', 'hiv_mand', 'ageapp',
                  'contraception', 'consent', 'relationships']].copy()
policy1["state"] = policy1["state"].str.strip().str.lower()

# average birth rate by state
nchs_sep = (
    nchs.groupby('state', as_index=False)['teen_birth_rate']
    .mean()
)

merged = pd.merge(nchs_sep, policy1, on="state", how="inner")
print("\nMerged NCHS + policy shape:", merged.shape)
print(merged.head())

policy_cols = ["sex_mand", "hiv_mand", "contraception", "consent", "relationships"]

print("\nUnique values per binary policy column:")
for col in policy_cols:
    if col in merged.columns:
        print(col, merged[col].unique())

# mean teen birth rate by 0 vs 1 (only if truly binary)
summary_table = {}
for col in policy_cols:
    if col in merged.columns:
        means = merged.groupby(col)["teen_birth_rate"].mean().round(2)
        # keep only columns that have both 0 and 1
        if len(means.index) == 2 and set(means.index) == {0, 1}:
            summary_table[col] = means.to_dict()

summary_df = pd.DataFrame(summary_table).T
summary_df.index.name = "Policy Feature"

if not summary_df.empty and summary_df.shape[1] == 2:
    summary_df.columns = ["No Policy (0)", "Policy Present (1)"]
else:
    print("\n[Info] No policy columns had both 0 and 1 after merge; skipping relabel.\n")

rename_map = {
    "sex_mand": "Sex Ed Mandated",
    "hiv_mand": "HIV Ed Mandated",
    "contraception": "Covers Contraception",
    "consent": "Covers Consent",
    "relationships": "Covers Healthy Relationships"
}
summary_df = summary_df.rename(index=rename_map).sort_index()

print("\nAverage Teen Birth Rate by Policy Feature (2019):\n")
print(summary_df)

if not summary_df.empty:
    plot_df2 = summary_df.reset_index().melt(
        id_vars="Policy Feature",
        var_name="Policy Status",
        value_name="Average Teen Birth Rate"
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df2,
        x="Policy Feature",
        y="Average Teen Birth Rate",
        hue="Policy Status",
        palette=["#E57373", "#64B5F6"]
    )
    plt.title("Average Teen Birth Rate by Policy Feature (2019)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("")
    plt.ylabel("Average Births per 1,000 Females (Ages 15–19)")
    plt.legend(title="")
    plt.tight_layout()
    plt.show()
else:
    print("No bar chart for policy vs birth rate (no binary variation).")


# 4. Grade scores vs teen birth rate (regression)
grades = pd.read_csv('Insights_US_Sex_ed.csv')

# clean grade text
for col in ["overall_grade", "sex_ed_requirement_grade", "content_grade"]:
    grades[col] = grades[col].astype(str).str.strip().str.lower()

# normalize state
grades["state"] = grades["state"].astype(str).str.strip().str.lower()
df_letter = (
    nchs.merge(grades[["state", "overall_grade"]], on="state", how="inner")
        .dropna(subset=["overall_grade", "teen_birth_rate"])
)

# Order grades from best to worst

grade_order_letter = ["a", "a-", "b+", "b", "b-", "c+", "c", "c-", "d+", "d", "d-", "f"]

# Compute average teen birth rate by content grade
content_means_letter = (
    df_letter.groupby("overall_grade")["teen_birth_rate"]
    .mean()
    .reindex(grade_order_letter)
    .dropna()
)

plt.figure(figsize=(10, 5))
sns.barplot(
    x=[g.upper() for g in content_means_letter.index],
    y=content_means_letter.values,
    palette="viridis"
)
plt.title("Average Teen Birth Rate by Overall Grade (SIECUS State Profiles, 2025)")
plt.xlabel("Overall Grade (Letter)")
plt.ylabel("Teen Birth Rate (per 1,000, ages 15–19)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

grade_map = {
    "a":4.0, "a-":3.7,
    "b+":3.3, "b":3.0, "b-":2.7,
    "c+":2.3, "c":2.0, "c-":1.7,
    "d+":1.3, "d":1.0, "d-":0.7,
    "f":0.0
}

grades["overall_num"] = grades["overall_grade"].map(grade_map)
grades["req_num"]     = grades["sex_ed_requirement_grade"].map(grade_map)
grades["content_num"] = grades["content_grade"].map(grade_map)

grades = grades[["state", "overall_num", "req_num", "content_num"]]

score_cols = {
    "overall_num": "Overall Grade (0–4)",
    "req_num": "Requirement Grade (0–4)",
    "content_num": "Content Grade (0–4)"
}

df = (
    nchs
      .merge(grades, on="state", how="inner")
      .dropna(subset=["overall_num", "req_num", "content_num", "teen_birth_rate"])
)

print("\nMerged NCHS + grade scores shape:", df.shape)
print(df.head())

# scatter + regression line plots
for col, xlabel in score_cols.items():
    plt.figure(figsize=(5, 4))
    sns.regplot(data=df, x=col, y="teen_birth_rate", scatter_kws={"alpha": 0.8})
    plt.title(f"Teen Birth Rate vs {xlabel}")
    plt.xlabel(xlabel)
    plt.ylabel("Teen Birth Rate (per 1,000, ages 15–19)")
    plt.tight_layout()
    plt.show()

# regression model
formula = "teen_birth_rate ~ overall_num + req_num + content_num"
model = smf.ols(formula, data=df).fit(cov_type="HC3")

print("\nOLS Regression Results:\n")
print(model.summary())

print("\nPlain-English effect sizes:")
for name, coef in model.params.items():
    if name == "Intercept":
        continue
    direction = "lower" if coef < 0 else "higher"
    print(f" • A 1-point increase in {name} → {abs(coef):.2f} {direction} teen births per 1,000 (holding other grades constant).")
