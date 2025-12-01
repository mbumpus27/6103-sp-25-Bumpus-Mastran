# %% [markdown]
# STI Rates Data Exploration

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# %% [markdown]
# CHLAMYDIA DATA

# %%
chlamydia = pd.read_csv('STI Data/chlamydia.csv')
chlamydia.head(10)

# %%
chlamydia.columns
chlamydia.describe()
chlamydia.info()
chlamydia['State'].is_unique

# %%
case = chlamydia['Cases']
plt.hist(case)

# %%
rate = chlamydia['Rate_per_100k']
plt.hist(rate)
plt.title("Distribution of Chlamydia Rates")
plt.xlabel("Rate per 100k")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# GONORRHEA

# %%
gonorrhea = pd.read_csv('STI Data/gonorrhea.csv')
gonorrhea.head(10)

# %%
gonorrhea.columns
gonorrhea.describe()
gonorrhea.info()
gonorrhea['State'].is_unique

# %%
case = gonorrhea['Cases']
plt.hist(case)

# %%
rate = gonorrhea['Rate_per_100k']
plt.hist(rate)
plt.title("Distribution of Gonorrhea Rates")
plt.xlabel("Rate per 100k")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# CONGENITAL SYPHILIS

# %%
cong_syph = pd.read_csv('STI Data/congenital_syphilis.csv')
cong_syph.head(10)

# %%
cong_syph.columns
cong_syph.describe()
cong_syph.info()
cong_syph['State'].is_unique

# %%
case = cong_syph['Cases']
plt.hist(case)

# %%
rate = cong_syph['Rate_per_100k']
plt.hist(rate)
plt.title("Distribution of Congenital Syphilis Rates")
plt.xlabel("Rate per 100k")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# PRIMARY SECONDARY SYPHILIS 

# %%
prim_syph = pd.read_csv('STI Data/primary_secondary_syphilis.csv')
prim_syph.head(10)

# %%
prim_syph.columns
prim_syph.describe()
prim_syph.info()
prim_syph['State'].is_unique

# %%
case = prim_syph['Cases']
plt.hist(case)

# %%
rate = prim_syph['Rate_per_100k']
plt.hist(rate)
plt.title("Distribution of Primary Secondary Syphilis Rates")
plt.xlabel("Rate per 100k")
plt.ylabel("Frequency")
plt.show()

# %% [markdown]
# STI Regression

# %%
state = pd.read_csv('newer_US_Sex_ed.csv')
state.head(10)
state = state.rename(columns={'state': 'State'})
state.columns

# %% [markdown]
# Chlamydia

# %%
chlamydia.shape

# %%
state.shape

# %%
combo_chla = pd.merge(chlamydia, state, on = "State")
combo_chla.shape

# %%
combo_chla.head(10)

# %%
model_chlam = ols(formula = 'Rate_per_100k ~ sex_ed_requirement_grade + content_grade', data = combo_chla)
model_chlam = model_chlam.fit()
print( model_chlam.summary() )

# %%
model_chlam = ols(formula = 'Rate_per_100k ~ overall_grade', data = combo_chla)
model_chlam = model_chlam.fit()
print( model_chlam.summary() )

# %% [markdown]
# Gonorrhea

# %%
combo_gon = pd.merge(gonorrhea, state, on = "State")
combo_gon.shape

# %%
combo_gon.head(10)

# %%
model_gon = ols(formula = 'Rate_per_100k ~ sex_ed_requirement_grade + content_grade', data = combo_gon)
model_gon = model_gon.fit()
print( model_gon.summary() )

# %%
model_gon = ols(formula = 'Rate_per_100k ~ overall_grade', data = combo_gon)
model_gon = model_gon.fit()
print( model_gon.summary() )

# %% [markdown]
# Primary Secondary Syphilis

# %%
combo_prim = pd.merge(prim_syph, state, on = "State")
combo_prim.shape

# %%
combo_prim.head(10)

# %%
model_prim = ols(formula = 'Rate_per_100k ~ sex_ed_requirement_grade + content_grade', data = combo_prim)
model_prim = model_prim.fit()
print( model_prim.summary() )

# %%
model_prim = ols(formula = 'Rate_per_100k ~ overall_grade', data = combo_prim)
model_prim = model_prim.fit()
print( model_prim.summary() )

# %% [markdown]
# Congenital Syphilis

# %%
cong_syph.shape

# %%
combo_cong = pd.merge(cong_syph, state, on = "State")
combo_cong.shape

# %%
combo_cong.head(10)

# %%
model_cong = ols(formula = 'Rate_per_100k ~ sex_ed_requirement_grade + content_grade', data = combo_cong)
model_cong = model_cong.fit()
print( model_cong.summary() )

# %%
model_cong = ols(formula = 'Rate_per_100k ~ overall_grade', data = combo_cong)
model_cong = model_cong.fit()
print( model_cong.summary() )

# %% [markdown]
# Bar Chart Comparison

# %% [markdown]
# Chlamydia

# %%
combo_chla.head(10)

# %%
avg_rates_chla = combo_chla.groupby("sex_ed_requirement_grade")["Rate_per_100k"].mean().sort_index()
avg_rates_chla.plot(kind="bar")

plt.xlabel("Sex Ed Requirement Grade")
plt.ylabel("Average Rate")
plt.title("Chlamydia Average Rate by Policy Score")
plt.show()

# %% [markdown]
# Gonorrhea

# %%
avg_rates_gon = combo_gon.groupby("sex_ed_requirement_grade")["Rate_per_100k"].mean().sort_index()
avg_rates_gon.plot(kind="bar")

plt.xlabel("Sex Ed Requirement Grade")
plt.ylabel("Average Rate")
plt.title("Gonorrhea Average Rate by Policy Score")
plt.show()

# %% [markdown]
# Primary Secondary Syphilis

# %%
avg_rates_prim = combo_prim.groupby("sex_ed_requirement_grade")["Rate_per_100k"].mean().sort_index()
avg_rates_prim.plot(kind="bar")

plt.xlabel("Sex Ed Requirement Grade")
plt.ylabel("Average Rate")
plt.title("Primary Secondary Syphilis Average Rate by Policy Score")
plt.show()

# %% [markdown]
# Congenital Syphilis

# %%
avg_rates_cong = combo_cong.groupby("sex_ed_requirement_grade")["Rate_per_100k"].mean().sort_index()
avg_rates_cong.plot(kind="bar")

plt.xlabel("Sex Ed Requirement Grade")
plt.ylabel("Average Rate")
plt.title("Congenital Syphilis Average Rate by Policy Score")
plt.show()


