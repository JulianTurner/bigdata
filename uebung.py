# %%
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler 
from sklearn.model_selection import train_test_split

# %% [markdown]
# Only column C-M needs to be imported

# %%
dataset = pd.read_csv('Churn-Rate.csv',usecols=range(2,13), sep=';', decimal=',')
dataset


# %%
dataset


# %%
dataset.dtypes

# %% [markdown]
# Lines without CreditScore must be deleted
# 
# `copy` damit keien Seiteneffekte entstehen können (write on copy)

# %%
dataset = dataset.dropna(subset='CreditScore').copy(deep=False)
dataset


# %% [markdown]
# Fix datatype of columns

# %%
dataset[['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'CreditScore']
        ] = dataset.loc[:, ['NumOfProducts', 'HasCrCard', 'IsActiveMember', 'CreditScore']].astype(int)


# %% [markdown]
# Convert Value wich are out of range or faulty

# %%
age_numeric = pd.to_numeric(dataset.loc[:, 'Age'], errors='coerce')

age_fault = age_numeric.replace(0,np.NaN)



dataset['Age'] = age_fault



# %% [markdown]
# Check for values

# %%
dataset[dataset.isnull().any(axis=1)].style.applymap(
    lambda x: 'background:coral' if pd.isna(x) else '')


# %% [markdown]
# Gender: Missing data should be imputed with the same distrubution of Male/Female of the
# rest of the data set

# %%
count_missing_gender = dataset.loc[:, 'Gender'].isna().sum()
count_missing_gender

count = dataset.loc[:, 'Gender'].value_counts(normalize=True)
count


gender_amount = (count * count_missing_gender).round().astype(int)
gender_amount

gender_colum_name = gender_amount.name

gener_data = np.repeat(gender_amount.keys().values, gender_amount.values)
gener_data


isna = dataset.loc[:, 'Gender'].isna()
missing_index = dataset[isna].index
missing_index

gender_update = pd.Series(
    data=gener_data, index=missing_index, name=gender_colum_name)

dataset.update(gender_update)
dataset

# %% [markdown]
#  Age: Impute with mean

# %%
age_tenure_median = dataset.loc[:, ['Age', 'Tenure']].mean()

age_tenure_median_nona = dataset.loc[:, ['Age', 'Tenure']].fillna(
    age_tenure_median).astype(int)


dataset[['Age', 'Tenure']] = age_tenure_median_nona


# %% [markdown]
#  Geography: Transform it with Pandas

# %%
dataset = pd.get_dummies(dataset, prefix='Geography', columns=['Geography'], dtype=int)
dataset

# %% [markdown]
# Gender & Exited: Transform it with OrdinalEncoder instead of LabelEncoder

# %%
enc = OrdinalEncoder()
gender_exited_ordinal = enc.fit_transform(dataset.loc[:,['Gender', 'Exited']]).astype(int)

dataset[['Gender', 'Exited']] = gender_exited_ordinal
gender_exited_ordinal


# %% [markdown]
# Split the data-set into the Training- and Test-set (80/20)

# %%
train, test = train_test_split(dataset, test_size=0.2, random_state=0)
display(train)
display(test)

# %% [markdown]
# Apply Feature Scaling 
# - Values wich can me put in a bucket sould be normalized
# - Valued wich differ a lot should be standardized
# 
# Age, NumOfProducts => normalized
# Balance, EstimatedSalary => standardized

# %%
min_max_scaler = MinMaxScaler()
train[['Age', 'NumOfProducts']] = min_max_scaler.fit_transform(
    train.loc[:, ['Age', 'NumOfProducts']])

test[['Age', 'NumOfProducts']] = min_max_scaler.fit_transform(
    test.loc[:, ['Age', 'NumOfProducts']])

display(train)
display(test)

# %%
standard_scaler = StandardScaler()
train[['Balance', 'EstimatedSalary']] = standard_scaler.fit_transform(
    train.loc[:, ['Balance', 'EstimatedSalary']])

test[['Balance', 'EstimatedSalary']] = standard_scaler.fit_transform(
    test.loc[:, ['Balance', 'EstimatedSalary']])

display(train)
display(test)


# %% [markdown]
# describe normalization

# %%

def desc_normalization(title: str, df):
    describe_data = df[['Age', 'NumOfProducts']].describe(percentiles=[]).loc[['min', 'max']]
    display(title)
    display(describe_data)

desc_normalization('train', train)
desc_normalization('test', test)


# %% [markdown]
# describe standardization

# %%
def desc_standardization(title: str, df):
    describe_data = df[['Balance', 'EstimatedSalary']].describe(
        percentiles=[]).loc[['mean', 'std']]
    display(title)
    display(describe_data)


desc_standardization('train', train)
desc_standardization('test', test)


# %%
display(train)
display(train.dtypes)
display(test)
display(test.dtypes)


# %% [markdown]
# save data

# %%
train.to_csv('result/train.csv', sep=';', index=False, decimal=',')
test.to_csv('result/test.csv', sep=';', index=False, decimal=',')



