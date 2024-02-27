---
layout: post
title: "Credit Risk Prediction"
featured-img: credit-risk
mathjax: true
categories: [Machine Learning, Data Science]
summary: This is a German credit risk dataset that can be found on Kaggle. My goal is to create a predictive model, use this model to generate a score for each client, and ultimately classify clients into risk profiles, differentiating between the riskiest and least risky
---
This is a German credit risk dataset that can be found on Kaggle [German Risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk). My goal is to create a predictive model, use this model to generate a score for each client, and ultimately classify clients into risk profiles, differentiating between the riskiest and least risky

# Tables of Content:

**1. [Introduction](#Introduction)**

**2. [Dataset](#Dataset)** 

**3. [EDA](#EDA)**

**5. [Preprocess](#Preprocessing)** 

**6.  [Model](#Training)** 

**7.  [Hyperparameter Optimization using Optuna](#Hyperparameter)**

**8. [Ranking the final model](#Ranking)**

# Introduction
Context

Each person is classified as having good or bad credit risk according to the set of attributes. The selected attributes are:
* Age (numeric)
* Sex (text: male, female)
* Job (numeric: 0 - unskilled and non-resident, 1 - unskilled and resident, 2 - skilled, 3 - highly skilled)
* Housing (text: own, rent, or free)
* Saving accounts (text - little, moderate, quite rich, rich)
* Checking account (text - little, moderate, rich)
* Credit amount (numeric, in DM)
* Duration (numeric, in month)
* Purpose(text: car, furniture/equipment, radio/TV, domestic appliances, repairs, education, business, vacation/others)
* Risk (Value target - Good or Bad Risk)

The business team came to you because they want to understand the behavior and the profile of the most risk clients and our goal here is to create a predictive model to help them

My goal here is to create a prediction model. I'll use Optuna for hyperparameter optimization then 'rank' the customer in scores and then use the shap to identify each variable as the most important

# Dataset


```python
import numpy as np
import pandas as pd
import sys
import timeit
import gc
import sklearn
from sklearn.model_selection import KFold
import seaborn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import lightgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import optuna
import matplotlib.pylab as plt
import seaborn as sns
import plotly.offline as py 
py.init_notebook_mode(connected=True) 
import plotly.graph_objs as go
import plotly.tools as tls 
from collections import Counter
```

```python
# read the dataset
df = pd.read_csv('german_credit_data.csv')
```


```python
df
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Sex</th>
      <th>Job</th>
      <th>Housing</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Purpose</th>
      <th>Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>67</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>NaN</td>
      <td>little</td>
      <td>1169</td>
      <td>6</td>
      <td>radio/TV</td>
      <td>good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>female</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>moderate</td>
      <td>5951</td>
      <td>48</td>
      <td>radio/TV</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>49</td>
      <td>male</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>2096</td>
      <td>12</td>
      <td>education</td>
      <td>good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>45</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>7882</td>
      <td>42</td>
      <td>furniture/equipment</td>
      <td>good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>53</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>4870</td>
      <td>24</td>
      <td>car</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>995</td>
      <td>31</td>
      <td>female</td>
      <td>1</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>1736</td>
      <td>12</td>
      <td>furniture/equipment</td>
      <td>good</td>
    </tr>
    <tr>
      <th>996</th>
      <td>996</td>
      <td>40</td>
      <td>male</td>
      <td>3</td>
      <td>own</td>
      <td>little</td>
      <td>little</td>
      <td>3857</td>
      <td>30</td>
      <td>car</td>
      <td>good</td>
    </tr>
    <tr>
      <th>997</th>
      <td>997</td>
      <td>38</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>little</td>
      <td>NaN</td>
      <td>804</td>
      <td>12</td>
      <td>radio/TV</td>
      <td>good</td>
    </tr>
    <tr>
      <th>998</th>
      <td>998</td>
      <td>23</td>
      <td>male</td>
      <td>2</td>
      <td>free</td>
      <td>little</td>
      <td>little</td>
      <td>1845</td>
      <td>45</td>
      <td>radio/TV</td>
      <td>bad</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>27</td>
      <td>male</td>
      <td>2</td>
      <td>own</td>
      <td>moderate</td>
      <td>moderate</td>
      <td>4576</td>
      <td>45</td>
      <td>car</td>
      <td>good</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 11 columns</p>
</div>



* Looking at the Type of Data
* Null Numbers or/and Unique values


```python
# knowing the shape of the data and search for missing
print(df.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 11 columns):
     #   Column            Non-Null Count  Dtype 
    ---  ------            --------------  ----- 
     0   Unnamed: 0        1000 non-null   int64 
     1   Age               1000 non-null   int64 
     2   Sex               1000 non-null   object
     3   Job               1000 non-null   int64 
     4   Housing           1000 non-null   object
     5   Saving accounts   817 non-null    object
     6   Checking account  606 non-null    object
     7   Credit amount     1000 non-null   int64 
     8   Duration          1000 non-null   int64 
     9   Purpose           1000 non-null   object
     10  Risk              1000 non-null   object
    dtypes: int64(5), object(6)
    memory usage: 86.1+ KB
    None



```python
# looking unique values
print(df.nunique())
```

    Unnamed: 0          1000
    Age                   53
    Sex                    2
    Job                    4
    Housing                3
    Saving accounts        4
    Checking account       3
    Credit amount        921
    Duration              33
    Purpose                8
    Risk                   2
    dtype: int64


# EDA
Let's start looking through the target variable and their distribution, here I'll show only some variables that I thought they have some interesting distribution, to see the others look the Notebook in the GitHub Repository

```python
df_age = df['Age'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Age'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Age'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_age,
    histnorm='probability',
    name="Overall Age"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Age", 
    xaxis = dict(title="Age")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![Age](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/age.png?raw=true)
We can see that people with Bad Credit tend to more youth

```python
df_housing = df['Housing'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Housing'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Housing'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_housing,
    histnorm='probability',
    name="Overall Housing"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Housing", 
    xaxis = dict(title="Housing")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![House](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/housing.png?raw=true)
People who own a house have better credit.


```python
df_saving = df['Saving accounts'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Saving accounts'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Saving accounts'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_saving,
    histnorm='probability',
    name="Overall saving"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Saving", 
    xaxis = dict(title="Saving")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![Saving](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/saving.png?raw=true)
People with more savings accounts also have better credit

```
df_checking = df['Checking account'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Checking account'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Checking account'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_checking,
    histnorm='probability',
    name="Overall checking account"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Checking Account", 
    xaxis = dict(title="Checking Account")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![Checking](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/checking.png?raw=true)

The same here, people with more checking account has better credit

```ptyhon
df_credit = df['Credit amount'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Credit amount'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Credit amount'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_credit,
    histnorm='probability',
    name="Overall Credit amount"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Credit amount", 
    xaxis = dict(title="Credit amount")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![Credit](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/credit_amount.png?raw=true)
People with more than 4k in credit amount have worse credit than people with less

```python
df_purpose = df['Purpose'].values.tolist()
df_good = df.loc[df["Risk"] == 'good']['Purpose'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Purpose'].values.tolist()

hist_1 = go.Histogram(
    x=df_good,
    histnorm='probability',
    name="Good Credit"
)

hist_2 = go.Histogram(
    x=df_bad,
    histnorm='probability',
    name="Bad Credit"
)

hist_3 = go.Histogram(
    x=df_purpose,
    histnorm='probability',
    name="Overall Purpose"
)

data = [hist_1, hist_2, hist_3]

layout = dict(
    title="Type of Credit by Purpose", 
    xaxis = dict(title="Purpose")
)

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='custom-sized-subplot-with-subplot-titles')
```
![Purpose](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/purpose.png?raw=true)
People that the purpose is to buy radio/TV have a better credit

Now let's see the distribution using two variables

```python
df_good = df.loc[df["Risk"] == 'good']['Checking account'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Checking account'].values.tolist()
box_1 = go.Box(
    x=df_good,
    y=df['Credit amount'],
    name="Good Credit"
)

box_2 = go.Box(
    x=df_bad,
    y=df['Credit amount'],
    name="Bad Credit"
)



data = [box_1, box_2]

layout = go.Layout(
    yaxis=dict(
        title='Credit Amount by Checking Account'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')
```
![Credit](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/amount_checking.png?raw=true)
The credit amount is also less in rich people (checking account), even in those bad credit

```python

df_good = df.loc[df["Risk"] == 'good']['Job'].values.tolist()
df_bad = df.loc[df["Risk"] == 'bad']['Job'].values.tolist()
box_1 = go.Box(
    x=df_good,
    y=df['Credit amount'],
    name="Good Credit"
)

box_2 = go.Box(
    x=df_bad,
    y=df['Credit amount'],
    name="Bad Credit"
)



data = [box_1, box_2]

layout = go.Layout(
    yaxis=dict(
        title='Job'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')
```
![Credit](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/amount_job.png?raw=true)
Unskilled and non-residents with bad credit have more credit amount than others

# Preprocessing

```python
df.dtypes
```




    Unnamed: 0           int64
    Age                  int64
    Sex                 object
    Job                  int64
    Housing             object
    Saving accounts     object
    Checking account    object
    Credit amount        int64
    Duration             int64
    Purpose             object
    Risk                object
    dtype: object




```python
df.isna().sum()
```




    Unnamed: 0            0
    Age                   0
    Sex                   0
    Job                   0
    Housing               0
    Saving accounts     183
    Checking account    394
    Credit amount         0
    Duration              0
    Purpose               0
    Risk                  0
    dtype: int64



We will use one-hot encoding for the sex, housing, and purpose variables.


```python
one_hot = {
    "Sex": "sex",
    "Housing": "hous",
    "Purpose": "purp"
}
```

And ordinal encoding for the others


```python
ordinal_encoding = {
    "Saving accounts": {
        None: 0,
        "little": 1,
        "moderate": 2,
        "quite rich": 3,
        "rich": 4,
    },
    "Checking account": {
        None: 0,
        "little": 1,
        "moderate": 2,
        "rich": 3,
    },
    "Risk": {
        "bad": 1,
        "good": 0,
    }
}
```


```python
def one_hot_enconding(df, col_prefix: dict):
    df = df.copy()
    for col, prefix in col_prefix.items():
        df = pd.get_dummies(data=df, prefix=prefix, columns=[col])
    return df
```


```python
def encode_ordinal(df, custom_ordinals: dict):
    df = df.copy()
    for col, map_dict in custom_ordinals.items():
        df[col] = df[col].replace(map_dict)
    return df
```


```python
df_encode = df.copy()
df_encode = one_hot_enconding(df_encode, one_hot)
df_encode = encode_ordinal(df_encode, ordinal_encoding)
```


```python
df_encode
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Age</th>
      <th>Job</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>Risk</th>
      <th>sex_female</th>
      <th>sex_male</th>
      <th>...</th>
      <th>hous_own</th>
      <th>hous_rent</th>
      <th>purp_business</th>
      <th>purp_car</th>
      <th>purp_domestic appliances</th>
      <th>purp_education</th>
      <th>purp_furniture/equipment</th>
      <th>purp_radio/TV</th>
      <th>purp_repairs</th>
      <th>purp_vacation/others</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>67</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1169</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>22</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5951</td>
      <td>48</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>49</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2096</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>45</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>7882</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>53</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>4870</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>995</td>
      <td>31</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1736</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>996</td>
      <td>40</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3857</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>997</td>
      <td>38</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>804</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>998</td>
      <td>23</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1845</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>999</td>
      <td>27</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>4576</td>
      <td>45</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 21 columns</p>
</div>




```python
df_encode.dtypes
```




    Unnamed: 0                  int64
    Age                         int64
    Job                         int64
    Saving accounts             int64
    Checking account            int64
    Credit amount               int64
    Duration                    int64
    Risk                        int64
    sex_female                  uint8
    sex_male                    uint8
    hous_free                   uint8
    hous_own                    uint8
    hous_rent                   uint8
    purp_business               uint8
    purp_car                    uint8
    purp_domestic appliances    uint8
    purp_education              uint8
    purp_furniture/equipment    uint8
    purp_radio/TV               uint8
    purp_repairs                uint8
    purp_vacation/others        uint8
    dtype: object




```python
df_encode.isna().sum()
```




    Unnamed: 0                  0
    Age                         0
    Job                         0
    Saving accounts             0
    Checking account            0
    Credit amount               0
    Duration                    0
    Risk                        0
    sex_female                  0
    sex_male                    0
    hous_free                   0
    hous_own                    0
    hous_rent                   0
    purp_business               0
    purp_car                    0
    purp_domestic appliances    0
    purp_education              0
    purp_furniture/equipment    0
    purp_radio/TV               0
    purp_repairs                0
    purp_vacation/others        0
    dtype: int64




```python
# Check for duplicate rows
df.duplicated().sum()
```




    0




```python
df_encode.corr()['Risk'].sort_values()
```




    hous_own                   -0.134589
    purp_radio/TV              -0.106922
    Age                        -0.091127
    sex_male                   -0.075493
    Saving accounts            -0.033871
    purp_domestic appliances    0.008016
    purp_repairs                0.020828
    purp_furniture/equipment    0.020971
    purp_car                    0.022621
    purp_vacation/others        0.028058
    Job                         0.032735
    Unnamed: 0                  0.034606
    purp_business               0.036129
    purp_education              0.049085
    sex_female                  0.075493
    hous_free                   0.081556
    hous_rent                   0.092785
    Credit amount               0.154739
    Checking account            0.197788
    Duration                    0.214927
    Risk                        1.000000
    Name: Risk, dtype: float64




```python
df_encode.columns
```




    Index(['Unnamed: 0', 'Age', 'Job', 'Saving accounts', 'Checking account',
           'Credit amount', 'Duration', 'Risk', 'sex_female', 'sex_male',
           'hous_free', 'hous_own', 'hous_rent', 'purp_business', 'purp_car',
           'purp_domestic appliances', 'purp_education',
           'purp_furniture/equipment', 'purp_radio/TV', 'purp_repairs',
           'purp_vacation/others'],
          dtype='object')



Getting all the coluns that we are going to use in our model.


```python
model_cols = ['Age', 'Job', 'Saving accounts', 'Checking account',
       'Credit amount', 'Duration', 'sex_female', 'sex_male',
       'hous_free', 'hous_own', 'hous_rent', 'purp_business', 'purp_car',
       'purp_domestic appliances', 'purp_education',
       'purp_furniture/equipment', 'purp_radio/TV', 'purp_repairs',
       'purp_vacation/others']
```


```python
df_encode.loc[df_encode['Risk']==0].mean()
```




    Unnamed: 0                   492.960000
    Age                           36.224286
    Job                            1.890000
    Saving accounts                1.211429
    Checking account               0.877143
    Credit amount               2985.457143
    Duration                      19.207143
    Risk                           0.000000
    sex_female                     0.287143
    sex_male                       0.712857
    hous_free                      0.091429
    hous_own                       0.752857
    hous_rent                      0.155714
    purp_business                  0.090000
    purp_car                       0.330000
    purp_domestic appliances       0.011429
    purp_education                 0.051429
    purp_furniture/equipment       0.175714
    purp_radio/TV                  0.311429
    purp_repairs                   0.020000
    purp_vacation/others           0.010000
    dtype: float64




```python
df_encode.loc[df_encode['Risk']==1].mean()
```




    Unnamed: 0                   514.760000
    Age                           33.963333
    Job                            1.936667
    Saving accounts                1.140000
    Checking account               1.290000
    Credit amount               3938.126667
    Duration                      24.860000
    Risk                           1.000000
    sex_female                     0.363333
    sex_male                       0.636667
    hous_free                      0.146667
    hous_own                       0.620000
    hous_rent                      0.233333
    purp_business                  0.113333
    purp_car                       0.353333
    purp_domestic appliances       0.013333
    purp_education                 0.076667
    purp_furniture/equipment       0.193333
    purp_radio/TV                  0.206667
    purp_repairs                   0.026667
    purp_vacation/others           0.016667
    dtype: float64



some correlation


```python
df_encode.astype(float).corr().abs().sort_values(by='Risk',ascending=False)['Risk']
```




    Risk                        1.000000
    Duration                    0.214927
    Checking account            0.197788
    Credit amount               0.154739
    hous_own                    0.134589
    purp_radio/TV               0.106922
    hous_rent                   0.092785
    Age                         0.091127
    hous_free                   0.081556
    sex_female                  0.075493
    sex_male                    0.075493
    purp_education              0.049085
    purp_business               0.036129
    Unnamed: 0                  0.034606
    Saving accounts             0.033871
    Job                         0.032735
    purp_vacation/others        0.028058
    purp_car                    0.022621
    purp_furniture/equipment    0.020971
    purp_repairs                0.020828
    purp_domestic appliances    0.008016
    Name: Risk, dtype: float64



Duration, checking account, credit amount, and owning house have the most Corr


```python
plt.figure(figsize=(15,15))
sns.heatmap(df_encode.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True,  linecolor='white', annot=True)
plt.show()
```


    
![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_60_0.png?raw=true)
    


# Training some Models


```python
X = df_encode.loc[:,model_cols]
y = df_encode.loc[:,'Risk']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape 
```




    ((700, 19), (300, 19), (700,), (300,))



Here we are going to train 5 models


```python
# prepare models
lgbmparameters = {'verbose': -1}
models = []
models.append(('XGB', XGBClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('LGBM', LGBMClassifier(**lgbmparameters)))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
# evaluate each model in turn
results = []
names = []
scoring = 'roc_auc'
n_splits = 10
for name, model in models:
        kfold = KFold(n_splits=n_splits)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        
```

    XGB: 0.720764 (0.055151)
    CART: 0.598842 (0.066576)
    LGBM: 0.733702 (0.062989)
    RF: 0.722074 (0.052949)
    NB: 0.685554 (0.082030)



```python
box_1 = go.Box(
    x=n_splits*['XGB'],
    y=results[0],
    name="XGB"
)
box_2 = go.Box(
    x=n_splits*['CART'],
    y=results[1],
    name="CART"
)
box_3 = go.Box(
    x=n_splits*['LGBM'],
    y=results[2],
    name="LGBM"
)
box_4 = go.Box(
    x=n_splits*['RF'],
    y=results[3],
    name="RF"
)
box_5 = go.Box(
    x=n_splits*['NB'],
    y=results[4],
    name="NB"
)

data = [box_1, box_2, box_3, box_4, box_5]
layout = go.Layout(
    yaxis=dict(
        title='Model Results'
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='box-age-cat')
```
![Models](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/models_train.png?raw=true)

The best models were RandomForest and LGBM, we are going to train this model and use Optuna for hyperparameter optimization.

```python
lgbm_model = LGBMClassifier(**lgbmparameters).fit(X_train, y_train)
y_prob_lgbm = lgbm_model.predict_proba(X_test)
```


```python
print('For the LGBM Model, the test AUC is: '+str(roc_auc_score(y_test,y_prob_lgbm[:,1])))
print('For the LGBM Model, the test Accu is: '+ str(accuracy_score(y_test,y_prob_lgbm[:,1].round())))
```

    For the LGBM Model, the test AUC is: 0.7434670592565329
    For the LGBM Model, the test Accu is: 0.7533333333333333



```python
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_prob_rf = rf_model.predict_proba(X_test)
```


```python
print('For the RandomForest Model, the test AUC is: '+str(roc_auc_score(y_test,y_prob_rf[:,1])))
print('For the RandomForest Model, the test Accu is: '+ str(accuracy_score(y_test,y_prob_rf[:,1].round())))
```

    For the RandomForest Model, the test AUC is: 0.7218308007781692
    For the RandomForest Model, the test Accu is: 0.7266666666666667


# Hyperparameter Optimization using Optuna


```python
def auc_ks_metric(y_test, y_prob):
    '''
    Input:
        y_prob: model predict prob
        y_test: target
    Output: Metrics of validation
        auc, ks (Kolmogorov-Smirnov)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)
    auc = metrics.auc(fpr, tpr)
    ks = max(tpr - fpr)
    return auc, ks

def objective(trial, X_train, y_train, X_test, y_test, balanced, method):
    '''
    Input:
        trial: trial of the test
        X_train:
        y_train:
        X_test:
        y_test:
        balanced:balanced or None
        method: XGBoost, CatBoost or LGBM
    Output: Metrics of validation
        auc, ks, log_loss
        auc_logloss_ks(y_test, y_pred)[0]
    '''
    gc.collect()
    if method=='LGBM':
        param_grid = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
                      'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                      'lambda_l1': trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                      'lambda_l2': trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                      'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
                      'max_depth': trial.suggest_int('max_depth', 5, 64),
                      'feature_fraction': trial.suggest_float("feature_fraction", 0.4, 1.0),
                      'bagging_fraction': trial.suggest_float("bagging_fraction", 0.4, 1.0),
                      'bagging_freq': trial.suggest_int("bagging_freq", 1, 7),
                      'verbose': -1
  
                     }
        model = LGBMClassifier(**param_grid,tree_method='gpu_hist',gpu_id=0)

        print('LGBM - Optimization using optuna')
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:,1]
    if method=='RF':
        param_grid = {
                      'max_features': trial.suggest_int('max_features', 4, 20),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 25),
                      'max_depth': trial.suggest_int('max_depth', 5, 64),
                      'min_samples_split': trial.suggest_int("min_samples_split", 2, 30),
                      'n_estimators': trial.suggest_int("n_estimators", 100, 2000)
  
                     }
        model = RandomForestClassifier(**param_grid)

        print('RandomForest - Optimization using optuna')
        model.fit(X_train, y_train)
        
        y_pred = model.predict_proba(X_test)[:,1]
        
    if method=='XGBoost':
        param_grid = {'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.1, log=True),
                      'max_depth': trial.suggest_int('max_depth', 3, 16),
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                      'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log = True),
                      'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log = True),
                      'lambda': trial.suggest_float('lambda', 0.0001, 10.0, log = True),
                      'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.8),
                      'booster': 'gbtree',
                      'random_state': 42,
                     }
        model = XGBClassifier(**param_grid,tree_method='gpu_hist',gpu_id=0)
        print('XGBoost - Optimization using optuna')
        model.fit(X_train, y_train,verbose=False)
        y_pred = model.predict_proba(X_test)[:,1]
    
    auc_res = auc_ks_metric(y_test, y_pred)[0]
    print('auc:'+str(auc_res))
    return auc_ks_metric(y_test, y_pred)[0]

def tuning(X_train, y_train, X_test, y_test, balanced, method):
    '''
    Input:
        trial: 
        x_train:
        y_train:
        X_test:
        y_test:
        balanced:balanced or not balanced
        method: XGBoost, CatBoost or LGBM
    Output: Metrics of validation
        auc, ks, log_loss
        auc_logloss_ks(y_test, y_pred)[0]
    '''
    study = optuna.create_study(direction='maximize', study_name=method+' Classifier')
    func = lambda trial: objective(trial, X_train, y_train, X_test, y_test, balanced, method)
    print('Starting the optimization')
    time_max_tuning = 60*30 # max time in seconds to stop
    study.optimize(func, timeout=time_max_tuning)
    return study

def train(X_train, y_train, X_test, y_test, balanced, method):
    '''
    Input:
        X_train:
        y_train:
        X_test:
        y_test:
        balanced:balanced or None
        method: XGBoost, CatBoost or LGBM
    Output: predict model
    '''
    print('Tuning')
    study = tuning(X_train, y_train, X_test, y_test, balanced, method)
    if method=='LGBM':
        model = LGBMClassifier(**study.best_params)
        print('Last Fit')
        model.fit(X_train, y_train, eval_set=[(X_test,y_test)],
                 callbacks = [lightgbm.early_stopping(stopping_rounds=100), lightgbm.log_evaluation(period=5000)])
    if method=='XGBoost':
        model = XGBClassifier(**study.best_params)
        print('Last Fit')
        model.fit(X_train, y_train, eval_set=[(X_test,y_test)],
                 early_stopping_rounds=100,verbose = False)
    if method=='RF':
        model = RandomForestClassifier(**study.best_params)
        print('Last Fit')
        model.fit(X_train, y_train)
    return model, study
```


```python
lgbm_model, study_lgbm = train(X_train, y_train, X_test, y_test, balanced='balanced', method='LGBM')
```

    [I 2023-09-25 07:48:59,504] A new study created in memory with name: LGBM Classifier
    [I 2023-09-25 07:48:59,607] Trial 0 finished with value: 0.7707818497292181 and parameters: {'learning_rate': 0.04564317750022488, 'num_leaves': 254, 'lambda_l1': 0.17602474289716696, 'lambda_l2': 2.936736356867574, 'min_data_in_leaf': 92, 'max_depth': 41, 'feature_fraction': 0.630771183128692, 'bagging_fraction': 0.8791863972428846, 'bagging_freq': 5}. Best is trial 0 with value: 0.7707818497292181.
    [I 2023-09-25 07:48:59,695] Trial 1 finished with value: 0.7673905042326096 and parameters: {'learning_rate': 0.03163545356039165, 'num_leaves': 93, 'lambda_l1': 5.331694642994698e-07, 'lambda_l2': 0.0016117988828970487, 'min_data_in_leaf': 62, 'max_depth': 17, 'feature_fraction': 0.5207793700543741, 'bagging_fraction': 0.6988688771949946, 'bagging_freq': 2}. Best is trial 0 with value: 0.7707818497292181.


    Tuning
    Starting the optimization
    LGBM - Optimization using optuna
    auc:0.7707818497292181
    LGBM - Optimization using optuna
    auc:0.7673905042326096
    LGBM - Optimization using optuna


    [I 2023-09-25 07:48:59,778] Trial 2 finished with value: 0.7639202902360797 and parameters: {'learning_rate': 0.00032903168736575527, 'num_leaves': 221, 'lambda_l1': 0.013182095631109458, 'lambda_l2': 0.004903360053701577, 'min_data_in_leaf': 50, 'max_depth': 35, 'feature_fraction': 0.81375010947157, 'bagging_fraction': 0.47255383236900694, 'bagging_freq': 6}. Best is trial 0 with value: 0.7707818497292181.
    [I 2023-09-25 07:48:59,870] Trial 3 finished with value: 0.7488301172511699 and parameters: {'learning_rate': 0.07583419812542502, 'num_leaves': 227, 'lambda_l1': 0.001263229821256988, 'lambda_l2': 0.6714031923624736, 'min_data_in_leaf': 23, 'max_depth': 48, 'feature_fraction': 0.47371647441012454, 'bagging_fraction': 0.5357410570154348, 'bagging_freq': 3}. Best is trial 0 with value: 0.7707818497292181.
    [I 2023-09-25 07:48:59,958] Trial 4 finished with value: 0.7586361007413639 and parameters: {'learning_rate': 0.0001610953746996855, 'num_leaves': 228, 'lambda_l1': 4.74483283120879, 'lambda_l2': 0.00011656154418021165, 'min_data_in_leaf': 88, 'max_depth': 60, 'feature_fraction': 0.5768682497889083, 'bagging_fraction': 0.9363888441877074, 'bagging_freq': 5}. Best is trial 0 with value: 0.7707818497292181.


    auc:0.7639202902360797
    LGBM - Optimization using optuna
    auc:0.7488301172511699
    LGBM - Optimization using optuna
    auc:0.7586361007413639
    LGBM - Optimization using optuna


    [I 2023-09-25 07:49:00,049] Trial 5 finished with value: 0.7586098112413902 and parameters: {'learning_rate': 0.0001249838804070837, 'num_leaves': 62, 'lambda_l1': 1.0950722639611093e-08, 'lambda_l2': 1.6247452419757427, 'min_data_in_leaf': 88, 'max_depth': 20, 'feature_fraction': 0.7390198578425595, 'bagging_fraction': 0.5451921961124094, 'bagging_freq': 4}. Best is trial 0 with value: 0.7707818497292181.
    [I 2023-09-25 07:49:00,135] Trial 6 finished with value: 0.7462274567537727 and parameters: {'learning_rate': 0.021165567217590234, 'num_leaves': 219, 'lambda_l1': 2.634681758289909e-06, 'lambda_l2': 2.1170808617536877e-06, 'min_data_in_leaf': 70, 'max_depth': 52, 'feature_fraction': 0.8886838094462373, 'bagging_fraction': 0.7972849312237408, 'bagging_freq': 5}. Best is trial 0 with value: 0.7707818497292181.
    [I 2023-09-25 07:49:00,229] Trial 7 finished with value: 0.7455702192544299 and parameters: {'learning_rate': 0.028054358171711414, 'num_leaves': 106, 'lambda_l1': 0.2794244438193041, 'lambda_l2': 0.02038032703976737, 'min_data_in_leaf': 41, 'max_depth': 8, 'feature_fraction': 0.9320015422653435, 'bagging_fraction': 0.97584085801718, 'bagging_freq': 1}. Best is trial 0 with value: 0.7707818497292181.

    [I 2023-09-25 08:18:59,736] Trial 5841 finished with value: 0.7210946947789053 and parameters: {'learning_rate': 0.0017454581670019865, 'num_leaves': 71, 'lambda_l1': 0.00030822208833371846, 'lambda_l2': 0.001178948478008598, 'min_data_in_leaf': 82, 'max_depth': 64, 'feature_fraction': 0.9975320930746641, 'bagging_fraction': 0.9612726835980929, 'bagging_freq': 1}. Best is trial 3444 with value: 0.7828487302171513.


    LGBM - Optimization using optuna
    auc:0.7210946947789053
    Last Fit

```python
y_prob_lgbm = lgbm_model.predict_proba(X_test)
```

    [LightGBM] [Warning] min_data_in_leaf is set=81, min_child_samples=20 will be ignored. Current value: min_data_in_leaf=81
    [LightGBM] [Warning] feature_fraction is set=0.5174527298564775, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.5174527298564775
    [LightGBM] [Warning] lambda_l2 is set=0.0003124668197733085, reg_lambda=0.0 will be ignored. Current value: lambda_l2=0.0003124668197733085
    [LightGBM] [Warning] lambda_l1 is set=2.524882043205203e-06, reg_alpha=0.0 will be ignored. Current value: lambda_l1=2.524882043205203e-06
    [LightGBM] [Warning] bagging_fraction is set=0.7962210156422196, subsample=1.0 will be ignored. Current value: bagging_fraction=0.7962210156422196
    [LightGBM] [Warning] bagging_freq is set=1, subsample_freq=0 will be ignored. Current value: bagging_freq=1



```python
print('For the LGBM Model, the test AUC is: '+str(roc_auc_score(y_test,y_prob_lgbm[:,1])))
print('For the LGBM Model, the KS is: '+str(auc_ks_metric(y_test,y_prob_lgbm[:,1])[1]))
print('For the LGBM Model, the test Accu is: '+ str(accuracy_score(y_test,y_prob_lgbm[:,1].round())))
```

    For the LGBM Model, the test AUC is: 0.7828487302171513
    For the LGBM Model, the KS is: 0.4945580735054419
    For the LGBM Model, the test Accu is: 0.7133333333333334



```python
confusion_hard = confusion_matrix(y_test, y_prob_lgbm[:,1].round())
plt.figure(figsize=(8, 6))
ax = sns.heatmap(confusion_hard, vmin=10, vmax=190,annot = True, fmt='d')
ax.set_title('Confusion Matrix')
```

Confusion Matrix in LGBM
![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_77_1.png?raw=true)
    



```python
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_prob_lgbm[:,1])

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_78_0.png?raw=true)
    

```python
optuna.visualization.plot_param_importances(study_lgbm)
```


```python
rf_model, study_rf = train(X_train, y_train, X_test, y_test, balanced='balanced', method='RF')
```

    [I 2023-09-25 08:32:34,955] A new study created in memory with name: RF Classifier


    Tuning
    Starting the optimization
    RandomForest - Optimization using optuna


    [I 2023-09-25 08:32:35,237] Trial 0 finished with value: 0.7599768652400232 and parameters: {'max_features': 7, 'min_samples_leaf': 21, 'max_depth': 25, 'min_samples_split': 24, 'n_estimators': 231}. Best is trial 0 with value: 0.7599768652400232.


    auc:0.7599768652400232
    RandomForest - Optimization using optuna


    [I 2023-09-25 08:32:36,115] Trial 1 finished with value: 0.7550344392449656 and parameters: {'max_features': 6, 'min_samples_leaf': 25, 'max_depth': 63, 'min_samples_split': 27, 'n_estimators': 1132}. Best is trial 0 with value: 0.7599768652400232.

    Last Fit



```python
y_prob_rf = rf_model.predict_proba(X_test)
```


```python
print('For the RandomForest Model, the test AUC is: '+str(roc_auc_score(y_test,y_prob_rf[:,1])))
print('For the RandomForest, the KS is: '+str(auc_ks_metric(y_test,y_prob_rf[:,1])[1]))
print('For the RandomForest Model, the test Accu is: '+ str(accuracy_score(y_test,y_prob_rf[:,1].round())))
```

    For the RandomForest Model, the test AUC is: 0.7489352752510646
    For the RandomForest, the KS is: 0.40080971659919035
    For the RandomForest Model, the test Accu is: 0.7233333333333334



```python
confusion_hard = confusion_matrix(y_test, y_prob_rf[:,1].round())
plt.figure(figsize=(8, 6))
ax = sns.heatmap(confusion_hard, vmin=10, vmax=190,annot = True, fmt='d')
ax.set_title('Confusion Matrix')
```


Confusion Matrix in RandomForest
![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_83_1.png?raw=true)
    

```python
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_prob_rf[:,1])

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```
    
![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_84_0.png?raw=true)
    

LGBM model has a better performance after the optimization using Optuna, so we'll this model as our final model.

# Ranking the final model


```python
import shap
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train,show=False)
```


SHAP results in LGBM
![png](https://github.com/joaomh/joaomh.github.io/blob/main/assets/post_img/german-credit-risk/output_88_1.png?raw=true)    

```python
df_test = pd.concat([X_test, y_test],axis=1)
```


```python
shap_test = explainer.shap_values(X_test)
```

    LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray


Here we are going to create new variables from shap. The goal is to make more easier to use our final model, for example we want to select the clients with high scores and have more cash in their checking account


```python
def shap_col(shap_):
    col = ['Age', 'Job', 'Saving accounts', 'Checking account', 'Credit amount',
       'Duration', 'sex_female', 'sex_male', 'hous_free', 'hous_own',
       'hous_rent', 'purp_business', 'purp_car', 'purp_domestic appliances',
       'purp_education', 'purp_furniture/equipment', 'purp_radio/TV',
       'purp_repairs', 'purp_vacation/others']
    df_shap = pd.DataFrame(shap_test[1],columns=col)
#     shap_cols = {}
#     shap_cols['shap_1'] = np.nan
#     shap_cols['shap_2'] = np.nan
#     shap_cols['shap_3'] = np.nan
#     shap_cols['shap_4'] = np.nan
#     shap_cols['shap_5'] = np.nan
#     shap_cols['shap_6'] = np.nan
    df_shap.loc[df_shap['Checking account']>0.2, 'shap_1'] = 'Little Check Account'
    df_shap.loc[df_shap['Duration']>0.2, 'shap_2'] = 'More Credit Duration'
    df_shap.loc[df_shap['Credit amount']>0.2, 'shap_3'] = 'More Credit Amount'
    df_shap.loc[df_shap['Age']>0.2, 'shap_4'] = 'More Junior Client'
    df_shap.loc[df_shap['hous_own']>0.2, 'shap_5'] = 'Have House'
    df_shap.loc[df_shap['purp_radio/TV']>0.2, 'shap_6'] = 'The purpose is to buy Radio/TV'
    df_shap.loc[df_shap['Checking account']<-0.2, 'shap_7'] = 'Moderate/Rich Check Account'
    df_shap.loc[df_shap['Duration']<-0.2, 'shap_8'] = 'Less Credit Duration'
    df_shap.loc[df_shap['Credit amount']<-0.2, 'shap_9'] = 'Less Credit Amount'
    df_shap.loc[df_shap['Age']<-0.2, 'shap_10'] = 'More Senior Client'
    df_shap.loc[df_shap['hous_own']<-0.2, 'shap_11'] = 'Does not have House'
# pd.DataFrame(shap_test[1],columns=col).apply(shap_col, axis=1, result_type='expand')

    return df_shap[['shap_1','shap_2','shap_3','shap_4','shap_5','shap_6',
                    'shap_7','shap_8','shap_9','shap_10','shap_11']]
```


```python
df_shap_arg = pd.DataFrame(shap_col(shap_test[1]))
```


```python
df_shap_arg
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shap_1</th>
      <th>shap_2</th>
      <th>shap_3</th>
      <th>shap_4</th>
      <th>shap_5</th>
      <th>shap_6</th>
      <th>shap_7</th>
      <th>shap_8</th>
      <th>shap_9</th>
      <th>shap_10</th>
      <th>shap_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Little Check Account</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Little Check Account</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Little Check Account</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Senior Client</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>297</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>More Credit Amount</td>
      <td>NaN</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Little Check Account</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 11 columns</p>
</div>




```python
df_final = pd.concat([df_test.reset_index() ,df_shap_arg],axis=1)
```


```python
df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>Age</th>
      <th>Job</th>
      <th>Saving accounts</th>
      <th>Checking account</th>
      <th>Credit amount</th>
      <th>Duration</th>
      <th>sex_female</th>
      <th>sex_male</th>
      <th>hous_free</th>
      <th>...</th>
      <th>shap_2</th>
      <th>shap_3</th>
      <th>shap_4</th>
      <th>shap_5</th>
      <th>shap_6</th>
      <th>shap_7</th>
      <th>shap_8</th>
      <th>shap_9</th>
      <th>shap_10</th>
      <th>shap_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>521</td>
      <td>24</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>3190</td>
      <td>18</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>737</td>
      <td>35</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>4380</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>740</td>
      <td>32</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>2325</td>
      <td>24</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>More Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>660</td>
      <td>23</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1297</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>411</td>
      <td>35</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>7253</td>
      <td>33</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Senior Client</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>468</td>
      <td>26</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>2764</td>
      <td>33</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>NaN</td>
      <td>Less Credit Amount</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>296</th>
      <td>935</td>
      <td>30</td>
      <td>3</td>
      <td>2</td>
      <td>2</td>
      <td>1919</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>More Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>297</th>
      <td>428</td>
      <td>20</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1313</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Moderate/Rich Check Account</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>298</th>
      <td>7</td>
      <td>35</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>6948</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>More Credit Duration</td>
      <td>More Credit Amount</td>
      <td>NaN</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>299</th>
      <td>155</td>
      <td>20</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1282</td>
      <td>12</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Credit Duration</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 32 columns</p>
</div>




```python
df_final = df_final.fillna(0)
```

creating a column score from our predeict_proba


```python
df_final['score'] = y_prob_lgbm[:,1]
```

we will divide our clients into 5 groups based on the score, this number we can use any number to see if our model is ordering some variables


```python
df_final['rank'] = pd.qcut(df_final['score'], 5,labels = False)
```

Group by some variables


```python
df_final.groupby('rank')[['Checking account','Duration', 'Age','Credit amount', 'hous_own',
                          'Saving accounts','purp_radio/TV','purp_car','sex_male','sex_female']].agg('mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Checking account</th>
      <th>Duration</th>
      <th>Age</th>
      <th>Credit amount</th>
      <th>hous_own</th>
      <th>Saving accounts</th>
      <th>purp_radio/TV</th>
      <th>purp_car</th>
      <th>sex_male</th>
      <th>sex_female</th>
    </tr>
    <tr>
      <th>rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.083333</td>
      <td>15.350000</td>
      <td>41.300000</td>
      <td>2261.750000</td>
      <td>0.900000</td>
      <td>1.183333</td>
      <td>0.400000</td>
      <td>0.300000</td>
      <td>0.766667</td>
      <td>0.233333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.716667</td>
      <td>16.266667</td>
      <td>35.266667</td>
      <td>2352.266667</td>
      <td>0.783333</td>
      <td>1.283333</td>
      <td>0.216667</td>
      <td>0.266667</td>
      <td>0.766667</td>
      <td>0.233333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.216667</td>
      <td>18.566667</td>
      <td>36.950000</td>
      <td>3140.466667</td>
      <td>0.766667</td>
      <td>1.033333</td>
      <td>0.250000</td>
      <td>0.300000</td>
      <td>0.683333</td>
      <td>0.316667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.450000</td>
      <td>18.466667</td>
      <td>33.133333</td>
      <td>2666.100000</td>
      <td>0.683333</td>
      <td>1.283333</td>
      <td>0.266667</td>
      <td>0.383333</td>
      <td>0.716667</td>
      <td>0.283333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.383333</td>
      <td>31.783333</td>
      <td>31.700000</td>
      <td>4354.200000</td>
      <td>0.383333</td>
      <td>1.033333</td>
      <td>0.166667</td>
      <td>0.333333</td>
      <td>0.600000</td>
      <td>0.400000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_final.groupby('rank')[['Checking account','Duration', 'Age','Credit amount', 'hous_own',
                          'Saving accounts','purp_radio/TV','purp_car','sex_male','sex_female']].agg('mean').style.bar(align='mid', color=['#d65f5f', '#5fba7d'])
```




<style type="text/css">
#T_7d04a_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 5.7%, transparent 5.7%);
}
#T_7d04a_row0_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 48.3%, transparent 48.3%);
}
#T_7d04a_row0_col2, #T_7d04a_row0_col4, #T_7d04a_row0_col6, #T_7d04a_row0_col8, #T_7d04a_row1_col5, #T_7d04a_row1_col8, #T_7d04a_row3_col0, #T_7d04a_row3_col5, #T_7d04a_row3_col7, #T_7d04a_row4_col1, #T_7d04a_row4_col3, #T_7d04a_row4_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_7d04a_row0_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 51.9%, transparent 51.9%);
}
#T_7d04a_row0_col5 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 92.2%, transparent 92.2%);
}
#T_7d04a_row0_col7, #T_7d04a_row2_col7, #T_7d04a_row4_col8 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 78.3%, transparent 78.3%);
}
#T_7d04a_row0_col9, #T_7d04a_row1_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 58.3%, transparent 58.3%);
}
#T_7d04a_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 49.4%, transparent 49.4%);
}
#T_7d04a_row1_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 51.2%, transparent 51.2%);
}
#T_7d04a_row1_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 85.4%, transparent 85.4%);
}
#T_7d04a_row1_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 54.0%, transparent 54.0%);
}
#T_7d04a_row1_col4, #T_7d04a_row4_col7 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 87.0%, transparent 87.0%);
}
#T_7d04a_row1_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 54.2%, transparent 54.2%);
}
#T_7d04a_row1_col7 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 69.6%, transparent 69.6%);
}
#T_7d04a_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 83.9%, transparent 83.9%);
}
#T_7d04a_row2_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 58.4%, transparent 58.4%);
}
#T_7d04a_row2_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 89.5%, transparent 89.5%);
}
#T_7d04a_row2_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 72.1%, transparent 72.1%);
}
#T_7d04a_row2_col4 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 85.2%, transparent 85.2%);
}
#T_7d04a_row2_col5, #T_7d04a_row4_col5 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 80.5%, transparent 80.5%);
}
#T_7d04a_row2_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 62.5%, transparent 62.5%);
}
#T_7d04a_row2_col8 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 89.1%, transparent 89.1%);
}
#T_7d04a_row2_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 79.2%, transparent 79.2%);
}
#T_7d04a_row3_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 58.1%, transparent 58.1%);
}
#T_7d04a_row3_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 80.2%, transparent 80.2%);
}
#T_7d04a_row3_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 61.2%, transparent 61.2%);
}
#T_7d04a_row3_col4 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 75.9%, transparent 75.9%);
}
#T_7d04a_row3_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 66.7%, transparent 66.7%);
}
#T_7d04a_row3_col8 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 93.5%, transparent 93.5%);
}
#T_7d04a_row3_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 70.8%, transparent 70.8%);
}
#T_7d04a_row4_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 95.4%, transparent 95.4%);
}
#T_7d04a_row4_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 76.8%, transparent 76.8%);
}
#T_7d04a_row4_col4 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 42.6%, transparent 42.6%);
}
#T_7d04a_row4_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 41.7%, transparent 41.7%);
}
</style>
<table id="T_7d04a">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_7d04a_level0_col0" class="col_heading level0 col0" >Checking account</th>
      <th id="T_7d04a_level0_col1" class="col_heading level0 col1" >Duration</th>
      <th id="T_7d04a_level0_col2" class="col_heading level0 col2" >Age</th>
      <th id="T_7d04a_level0_col3" class="col_heading level0 col3" >Credit amount</th>
      <th id="T_7d04a_level0_col4" class="col_heading level0 col4" >hous_own</th>
      <th id="T_7d04a_level0_col5" class="col_heading level0 col5" >Saving accounts</th>
      <th id="T_7d04a_level0_col6" class="col_heading level0 col6" >purp_radio/TV</th>
      <th id="T_7d04a_level0_col7" class="col_heading level0 col7" >purp_car</th>
      <th id="T_7d04a_level0_col8" class="col_heading level0 col8" >sex_male</th>
      <th id="T_7d04a_level0_col9" class="col_heading level0 col9" >sex_female</th>
    </tr>
    <tr>
      <th class="index_name level0" >rank</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_7d04a_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_7d04a_row0_col0" class="data row0 col0" >0.083333</td>
      <td id="T_7d04a_row0_col1" class="data row0 col1" >15.350000</td>
      <td id="T_7d04a_row0_col2" class="data row0 col2" >41.300000</td>
      <td id="T_7d04a_row0_col3" class="data row0 col3" >2261.750000</td>
      <td id="T_7d04a_row0_col4" class="data row0 col4" >0.900000</td>
      <td id="T_7d04a_row0_col5" class="data row0 col5" >1.183333</td>
      <td id="T_7d04a_row0_col6" class="data row0 col6" >0.400000</td>
      <td id="T_7d04a_row0_col7" class="data row0 col7" >0.300000</td>
      <td id="T_7d04a_row0_col8" class="data row0 col8" >0.766667</td>
      <td id="T_7d04a_row0_col9" class="data row0 col9" >0.233333</td>
    </tr>
    <tr>
      <th id="T_7d04a_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_7d04a_row1_col0" class="data row1 col0" >0.716667</td>
      <td id="T_7d04a_row1_col1" class="data row1 col1" >16.266667</td>
      <td id="T_7d04a_row1_col2" class="data row1 col2" >35.266667</td>
      <td id="T_7d04a_row1_col3" class="data row1 col3" >2352.266667</td>
      <td id="T_7d04a_row1_col4" class="data row1 col4" >0.783333</td>
      <td id="T_7d04a_row1_col5" class="data row1 col5" >1.283333</td>
      <td id="T_7d04a_row1_col6" class="data row1 col6" >0.216667</td>
      <td id="T_7d04a_row1_col7" class="data row1 col7" >0.266667</td>
      <td id="T_7d04a_row1_col8" class="data row1 col8" >0.766667</td>
      <td id="T_7d04a_row1_col9" class="data row1 col9" >0.233333</td>
    </tr>
    <tr>
      <th id="T_7d04a_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_7d04a_row2_col0" class="data row2 col0" >1.216667</td>
      <td id="T_7d04a_row2_col1" class="data row2 col1" >18.566667</td>
      <td id="T_7d04a_row2_col2" class="data row2 col2" >36.950000</td>
      <td id="T_7d04a_row2_col3" class="data row2 col3" >3140.466667</td>
      <td id="T_7d04a_row2_col4" class="data row2 col4" >0.766667</td>
      <td id="T_7d04a_row2_col5" class="data row2 col5" >1.033333</td>
      <td id="T_7d04a_row2_col6" class="data row2 col6" >0.250000</td>
      <td id="T_7d04a_row2_col7" class="data row2 col7" >0.300000</td>
      <td id="T_7d04a_row2_col8" class="data row2 col8" >0.683333</td>
      <td id="T_7d04a_row2_col9" class="data row2 col9" >0.316667</td>
    </tr>
    <tr>
      <th id="T_7d04a_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_7d04a_row3_col0" class="data row3 col0" >1.450000</td>
      <td id="T_7d04a_row3_col1" class="data row3 col1" >18.466667</td>
      <td id="T_7d04a_row3_col2" class="data row3 col2" >33.133333</td>
      <td id="T_7d04a_row3_col3" class="data row3 col3" >2666.100000</td>
      <td id="T_7d04a_row3_col4" class="data row3 col4" >0.683333</td>
      <td id="T_7d04a_row3_col5" class="data row3 col5" >1.283333</td>
      <td id="T_7d04a_row3_col6" class="data row3 col6" >0.266667</td>
      <td id="T_7d04a_row3_col7" class="data row3 col7" >0.383333</td>
      <td id="T_7d04a_row3_col8" class="data row3 col8" >0.716667</td>
      <td id="T_7d04a_row3_col9" class="data row3 col9" >0.283333</td>
    </tr>
    <tr>
      <th id="T_7d04a_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_7d04a_row4_col0" class="data row4 col0" >1.383333</td>
      <td id="T_7d04a_row4_col1" class="data row4 col1" >31.783333</td>
      <td id="T_7d04a_row4_col2" class="data row4 col2" >31.700000</td>
      <td id="T_7d04a_row4_col3" class="data row4 col3" >4354.200000</td>
      <td id="T_7d04a_row4_col4" class="data row4 col4" >0.383333</td>
      <td id="T_7d04a_row4_col5" class="data row4 col5" >1.033333</td>
      <td id="T_7d04a_row4_col6" class="data row4 col6" >0.166667</td>
      <td id="T_7d04a_row4_col7" class="data row4 col7" >0.333333</td>
      <td id="T_7d04a_row4_col8" class="data row4 col8" >0.600000</td>
      <td id="T_7d04a_row4_col9" class="data row4 col9" >0.400000</td>
    </tr>
  </tbody>
</table>




We managed to create a good discrimination between our audience with higher scores and those with lower scores


```python
df_final.groupby('rank')[['Risk']].agg('sum').style.bar(align='mid', color=['#d65f5f', '#5fba7d'])
```




<style type="text/css">
#T_5712f_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 11.1%, transparent 11.1%);
}
#T_5712f_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 22.2%, transparent 22.2%);
}
#T_5712f_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 36.1%, transparent 36.1%);
}
#T_5712f_row3_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 83.3%, transparent 83.3%);
}
#T_5712f_row4_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
</style>
<table id="T_5712f">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_5712f_level0_col0" class="col_heading level0 col0" >Risk</th>
    </tr>
    <tr>
      <th class="index_name level0" >rank</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_5712f_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_5712f_row0_col0" class="data row0 col0" >4</td>
    </tr>
    <tr>
      <th id="T_5712f_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_5712f_row1_col0" class="data row1 col0" >8</td>
    </tr>
    <tr>
      <th id="T_5712f_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_5712f_row2_col0" class="data row2 col0" >13</td>
    </tr>
    <tr>
      <th id="T_5712f_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_5712f_row3_col0" class="data row3 col0" >30</td>
    </tr>
    <tr>
      <th id="T_5712f_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_5712f_row4_col0" class="data row4 col0" >36</td>
    </tr>
  </tbody>
</table>




We were able to order the amount of bad credit

we can see using other's numbers to divide


```python
df_final['rank'] = pd.qcut(df_final['score'], 3,labels = False)
df_final.groupby('rank')[['Checking account','Duration', 'Age','Credit amount', 'hous_own',
                          'Saving accounts','purp_radio/TV','purp_car','sex_male','sex_female']].agg('mean')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Checking account</th>
      <th>Duration</th>
      <th>Age</th>
      <th>Credit amount</th>
      <th>hous_own</th>
      <th>Saving accounts</th>
      <th>purp_radio/TV</th>
      <th>purp_car</th>
      <th>sex_male</th>
      <th>sex_female</th>
    </tr>
    <tr>
      <th>rank</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.31</td>
      <td>15.43</td>
      <td>39.69</td>
      <td>2282.97</td>
      <td>0.86</td>
      <td>1.17</td>
      <td>0.36</td>
      <td>0.28</td>
      <td>0.77</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.17</td>
      <td>18.19</td>
      <td>35.30</td>
      <td>2868.43</td>
      <td>0.77</td>
      <td>1.18</td>
      <td>0.20</td>
      <td>0.31</td>
      <td>0.71</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.43</td>
      <td>26.64</td>
      <td>32.02</td>
      <td>3713.47</td>
      <td>0.48</td>
      <td>1.14</td>
      <td>0.22</td>
      <td>0.36</td>
      <td>0.64</td>
      <td>0.36</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_final.groupby('rank')[['Checking account','Duration', 'Age','Credit amount', 'hous_own',
                          'Saving accounts','purp_radio/TV','purp_car','sex_male','sex_female']].agg('mean').style.bar(align='mid', color=['#d65f5f', '#5fba7d'])
```




<style type="text/css">
#T_ee254_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 21.7%, transparent 21.7%);
}
#T_ee254_row0_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 57.9%, transparent 57.9%);
}
#T_ee254_row0_col2, #T_ee254_row0_col4, #T_ee254_row0_col6, #T_ee254_row0_col8, #T_ee254_row1_col5, #T_ee254_row2_col0, #T_ee254_row2_col1, #T_ee254_row2_col3, #T_ee254_row2_col7, #T_ee254_row2_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
#T_ee254_row0_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 61.5%, transparent 61.5%);
}
#T_ee254_row0_col5 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 99.2%, transparent 99.2%);
}
#T_ee254_row0_col7 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 77.8%, transparent 77.8%);
}
#T_ee254_row0_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 63.9%, transparent 63.9%);
}
#T_ee254_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 81.8%, transparent 81.8%);
}
#T_ee254_row1_col1 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 68.3%, transparent 68.3%);
}
#T_ee254_row1_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 88.9%, transparent 88.9%);
}
#T_ee254_row1_col3 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 77.2%, transparent 77.2%);
}
#T_ee254_row1_col4 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 89.5%, transparent 89.5%);
}
#T_ee254_row1_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 55.6%, transparent 55.6%);
}
#T_ee254_row1_col7 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 86.1%, transparent 86.1%);
}
#T_ee254_row1_col8 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 92.2%, transparent 92.2%);
}
#T_ee254_row1_col9 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 80.6%, transparent 80.6%);
}
#T_ee254_row2_col2 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 80.7%, transparent 80.7%);
}
#T_ee254_row2_col4 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 55.8%, transparent 55.8%);
}
#T_ee254_row2_col5 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 96.6%, transparent 96.6%);
}
#T_ee254_row2_col6 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 61.1%, transparent 61.1%);
}
#T_ee254_row2_col8 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 83.1%, transparent 83.1%);
}
</style>
<table id="T_ee254">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_ee254_level0_col0" class="col_heading level0 col0" >Checking account</th>
      <th id="T_ee254_level0_col1" class="col_heading level0 col1" >Duration</th>
      <th id="T_ee254_level0_col2" class="col_heading level0 col2" >Age</th>
      <th id="T_ee254_level0_col3" class="col_heading level0 col3" >Credit amount</th>
      <th id="T_ee254_level0_col4" class="col_heading level0 col4" >hous_own</th>
      <th id="T_ee254_level0_col5" class="col_heading level0 col5" >Saving accounts</th>
      <th id="T_ee254_level0_col6" class="col_heading level0 col6" >purp_radio/TV</th>
      <th id="T_ee254_level0_col7" class="col_heading level0 col7" >purp_car</th>
      <th id="T_ee254_level0_col8" class="col_heading level0 col8" >sex_male</th>
      <th id="T_ee254_level0_col9" class="col_heading level0 col9" >sex_female</th>
    </tr>
    <tr>
      <th class="index_name level0" >rank</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
      <th class="blank col6" >&nbsp;</th>
      <th class="blank col7" >&nbsp;</th>
      <th class="blank col8" >&nbsp;</th>
      <th class="blank col9" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_ee254_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_ee254_row0_col0" class="data row0 col0" >0.310000</td>
      <td id="T_ee254_row0_col1" class="data row0 col1" >15.430000</td>
      <td id="T_ee254_row0_col2" class="data row0 col2" >39.690000</td>
      <td id="T_ee254_row0_col3" class="data row0 col3" >2282.970000</td>
      <td id="T_ee254_row0_col4" class="data row0 col4" >0.860000</td>
      <td id="T_ee254_row0_col5" class="data row0 col5" >1.170000</td>
      <td id="T_ee254_row0_col6" class="data row0 col6" >0.360000</td>
      <td id="T_ee254_row0_col7" class="data row0 col7" >0.280000</td>
      <td id="T_ee254_row0_col8" class="data row0 col8" >0.770000</td>
      <td id="T_ee254_row0_col9" class="data row0 col9" >0.230000</td>
    </tr>
    <tr>
      <th id="T_ee254_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_ee254_row1_col0" class="data row1 col0" >1.170000</td>
      <td id="T_ee254_row1_col1" class="data row1 col1" >18.190000</td>
      <td id="T_ee254_row1_col2" class="data row1 col2" >35.300000</td>
      <td id="T_ee254_row1_col3" class="data row1 col3" >2868.430000</td>
      <td id="T_ee254_row1_col4" class="data row1 col4" >0.770000</td>
      <td id="T_ee254_row1_col5" class="data row1 col5" >1.180000</td>
      <td id="T_ee254_row1_col6" class="data row1 col6" >0.200000</td>
      <td id="T_ee254_row1_col7" class="data row1 col7" >0.310000</td>
      <td id="T_ee254_row1_col8" class="data row1 col8" >0.710000</td>
      <td id="T_ee254_row1_col9" class="data row1 col9" >0.290000</td>
    </tr>
    <tr>
      <th id="T_ee254_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_ee254_row2_col0" class="data row2 col0" >1.430000</td>
      <td id="T_ee254_row2_col1" class="data row2 col1" >26.640000</td>
      <td id="T_ee254_row2_col2" class="data row2 col2" >32.020000</td>
      <td id="T_ee254_row2_col3" class="data row2 col3" >3713.470000</td>
      <td id="T_ee254_row2_col4" class="data row2 col4" >0.480000</td>
      <td id="T_ee254_row2_col5" class="data row2 col5" >1.140000</td>
      <td id="T_ee254_row2_col6" class="data row2 col6" >0.220000</td>
      <td id="T_ee254_row2_col7" class="data row2 col7" >0.360000</td>
      <td id="T_ee254_row2_col8" class="data row2 col8" >0.640000</td>
      <td id="T_ee254_row2_col9" class="data row2 col9" >0.360000</td>
    </tr>
  </tbody>
</table>





```python
df_final.groupby('rank')[['Risk']].agg('sum').style.bar(align='mid', color=['#d65f5f', '#5fba7d'])
```




<style type="text/css">
#T_0f6f8_row0_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 14.0%, transparent 14.0%);
}
#T_0f6f8_row1_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 45.6%, transparent 45.6%);
}
#T_0f6f8_row2_col0 {
  width: 10em;
  background: linear-gradient(90deg, #5fba7d 100.0%, transparent 100.0%);
}
</style>
<table id="T_0f6f8">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_0f6f8_level0_col0" class="col_heading level0 col0" >Risk</th>
    </tr>
    <tr>
      <th class="index_name level0" >rank</th>
      <th class="blank col0" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_0f6f8_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_0f6f8_row0_col0" class="data row0 col0" >8</td>
    </tr>
    <tr>
      <th id="T_0f6f8_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_0f6f8_row1_col0" class="data row1 col0" >26</td>
    </tr>
    <tr>
      <th id="T_0f6f8_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_0f6f8_row2_col0" class="data row2 col0" >57</td>
    </tr>
  </tbody>
</table>




Our final model is:


```python
df_final[['score','rank','shap_1','shap_2','shap_3','shap_4','shap_5','shap_6',
        'shap_7','shap_8','shap_9','shap_10','shap_11']]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>rank</th>
      <th>shap_1</th>
      <th>shap_2</th>
      <th>shap_3</th>
      <th>shap_4</th>
      <th>shap_5</th>
      <th>shap_6</th>
      <th>shap_7</th>
      <th>shap_8</th>
      <th>shap_9</th>
      <th>shap_10</th>
      <th>shap_11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.344688</td>
      <td>2</td>
      <td>Little Check Account</td>
      <td>0</td>
      <td>0</td>
      <td>More Junior Client</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Less Credit Amount</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.296853</td>
      <td>1</td>
      <td>Little Check Account</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.410630</td>
      <td>2</td>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Less Credit Amount</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.427560</td>
      <td>2</td>
      <td>Little Check Account</td>
      <td>0</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>0</td>
      <td>0</td>
      <td>Less Credit Duration</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.184806</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>More Credit Amount</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Moderate/Rich Check Account</td>
      <td>0</td>
      <td>0</td>
      <td>More Senior Client</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>0.187536</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>More Junior Client</td>
      <td>0</td>
      <td>0</td>
      <td>Moderate/Rich Check Account</td>
      <td>0</td>
      <td>Less Credit Amount</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>296</th>
      <td>0.245917</td>
      <td>1</td>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>297</th>
      <td>0.167448</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>0</td>
      <td>0</td>
      <td>Moderate/Rich Check Account</td>
      <td>Less Credit Duration</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>298</th>
      <td>0.658138</td>
      <td>2</td>
      <td>Little Check Account</td>
      <td>More Credit Duration</td>
      <td>More Credit Amount</td>
      <td>0</td>
      <td>Have House</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>299</th>
      <td>0.651962</td>
      <td>2</td>
      <td>Little Check Account</td>
      <td>0</td>
      <td>More Credit Amount</td>
      <td>More Junior Client</td>
      <td>Have House</td>
      <td>0</td>
      <td>0</td>
      <td>Less Credit Duration</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 13 columns</p>
</div>
