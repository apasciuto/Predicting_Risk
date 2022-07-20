# Lending Club

**Identify loans that have a better chance of being paid off on time.**  

# 1. About Lending Club

We will be working with financial lending data from [Lending Club](https://www.lendingclub.com/). **Lending Club** is America’s largest marketplace connecting borrowers and investors, where consumers and small business owners lower the cost of their credit and enjoy a better experience than traditional bank lending, and investors earn attractive risk-adjusted returns.  

**[How does an online credit marketplace work?](https://www.lendingclub.com/public/how-peer-lending-works.action)**  

- Customers interested in a loan complete a simple application at LendingClub.com
- We leverage online data and technology to quickly assess risk, determine a credit rating and assign appropriate interest rates. Qualified applicants receive offers in just minutes and can evaluate loan options with no impact to their credit score
- Investors ranging from individuals to institutions select loans in which to invest and can earn monthly returns  

**Here is a diagram of the process:**
![HOW-PEER-LENDING-WORKS](/assets/media/loans/how-peer-lending-works.jpg)

# 2. The Data

Lending Club releases data for all of the approved and declined loan applications periodically on their [website](https://www.lendingclub.com/info/download-data.action). We will be focusing on the approved loans data from 2007 to 2011, since a good number of the loans have already finished. In the datasets for later years, many of the loans are current and still being paid off.

The Data Dictionary contains information about what each column represents in the dataset and can be found [here](https://goo.gl/K4Hi5j).  

There are three sections to the Data Dictionary:
- **LoanStats** describes the approved loans dataset
- **RejectStats** describes the rejected loans dataset
- **BrowseNotes** describes additional information on loans in the dataset  

Since rejected applications don't appear on the Lending Club marketplace and aren't available for investment, we will be focusing on data for approved loans only.  

The approved loans datasets contain information on current loans, completed loans, and defaulted loans.  

Before we can start doing machine learning, we need to define what features we want to use and which column repesents the target column we want to predict. Let's start by reading in the dataset and exploring it.

# 3. Acquire The Data


```python
import numpy as np
import pandas as pd

loans_2007 = pd.read_csv("LoanStats3a.csv")
loans_2007.head()
```

We need to format the data as followed:  

1. Remove the first row:  
    - The first row contains notes by [Prospectus](https://www.lendingclub.com/info/prospectus.action) instead of the column names and prevents the dataset from being parsed correctly  

2. Remove the `desc` column:  
    - Contains a long text explanation for each loan  

3. Remove the `url` column:  
    - Contains a link to each loan on Lending Club which can only be accessed with an investor account   

4. Remove all columns containing more than 50% missing values:  
    - Allows us to move more efficiently since we can spend less time trying to fill in these values


```python
loans_2007 = pd.read_csv('LoanStats3a.csv', skiprows=1)
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
loans_2007 = loans_2007.drop(['desc', 'url'],axis=1)
loans_2007.to_csv('loans_2007.csv', index=False)
```

We will maintain the original dataset `LoanStats3a.csv` by filtering out our newly parsed dataset into `loans_2007.csv`

### 3.1 Creating a Pandas DataFrame 


```python
import numpy as np
import pandas as pd
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

loans_2007 = pd.read_csv("loans_2007.csv", low_memory=False)
loans_2007.drop_duplicates()
print(loans_2007.shape)
loans_2007.head(5)
```

    (42538, 52)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>...</th>
      <th>last_pymnt_amnt</th>
      <th>last_credit_pull_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1077501</td>
      <td>1296599.0</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>...</td>
      <td>171.62</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1077430</td>
      <td>1314167.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>...</td>
      <td>119.66</td>
      <td>Sep-2013</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1077175</td>
      <td>1313524.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>...</td>
      <td>649.91</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1076863</td>
      <td>1277178.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>C</td>
      <td>C1</td>
      <td>...</td>
      <td>357.48</td>
      <td>Apr-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1075358</td>
      <td>1311748.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>B</td>
      <td>B5</td>
      <td>...</td>
      <td>67.79</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 52 columns</p>
</div>



# 4. Parse The Data

By exploring the data set and utilizing our understanding of the Data Dictionary we are going to:  

1. Seperate the dataset into 3 groups

2. Watch out for data leakage
    - This can cause our model to overfit, and occures when the model uses data about the target column that wouldn't be available when using the model on future loans.  

3. Identify features that don't affect a borrower's ability to pay back a loan
    - (e.g. a randomly generated ID value by Lending Club)  

4. Clean poorly formatted features

5. Explore features that require more data or a lot of processing to be useful

6. Determine if any of the features contain redundant information

7. Select the target column for when we move on to the machine learning phase 

### 4.1 Group 1
![GROUP-ONE](/assets/media/loans/group-one.png)

After analyzing each column, we can conclude that the following features need to be removed:  

- `id`: randomly generated field by Lending Club for unique identification purposes only
- `member_id`: also a randomly generated field by Lending Club for unique identification purposes only
- `funded_amnt`: leaks data from the future (after the loan is already started to be funded)
- `funded_amnt_inv`: also leaks data from the future (after the loan is already started to be funded)
- `grade`: contains redundant information as the interest rate column (int_rate)
- `sub_grade`: also contains redundant information as the interest rate column (int_rate)
- `emp_title`: requires other data and a lot of processing to potentially be useful
- `issue_d`: leaks data from the future (after the loan is already completed funded)  


```python
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
```

Lending Club assigns a grade and a sub-grade based on the borrower's interest rate. While the `grade` and `sub_grade` values are categorical, the `int_rate` column contains continuous values, which are better suited for machine learning.

### 4.2 Group 2
![GROUP-TWO](/assets/media/loans/group-two.png)

Within this group of columns, we need to drop the following columns:  

- `zip_code`: redundant with the addr_state column since only the first 3 digits of the 5 digit zip code are visible (which only can be used to identify the state the borrower lives in)
- `out_prncp`: leaks data from the future, (after the loan already started to be paid off)
- `out_prncp_inv`: also leaks data from the future, (after the loan already started to be paid off)
- `total_pymnt`: also leaks data from the future, (after the loan already started to be paid off)
- `total_pymnt_inv`: also leaks data from the future, (after the loan already started to be paid off)
- `total_rec_prncp`: also leaks data from the future, (after the loan already started to be paid off)  

The `out_prncp` and `out_prncp_inv` both describe the outstanding principal amount for a loan, which is the remaining amount the borrower still owes. These 2 columns as well as the `total_pymnt` column describe properties of the loan after it's fully funded and started to be paid off. This information isn't available to an investor before the loan is fully funded and we don't want to include it in our model.


```python
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
```

### 4.3 Group 3
![GROUP-THREE](/assets/media/loans/group-three.png)

In the last group of columns, we need to drop the following columns:  

- `total_rec_int`: leaks data from the future, (after the loan already started to be paid off)
- `total_rec_late_fee`: also leaks data from the future, (after the loan already started to be paid off)
- `recoveries`: also leaks data from the future, (after the loan already started to be paid off)
- `collection_recovery_fee`: also leaks data from the future, (after the loan already started to be paid off)
- `last_pymnt_d`: also leaks data from the future, (after the loan already started to be paid off)
- `last_pymnt_amnt`: also leaks data from the future, (after the loan already started to be paid off)  

All of these columns leak data from the future, meaning that they're describing aspects of the loan after it's already been fully funded and started to be paid off by the borrower.


```python
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
```


```python
print(loans_2007.shape)
loans_2007.head()
```

    (42538, 32)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>...</th>
      <th>initial_list_status</th>
      <th>last_credit_pull_d</th>
      <th>collections_12_mths_ex_med</th>
      <th>policy_code</th>
      <th>application_type</th>
      <th>acc_now_delinq</th>
      <th>chargeoff_within_12_mths</th>
      <th>delinq_amnt</th>
      <th>pub_rec_bankruptcies</th>
      <th>tax_liens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>36 months</td>
      <td>10.65%</td>
      <td>162.87</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27%</td>
      <td>59.83</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Sep-2013</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96%</td>
      <td>84.33</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>36 months</td>
      <td>13.49%</td>
      <td>339.31</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>49200.0</td>
      <td>Source Verified</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Apr-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3000.0</td>
      <td>60 months</td>
      <td>12.69%</td>
      <td>67.79</td>
      <td>1 year</td>
      <td>RENT</td>
      <td>80000.0</td>
      <td>Source Verified</td>
      <td>Current</td>
      <td>n</td>
      <td>...</td>
      <td>f</td>
      <td>Jun-2016</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>INDIVIDUAL</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32 columns</p>
</div>



We were able to reduce the number of columns from 52 to 32 columns. Now we need to decide on a target column that we want to use for modeling.

### 4.4 Target Column

The **loan_status** column is the only column that directly describes if a loan was paid off on time, had delayed payments, or was defaulted on the borrower. This column contains text values and needs to be converted to a numerical column for training our model.

We will use the `value_counts()` method to display the frequency of each unique value. This will allow us to explore the different values in this column and come up with a strategy for converting them.


```python
loans_2007['loan_status'].value_counts()
```




    Fully Paid                                             33136
    Charged Off                                             5634
    Does not meet the credit policy. Status:Fully Paid      1988
    Current                                                  961
    Does not meet the credit policy. Status:Charged Off      761
    Late (31-120 days)                                        24
    In Grace Period                                           20
    Late (16-30 days)                                          8
    Default                                                    3
    Name: loan_status, dtype: int64



### 4.5 The Quality of The Data

#### Binary Classification

There are 8 different possible values for the `loan_status` column. The following table provides insight on each column as well as the counts in the Dataframe:  

![LOAN-STATUS](/assets/media/loans/loan_status.png)

From an investor's point of view, we are interested in trying to predict if a loan will be paid off on time or not. Only the `Fully Paid` and `Charged Off` values describe the final outcome of the loan. The other values describe loans that are still on going. Loans with a `Charged Off` status essentially have no chance of being repaid, and loans marked with a `Default` status have only a small chance of being repaid.  

Since we are only interested in predicting which of these 2 values (`Fully Paid` and `Charged Off`) a loan will fall under, we can establish that this is a **binary classification** problem.  

We can start by removing all the loans that don't contain either a `Fully Paid` or `Charged Off` loan status and then assign the `Fully Paid` values to `1` for the positive case and the `Charged Off` values to `0` for the negative case. While there are a few different ways to transform all of the values in a column, we'll use the Dataframe method replace.

#### Class Imbalance

One thing to keep in mind is the **class imbalance** between the positive and negative cases. While there are 33,136 loans that have been fully paid off, there are only 5,634 that were charged off. This class imbalance may result in the model having a stronger bias towards predicting the class with more observations. The stronger the imbalance, the more biased the model becomes.


```python
loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | (loans_2007['loan_status'] == "Charged Off")]

status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}

loans_2007 = loans_2007.replace(status_replace)
```

#### Reducing Data Size

We can also further reduce the size of our data set by identifying columns that contain only one unique value. These columns won't be useful for the model since they don't add any information to each loan application. 


```python
orig_columns = loans_2007.columns
drop_columns = []
for col in orig_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans_2007 = loans_2007.drop(drop_columns, axis=1)
print(drop_columns)
```

    ['pymnt_plan', 'initial_list_status', 'collections_12_mths_ex_med', 'policy_code', 'application_type', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens']


We were able to remove 9 more columns since they only contained 1 unique value.

Let's export our DataFrame and save it to a new `filtered_loans_2007.csv` file


```python
loans_2007.to_csv('filtered_loans_2007.csv', index=False)
```

# 5. Mine The Data

We need to make sure there are no missing values in the DataFrame, otherwise Scikit-learn will throw an error if you try to train a model using data that contain any missing or non-numerical values.


```python
loans = pd.read_csv('filtered_loans_2007.csv')
null_counts = loans.isnull().sum()
null_counts
```




    loan_amnt                 0
    term                      0
    int_rate                  0
    installment               0
    emp_length                0
    home_ownership            0
    annual_inc                0
    verification_status       0
    loan_status               0
    purpose                   0
    title                    10
    addr_state                0
    dti                       0
    delinq_2yrs               0
    earliest_cr_line          0
    inq_last_6mths            0
    open_acc                  0
    pub_rec                   0
    revol_bal                 0
    revol_util               50
    total_acc                 0
    last_credit_pull_d        2
    pub_rec_bankruptcies    697
    dtype: int64



### 5.1 Handling Missing Values

We can keep the following columns and just remove columns where more than 1% of the rows contain missing values:  

1. `pub_rec_bankruptcies` 
    - We should remove this column first because it contains the most rows with missing values  

2. `title` 

3. `revol_util` 

4. `last_credit_pull_d`


```python
loans = loans.drop("pub_rec_bankruptcies", axis=1)
loans = loans.dropna(axis=0)
```

### 5.2 Handling Non-numeric Values

We need to identify each columns data type and return the count


```python
loans.dtypes.value_counts()
```




    object     11
    float64    10
    int64       1
    dtype: int64



The object columns that contain text need to be converted to numerical data types.


```python
object_columns_df = loans.select_dtypes(include=["object"])
object_columns_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>term</th>
      <th>int_rate</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>verification_status</th>
      <th>purpose</th>
      <th>title</th>
      <th>addr_state</th>
      <th>earliest_cr_line</th>
      <th>revol_util</th>
      <th>last_credit_pull_d</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>36 months</td>
      <td>10.65%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Verified</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>AZ</td>
      <td>Jan-1985</td>
      <td>83.7%</td>
      <td>Jun-2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60 months</td>
      <td>15.27%</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>car</td>
      <td>bike</td>
      <td>GA</td>
      <td>Apr-1999</td>
      <td>9.4%</td>
      <td>Sep-2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>36 months</td>
      <td>15.96%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Not Verified</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>IL</td>
      <td>Nov-2001</td>
      <td>98.5%</td>
      <td>Jun-2016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36 months</td>
      <td>13.49%</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>other</td>
      <td>personel</td>
      <td>CA</td>
      <td>Feb-1996</td>
      <td>21%</td>
      <td>Apr-2016</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36 months</td>
      <td>7.90%</td>
      <td>3 years</td>
      <td>RENT</td>
      <td>Source Verified</td>
      <td>wedding</td>
      <td>My wedding loan I promise to pay back</td>
      <td>AZ</td>
      <td>Nov-2004</td>
      <td>28.3%</td>
      <td>Jan-2016</td>
    </tr>
  </tbody>
</table>
</div>



# 6. Refine The Data

Columns that represent categorical values:  

- `home_ownership`: Can only be 1 of 4 categorical values based on the Data Dictionary
- `verification_status`: Indicates if income was verified by Lending Club
- `emp_length`: Number of years the borrower was employed at the time of applying
- `term`: Number of payments on the loan, either 36 or 60
- `addr_state`: Borrower's state of residence
- `purpose`: A category provided by the borrower for the loan request
- `title`: Loan title provided the borrower  

Columns that represent numeric values:

- `int_rate`: Interest rate of the loan in %
- `revol_util`: Revolving Line Utilization Rate or the amount of credit the borrower is using relative to all available credit  


Columns containing date values that require feature engineering will be removed:

- `earliest_cr_line`: The month the borrower's earliest reported credit line was opened
- `last_credit_pull_d`: The most recent month Lending Club pulled credit for this loan  

### 6.1 Categorical Values
Here is the unique value counts of the columnns that could potentially contain categorical values.


```python
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state']
for c in cols:
    print(loans[c].value_counts())
```

    RENT        18513
    MORTGAGE    17112
    OWN          2984
    OTHER          96
    NONE            3
    Name: home_ownership, dtype: int64
    Not Verified       16696
    Verified           12290
    Source Verified     9722
    Name: verification_status, dtype: int64
    10+ years    8545
    < 1 year     4513
    2 years      4303
    3 years      4022
    4 years      3353
    5 years      3202
    1 year       3176
    6 years      2177
    7 years      1714
    8 years      1442
    9 years      1229
    n/a          1032
    Name: emp_length, dtype: int64
     36 months    29041
     60 months     9667
    Name: term, dtype: int64
    CA    6958
    NY    3713
    FL    2791
    TX    2667
    NJ    1798
    IL    1483
    PA    1473
    VA    1376
    GA    1364
    MA    1301
    OH    1179
    MD    1026
    AZ     850
    WA     822
    CO     770
    NC     753
    CT     730
    MI     712
    MO     671
    MN     603
    NV     481
    SC     462
    WI     441
    AL     437
    OR     436
    LA     430
    KY     315
    OK     290
    KS     260
    UT     254
    AR     237
    DC     209
    RI     196
    NM     184
    WV     172
    HI     166
    NH     166
    DE     113
    MT      83
    WY      80
    AK      78
    SD      61
    VT      53
    MS      19
    TN      17
    IN       9
    ID       6
    NE       5
    IA       5
    ME       3
    Name: addr_state, dtype: int64


Below we will display the unique values of the following columns:

- `purpose`
- `title`


```python
print(loans["purpose"].value_counts())
print(loans["title"].value_counts())
```

    debt_consolidation    18130
    credit_card            5039
    other                  3864
    home_improvement       2897
    major_purchase         2155
    small_business         1762
    car                    1510
    wedding                 929
    medical                 680
    moving                  576
    vacation                375
    house                   369
    educational             320
    renewable_energy        102
    Name: purpose, dtype: int64
    Debt Consolidation                                  2104
    Debt Consolidation Loan                             1632
    Personal Loan                                        642
    Consolidation                                        494
    debt consolidation                                   485
    Credit Card Consolidation                            353
    Home Improvement                                     346
    Debt consolidation                                   324
    Small Business Loan                                  310
    Credit Card Loan                                     305
    Personal                                             302
    Consolidation Loan                                   251
    Home Improvement Loan                                234
    personal loan                                        227
    personal                                             211
    Loan                                                 208
    Wedding Loan                                         201
    Car Loan                                             195
    consolidation                                        193
    Other Loan                                           181
    Credit Card Payoff                                   150
    Wedding                                              149
    Credit Card Refinance                                143
    Major Purchase Loan                                  139
    Consolidate                                          125
    Medical                                              118
    Credit Card                                          115
    home improvement                                     107
    My Loan                                               92
    Credit Cards                                          91
                                                        ... 
    Moving expense loan                                    1
    get rid of my credit cards loans                       1
    My Cessna Airplane Needs Engine Overhaul               1
    Remodel office space so it can be leased               1
    Happy Day                                              1
    Consolidation Nov 2011                                 1
    credit card and medical bill consolidati               1
    Goodbye to Debt                                        1
    Operation Pay Off Parents                              1
    Velma Loan                                             1
    Med Bills                                              1
    Young Engineer Looking For Home Help                   1
    Major Buy                                              1
    megical                                                1
    Debt Consildate CC                                     1
    BOA                                                    1
    debt free in 36!                                       1
    Chase-Citi-Dell-Apple-Exxon                            1
    Recently had a baby...                                 1
    Debt Free Me !                                         1
    Gabi's Car Loan                                        1
    Maria Elena                                            1
    Credit consolidation and refinance                     1
    better opportunities for money saved on interest       1
    Consolidation for Wedding Budget                       1
    Time to paint the exterior!                            1
    CC consolidator                                        1
    Education/Furniture                                    1
    Riddle family debt consolidation.                      1
    Duplex Investment                                      1
    Name: title, Length: 19332, dtype: int64


**Observation:**  

- The `purpose` and `title` columns contain overlapping information. 
- The `purpose` column since it contains a few discrete values. 
- The `title` column has data quality issues. Many of the values are repeated with slight modifications (for example: `Debt Consolidation` and `Debt Consolidation Loan` and `debt consolidation`).

The `emp_length` column needs to be cleaned and converted to numeric values since they have ordering.  

We can use the following mapping to clean the `emp_length` column:  

- "10+ years": 10
- "9 years": 9
- "8 years": 8
- "7 years": 7
- "6 years": 6
- "5 years": 5
- "4 years": 4
- "3 years": 3
- "2 years": 2
- "1 year": 1
- "< 1 year": 0
- "n/a": 0  

We are going to map all individuals who may have been working more than 10 year to the value of 10.  

For the individuals who have worked less than a year or have missing data will be assigned the value of 0.


```python
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state", "title"], axis=1)
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
loans = loans.replace(mapping_dict)
```

### 6.2 Dummy Variables

The `home_ownership`, `verification_status`, `emp_length`, and `term` columns each contain a few discrete categorical values. We should encode these columns as dummy variables:  

- Use the Series method `astype` to convert each column to the category data type
- Use the `get_dummies` function to return a DataFrame containing the dummy columns
- Use the `concat` method to add these dummy columns back to `loans`
- Remove the original, non-dummy columns (`home_ownership`, `verification_status`, `purpose`, and `term`) from `loans`


```python
cat_columns = ["home_ownership", "verification_status", "purpose", "term"]
dummy_df = pd.get_dummies(loans[cat_columns])
loans = pd.concat([loans, dummy_df], axis=1)
loans = loans.drop(cat_columns, axis=1)
```

Save our newly formatted Data:


```python
loans.to_csv('cleaned_loans_2007.csv', index=False)
```

Import newly formatted Data:


```python
import numpy as np
import pandas as pd
loans = pd.read_csv("cleaned_loans_2007.csv")
print(loans.info())
loans.head()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 38708 entries, 0 to 38707
    Data columns (total 38 columns):
    loan_amnt                              38708 non-null float64
    int_rate                               38708 non-null float64
    installment                            38708 non-null float64
    emp_length                             38708 non-null int64
    annual_inc                             38708 non-null float64
    loan_status                            38708 non-null int64
    dti                                    38708 non-null float64
    delinq_2yrs                            38708 non-null float64
    inq_last_6mths                         38708 non-null float64
    open_acc                               38708 non-null float64
    pub_rec                                38708 non-null float64
    revol_bal                              38708 non-null float64
    revol_util                             38708 non-null float64
    total_acc                              38708 non-null float64
    home_ownership_MORTGAGE                38708 non-null int64
    home_ownership_NONE                    38708 non-null int64
    home_ownership_OTHER                   38708 non-null int64
    home_ownership_OWN                     38708 non-null int64
    home_ownership_RENT                    38708 non-null int64
    verification_status_Not Verified       38708 non-null int64
    verification_status_Source Verified    38708 non-null int64
    verification_status_Verified           38708 non-null int64
    purpose_car                            38708 non-null int64
    purpose_credit_card                    38708 non-null int64
    purpose_debt_consolidation             38708 non-null int64
    purpose_educational                    38708 non-null int64
    purpose_home_improvement               38708 non-null int64
    purpose_house                          38708 non-null int64
    purpose_major_purchase                 38708 non-null int64
    purpose_medical                        38708 non-null int64
    purpose_moving                         38708 non-null int64
    purpose_other                          38708 non-null int64
    purpose_renewable_energy               38708 non-null int64
    purpose_small_business                 38708 non-null int64
    purpose_vacation                       38708 non-null int64
    purpose_wedding                        38708 non-null int64
    term_ 36 months                        38708 non-null int64
    term_ 60 months                        38708 non-null int64
    dtypes: float64(12), int64(26)
    memory usage: 11.2 MB
    None





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>emp_length</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>...</th>
      <th>purpose_major_purchase</th>
      <th>purpose_medical</th>
      <th>purpose_moving</th>
      <th>purpose_other</th>
      <th>purpose_renewable_energy</th>
      <th>purpose_small_business</th>
      <th>purpose_vacation</th>
      <th>purpose_wedding</th>
      <th>term_ 36 months</th>
      <th>term_ 60 months</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>10</td>
      <td>24000.0</td>
      <td>1</td>
      <td>27.65</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2500.0</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>0</td>
      <td>30000.0</td>
      <td>0</td>
      <td>1.00</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2400.0</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>10</td>
      <td>12252.0</td>
      <td>1</td>
      <td>8.72</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>13.49</td>
      <td>339.31</td>
      <td>10</td>
      <td>49200.0</td>
      <td>1</td>
      <td>20.00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5000.0</td>
      <td>7.90</td>
      <td>156.46</td>
      <td>3</td>
      <td>36000.0</td>
      <td>1</td>
      <td>11.20</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



# 7. Build A Data Model

### 7.1 Calculating Error Metric

Our error metric will help us figure out whether or not our model is performing well, and will ultimately determine if our algorithm will make us money or lose us money.  

In this case, we are primarily concerned with the following misclassifications: false positives and false negatives. Both of these are different types of misclassifications.  

**1. False Positive**  
- We predict that a loan will be paid off on time, but it actually isn't.  

**2. False Negative**  
- We predict that a loan won't be paid off on time, but it actually is paid off on time.    

We need to treat false positives differently than false negatives. We want to minimize risk, and avoid false positives as much as possible because we would prefer to miss out on opportunities (false negatives) instead of funding a risky loan (false positives).  

In the `loan_status` and `prediction` columns, a `0` means that the loan wont be paid off on time, and a `1` means that it will be paid off on time.  

1. Find the number of true negatives
    - Find the number of items where `predictions` is `0`, and the corresponding entry in `loans["loan_status"]` is also `0`
    - Assign the result to `tn`  
2. Find the number of true positives
    - Find the number of items where `predictions` is `1`, and the corresponding entry in `loans["loan_status"]` is also `1`
    - Assign the result to `tp`  
3. Find the number of false negatives
    - Find the number of items where `predictions` is `0`, and the corresponding entry in `loans["loan_status"]` is `1`
    - Assign the result to `fn`  
4. Find the number of false positives
    - Find the number of items where `predictions` is `1`, and the corresponding entry in `loans["loan_status"]` is `0`
    - Assign the result to `fp`  


There is **class imbalance** in the `loan_status` column. There are `6` times as many loans that were paid off on time (`1`), than loans that weren't paid off on time (`0`). This causes a major issue when we use accuracy as a metric. This is because due to the class imbalance, a classifier can predict `1` for every row, and still have high accuracy.  

it's important to always be aware of imbalanced classes in machine learning models, and to adjust your error metric accordingly. In this case, we don't want to use accuracy, and should instead use metrics that tell us the number of false positives and false negatives.  

This means that we should optimize for:  

- high recall (true positive rate)
- low fall-out (false positive rate)  

We can calculate false positive rate and true positive rate, using the numbers of true positives, true negatives, false negatives, and false positives.  

1. False Positive Rate: "what percentage of my 1 predictions are incorrect?"  
    - In this case, "what percentage of the loans that I fund would not be repaid?"  

2. True Positive Rate: "what percentage of all the possible 1 predictions am I making?"  
    - In this case, "what percentage of loans that could be funded would I fund?" 


```python
# Predict that all loans will be paid off on time
predictions = pd.Series(np.ones(loans.shape[0]))
# False positives
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
```

    1.0
    1.0


both `fpr` and `tpr` were `1`. This is because we predicted `1` for each row. This means that we correctly identified all of the good loans (true positive rate), but we also incorrectly identified all of the bad loans (false positive rate). Now that we've setup error metrics, we can move on to making predictions using a machine learning algorithm.

### 7.2 Logistic Regression

A good first algorithm to apply to binary classification problems is logistic regression, for the following reasons:  

- it's quick to train and we can iterate more quickly,
- it's less prone to overfitting than more complex models like decision trees,
- it's easy to interpret.


```python
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

cols = loans.columns
train_cols = cols.drop("loan_status")

features = loans[train_cols]

target = loans["loan_status"]

lr.fit(features, target)

predictions = lr.predict(features)
```

In order to get a realistic depiction of the accuracy of the algorithm, we'll need to use `cross validation` to generate predictions. Cross validation splits the dataset into groups, then makes predictions on each group using the other groups as training data. This ensures that we don't overfit by generating predictions on the same data that we train our algorithm with.  

We can perform cross validation using the `cross_val_predict` method of scikit-learn:  

1. cross_val_predict allows us to pass in a classifier, the features, and the target.  

2. We will create an instance of `KFold`, which will perform `3` fold cross validation across our dataset. We set `random_state` to `1` to ensure that the folds are always consistent, and we can compare scores between runs. If we don't, each fold will be randomized every time, making it hard to tell if we're improving our model or not.  

3. We can pass the instance of `KFold` into `cross_val_predict`, and it will then perform `3` fold cross validation to generate unbiased predictions.  

Once we have cross validated predictions, we can compute true positive rate and false positive rate.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, KFold
lr = LogisticRegression()
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
```

0.9988517209077449  
0.9969723953695458

#### Penalizing The Classifier

Our classifier is not accounting for the imbalance in the classes and we need to tell the classifier to penalize misclassifications of the less prevalent class more than the other class.  

We can do this by setting the `class_weight` parameter to `balanced` when creating the LogisticRegression instance. This tells scikit-learn to penalize the misclassification of the minority class during the training process. The penalty means that the logistic regression classifier pays more attention to correctly classifying rows where `loan_status` is `0`. This lowers accuracy when `loan_status` is `1`, but raises accuracy when `loan_status` is `0`.  

By setting the class_weight parameter to balanced, the penalty is set to be inversely proportional to the class frequencies. This would mean that for the classifier, correctly classifying a row where `loan_status` is `0` is 6 times more important than correctly classifying a row where `loan_status` is `1`.


```python
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
lr = LogisticRegression(class_weight="balanced")
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
```

0.6744024416039646  
0.39269813000890474

#### Manual Penalties

We significantly improved false positive rate by balancing the classes, which reduced true positive rate. Our true positive rate is now around `67%`, and our false positive rate is around `40%`. From a conservative investor's standpoint, it's reassuring that the false positive rate is lower because it means that we'll be able to do a better job at avoiding bad loans than if we funded everything. However, we'd only ever decide to fund `67%` of the total loans (true positive rate), so we'd immediately reject a good amount of loans.  

We can try to lower the false positive rate further by assigning a harsher penalty for misclassifying the negative class. While setting class_weight to balanced will automatically set a penalty based on the number of `1s` and `0s` in the column, we can also set a manual penalty. In the last screen, the penalty scikit-learn imposed for misclassifying a `0` would have been around `5.89` (since there are `5.89` times as many `1s` as `0s`). 

We can also specify a penalty manually if we want to adjust the rates more. To do this, we need to pass in a dictionary of penalty values to the `class_weight` parameter:


```python
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict
penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight=penalty)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
```

0.23340283443628562  
0.08637577916295637

It looks like assigning manual penalties lowered the false positive rate to around `8%`, and thus lowered our risk. Note that this comes at the expense of true positive rate. While we have fewer false positives, we're also missing opportunities to fund more loans and potentially make more money. Given that we're approaching this as a conservative investor, this strategy makes sense, but it's worth keeping in mind the tradeoffs. 

# 8. Random Forest

Random Forests are able to work with nonlinear data, and learn complex conditionals. Logistic Regressions are only able to work with linear data. Training a Random Forest algorithm may enable us to get more accuracy due to columns that correlate nonlinearly with `loan_status`.

We can use the `RandomForestClassifer` class from scikit-learn to do this:


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict
rf = RandomForestClassifier(class_weight="balanced", random_state=1)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(rf, features, target, cv=kf)
predictions = pd.Series(predictions)

# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)
```

0.971806726498051  
0.9330365093499555

# 9. Results

Unfortunately, using a RandomForestClassifier didn't improve our false positive rate. The model is likely weighting too heavily on the `1` class, and still mostly predicting `1s`. We could fix this by applying a harsher penalty for misclassifications of `0s`.  

Ultimately, our best model had a false positive rate of `8%`, and a true positive rate of `23%`. For a conservative investor, this means that they make money as long as the interest rate is high enough to offset the losses from `8%` of borrowers defaulting, and that the pool of `23%` of borrowers is large enough to make enough interest money to offset the losses.  

If we had randomly picked loans to fund, borrowers would have defaulted on `14.5%` of them, and our model is better than that, although we're excluding more loans than a random strategy would.

# 9. Recommendations
 
1. We can tweak the penalties further.
2. We can try models other than a random forest and logistic regression.
3. We can use some of the columns we discarded to generate better features.
4. We can ensemble multiple models to get more accurate predictions.
5. We can tune the parameters of the algorithm to achieve higher performance.
