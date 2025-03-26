
# 📚 Dataset Preprocessing Guide & Data Overview 📌


## 1️⃣ General Preprocessing Requirements

To ensure that the dataset complies with the code, all datasets must meet the following requirements:

- File Format: CSV files with column names in the first row.

- Column Order:

       * Sensitive features (S): These should be binary where $1$ refer to the males and $0$ refers to females

       *  Covariates (X)

       *  Treatment decisions (Z)

       * Outcome (Y): These should be binary where $1$ will refer to a positive class i.e non-default loan, and $0$ refers to a negative class, i.e., default loan.

- Encoding:

       -  Categorical variables must be label-encoded.

       -  Numerical variables are handled by the causal normalizing flow, so no need for additional normalization.

- Handling Missing Values:

Missing values must either be imputed or dropped, depending on dataset constraints.

##  2️⃣ 🚀 How to add support for a new dataset

We provide a ```preprocess.py``` script which includes functions for feature selection, missing value handling, encoding categorical variables, and reordering columns.

To preprocess a new dataset, follow these steps:

**Step 1**: Modify `DATASET_CONFIG` in `preprocess.py` to include the new dataset. NB: This can be modified as needed to accommodate  data-specific details. For the current datasets, we provide the following:
- `selected_columns`: List of columns to keep if not interested in all columns
- `drop_columns`: Columns to remove if they exist
- `drop_values`: Values to exclude (e.g., invalid categories)
- `convert_columns`: Columns that need transformation (e.g., date to years)
- `sensitive_attributes`:e.g., gender, race
- `treatment_attributes`: e.g duration, credit amount
- `target_attribute`: The label to predict
- `categorical_features`: Features that require encoding
- `reorder_columns`: Boolean flag to reorder columns (sensitive, covariates, treatments, outcome) after processing

**Example: Adding `new_dataset` to `DATASET_CONFIG`**
```python
DATASET_CONFIG = {
    'new_dataset': {
        'selected_columns': ['feature1', 'feature2', 'feature3', 'target'],
        'drop_columns': ['irrelevant_feature1', 'irrelevant_feature2'],
       'drop_values':{'CODE_GENDER': ['XNA']},
        'convert_columns': ['date_column'],  # Convert days to years if needed
        'sensitive_attributes': ['gender', 'age'],  
        'treatment_attributes': ['feature1', 'feature2'],  
        'target_attribute': ['target'],
        'categorical_features': ['feature1', 'feature2'],  
        'reorder_columns': True  
    }
} 
```

**Step 2**: Include a new preprocess function for the dataset following the pattern of the existing datasets.


**Step 3**: You can simply then use the preprocess script by:

```py
from utils import preprocess
import pandas as pd
filepath = 'path/to/saved/data.csv'
output_file = 'data.csv'
data = pd.read_csv(filepath)
preprocess.preprocess_german(data, output_file)
```
Keep the preprocessing steps consistent across datasets to ensure compatibility.

**Step 3**: Save the preprocessed and re-ordered datasets  as `data.csv` and place it in the folder `./Data/treatment_{new dataset name}`. Ensure that all datasets are always stored in this folder and follow the naming convention specified.


##  3️⃣ Synthetic Data Generation

We provide scripts to generate synthetic datasets. The synthetic dataset can be generated using the [generate synthetic dataset](https://github.com/ayanmaj92/beyond-bin-decisions/blob/public/notebooks/generate_synthetic_data.ipynb), simply run the notebook and then have the datasets saved in the folder ./Data/treatment_synthetic 


##  4️⃣ 📊 Dataset-Specific Preprocessing for this paper


###  1.🇩🇪💳[German Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

The dataset was downloaded from [Statlog](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data). It contains 1000 data points with no NaN and 21 features. We only converted the personal status which had information on people's gender and marital status to only a gender column. 

```py
gender_dict = {
            "'male single'": "male",
            "'female div/dep/mar'": "female",
            "'male div/sep'": "male",
            "'male mar/wid'": "male"}
```
            
Based on this we encoded `{Male: 1, Female:0}`; and the Class default  : `{Good:1, Bad:0}`. 
We provide the full dataset overview below and also information on what we used as a categorical feature.
Most importantly, we need to select the features for each of the endogenous variables of our causal graph carefully.

- **Sensitive Attribute(S)**: {`gender`, `age`}.

- **Covariates(X)**: {`checking_status`, `credit_history`, `purpose`,
       `savings_status`, `employment`, `other_parties`, `residence_since`,
       `property_magnitude`, `other_payment_plans`, `housing`,
       `existing_credits`, `job`, `num_dependents`, `own_telephone`,
       `foreign_worker`}.

- **Treatments(Z)**: {`installment_commitment`,`credit_amount` ,`duration`}.

- **Target(Y)**: {`class`}.

Below is a summary table describing each column in the processed dataset, including its type and description. 

| Feature Name       | Description                         | Categorical Type|
|-------------------|-------------------------------------|------------------|
| `gender`          | The gender of the applicant         | Yes              |
| `age`             | The age of the applicant.           |                  |
| `checking_status` | Status of existing checking account |Yes               |
| `credit_history`  | Credit history of the applicant     | Yes              |
| `purpose`         | Purpose of the loan                 |   Yes            |
| `savings_status`  | Savings account or bonds            |   Yes            |
| `employment`      | Present Employment                    | Yes            |
| `other_parties`   | Other parties involved in the loan  | Yes              |
| `property_magnitude` |  property if real estate or no proprty etc |  Yes   |
| `other_payment_plans` | Other payment plans             |  Yes             |
| `housing`         | Housing status                      | Yes              |
| `residence_since` | Present residence since      |       |
| `existing_credits`| Number of existing credits at this bank         |       |
| `job`             | Job                         |  Yes   |
| `num_dependents`  | Number of people being liable to provide maintenance for                |       |
| `own_telephone`   | Whether the applicant owns a telephone |  Yes    |
| `foreign_worker`  | Whether the applicant is a foreign worker | Yes  |
| `duration`        | Duration of the credit in months    |       |
| `credit_amount`   | Amount of credit in Deutsch Mark                   |       |
| `installment_commitment` | Installment rate in % of disposable income |     |
| `class` |default if someone paid back or not |  Yes   |



### 2.  🏠💰 [Homecredit Dataset](https://www.kaggle.com/c/home-credit-default-risk/discussion/63032)

This dataset is from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/discussion/63032) and has a number of CSV files (application train, bureau, bureau balance, credit card balance, installments payments, previous applications). However, we only use the application_train.csv file for our analysis.

Following the feature importance done by [this study](https://libstore.ugent.be/fulltxt/RUG01/002/790/702/RUG01-002790702_2019_0001_AC.pdf) using random forest and the information entropy, we selected the most important features for our study. These features were: 
`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `DAYS_BIRTH`, `NAME_EDUCATION_TYPE`, `CODE_GENDER`, `DAYS_EMPLOYED`, `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`, `AMT_CREDIT`, `AMT_ANNUITY`, `AMT_GOODS_PRICE`, `REGION_POPULATION_RELATIVE`, `TARGET`.

Based on this subset of data, we then removed the missing values in the dataset and converted the days of birth and days of employment to years using $\text{Days of birth or employment} / -365$. We divided by -365 because these values were in negatives. However, after doing this we still saw a few values of the days of employment in negatives, which we dropped. The complete dataset had a size of $98,859$ with $14$ features.

Additionally, for our causal graph these were the features:

- **Sensitive Attribute(S)**: {`CODE_GENDER`, `age`}.

- **Covariates(X)**: {`EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1`, `NAME_EDUCATION_TYPE`, `DAYS_EMPLOYED`, `NAME_INCOME_TYPE`, `ORGANIZATION_TYPE`, `AMT_GOODS_PRICE`, `REGION_POPULATION_RELATIVE`}.

- **Treatments(Z)**: {`AMT_CREDIT`, `AMT_ANNUITY`}.

- **Target(Y)**: {`TARGET`}.

Below is a summary table describing each column in the processed dataset, including its type and description. 


| Feature Name                  | Description                                          | Categorical Type |
|-------------------------------|------------------------------------------------------|------------------|
| `CODE_GENDER`                 | Gender of the applicant                              | Yes              |
| `age`                         | Age of the applicant                                 |                  |
| `EXT_SOURCE_2`                | External source 2 score                              |                  |
| `EXT_SOURCE_3`                | External source 3 score                              |                  |
| `EXT_SOURCE_1`                | External source 1 score                              |                  |
| `NAME_EDUCATION_TYPE`         | Education level of the applicant                     | Yes              |
| `DAYS_EMPLOYED`               | Number of days employed converted to years           |                  |
| `NAME_INCOME_TYPE`            | Type of income                                       | Yes              |
| `ORGANIZATION_TYPE`           | Type of organization where the applicant is employed | Yes              |
| `AMT_GOODS_PRICE`             | Price of goods for which the loan is requested       |                  |
| `REGION_POPULATION_RELATIVE`  | Relative population of the region                    |                  |
| `AMT_CREDIT`                  | Amount of credit requested                           |                  |
| `AMT_ANNUITY`                 | Annuity amount                                       |                  |
| `TARGET`                      | Target variable (whether the loan was repaid or not) | Yes              |


### 3. 🤠🏦 [HMDA 2017 Texas](https://github.com/pasta41/hmda-data-2017/tree/main/2017/TX)

HMDA dataset meaning the Home Mortgage Disclosure Act (HMDA) is a dataset about mortgage applications which will inform whether a consumer got a mortgage or not or something else happened. The dataset covers states in the United States, but we focused on just two states (New York, Texas) as done in [this study](https://arxiv.org/pdf/2301.11562#page=31.36). However, we treat them as separate datasets, though the preprocessing is similar for both datasets. Following the data preprocessing used by [this study](https://arxiv.org/pdf/2301.11562#page=31.36), we binarized the gender column into Male=1 and Female=0. Equally, we binarized the race which had originally $8$ possible values, so we utilized only Native White = 0 and White = 1. However, with the action taken which had $8$ values, we filtered out ${3-8}$ and kept 1-Loan originated as the applicant and the bank decision agreed and 2- which is application approved but not accepted to mean a disagreement. After the analysis we had a total dataset of size $399354$ with $17$ features.

We downloaded the dataset for New York [here](https://github.com/pasta41/hmda-data-2017/tree/main/2017/NY) and the Texas one from [here](https://github.com/pasta41/hmda-data-2017/tree/main/2017/TX).The descriptions of features: [Code sheet](https://files.consumerfinance.gov/hmda-historic-data-dictionaries/lar_record_codes.pdf).
For HMDA, it is additionally required to first load the separate CSV files of the features, protected, and target and concatenate them. This concatenated dataframe can then be used by our HMDA preprocessing function.

#### Merge data files

For this dataset, we first merge the  the csvs of the features, protected attributes and the target.

```py
tx_features = pd.read_csv('./data/hmda/2017-TX-features.csv')
tx_protected = pd.read_csv('./data/hmda/2017-TX-protected.csv')
tx_target = pd.read_csv('./data/hmda/2017-TX-target.csv')
texas_df = pd.concat([tx_features, tx_protected, tx_target])
data_hmda_texas = pd.concat([tx_features, tx_protected, tx_target], axis =1)
```

##### Missing Data

The dataset also contained a lot of missing values, hence we first removed the columns that already had a lot of null values and also information regarding the co-applicant. These features were removed:

[ `as_of_year`, `minority_population`, `respondent_id`, `edit_status`, `agency_abbr`,
        `sequence_number`,`agency_code`, `state_abbr`, `state_code`, `population`, `county_code`, 
        `application_date_indicator`, `applicant_race_2`, `applicant_race_3`, `co_applicant_race_4`,
        `co_applicant_race_5`,`denial_reason_1`, `denial_reason_2`, `denial_reason_3`, 
        `applicant_race_4`, `co_applicant_race_2`,`co_applicant_race_3`,`applicant_race_5`,
        `has_co_applicant`, `co_applicant_ethnicity`,`co_applicant_race_1`, `co_applicant_sex`, `applicant_ethnicity`].

Additionally, for our causal graph these were the features:

- **Sensitive Attributes(S)**:{ `applicant_sex`, `applicant_race_1`}.

- **Covariates(X)**: {`property_type`, `loan_purpose`, `owner_occupancy`, `msamd`, `census_tract_number`, `applicant_income_000s`, `purchaser_type`, `hud_median_family_income`, `tract_to_msamd_income`, `number_of_owner_occupied_units`, `number_of_1_to_4_family_units`,`loan_type`}.

- **Treatments(Z)**: {`loan_amount_000s`, `preapproval`}.

- **Target(Y)**: {`action_taken`}.

Below is a summary table describing each column in the processed dataset, including its type and description. 

| Feature Name                    | Description                                                          | Categorical Type |
|---------------------------------|----------------------------------------------------------------------|------------------|
| `applicant_sex`                 | Gender of the applicant                                                 | Yes              |
| `applicant_race_1`              | Race of the applicant                                                | Yes              |
| `property_type`                 | Type of property (1-4 family, multi-family, or manufactured housing) | Yes              |
| `loan_purpose`                  | Purpose of the loan                                                  | Yes              |
| `owner_occupancy`               | Whether the property is owner-occupied                               | Yes              |
| `msamd`                         | Metropolitan Statistical Area/Metropolitan Division                  |                  |
| `census_tract_number`           | Census tract number                                                  |                  |
| `applicant_income_000s`         | Applicant income in thousands of dollars                             |                  |
| `purchaser_type`                | Type of purchaser of the loan                                        | Yes              |
| `hud_median_family_income`      | HUD median family income                                             |                  |
| `tract_to_msamd_income`         | % of tract median family income                                      |                  |
| `number_of_owner_occupied_units`| Number of dwellings including individual condominiums that are lived in by the owner|   |
| `number_of_1_to_4_family_units` | Dwellings that are built to house fewer than 5 families              |                  |
| `loan_type`                     | Type of loan (Conventional, FHA-insured, VA-guaranteed, FSA/RHS)     | Yes              |
| `loan_amount_000s`              | Loan amount in thousands of dollars                                  |                  |
| `preapproval`                   | Preapproval requested, not requested, or not applicable              | Yes              |
| `action_taken`                  | Action taken on the application                                      | Yes              |


### 4. 🏙️🏦 [HMDA New York](https://github.com/pasta41/hmda-data-2017/tree/main/2017/NY)


Similar analysis as Texas.  But we had a dataset of size 166637 with 17 features.
