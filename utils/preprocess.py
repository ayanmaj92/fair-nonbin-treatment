import os
from sklearn.preprocessing import LabelEncoder


def preprocess_german(df, output_file):
    """ Preprocess German dataset: map gender, encode labels, and drop columns. """
    config = DATASET_CONFIG['german']
    df['gender'] = df['personal_status'].map(config['gender_map'])
    df = df.drop(columns=config['drop_columns'], axis=1)  # drop personal status column
    categorical_features = config.get('categorical_features', [])
    data_encoded = label_encoding(df, categorical_features)
    if config['reorder_columns']:
        data_ordered = reorder_columns(data_encoded, output_file, config['sensitive_attributes'],
                                       config['treatment_attributes'], config['target_attribute'])
    else:
        data_ordered = data_encoded
    return data_ordered.to_csv(output_file, index=False)


def preprocess_homecredit(df, output_file):
    """ Preprocess HomeCredit dataset: select columns, remove invalid gender, convert days to years. """
    config = DATASET_CONFIG['homecredit']
    df = df[config['selected_columns']]
    for col, values in config.get('drop_values', {}).items():
        df = df[~df[col].isin(values)]
    df[config['convert_columns']] = df[config['convert_columns']].apply(lambda x: x / -365)
    # Ensure no negative values remain
    df = df[(df['DAYS_BIRTH'] >= 0) & (df['DAYS_EMPLOYED'] >= 0)]
    # rename age column
    df['age'] = df.pop(config['age_column'])
    categorical_features = config.get('categorical_features', [])
    data_encoded = label_encoding(df, categorical_features)
    data_removed_missing = handle_missen_values(data_encoded, method='dropall', output_file=output_file)
    if config['reorder_columns']:
        data_ordered = reorder_columns(data_removed_missing, output_file, config['sensitive_attributes'],
                                       config['treatment_attributes'], config['target_attribute'])
    else:
        data_ordered = data_removed_missing
    return data_ordered.to_csv(output_file, index=False)


def preprocess_hmda(df, output_file):
    """ Preprocess HMDA dataset: drop columns, encode labels, and reorder columns. """
    config = DATASET_CONFIG['hmda']
    df.drop(columns=[col for col in config['columns_to_drop'] if col in df.columns], axis=1, inplace=True)

    # Binarize action_taken
    df = df.loc[~df['action_taken'].isin([3, 4, 5, 6, 7, 8])]  # filter out values {3-8}
    df.loc[:, 'action_taken'] = df['action_taken'].apply(lambda x: 1 if x == 1 else 0)

    # Binarize applicant_sex
    df = df.loc[~df['applicant_sex'].isin([3, 4])]
    df.loc[:, 'applicant_sex'] = df['applicant_sex'].apply(lambda x: 1 if x == 1 else 0)

    # Binarize applicant_race_1
    df = df.loc[~df['applicant_race_1'].isin([1, 2, 3])]
    df.loc[:, 'applicant_race_1'] = df['applicant_race_1'].apply(lambda x: 1 if x == 5 else 0)
    categorical_features = config.get('categorical_features', [])
    data_encoded = label_encoding(df, categorical_features)
    if reorder_columns:
        data_ordered = reorder_columns(data_encoded, output_file, config['sensitive_attributes'],
                                       config['treatment_attributes'], config['target_attribute'])
    else:
        data_ordered = data_encoded
    return data_ordered.to_csv(output_file, index=False)


def count_missen_columns(data):
    """ Returns the number of columns with missing values and a dictionary of missing values per column."""
    return data.isnull().sum().sum()


def handle_missen_values(data, method='dropall', output_file=None):
    if method == 'dropall':
        df_dropped_na = data.dropna()
        if output_file:
            df_dropped_na.to_csv(output_file, index=False)
        return df_dropped_na
    else:
        print(f"Unsupported method: {method}: Function can be edited to include new method")


def label_encoding(data, categorical_features):
    """ Apply label encoding to categorical features in the dataset. """
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])
        label_encoders[feature] = le
    return data


def reorder_columns(df, output_file, sensitive_attributes, treatment_attributes, target_attribute):
    """ Reorders dataset columns and applies label encoding to categorical features. """
    output_dir = './preprocess_data'
    os.makedirs(output_dir, exist_ok=True)
    # Reorder columns
    covariates = [col for col in df.columns if
                  col not in sensitive_attributes + treatment_attributes + target_attribute]
    reordered_columns = sensitive_attributes + covariates + treatment_attributes + target_attribute
    df_reordered = df[reordered_columns]
    df_reordered.to_csv(os.path.join(output_dir, output_file), index=False)
    print(f"Reordered columns: {df_reordered.columns}")
    return df_reordered


# Configuration dictionary for dataset-specific processing
DATASET_CONFIG = {
    'german': {
        'gender_map': {
            "'male single'": "male",
            "'female div/dep/mar'": "female",
            "'male div/sep'": "male",
            "'male mar/wid'": "male"
        },
        'drop_columns': ['personal_status'],
        'sensitive_attributes': ['gender', 'age'],
        'treatment_attributes': ['installment_commitment', 'duration', 'credit_amount'],
        'target_attribute': ['class'],
        'categorical_features': ['age',
                                 'gender', 'checking_status', 'credit_history', 'purpose', 'savings_status',
                                 'employment',
                                 'other_parties', 'property_magnitude', 'other_payment_plans',
                                 'housing', 'job', 'own_telephone', 'foreign_worker', 'class'
                                 ],
        'reorder_columns': True
    },
    'homecredit': {
        'selected_columns': [
            'EXT_SOURCE_2', 'EXT_SOURCE_3', 'EXT_SOURCE_1', 'DAYS_BIRTH', 'NAME_EDUCATION_TYPE',
            'CODE_GENDER', 'DAYS_EMPLOYED', 'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE',
            'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'TARGET'
        ],
        'drop_values': {'CODE_GENDER': ['XNA']},
        'convert_columns': ['DAYS_BIRTH', 'DAYS_EMPLOYED'],  # convert to years
        'age_column': 'DAYS_BIRTH',
        'sensitive_attributes': ['CODE_GENDER', 'age'],
        'treatment_attributes': ['AMT_CREDIT', 'AMT_ANNUITY'],
        'target_attribute': ['TARGET'],
        'categorical_features': ['NAME_EDUCATION_TYPE', 'CODE_GENDER', 'NAME_INCOME_TYPE', 'ORGANIZATION_TYPE',
                                 'TARGET'],
        'reorder_columns': True,

    },
    'hmda': {
        'columns_to_drop': [
            'as_of_year', 'minority_population', 'respondent_id', 'edit_status', 'agency_abbr',
            'sequence_number', 'agency_code', 'state_abbr', 'state_code', 'population', 'county_code',
            'application_date_indicator', 'applicant_race_2', 'applicant_race_3', 'co_applicant_race_4',
            'co_applicant_race_5', 'denial_reason_1', 'denial_reason_2', 'denial_reason_3',
            'applicant_race_4', 'co_applicant_race_2', 'co_applicant_race_3', 'applicant_race_5',
            'has_co_applicant', 'co_applicant_ethnicity', 'co_applicant_race_1', 'co_applicant_sex',
            'applicant_ethnicity'
        ],
        'sensitive_attributes': ['applicant_sex', 'applicant_race_1'],
        'treatment_attributes': ['loan_amount_000s', 'preapproval'],
        'target_attribute': ['action_taken'],
        'categorical_features': [
            'loan_type', 'property_type', 'loan_purpose', 'owner_occupancy', 'preapproval',
            'purchaser_type', 'applicant_race_1', 'applicant_sex', 'action_taken'
        ],
        'reorder_columns': True
    }
}
