import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath=None, n_samples=500):
    """
    Load clinical data from filepath or generate synthetic data.
    """
    if filepath:
        df = pd.read_csv(filepath)
        print(f"Data loaded from {filepath} — Shape: {df.shape}")
    else:
        print("No filepath provided — generating synthetic clinical data...")
        np.random.seed(42)
        n = n_samples
        df = pd.DataFrame({
            'patient_id': range(1, n+1),
            'age': np.random.randint(30, 85, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'diagnosis': np.random.choice(['Type A', 'Type B', 'Type C'], n),
            'comorbidities': np.random.randint(0, 5, n),
            'treatment': np.random.choice(['Treatment 1', 'Treatment 2', 'Control'], n),
            'duration': np.random.exponential(365, n).astype(int),
            'event_observed': np.random.choice([0, 1], n, p=[0.4, 0.6])
        })
        print(f"Synthetic data generated — Shape: {df.shape}")
    return df


def check_missing_values(df):
    """
    Report missing values in the dataset.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report = pd.DataFrame({
        'Missing Count': missing,
        'Missing %': missing_pct.round(2)
    })
    report = report[report['Missing Count'] > 0]
    if len(report) == 0:
        print("No missing values found!")
    else:
        print("=== Missing Values Report ===")
        print(report)
    return report


def impute_missing(df, strategy='median'):
    """
    Impute missing values.
    Numeric columns: median/mean
    Categorical columns: most frequent
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        print(f"Numeric columns imputed with {strategy}: {numeric_cols}")

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        print(f"Categorical columns imputed with most_frequent: {categorical_cols}")

    return df


def encode_categoricals(df, columns):
    """
    Label encode categorical columns.
    Returns encoded dataframe and encoder dict.
    """
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"Encoded: {col} → {col}_enc | Classes: {list(le.classes_)}")
    return df, encoders


def add_risk_group(df, age_col='age'):
    """
    Add risk group based on age bins.
    """
    df = df.copy()
    df['risk_group'] = pd.cut(
        df[age_col],
        bins=[0, 50, 65, 100],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    print("Risk groups added based on age bins.")
    print(df['risk_group'].value_counts())
    return df


def scale_features(df, columns):
    """
    Standardize numeric features using StandardScaler.
    """
    df = df.copy()
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"Scaled features: {columns}")
    return df, scaler


def remove_outliers(df, column, threshold=3.0):
    """
    Remove outliers using Z-score method.
    """
    df = df.copy()
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    original_len = len(df)
    df = df[z_scores < threshold]
    removed = original_len - len(df)
    print(f"Outlier removal on '{column}': {removed} rows removed | Remaining: {len(df)}")
    return df


def prepare_cox_features(df):
    """
    Full preprocessing pipeline for Cox PH model.
    Returns model-ready dataframe.
    """
    print("\n" + "="*50)
    print("  STARTING PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1: Check missing values
    print("\n[Step 1] Checking missing values...")
    check_missing_values(df)

    # Step 2: Impute missing values
    print("\n[Step 2] Imputing missing values...")
    df = impute_missing(df)

    # Step 3: Add risk group
    print("\n[Step 3] Adding risk groups...")
    df = add_risk_group(df)

    # Step 4: Encode categoricals
    print("\n[Step 4] Encoding categorical variables...")
    cat_cols = ['gender', 'diagnosis', 'treatment']
    df, encoders = encode_categoricals(df, cat_cols)

    # Step 5: Remove outliers on duration
    print("\n[Step 5] Removing outliers...")
    df = remove_outliers(df, 'duration')

    # Step 6: Select final features
    print("\n[Step 6] Selecting final features...")
    feature_cols = [
        'duration', 'event_observed',
        'age', 'gender_enc',
        'comorbidities', 'diagnosis_enc', 'treatment_enc'
    ]
    final_df = df[feature_cols]

    print("\n" + "="*50)
    print("  PREPROCESSING COMPLETE")
    print(f"  Final shape: {final_df.shape}")
    print("="*50 + "\n")

    return final_df, encoders


def data_quality_report(df):
    """
    Generate a full data quality report.
    """
    print("\n" + "="*50)
    print("       DATA QUALITY REPORT")
    print("="*50)
    print(f"Total Rows         : {len(df)}")
    print(f"Total Columns      : {df.shape[1]}")
    print(f"Duplicate Rows     : {df.duplicated().sum()}")
    print(f"Missing Values     : {df.isnull().sum().sum()}")
    print(f"Numeric Columns    : {len(df.select_dtypes(include=np.number).columns)}")
    print(f"Categorical Columns: {len(df.select_dtypes(include='object').columns)}")
    print("="*50)
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nBasic Statistics:")
    print(df.describe())


# -------------------------
# Run pipeline if executed directly
# -------------------------
if __name__ == "__main__":
    # Load data
    df = load_data()

    # Full quality report
    data_quality_report(df)

    # Run full preprocessing
    final_df, encoders = prepare_cox_features(df)

    print("Sample of processed data:")
    print(final_df.head())
