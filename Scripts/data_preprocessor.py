# import all necessary libraries here
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score




# 1. Impute Missing Values
def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    :param data: pandas DataFrame    
    :param strategy: str, imputation method ('mean', 'median', 'mode')
    :return: pandas DataFrame
    
    The function performs the following steps:
    1) identify type of column
    2) iterate over each column, based on strategy fill missing values
    3) Returns imputed data
    
    """
    
    data_imputed = data.copy() # make copy to track back to original
    
    # identify numerical and object columns, since object will take only mode strategy
    numerical_col_name = list(data_imputed.select_dtypes(include=['number']).columns)
    categorical_col_name = list(data_imputed.select_dtypes(include=['object']).columns)

    strategy_opt = strategy
    
    
    for colname in numerical_col_name[1:]:    #  DON'T TOUCH target, lesson learned hard     
            
                if strategy_opt == 'mean':
                        data_imputed[colname].fillna(data_imputed[colname].mean(), inplace=True)
                elif strategy_opt == 'median':
                        data_imputed[colname].fillna(data_imputed[colname].median(), inplace=True)
                elif strategy_opt == 'mode':
                        data_imputed[colname].fillna(data_imputed[colname].mode()[0], inplace=True)


    for colname in categorical_col_name:
            
            if strategy_opt == 'mode':
                    data_imputed[colname].fillna(data_imputed[colname].mode()[0], inplace=True)
    
    return data_imputed
    

   

# 2. Remove Duplicates
def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    :param data: pandas DataFrame
    :return: pandas DataFrame
    
    Removes duplicated row if any
    Returns refined data
    """
    data_no_duplicates = data.copy().drop_duplicates() 
    
    return data_no_duplicates



# 3. Normalize Numerical Data
def normalize_data(data,method='minmax'):
    """Apply normalization to numerical features.
    :param data: pandas DataFrame
    :param method: str, normalization method ('minmax' (default) or 'standard')
    
    Function will take numeric columns and scaler them based on method
    Returns normalized data
    """
    data_scaled= data.copy()
    # scaling will apply only on numeric columns
    numerical_col_name = list(data_scaled.select_dtypes(include=['number']).columns)[1:] #exclude target column
      
    method_use = method
    # apply scaling on numeric columns only
    if method_use == "minmax":
      scaler = MinMaxScaler()
      data_scaled[numerical_col_name] =  scaler.fit_transform(data_scaled[numerical_col_name])
    elif method_use == "standard":
      scaler = StandardScaler()
      data_scaled[numerical_col_name] =  scaler.fit_transform(data_scaled[numerical_col_name])
          
    return data_scaled

# 4. Remove Redundant Features   
def remove_redundant_features(data, threshold=0.9):
    """Remove redundant or duplicate columns.
    :param data: pandas DataFrame
    :param threshold: float, correlation threshold
    :return: pandas DataFrame
    
    1) will create correlation matrix on numeric columns, transform it to data frame
    2) will search extract rows within treshold criteria
    3) identify row/column names and remove from actual dataset
    
    Returns refined data
    
    """
    corr_data = data.copy()
    numerical_col_name = list(corr_data.select_dtypes(include=['number']).columns)
    
    threshold_value = threshold
    
    corr_matrix = pd.DataFrame(corr_data[numerical_col_name].corr().abs())
    mask = (corr_matrix > threshold_value) & (corr_matrix < 1.0) # 1 correspond to same column if we don't exclude it we will lose all data
    rows = np.where(mask)[0]
    row_names = list(corr_matrix.index[rows].unique())
    corr_data = corr_data.drop(row_names, axis=1)


    return corr_data
    

# ---------------------------------------------------

def simple_model(input_data, split_data=True, scale_data=False, print_report=False):
    """
    A simple logistic regression model for target classification.
    Parameters:
    input_data (pd.DataFrame): The input data containing features and the target variable 'target' (assume 'target' is the first column).
    split_data (bool): Whether to split the data into training and testing sets. Default is True.
    scale_data (bool): Whether to scale the features using StandardScaler. Default is False.
    print_report (bool): Whether to print the classification report. Default is False.
    Returns:
    None
    The function performs the following steps:
    1. Removes columns with missing data.
    2. Splits the input data into features and target.
    3. Encodes categorical features using one-hot encoding.
    4. Splits the data into training and testing sets (if split_data is True).
    5. Scales the features using StandardScaler (if scale_data is True).
    6. Instantiates and fits a logistic regression model.
    7. Makes predictions on the test set.
    8. Evaluates the model using accuracy score and classification report.
    9. Prints the accuracy and classification report (if print_report is True).
    """

    # if there's any missing data, remove the columns
    input_data.dropna(inplace=True)

    # split the data into features and target
    target = input_data.copy()[input_data.columns[0]]
    features = input_data.copy()[input_data.columns[1:]]

    # if the column is not numeric, encode it (one-hot)
    for col in features.columns:
        if features[col].dtype == 'object':
            features = pd.concat([features, pd.get_dummies(features[col], prefix=col)], axis=1)
            features.drop(col, axis=1, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target, random_state=42)

    if scale_data:
        # scale the data
        X_train = normalize_data(X_train)
        X_test = normalize_data(X_test)
        
    # instantiate and fit the model
    log_reg = LogisticRegression(random_state=42, max_iter=100, solver='liblinear', penalty='l2', C=1.0)
    log_reg.fit(X_train, y_train)

    # make predictions and evaluate the model
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    
    # if specified, print the classification report
    if print_report:
        print('Classification Report:')
        print(report)
        print('Read more about the classification report: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html and https://www.nb-data.com/p/breaking-down-the-classification')
    
    return None

