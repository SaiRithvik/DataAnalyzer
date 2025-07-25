import pandas as pd
import numpy as np
from scipy import stats

def get_basic_statistics(df, numeric_columns):
    """
    Calculate basic statistics for numeric columns in the dataframe.
    
    Args:
        df: Pandas DataFrame
        numeric_columns: List of numeric column names
        
    Returns:
        DataFrame with statistics
    """
    stats_data = []
    
    for col in numeric_columns:
        col_data = df[col].dropna()
        
        if len(col_data) > 0:
            stats_dict = {
                'Column': col,
                'Count': len(col_data),
                'Missing': df[col].isnull().sum(),
                'Mean': col_data.mean(),
                'Median': col_data.median(),
                'Mode': col_data.mode().iloc[0],
                'Std Dev': col_data.std(),
                'Min': col_data.min(),
                'Max': col_data.max(),
                '25%': col_data.quantile(0.25),
                '75%': col_data.quantile(0.75),
                'Skewness': stats.skew(col_data),
                'Kurtosis': stats.kurtosis(col_data)
            }
            
            stats_data.append(stats_dict)
    
    # Create DataFrame and reorder columns
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        col_order = ['Column', 'Count', 'Missing', 'Mean', 'Median', 'Mode', 'Std Dev', 
                     'Min', '25%', '75%', 'Max', 'Skewness', 'Kurtosis']
        stats_df = stats_df[col_order]
    else:
        stats_df = pd.DataFrame()
    
    return stats_df

def analyze_correlations(df, numeric_columns, method='pearson'):
    """
    Analyze correlations between numeric columns.
    
    Args:
        df: Pandas DataFrame
        numeric_columns: List of numeric column names
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        DataFrame with pair-wise correlations
    """
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr(method=method)
    
    # Convert to long format
    corr_data = []
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only include each pair once (and exclude self-correlations)
                corr_value = corr_matrix.loc[col1, col2]
                corr_data.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': corr_value,
                    'Absolute Correlation': abs(corr_value)
                })
    
    # Create DataFrame and sort by absolute correlation (descending)
    corr_df = pd.DataFrame(corr_data)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('Absolute Correlation', ascending=False)
    
    return corr_df

def detect_outliers(df, column, method='IQR', iqr_factor=1.5, z_threshold=3):
    """
    Detect outliers in a numeric column using IQR or Z-Score method.
    
    Args:
        df: Pandas DataFrame
        column: Column name to analyze
        method: Detection method ('IQR' or 'Z-Score')
        iqr_factor: Factor multiplied with IQR for outlier detection (for IQR method)
        z_threshold: Z-score threshold for outlier detection (for Z-Score method)
        
    Returns:
        Tuple of (outliers DataFrame, outlier indices, summary dict)
    """
    data = df[column].dropna()
    outlier_indices = []
    
    if method == 'IQR':
        # IQR method
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        
        # Get outlier indices
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_indices = df[outlier_mask].index.tolist()
        
        summary = {
            'Method': 'IQR',
            'IQR Factor': iqr_factor,
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outliers Count': len(outlier_indices),
            'Outliers Percentage': f"{(len(outlier_indices) / len(data) * 100):.2f}%"
        }
    
    else:  # Z-Score method
        z_scores = np.abs(stats.zscore(data))
        
        # Get outlier indices
        outlier_mask = z_scores > z_threshold
        outlier_indices = df[outlier_mask].index.tolist()
        
        summary = {
            'Method': 'Z-Score',
            'Z-Score Threshold': z_threshold,
            'Outliers Count': len(outlier_indices),
            'Outliers Percentage': f"{(len(outlier_indices) / len(data) * 100):.2f}%"
        }
    
    # Get outlier rows
    outliers = df.loc[outlier_indices]
    
    return outliers, outlier_indices, summary

def analyze_missing_data(df):
    """
    Analyze missing data in the dataframe.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        DataFrame with missing data statistics
    """
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    # Create summary DataFrame
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count.values,
        'Missing Percentage': missing_percent.values
    })
    
    # Sort by missing percentage (descending)
    missing_df = missing_df.sort_values('Missing Percentage', ascending=False)
    
    return missing_df

def generate_data_profile(df):
    """Generate a comprehensive data profile."""
    profile = {
        'Shape': df.shape,
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB",
        'Duplicate Rows': df.duplicated().sum(),
        'Column Types': df.dtypes,
        'Column Details': []
    }
    
    for col in df.columns:
        col_profile = {
            'name': col,
            'type': str(df[col].dtype),
            'missing': df[col].isnull().sum(),
            'unique_values': df[col].nunique()
        }
        if df[col].dtype in ['int64', 'float64']:
            col_profile.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            })
        profile['Column Details'].append(col_profile)
    
    return profile
