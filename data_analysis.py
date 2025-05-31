import polars as pl
import numpy as np
from scipy import stats

def get_basic_statistics(df, numeric_columns):
    """
    Calculate basic statistics for numeric columns in the dataframe.
    
    Args:
        df: Polars DataFrame
        numeric_columns: List of numeric column names
        
    Returns:
        DataFrame with statistics
    """
    stats_data = []
    
    for col in numeric_columns:
        col_data = df.select(pl.col(col)).drop_nulls()
        
        if col_data.height > 0:
            stats_dict = {
                'Column': col,
                'Count': col_data.height,
                'Missing': df.select(pl.col(col).null_count()).item(),
                'Mean': col_data.select(pl.col(col).mean()).item(),
                'Median': col_data.select(pl.col(col).median()).item(),
                'Mode': col_data.select(pl.col(col).mode().first()).item(),
                'Std Dev': col_data.select(pl.col(col).std()).item(),
                'Min': col_data.select(pl.col(col).min()).item(),
                'Max': col_data.select(pl.col(col).max()).item(),
                '25%': col_data.select(pl.col(col).quantile(0.25)).item(),
                '75%': col_data.select(pl.col(col).quantile(0.75)).item(),
                'Skewness': stats.skew(col_data.select(pl.col(col)).to_series()),
                'Kurtosis': stats.kurtosis(col_data.select(pl.col(col)).to_series())
            }
            
            stats_data.append(stats_dict)
    
    # Create DataFrame and reorder columns
    if stats_data:
        stats_df = pl.DataFrame(stats_data)
        col_order = ['Column', 'Count', 'Missing', 'Mean', 'Median', 'Mode', 'Std Dev', 
                     'Min', '25%', '75%', 'Max', 'Skewness', 'Kurtosis']
        stats_df = stats_df.select(col_order)
    else:
        stats_df = pl.DataFrame()
    
    return stats_df

def analyze_correlations(df, numeric_columns, method='pearson'):
    """
    Analyze correlations between numeric columns.
    
    Args:
        df: Polars DataFrame
        numeric_columns: List of numeric column names
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        DataFrame with pair-wise correlations
    """
    # Calculate correlation matrix
    corr_matrix = df.select(numeric_columns).corr()
    
    # Convert to long format
    corr_data = []
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Only include each pair once (and exclude self-correlations)
                corr_value = corr_matrix.select(pl.col(col1).corr(pl.col(col2))).item()
                corr_data.append({
                    'Variable 1': col1,
                    'Variable 2': col2,
                    'Correlation': corr_value,
                    'Absolute Correlation': abs(corr_value)
                })
    
    # Create DataFrame and sort by absolute correlation (descending)
    corr_df = pl.DataFrame(corr_data)
    if corr_df.height > 0:
        corr_df = corr_df.sort('Absolute Correlation', descending=True)
    
    return corr_df

def detect_outliers(df, column, method='IQR', iqr_factor=1.5, z_threshold=3):
    """
    Detect outliers in a numeric column using IQR or Z-Score method.
    
    Args:
        df: Polars DataFrame
        column: Column name to analyze
        method: Detection method ('IQR' or 'Z-Score')
        iqr_factor: Factor multiplied with IQR for outlier detection (for IQR method)
        z_threshold: Z-score threshold for outlier detection (for Z-Score method)
        
    Returns:
        Tuple of (outliers DataFrame, outlier indices, summary dict)
    """
    data = df.select(pl.col(column)).drop_nulls()
    outlier_indices = []
    
    if method == 'IQR':
        # IQR method
        q1 = data.select(pl.col(column).quantile(0.25)).item()
        q3 = data.select(pl.col(column).quantile(0.75)).item()
        iqr = q3 - q1
        
        lower_bound = q1 - iqr_factor * iqr
        upper_bound = q3 + iqr_factor * iqr
        
        # Get outlier indices
        outlier_mask = df.select(
            (pl.col(column) < lower_bound) | (pl.col(column) > upper_bound)
        ).to_series()
        outlier_indices = df.filter(outlier_mask).row_nr().to_list()
        
        summary = {
            'Method': 'IQR',
            'IQR Factor': iqr_factor,
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outliers Count': len(outlier_indices),
            'Outliers Percentage': f"{(len(outlier_indices) / data.height * 100):.2f}%"
        }
    
    else:  # Z-Score method
        z_scores = np.abs(stats.zscore(data.to_series()))
        
        # Get outlier indices
        outlier_mask = z_scores > z_threshold
        outlier_indices = df.filter(outlier_mask).row_nr().to_list()
        
        summary = {
            'Method': 'Z-Score',
            'Z-Score Threshold': z_threshold,
            'Outliers Count': len(outlier_indices),
            'Outliers Percentage': f"{(len(outlier_indices) / data.height * 100):.2f}%"
        }
    
    # Get outlier rows
    outliers = df.filter(pl.Series(name="row_nr", values=outlier_indices))
    
    return outliers, outlier_indices, summary

def analyze_missing_data(df):
    """
    Analyze missing data in the dataframe.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        DataFrame with missing data statistics
    """
    # Calculate missing values
    missing_count = df.null_count()
    missing_percent = missing_count / df.height * 100
    
    # Create summary DataFrame
    missing_df = pl.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_count.row(0),
        'Missing Percentage': missing_percent.row(0)
    })
    
    # Sort by missing percentage (descending)
    missing_df = missing_df.sort('Missing Percentage', descending=True)
    
    return missing_df

def generate_data_profile(df):
    """Generate a comprehensive data profile."""
    profile = {
        'Shape': (df.height, df.width),
        'Memory Usage': f"{df.estimated_size() / 1024 / 1024:.2f} MB",
        'Duplicate Rows': df.unique().height - df.height,
        'Column Types': df.dtypes,
        'Column Details': []
    }
    
    for col in df.columns:
        col_profile = {
            'name': col,
            'type': str(df.select(pl.col(col)).dtypes[0]),
            'missing': df.select(pl.col(col).null_count()).item(),
            'unique_values': df.select(pl.col(col).n_unique()).item()
        }
        if df.select(pl.col(col)).dtypes[0] in [pl.Int64, pl.Float64]:
            col_profile.update({
                'mean': df.select(pl.col(col).mean()).item(),
                'std': df.select(pl.col(col).std()).item(),
                'min': df.select(pl.col(col).min()).item(),
                'max': df.select(pl.col(col).max()).item()
            })
        profile['Column Details'].append(col_profile)
    
    return profile
