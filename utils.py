import polars as pl
import numpy as np
import io

def get_data_types(df):
    """
    Categorize columns into numeric and categorical types.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        Tuple of (numeric_columns, categorical_columns)
    """
    numeric_columns = []
    categorical_columns = []
    
    for column in df.columns:
        # Check if column is numeric
        if df.select(pl.col(column)).dtypes[0] in [pl.Int64, pl.Float64]:
            numeric_columns.append(column)
        else:
            # Try to convert to numeric to catch strings that are actually numbers
            try:
                df.select(pl.col(column).cast(pl.Float64))
                numeric_columns.append(column)
            except:
                categorical_columns.append(column)
    
    return numeric_columns, categorical_columns

def convert_df_to_csv(df):
    """
    Convert DataFrame to CSV for download.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        CSV string
    """
    return df.write_csv().encode('utf-8')

def convert_df_to_excel(df):
    """
    Convert DataFrame to Excel for download.
    
    Args:
        df: Polars DataFrame
        
    Returns:
        Excel bytes object
    """
    # Convert to pandas for Excel export
    df_pd = df.to_pandas()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_pd.to_excel(writer, index=False, sheet_name='Data')
    
    output.seek(0)
    return output.getvalue()

def generate_sample_data(rows=100):
    """
    Generate sample data for demonstration purposes.
    
    Args:
        rows: Number of rows to generate
        
    Returns:
        Polars DataFrame with sample data
    """
    np.random.seed(42)
    
    # Generate some random data
    data = {
        'ID': range(1, rows + 1),
        'Name': [f"Person {i}" for i in range(1, rows + 1)],
        'Age': np.random.randint(18, 65, rows),
        'Salary': np.random.normal(50000, 15000, rows).round(2),
        'Department': np.random.choice(['HR', 'IT', 'Finance', 'Marketing', 'Sales'], rows),
        'Years': np.random.randint(0, 20, rows),
        'Performance': np.random.normal(3.5, 0.5, rows).round(1),
        'Satisfaction': np.random.uniform(1, 5, rows).round(1)
    }
    
    # Add some missing values
    for col in ['Age', 'Salary', 'Years', 'Performance', 'Satisfaction']:
        mask = np.random.random(rows) < 0.1  # 10% missing
        data[col] = [None if mask[i] else data[col][i] for i in range(rows)]
    
    # Create DataFrame
    df = pl.DataFrame(data)
    
    return df
