import pandas as pd
import openai
import json

def analyze_dataset_with_gpt(df, api_key, max_rows=100, max_cols=20):
    """
    Analyze a dataset using OpenAI's GPT-4o-mini model.
    
    Args:
        df: Pandas DataFrame
        api_key: OpenAI API key
        max_rows: Maximum number of rows to include in analysis
        max_cols: Maximum number of columns to include in analysis
        
    Returns:
        Dictionary containing analysis results
    """
    # Limit dataset size if needed
    if df.shape[0] > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    if df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
    
    # Calculate basic statistics
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns
    
    stats = {}
    for col in numeric_columns:
        stats[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q1': df[col].quantile(0.25),
            'q3': df[col].quantile(0.75)
        }
    
    for col in categorical_columns:
        stats[col] = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].value_counts().head(3).to_dict()
        }
    
    # Get sample records
    sample_records = df.head(5).to_dict('records')
    
    # Prepare dataset information
    dataset_info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'statistics': stats,
        'sample_records': sample_records
    }
    
    # Prepare prompt for GPT
    prompt = f"""
    Please analyze this dataset and provide insights:
    
    Dataset Information:
    {json.dumps(dataset_info, indent=2)}
    
    Please provide:
    1. A brief overview of the dataset
    2. Key insights about the data
    3. Potential patterns or relationships
    4. Suggestions for further analysis
    5. Any data quality issues or concerns
    """
    
    # Call OpenAI API
    openai.api_key = api_key
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analysis expert. Provide clear, concise insights about the dataset."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        analysis = response.choices[0].message.content
        
        return {
            'success': True,
            'analysis': analysis,
            'dataset_info': dataset_info
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }