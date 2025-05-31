import polars as pl
import json
from openai import OpenAI

def analyze_dataset_with_gpt(df, api_key, max_rows=100, max_cols=20):
    """
    Analyze a dataset using OpenAI's GPT-4o-mini model.
    
    Args:
        df: Polars DataFrame
        api_key: OpenAI API key
        max_rows: Maximum rows to send to the API
        max_cols: Maximum columns to send to the API
        
    Returns:
        List of insights about the dataset
    """
    # Limit dataset size for API request
    sample_df = df
    if df.height > max_rows:
        sample_df = df.sample(n=max_rows, seed=42)
    
    # Limit columns if too many
    if df.width > max_cols:
        sample_df = sample_df.select(df.columns[:max_cols])
    
    # Get basic stats
    numeric_cols = [col for col in df.columns if df.select(pl.col(col)).dtypes[0] in [pl.Int64, pl.Float64]]
    stats = {}
    if numeric_cols:
        stats = {
            col: {
                'count': sample_df.select(pl.col(col).count()).item(),
                'mean': sample_df.select(pl.col(col).mean()).item(),
                'std': sample_df.select(pl.col(col).std()).item(),
                'min': sample_df.select(pl.col(col).min()).item(),
                '25%': sample_df.select(pl.col(col).quantile(0.25)).item(),
                '50%': sample_df.select(pl.col(col).median()).item(),
                '75%': sample_df.select(pl.col(col).quantile(0.75)).item(),
                'max': sample_df.select(pl.col(col).max()).item()
            }
            for col in numeric_cols
        }
    
    # Convert sample to records for analysis
    sample_records = sample_df.head(10).to_dicts()
    
    # Prepare dataset info
    dataset_info = {
        "columns": df.columns,
        "shape": (df.height, df.width),
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "sample_data": sample_records,
        "basic_stats": stats,
        "missing_values": {col: df.select(pl.col(col).null_count()).item() for col in df.columns}
    }
    
    # Create the prompt for GPT
    prompt = f"""
    You are a data analysis expert. I'm providing you with information about a dataset. 
    Please analyze this information and provide the TOP 5 most valuable insights about the data.
    
    Dataset Information:
    - Shape: {dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns
    - Columns: {', '.join(dataset_info['columns'])}
    
    Sample Data:
    {json.dumps(dataset_info['sample_data'], indent=2)}
    
    Basic Statistics:
    {json.dumps(dataset_info['basic_stats'], indent=2)}
    
    Missing Values:
    {json.dumps(dataset_info['missing_values'], indent=2)}
    
    Please provide:
    1. The top 5 most valuable and specific insights about this dataset
    2. Format each insight as a bullet point starting with "Insight #:"
    3. Focus on patterns, distributions, correlations, anomalies, or any other interesting aspects
    4. Make your insights specific, data-driven, and actionable
    5. Do NOT include generic statements or observations that could apply to any dataset
    """
    
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make API call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis expert who provides clear, concise insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        # Return the insights
        insights = response.choices[0].message.content.strip()
        return insights
        
    except Exception as e:
        return f"Error analyzing dataset: {str(e)}"