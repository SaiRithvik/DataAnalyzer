import pandas as pd
import json
from openai import OpenAI

def analyze_dataset_with_gpt(df, api_key, max_rows=100, max_cols=20):
    """
    Analyze a dataset using OpenAI's GPT-4o-mini model.
    
    Args:
        df: Pandas DataFrame
        api_key: OpenAI API key
        max_rows: Maximum rows to send to the API
        max_cols: Maximum columns to send to the API
        
    Returns:
        List of insights about the dataset
    """
    # Limit dataset size for API request
    sample_df = df
    if len(df) > max_rows:
        sample_df = df.sample(max_rows, random_state=42)
    
    # Limit columns if too many
    if len(df.columns) > max_cols:
        sample_df = sample_df.iloc[:, :max_cols]
    
    # Get basic stats
    numeric_cols = sample_df.select_dtypes(include=['number']).columns.tolist()
    stats = {}
    if numeric_cols:
        stats = sample_df[numeric_cols].describe().to_dict()
    
    # Convert sample to records for analysis
    sample_records = sample_df.head(10).to_dict(orient='records')
    
    # Prepare dataset info
    dataset_info = {
        "columns": sample_df.columns.tolist(),
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample_data": sample_records,
        "basic_stats": stats,
        "missing_values": df.isnull().sum().to_dict()
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