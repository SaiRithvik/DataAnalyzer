import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

def plot_histogram(df, column, bins=20, highlight_outliers=False, outlier_indices=None):
    """
    Create a histogram for a numeric column.
    
    Args:
        df: Pandas DataFrame
        column: Column name to plot
        bins: Number of bins
        highlight_outliers: Whether to highlight outliers
        outlier_indices: List of outlier indices
        
    Returns:
        Plotly figure object
    """
    fig = px.histogram(
        df, 
        x=column,
        nbins=bins,
        title=f"Histogram of {column}",
        labels={column: column},
        opacity=0.7,
        marginal="box"  # add a box plot at the margin
    )
    
    # Add normal distribution curve
    data = df[column].dropna()
    mean = data.mean()
    std = data.std()
    
    x = np.linspace(data.min(), data.max(), 100)
    y = ((1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)) * len(data) * (data.max() - data.min()) / bins
    
    fig.add_trace(
        go.Scatter(
            x=x, 
            y=y, 
            mode='lines', 
            name='Normal Distribution',
            line=dict(color='red', width=2)
        )
    )
    
    # Highlight outliers if requested
    if highlight_outliers and outlier_indices is not None:
        outlier_data = df.loc[outlier_indices, column].dropna()
        
        fig.add_trace(
            go.Histogram(
                x=outlier_data,
                nbinsx=bins,
                name='Outliers',
                marker_color='red',
                opacity=0.7
            )
        )
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        legend_title="Legend",
        barmode='overlay'
    )
    
    return fig

def plot_box_plot(df, columns, highlight_outliers=False, outlier_indices=None):
    """
    Create a box plot for one or more numeric columns.
    
    Args:
        df: Pandas DataFrame
        columns: List of column names to plot
        highlight_outliers: Whether to highlight outliers
        outlier_indices: List of outlier indices
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for column in columns:
        fig.add_trace(
            go.Box(
                y=df[column].dropna(),
                name=column,
                boxpoints='outliers',  # show outliers
                jitter=0.3,
                pointpos=-1.8,
                boxmean=True  # show mean
            )
        )
        
        # Highlight specific outliers if requested
        if highlight_outliers and outlier_indices is not None:
            outlier_data = df.loc[outlier_indices, column].dropna()
            
            if not outlier_data.empty:
                fig.add_trace(
                    go.Box(
                        y=outlier_data,
                        name=f"{column} (Outliers)",
                        boxpoints='all',
                        jitter=0,
                        fillcolor='rgba(255,0,0,0.5)',
                        marker=dict(color='red'),
                        line=dict(color='red'),
                        showlegend=True
                    )
                )
    
    fig.update_layout(
        title="Box Plot",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def plot_scatter_matrix(df, columns):
    """
    Create a scatter plot matrix for numeric columns.
    
    Args:
        df: Pandas DataFrame
        columns: List of column names to plot
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter_matrix(
        df,
        dimensions=columns,
        title="Scatter Plot Matrix",
        opacity=0.7
    )
    
    # Update layout for better readability
    fig.update_layout(
        height=100 * len(columns),
        width=100 * len(columns)
    )
    
    # Update axis titles to make them more readable
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i != j:
                fig.update_xaxes(title_text=columns[j], row=i+1, col=j+1)
                fig.update_yaxes(title_text=columns[i], row=i+1, col=j+1)
    
    return fig

def plot_correlation_heatmap(df, numeric_columns):
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Pandas DataFrame
        numeric_columns: List of numeric column names
        
    Returns:
        Plotly figure object
    """
    # Calculate correlation matrix
    corr_matrix = df[numeric_columns].corr()
    
    # Create heatmap
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation"),
        title="Correlation Heatmap"
    )
    
    fig.update_layout(
        height=600,
        width=700
    )
    
    return fig

def plot_missing_data_heatmap(df):
    """
    Create a heatmap of missing data.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Plotly figure object
    """
    # Create a boolean mask for missing values
    missing_mask = df.isnull()
    
    # Calculate the percentage of missing values per column
    missing_percent = missing_mask.mean().sort_values(ascending=False)
    
    # Only include columns with missing values
    missing_cols = missing_percent[missing_percent > 0].index.tolist()
    
    if not missing_cols:
        # Create empty figure with message if no missing data
        fig = go.Figure()
        fig.add_annotation(
            text="No missing data in the dataset",
            showarrow=False,
            font=dict(size=20)
        )
        return fig
    
    # Sample rows if there are too many (for performance)
    sample_size = min(100, df.shape[0])
    if df.shape[0] > sample_size:
        sampled_indices = np.random.choice(df.index, sample_size, replace=False)
        missing_mask_sampled = missing_mask.loc[sampled_indices, missing_cols]
    else:
        missing_mask_sampled = missing_mask[missing_cols]
    
    # Create heatmap
    fig = px.imshow(
        missing_mask_sampled.T,  # Transpose to get columns as y-axis
        color_continuous_scale=[[0, 'lightblue'], [1, 'red']],
        labels=dict(color="Missing"),
        title="Missing Data Heatmap (Sample)"
    )
    
    # Add percentage of missing values as y-axis text
    missing_percent_text = [f"{col} ({missing_percent[col]:.1f}%)" for col in missing_cols]
    
    fig.update_layout(
        height=max(400, 30 * len(missing_cols)),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(missing_cols))),
            ticktext=missing_percent_text
        ),
        yaxis_title="Columns",
        xaxis_title="Rows (Sample)"
    )
    
    return fig

def plot_line_chart(df, x_column, y_columns):
    """
    Create a line chart.
    
    Args:
        df: Pandas DataFrame
        x_column: Column for x-axis
        y_columns: List of columns for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    for column in y_columns:
        # Skip if column doesn't exist (should never happen but just in case)
        if column not in df.columns:
            continue
            
        # Get data and sort by x_column
        plot_df = df[[x_column, column]].dropna().sort_values(by=x_column)
        
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_column],
                y=plot_df[column],
                mode='lines+markers',
                name=column
            )
        )
    
    fig.update_layout(
        title=f"Line Chart",
        xaxis_title=x_column,
        yaxis_title="Value",
        legend_title="Columns"
    )
    
    return fig

def plot_bar_chart(df, category_column, value_column=None):
    """
    Create a bar chart for categorical data.
    
    Args:
        df: Pandas DataFrame
        category_column: Column with categories
        value_column: Optional column for values (uses count if None)
        
    Returns:
        Plotly figure object
    """
    if value_column:
        # Aggregate data by category
        agg_data = df.groupby(category_column)[value_column].sum().reset_index()
        
        fig = px.bar(
            agg_data,
            x=category_column,
            y=value_column,
            title=f"Bar Chart: Sum of {value_column} by {category_column}",
            labels={
                category_column: category_column,
                value_column: f"Sum of {value_column}"
            }
        )
    else:
        # Use value counts
        value_counts = df[category_column].value_counts().reset_index()
        value_counts.columns = [category_column, 'Count']
        
        fig = px.bar(
            value_counts,
            x=category_column,
            y='Count',
            title=f"Bar Chart: Count by {category_column}",
            labels={
                category_column: category_column,
                'Count': 'Count'
            }
        )
    
    # Add data labels
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    
    fig.update_layout(
        xaxis_title=category_column,
        yaxis_title="Value" if value_column else "Count"
    )
    
    return fig

def plot_pie_chart(df, category_column, value_column=None):
    """
    Create a pie chart for categorical data.
    
    Args:
        df: Pandas DataFrame
        category_column: Column with categories
        value_column: Optional column for values (uses count if None)
        
    Returns:
        Plotly figure object
    """
    if value_column:
        # Aggregate data by category
        agg_data = df.groupby(category_column)[value_column].sum().reset_index()
        
        fig = px.pie(
            agg_data,
            names=category_column,
            values=value_column,
            title=f"Pie Chart: Sum of {value_column} by {category_column}"
        )
    else:
        # Use value counts
        value_counts = df[category_column].value_counts().reset_index()
        value_counts.columns = [category_column, 'Count']
        
        fig = px.pie(
            value_counts,
            names=category_column,
            values='Count',
            title=f"Pie Chart: Count by {category_column}"
        )
    
    fig.update_traces(textinfo='percent+label')
    
    return fig
