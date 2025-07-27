import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import pandas as pd

def plot_histogram(
        df,
        column,
        *,
        bins=20,
        bin_width=None,          # NEW: explicit width (data units)
        highlight_outliers=False,
        outlier_indices=None):
    """
    Create a histogram for a numeric column. Either
    • bins (approx. number of bars) or
    • bin_width (exact bar width) may be provided.
    """
    # base histogram
    fig = px.histogram(
        df,
        x=column,
        nbins=bins,
        title=f"Histogram of {column}",
        labels={column: column},
        opacity=0.7,
        marginal="box"
    )

    # if caller gave a width, override Plotly’s binning
    if bin_width is not None:
        for t in fig.data:
            if isinstance(t, go.Histogram):
                t.xbins = dict(size=bin_width)

    # normal-distribution curve
    data = df[column].dropna()
    mean, std = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    y = ((1 / (np.sqrt(2 * np.pi) * std))
         * np.exp(-0.5 * ((x - mean) / std) ** 2)
         * len(data) * (data.max() - data.min()) / (bin_width or bins))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Normal Distribution",
            line=dict(color="red", width=2)
        )
    )

    # optional outlier overlay
    if highlight_outliers and outlier_indices is not None:
        outlier_data = df.loc[outlier_indices, column].dropna()
        fig.add_trace(
            go.Histogram(
                x=outlier_data,
                nbinsx=bins,
                name="Outliers",
                marker_color="red",
                opacity=0.7
            )
        )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequency",
        legend_title="Legend",
        barmode="overlay"
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
        data = df[column].dropna()
        
        fig.add_trace(
            go.Box(
                y=data,
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
            
            if len(outlier_data) > 0:
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
    
    fig = px.imshow(
        missing_mask,
        color_continuous_scale='Reds',
        labels=dict(color="Missing Values"),
        title="Missing Data Heatmap"
    )
    
    fig.update_layout(
        height=400,
        width=800
    )
    
    return fig

def plot_line_chart(df, x_column, y_columns):
    """
    Create a line chart for one or more numeric columns.
    
    Args:
        df: Pandas DataFrame
        x_column: Column name for x-axis
        y_columns: List of column names for y-axis
        
    Returns:
        Plotly figure object
    """
    fig = px.line(
        df,
        x=x_column,
        y=y_columns,
        title="Line Chart",
        labels={x_column: x_column},
        markers=True
    )
    
    fig.update_layout(
        xaxis_title=x_column,
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def plot_bar_chart(df, category_column, value_column=None):
    """
    Create a bar chart.
    
    Args:
        df: Pandas DataFrame
        category_column: Column name for categories
        value_column: Column name for values (optional)
        
    Returns:
        Plotly figure object
    """
    if value_column:
        fig = px.bar(
            df,
            x=category_column,
            y=value_column,
            title=f"Bar Chart: {value_column} by {category_column}",
            labels={category_column: category_column, value_column: value_column}
        )
    else:
        # Count occurrences if no value column provided
        value_counts = df[category_column].value_counts().reset_index()
        value_counts.columns = [category_column, 'counts']
        fig = px.bar(
            value_counts,
            x=category_column,
            y='counts',
            title=f"Bar Chart: Count of {category_column}",
            labels={category_column: category_column, 'counts': 'Count'}
        )
    
    fig.update_layout(
        xaxis_title=category_column,
        yaxis_title="Value" if value_column else "Count",
        showlegend=False
    )
    
    return fig

def plot_pie_chart(df, category_column, value_column=None):
    """
    Create a pie chart.
    
    Args:
        df: Pandas DataFrame
        category_column: Column name for categories
        value_column: Column name for values (optional)
        
    Returns:
        Plotly figure object
    """
    if value_column:
        fig = px.pie(
            df,
            names=category_column,
            values=value_column,
            title=f"Pie Chart: {value_column} by {category_column}"
        )
    else:
        # Count occurrences if no value column provided
        value_counts = df[category_column].value_counts().reset_index()
        value_counts.columns = [category_column, 'counts']
        fig = px.pie(
            value_counts,
            names=category_column,
            values='counts',
            title=f"Pie Chart: Distribution of {category_column}"
        )
    
    fig.update_layout(
        showlegend=True
    )
    
    return fig
