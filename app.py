import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import os

from data_analysis import (
    get_basic_statistics, 
    analyze_correlations, 
    detect_outliers, 
    analyze_missing_data
)
from data_visualization import (
    plot_histogram,
    plot_box_plot,
    plot_scatter_matrix,
    plot_correlation_heatmap,
    plot_missing_data_heatmap,
    plot_line_chart,
    plot_bar_chart,
    plot_pie_chart
)
from utils import (
    get_data_types,
    convert_df_to_csv,
    convert_df_to_excel
)
from ai_analysis import analyze_dataset_with_gpt

# Set page config
st.set_page_config(
    page_title="Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None
if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None

# Header
st.title("📊 Data Analyzer")
st.write("Upload a CSV or Excel file to analyze your data")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# Process the uploaded file
if uploaded_file is not None:
    try:
        # Get file extension
        file_extension = uploaded_file.name.split('.')[-1]
        
        # Read the file
        if file_extension.lower() == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension.lower() in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name
        
        # Get column types
        numeric_cols, categorical_cols = get_data_types(df)
        st.session_state.numeric_columns = numeric_cols
        st.session_state.categorical_columns = categorical_cols
        
        st.success(f"Successfully loaded {uploaded_file.name} with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")

# Show analysis options only if data is loaded
if st.session_state.df is not None:
    df = st.session_state.df
    numeric_columns = st.session_state.numeric_columns
    categorical_columns = st.session_state.categorical_columns
    
    # Data exploration tab
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Preview", "Basic Statistics", "Visualizations", "Advanced Analysis", "Custom Plots", "AI Analysis", "Export"])
    
    with tab1:
        st.header("Data Preview")
        
        # Display data sample
        sample_size = st.slider("Number of rows to display", min_value=5, max_value=min(100, df.shape[0]), value=10)
        st.dataframe(df.head(sample_size))
        
        # Data filtering
        st.subheader("Data Filtering")
        
        # Column selection
        selected_columns = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist()[:5])
        
        if selected_columns:
            filtered_df = df[selected_columns]
            
            # Filter by value (only for categorical columns that are selected)
            categorical_filter_cols = [col for col in selected_columns if col in categorical_columns]
            if categorical_filter_cols:
                filter_column = st.selectbox("Filter by categorical column", categorical_filter_cols)
                unique_values = df[filter_column].unique().tolist()
                filter_values = st.multiselect("Select values", unique_values, default=unique_values[:5])
                
                if filter_values:
                    filtered_df = filtered_df[filtered_df[filter_column].isin(filter_values)]
            
            # Sorting
            sort_column = st.selectbox("Sort by", ["None"] + selected_columns)
            if sort_column != "None":
                sort_order = st.radio("Sort order", ["Ascending", "Descending"])
                ascending = sort_order == "Ascending"
                filtered_df = filtered_df.sort_values(by=sort_column, ascending=ascending)
            
            st.subheader("Filtered Data")
            st.dataframe(filtered_df)

    with tab2:
        st.header("Basic Statistics")
        
        # Show data types
        st.subheader("Data Types")
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
        dtypes_df = dtypes_df.reset_index().rename(columns={"index": "Column"})
        st.dataframe(dtypes_df)
        
        # Show statistics for numeric columns
        if numeric_columns:
            st.subheader("Numeric Columns Statistics")
            stats_df = get_basic_statistics(df, numeric_columns)
            st.dataframe(stats_df)
        else:
            st.info("No numeric columns found in the dataset.")
        
        # Show value counts for categorical columns
        if categorical_columns:
            st.subheader("Categorical Columns Value Counts")
            cat_col = st.selectbox("Select a categorical column", categorical_columns)
            value_counts = df[cat_col].value_counts().reset_index()
            value_counts.columns = [cat_col, 'Count']
            st.dataframe(value_counts)
        else:
            st.info("No categorical columns found in the dataset.")
        
        # Missing data analysis
        st.subheader("Missing Data Analysis")
        missing_data = analyze_missing_data(df)
        st.dataframe(missing_data)

    with tab3:
        st.header("Data Visualizations")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            ["Histogram", "Box Plot", "Scatter Plot Matrix", "Correlation Heatmap", 
             "Missing Data Heatmap", "Line Chart", "Bar Chart", "Pie Chart"]
        )
        
        if viz_type == "Histogram":
            if not numeric_columns:
                st.warning("No numeric columns available for histogram.")
            else:
                col = st.selectbox("Select column for histogram", numeric_columns)
                bins = st.slider("Number of bins", 5, 100, 20)
                fig = plot_histogram(df, col, bins)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Box Plot":
            if not numeric_columns:
                st.warning("No numeric columns available for box plot.")
            else:
                cols = st.multiselect("Select columns for box plot", numeric_columns, default=numeric_columns[:min(5, len(numeric_columns))])
                if cols:
                    fig = plot_box_plot(df, cols)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Scatter Plot Matrix":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for scatter plot matrix.")
            else:
                # Limit selection to prevent large matrices
                max_cols = min(6, len(numeric_columns))
                cols = st.multiselect(
                    "Select columns for scatter plot matrix (max 6)", 
                    numeric_columns, 
                    default=numeric_columns[:min(4, max_cols)]
                )
                if 2 <= len(cols) <= 6:
                    fig = plot_scatter_matrix(df, cols)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select between 2 and 6 columns.")
        
        elif viz_type == "Correlation Heatmap":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap.")
            else:
                fig = plot_correlation_heatmap(df, numeric_columns)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Missing Data Heatmap":
            fig = plot_missing_data_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Line Chart":
            if not numeric_columns:
                st.warning("No numeric columns available for line chart.")
            else:
                y_cols = st.multiselect(
                    "Select columns for Y-axis", 
                    numeric_columns, 
                    default=numeric_columns[:min(3, len(numeric_columns))]
                )
                x_col = st.selectbox("Select column for X-axis", df.columns.tolist())
                if y_cols:
                    fig = plot_line_chart(df, x_col, y_cols)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            if not categorical_columns:
                st.warning("No categorical columns available for bar chart.")
            else:
                cat_col = st.selectbox("Select categorical column", categorical_columns)
                
                # Option to select a numeric column for values
                use_numeric = st.checkbox("Use numeric column for values")
                num_col = None
                if use_numeric and numeric_columns:
                    num_col = st.selectbox("Select numeric column for values", numeric_columns)
                
                fig = plot_bar_chart(df, cat_col, num_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            if not categorical_columns:
                st.warning("No categorical columns available for pie chart.")
            else:
                cat_col = st.selectbox("Select categorical column for pie chart", categorical_columns)
                # Option to select a numeric column for values
                use_numeric = st.checkbox("Use numeric column for pie chart values")
                num_col = None
                if use_numeric and numeric_columns:
                    num_col = st.selectbox("Select numeric column for pie chart values", numeric_columns)
                
                fig = plot_pie_chart(df, cat_col, num_col)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Advanced Analysis")
        
        analysis_type = st.selectbox(
            "Select analysis type",
            ["Correlation Analysis", "Outlier Detection"]
        )
        
        if analysis_type == "Correlation Analysis":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for correlation analysis.")
            else:
                st.subheader("Correlation Analysis")
                corr_method = st.radio("Correlation method", ["pearson", "spearman", "kendall"])
                corr_df = analyze_correlations(df, numeric_columns, corr_method)
                st.dataframe(corr_df)
                
                # Strong correlations
                threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.7, 0.05)
                strong_corr = corr_df[corr_df['Correlation'].abs() >= threshold]
                
                if not strong_corr.empty:
                    st.subheader(f"Strong Correlations (|r| >= {threshold})")
                    st.dataframe(strong_corr)
                else:
                    st.info(f"No correlations with absolute value >= {threshold} found.")
        
        elif analysis_type == "Outlier Detection":
            if not numeric_columns:
                st.warning("No numeric columns available for outlier detection.")
            else:
                st.subheader("Outlier Detection")
                col = st.selectbox("Select column for outlier detection", numeric_columns)
                method = st.radio("Detection method", ["IQR", "Z-Score"])
                
                if method == "IQR":
                    iqr_mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
                    outliers, outlier_indices, summary = detect_outliers(df, col, method, iqr_factor=iqr_mult)
                else:  # Z-Score
                    z_threshold = st.slider("Z-Score threshold", 1.0, 5.0, 3.0, 0.1)
                    outliers, outlier_indices, summary = detect_outliers(df, col, method, z_threshold=z_threshold)
                
                # Show summary
                st.write(summary)
                
                # Show outliers
                if not outliers.empty:
                    st.subheader(f"Outliers Detected ({len(outliers)} rows)")
                    st.dataframe(outliers)
                    
                    # Option to show in plot
                    show_plot = st.checkbox("Show outliers in plot")
                    if show_plot:
                        if method == "IQR":
                            fig = plot_box_plot(df, [col], highlight_outliers=True, outlier_indices=outlier_indices)
                        else:  # Z-Score
                            fig = plot_histogram(df, col, highlight_outliers=True, outlier_indices=outlier_indices)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No outliers detected in column '{col}' using {method} method.")

    with tab5:
        st.header("Custom Plots")
        
        # Choose plot type (2D or 3D)
        plot_dimension = st.radio("Select plot dimension", ["2D Plot", "3D Plot"])
        
        if plot_dimension == "2D Plot":
            if len(numeric_columns) < 2:
                st.warning("Need at least 2 numeric columns for custom 2D plots.")
            else:
                # Choose plot type
                plot_type = st.selectbox(
                    "Select 2D plot type",
                    ["Scatter Plot", "Line Plot", "Bubble Chart", "Contour Plot"]
                )
                
                # Common settings for 2D plots
                x_col = st.selectbox("X-axis column", numeric_columns)
                y_col = st.selectbox("Y-axis column", numeric_columns, index=min(1, len(numeric_columns)-1))
                
                # Color by column (optional)
                use_color = st.checkbox("Color by column")
                color_col = None
                if use_color:
                    color_options = df.columns.tolist()
                    color_col = st.selectbox("Select color column", color_options)
                
                # Create plot based on type
                if plot_type == "Scatter Plot":
                    fig = px.scatter(
                        df, 
                        x=x_col, 
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}",
                        labels={x_col: x_col, y_col: y_col},
                        hover_data=df.columns
                    )
                
                elif plot_type == "Line Plot":
                    # Optional group by for multiple lines
                    use_group = st.checkbox("Group by column (for multiple lines)")
                    group_col = None
                    if use_group and categorical_columns:
                        group_col = st.selectbox("Select grouping column", categorical_columns)
                    
                    # Create a sorted dataframe for proper line plotting
                    plot_df = df.sort_values(by=x_col)
                    
                    fig = px.line(
                        plot_df, 
                        x=x_col, 
                        y=y_col,
                        color=group_col if use_group else color_col,
                        title=f"Line Plot: {y_col} vs {x_col}",
                        labels={x_col: x_col, y_col: y_col},
                        markers=True
                    )
                
                elif plot_type == "Bubble Chart":
                    if len(numeric_columns) < 3:
                        st.warning("Need at least 3 numeric columns for bubble chart (x, y, and size).")
                    else:
                        size_col = st.selectbox("Bubble size column", numeric_columns, index=min(2, len(numeric_columns)-1))
                        
                        fig = px.scatter(
                            df, 
                            x=x_col, 
                            y=y_col,
                            size=size_col,
                            color=color_col,
                            title=f"Bubble Chart: {y_col} vs {x_col} (Size: {size_col})",
                            labels={x_col: x_col, y_col: y_col, size_col: size_col},
                            hover_data=df.columns
                        )
                
                elif plot_type == "Contour Plot":
                    if len(df) < 5:
                        st.warning("Not enough data points for a contour plot.")
                    else:
                        z_col = st.selectbox("Z-axis (contour values) column", numeric_columns, index=min(2, len(numeric_columns)-1))
                        
                        fig = px.density_contour(
                            df, 
                            x=x_col, 
                            y=y_col,
                            z=z_col,
                            title=f"Contour Plot: {z_col} values for {y_col} vs {x_col}",
                            labels={x_col: x_col, y_col: y_col}
                        )
                        
                        # Option to add points
                        show_points = st.checkbox("Show data points")
                        if show_points:
                            fig.add_trace(
                                go.Scatter(
                                    x=df[x_col], 
                                    y=df[y_col], 
                                    mode='markers',
                                    marker=dict(size=5, color='white', line=dict(width=1, color='black')),
                                    name='Data points'
                                )
                            )
                
                # Plot customization options
                st.subheader("Plot Customization")
                
                # Grid lines
                show_grid = st.checkbox("Show grid lines", value=True)
                
                # Log scale options
                col1, col2 = st.columns(2)
                with col1:
                    log_x = st.checkbox("Log scale for X-axis")
                with col2:
                    log_y = st.checkbox("Log scale for Y-axis")
                
                # Apply customizations
                fig.update_layout(
                    xaxis=dict(showgrid=show_grid, type='log' if log_x else 'linear'),
                    yaxis=dict(showgrid=show_grid, type='log' if log_y else 'linear')
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
        
        elif plot_dimension == "3D Plot":
            if len(numeric_columns) < 3:
                st.warning("Need at least 3 numeric columns for 3D plots.")
            else:
                # Choose 3D plot type
                plot3d_type = st.selectbox(
                    "Select 3D plot type",
                    ["3D Scatter", "3D Surface", "3D Line"]
                )
                
                # Common settings for 3D plots
                x_col = st.selectbox("X-axis column", numeric_columns)
                y_col = st.selectbox("Y-axis column", numeric_columns, index=min(1, len(numeric_columns)-1))
                z_col = st.selectbox("Z-axis column", numeric_columns, index=min(2, len(numeric_columns)-1))
                
                # Color by column (optional)
                use_color = st.checkbox("Color by column", key="3d_color")
                color_col = None
                if use_color:
                    color_options = df.columns.tolist()
                    color_col = st.selectbox("Select color column", color_options, key="3d_color_col")
                
                # Create plot based on type
                if plot3d_type == "3D Scatter":
                    fig = px.scatter_3d(
                        df,
                        x=x_col,
                        y=y_col,
                        z=z_col,
                        color=color_col,
                        title=f"3D Scatter Plot: {x_col}, {y_col}, {z_col}",
                        labels={x_col: x_col, y_col: y_col, z_col: z_col}
                    )
                
                elif plot3d_type == "3D Surface":
                    # Surface plots need a grid of data
                    st.info("Surface plots work best with gridded data. Using available data points.")
                    
                    # Create a figure
                    fig = go.Figure(data=[go.Mesh3d(
                        x=df[x_col],
                        y=df[y_col],
                        z=df[z_col],
                        colorbar_title=z_col,
                        colorscale='Viridis',
                        opacity=0.8
                    )])
                    
                    fig.update_layout(
                        title=f"3D Surface Plot: {x_col}, {y_col}, {z_col}",
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col
                        )
                    )
                
                elif plot3d_type == "3D Line":
                    # Sort by x and y for better line visualization
                    plot_df = df.sort_values(by=[x_col, y_col])
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=plot_df[x_col],
                        y=plot_df[y_col],
                        z=plot_df[z_col],
                        mode='lines+markers',
                        marker=dict(
                            size=4,
                            color=plot_df[color_col] if color_col else plot_df[z_col],
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        line=dict(
                            color='darkblue',
                            width=2
                        )
                    )])
                    
                    fig.update_layout(
                        title=f"3D Line Plot: {x_col}, {y_col}, {z_col}",
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col
                        )
                    )
                
                # Plot customization options
                st.subheader("3D Plot Customization")
                
                # Camera settings
                camera_distance = st.slider("Camera distance", 1.0, 5.0, 2.0, 0.1)
                
                # Apply camera settings
                camera = dict(
                    eye=dict(x=camera_distance, y=camera_distance, z=camera_distance)
                )
                
                fig.update_layout(
                    scene_camera=camera,
                    scene=dict(
                        aspectmode='cube'
                    )
                )
                
                # Display the plot
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("AI Analysis")
        
        st.write("Use OpenAI's GPT-4o-mini to analyze your dataset and get valuable insights.")
        
        # API Key input
        api_key = st.text_input("Enter your OpenAI API Key", type="password", help="Your API key will not be stored")
        
        if api_key:
            # Cache the analysis results to avoid repeated API calls
            if 'ai_insights' not in st.session_state:
                st.session_state.ai_insights = None
                
            # Button to trigger analysis
            if st.button("Generate AI Insights") or st.session_state.ai_insights:
                with st.spinner("Analyzing your data with GPT-4o-mini..."):
                    if not st.session_state.ai_insights:
                        # Only make the API call if we don't already have results
                        insights = analyze_dataset_with_gpt(df, api_key)
                        st.session_state.ai_insights = insights
                    
                    # Display the insights
                    st.subheader("AI-Generated Insights")
                    st.markdown(st.session_state.ai_insights)
                    
                    # Option to regenerate
                    if st.button("Regenerate Analysis"):
                        st.session_state.ai_insights = None
                        st.rerun()
        else:
            st.info("Please enter your OpenAI API key to use this feature.")
            st.markdown("""
            ### How to get an OpenAI API key:
            1. Go to [OpenAI API](https://platform.openai.com/signup)
            2. Create an account or log in
            3. Navigate to API keys section
            4. Create a new API key
            """)
    
    with tab7:
        st.header("Export Data & Results")
        
        st.subheader("Export Original Dataset")
        export_format = st.radio("Export format", ["CSV", "Excel"])
        
        if export_format == "CSV":
            csv = convert_df_to_csv(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name=f"{st.session_state.file_name.split('.')[0]}_analyzed.csv",
                mime="text/csv",
            )
        else:  # Excel
            excel_data = convert_df_to_excel(df)
            st.download_button(
                label="Download data as Excel",
                data=excel_data,
                file_name=f"{st.session_state.file_name.split('.')[0]}_analyzed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        
        # Export statistics
        if numeric_columns:
            st.subheader("Export Statistics Report")
            stats_df = get_basic_statistics(df, numeric_columns)
            
            if export_format == "CSV":
                stats_csv = convert_df_to_csv(stats_df)
                st.download_button(
                    label="Download statistics as CSV",
                    data=stats_csv,
                    file_name=f"{st.session_state.file_name.split('.')[0]}_statistics.csv",
                    mime="text/csv",
                )
            else:  # Excel
                stats_excel = convert_df_to_excel(stats_df)
                st.download_button(
                    label="Download statistics as Excel",
                    data=stats_excel,
                    file_name=f"{st.session_state.file_name.split('.')[0]}_statistics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            
            # Export correlation matrix
            if len(numeric_columns) >= 2:
                st.subheader("Export Correlation Matrix")
                corr_df = df[numeric_columns].corr()
                
                if export_format == "CSV":
                    corr_csv = convert_df_to_csv(corr_df)
                    st.download_button(
                        label="Download correlation matrix as CSV",
                        data=corr_csv,
                        file_name=f"{st.session_state.file_name.split('.')[0]}_correlation.csv",
                        mime="text/csv",
                    )
                else:  # Excel
                    corr_excel = convert_df_to_excel(corr_df)
                    st.download_button(
                        label="Download correlation matrix as Excel",
                        data=corr_excel,
                        file_name=f"{st.session_state.file_name.split('.')[0]}_correlation.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
else:
    # Show example when no file is uploaded
    st.info("👆 Please upload a CSV or Excel file to begin analysis.")
    
    st.subheader("Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Data Overview")
        st.markdown("- View data sample")
        st.markdown("- Filter and sort data")
        st.markdown("- Data type information")
    
    with col2:
        st.markdown("#### 📈 Statistical Analysis")
        st.markdown("- Basic statistics")
        st.markdown("- Correlation analysis")
        st.markdown("- Missing data identification")
        st.markdown("- Outlier detection")
    
    with col3:
        st.markdown("#### 🎨 Visualizations")
        st.markdown("- Histograms")
        st.markdown("- Box plots")
        st.markdown("- Scatter plot matrices")
        st.markdown("- Correlation heatmaps")
        st.markdown("- And more...")
