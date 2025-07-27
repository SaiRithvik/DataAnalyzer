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
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Exclude boolean columns from the list of numeric columns
        bool_cols = df.select_dtypes(include=np.bool_).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in bool_cols]
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Optionally, you can treat boolean columns as categorical
        categorical_cols.extend(bool_cols)



        #numeric_cols, categorical_cols = get_data_types(df)
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
        selected_columns = st.multiselect("Select columns to display", df.columns, default=df.columns[:5])
        
        if selected_columns:
            filtered_df = df.loc[:, selected_columns]
            
            # Filter by value (only for categorical columns that are selected)
            categorical_filter_cols = [col for col in selected_columns if col in categorical_columns]
            if categorical_filter_cols:
                filter_column = st.selectbox("Filter by categorical column", categorical_filter_cols)
                unique_values = df[filter_column].unique()
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
        dtypes_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.apply(lambda x: x.name)
        })
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
            value_counts = df[cat_col].value_counts()
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
                if cols:
                    fig = plot_scatter_matrix(df, cols)
                    st.plotly_chart(fig, use_container_width=True)
        
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
                x_col = st.selectbox("Select x-axis column", df.columns)
                y_cols = st.multiselect("Select y-axis columns", numeric_columns)
                if y_cols:
                    fig = plot_line_chart(df, x_col, y_cols)
                    st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Bar Chart":
            if not categorical_columns:
                st.warning("No categorical columns available for bar chart.")
            else:
                cat_col = st.selectbox("Select category column", categorical_columns)
                if numeric_columns:
                    value_col = st.selectbox("Select value column (optional)", ["None"] + numeric_columns)
                    if value_col != "None":
                        fig = plot_bar_chart(df, cat_col, value_col)
                    else:
                        fig = plot_bar_chart(df, cat_col)
                else:
                    fig = plot_bar_chart(df, cat_col)
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_type == "Pie Chart":
            if not categorical_columns:
                st.warning("No categorical columns available for pie chart.")
            else:
                cat_col = st.selectbox("Select category column", categorical_columns)
                if numeric_columns:
                    value_col = st.selectbox("Select value column (optional)", ["None"] + numeric_columns)
                    if value_col != "None":
                        fig = plot_pie_chart(df, cat_col, value_col)
                    else:
                        fig = plot_pie_chart(df, cat_col)
                else:
                    fig = plot_pie_chart(df, cat_col)
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Advanced Analysis")
        
        # Correlation analysis
        st.subheader("Correlation Analysis")
        if len(numeric_columns) >= 2:
            method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
            corr_df = analyze_correlations(df, numeric_columns, method)
            st.dataframe(corr_df)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        
        # Outlier detection
        st.subheader("Outlier Detection")
        if numeric_columns:
            col = st.selectbox("Select column for outlier detection", numeric_columns)
            method = st.selectbox("Detection method", ["IQR", "Z-Score"])
            
            if method == "IQR":
                iqr_factor = st.slider("IQR factor", 1.0, 3.0, 1.5, 0.1)
                outliers, indices, summary = detect_outliers(df, col, method="IQR", iqr_factor=iqr_factor)
            else:
                z_threshold = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
                outliers, indices, summary = detect_outliers(df, col, method="Z-Score", z_threshold=z_threshold)
            
            st.write("Outlier Summary:")
            st.json(summary)
            
            if len(indices) > 0:
                st.write("Outlier Data:")
                st.dataframe(outliers)
                
                # Plot with outliers highlighted
                fig = plot_histogram(df, col, highlight_outliers=True, outlier_indices=indices)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No outliers detected.")
        else:
            st.warning("No numeric columns available for outlier detection.")

    with tab5:
        st.header("Custom Plots")
        
        # Custom scatter plot
        st.subheader("Custom Scatter Plot")
        if len(numeric_columns) >= 2:
            x_col = st.selectbox("Select x-axis column", numeric_columns)
            y_col = st.selectbox("Select y-axis column", [col for col in numeric_columns if col != x_col])
            
            if categorical_columns:
                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_columns)
                if color_col != "None":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        color=color_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}"
                    )
                else:
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"Scatter Plot: {y_col} vs {x_col}"
                    )
            else:
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    title=f"Scatter Plot: {y_col} vs {x_col}"
                )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for scatter plot.")

    with tab6:
        st.header("AI Analysis")
        
        # OpenAI API key input
        api_key = st.text_input("Enter your OpenAI API key", type="password")
        
        if api_key:
            if st.button("Analyze with AI"):
                with st.spinner("Analyzing data with AI..."):
                    insights = analyze_dataset_with_gpt(df, api_key)
                    st.session_state.ai_insights = insights
                    st.markdown(insights)
        else:
            st.info("Please enter your OpenAI API key to use AI analysis.")

    with tab7:
        st.header("Export Data")
        
        # Export options
        export_format = st.selectbox("Select export format", ["CSV", "Excel"])
        
        if export_format == "CSV":
            csv = convert_df_to_csv(df)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{st.session_state.file_name.split('.')[0]}_export.csv",
                mime="text/csv"
            )
        else:
            excel = convert_df_to_excel(df)
            st.download_button(
                label="Download Excel",
                data=excel,
                file_name=f"{st.session_state.file_name.split('.')[0]}_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
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
