@ -4,6 +4,7 @@ import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import os

from data_analysis import (
    get_basic_statistics, 
@ -26,6 +27,7 @@ from utils import (
    convert_df_to_csv,
    convert_df_to_excel
)
from ai_analysis import analyze_dataset_with_gpt

# Set page config
st.set_page_config(
@ -43,6 +45,8 @@ if 'numeric_columns' not in st.session_state:
    st.session_state.numeric_columns = []
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None

# Header
st.title("📊 Data Analyzer")
@ -83,7 +87,7 @@ if st.session_state.df is not None:
    categorical_columns = st.session_state.categorical_columns
    
    # Data exploration tab
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Data Preview", "Basic Statistics", "Visualizations", "Advanced Analysis", "Custom Plots", "Export"])
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Data Preview", "Basic Statistics", "Visualizations", "Advanced Analysis", "Custom Plots", "AI Analysis", "Export"])
    
    with tab1:
        st.header("Data Preview")
@ -548,96 +552,135 @@
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
