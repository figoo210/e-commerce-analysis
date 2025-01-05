import streamlit as st
import pandas as pd

from helper_functions import *


# Functions
def arima_model_analysis(sector_name=""):
    st.write("### Time Series Analysis (ARIMA Model & State-Space Model)")
    if use_arima:
        st.write("#### ARIMA Model")
        arima_input_col1, arima_input_col2, arima_input_col3 = st.columns(3)
        p = arima_input_col1.number_input("Autoregressive (AR) Order", 0, 5, 0)
        d = arima_input_col2.number_input("Differencing Order", 0, 2, 0)
        q = arima_input_col3.number_input("Moving Average (MA) Order", 0, 5, 0)
        arima_output = analyze_arima_relationships_interactive(
            df, sector_name, p=p, d=d, q=q
        )
        if arima_output:
            for key, value in arima_output.items():
                if key != "plot" and key != "results":
                    st.write(f"#### {key}")
                    st.code(value)
                elif key == "plot":
                    st.plotly_chart(value)

        st.write("#### State-Space Model")
        (
            state_space_input_col1,
            state_space_input_col2,
            state_space_input_col3,
            state_space_input_col4,
        ) = st.columns(4)
        p = state_space_input_col1.number_input(
            "Seasonal Autoregressive (AR) Order", 0, 5, 0
        )
        d = state_space_input_col2.number_input("Seasonal Differencing Order", 0, 2, 0)
        q = state_space_input_col3.number_input(
            "Seasonal Moving Average (MA) Order", 0, 5, 0
        )
        s = state_space_input_col4.number_input(
            "Seasonal Period", 0, 365, 12
        )  # default 12 months
        state_space_output = analyze_state_space(
            df, sector_name, p=p, d=d, q=q, seasonal_order=(p, d, q, s)
        )
        if state_space_output:
            for key, value in state_space_output.items():
                if key != "plot" and key != "results":
                    st.write(f"#### {key}")
                    st.code(value)
                elif key == "plot":
                    st.plotly_chart(value)
    else:
        st.write("#### You can enable ARIMA Model from the sidebar to see the results.")


# Global variables
st.set_page_config(
    page_title="E-commerce Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Title
st.title("E-commerce Data Analysis")


# Load Dataset
@st.cache_data
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# Add a download button for the sample CSV file
st.download_button(
    label="Download Sample CSV",
    data=generate_sample_csv(),
    file_name="sample_ecommerce.csv",
    mime="text/csv",
)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write("Dataset uploaded successfully.")
else:
    df = pd.read_csv("./files/prepared_ecommerce.csv")

if not df.empty:
    st.write("Used default dataset.")
    # Sidebar
    st.sidebar.title("Data Science Lab")
    st.sidebar.write("### Data Analysis Options")
    # Select Sector for Analysis, default none and the rest comes from df["Sector"].unique()
    sector = st.sidebar.selectbox(
        "Select Sector for Analysis", ["None"] + list(df["Sector"].unique())
    )
    # Add radio button at the end of the side bar to enable ARIMA model
    use_arima = st.sidebar.checkbox(
        "Enable ARIMA Model & State-Space Model Analysis",
        help="It may take a while to run the ARIMA model.\n(It uses a lot of computational resources)",
    )

    # Sector Analysis
    if sector != "None":
        st.write(f"### Selected Sector: {sector}")
        sector_df = df[df["Sector"] == sector].copy()

        radio_option = None
        if sector == "Pesca":
            radio_option = st.sidebar.radio(
                "Enrich the Pesca sector with events to see the impact of events on sales.",
                [None, "Pesca Sector Enrichment"],
            )

        if sector == "Calcio":
            radio_option = st.sidebar.radio(
                "Enrich the Calcio sector with events to see the impact of events on sales.",
                [None, "Calcio Sector Enrichment"],
            )

        # Analysis options
        if radio_option is None:
            options = st.sidebar.selectbox(
                "Select Sector Analysis Features",
                [
                    "Overview",
                    "Seasonality",
                    "Economic Correlation",
                    "Holiday Impact",
                    "Lockdown Impact",
                    "ARIMA Model & State-Space Model",
                ],
            )

        if radio_option == "Pesca Sector Enrichment":
            st.write("### Data Enrichment for Pesca Sector")
            output = enrich_pesca_sector(sector_df)
            # loop over the output and display the results
            for key, value in output.items():
                if key != "plot":
                    st.code(value)
                else:
                    st.plotly_chart(value)

        elif radio_option == "Calcio Sector Enrichment":
            st.write("### Data Enrichment for Calcio Sector")
            output, plot = enrich_calcio_sector(sector_df)
            st.code(output)
            st.plotly_chart(plot)

        elif "Overview" in options:
            st.subheader("Sector Overview")
            output = sector_explore(df, sector)
            # loop over the output and display the results
            for key, value in output.items():
                if key != "plots":
                    st.write(f"#### {key}")
                    st.code(value)
                else:
                    for plot_name, plot in value.items():
                        st.plotly_chart(plot)

        elif "Seasonality" in options:
            st.subheader("Seasonality Analysis")
            # seasonality analysis
            fig = plot_interactive_decomposition(sector_df)
            st.plotly_chart(fig)

            fig2 = plot_interactive_average_sales_by_season(sector_df)
            st.plotly_chart(fig2)

        elif "Economic Correlation" in options:
            st.subheader("Economic Indicators Correlation")
            # economic correlation
            fig = plot_interactive_correlation_matrix(sector_df)
            st.plotly_chart(fig)

        elif "Holiday Impact" in options:
            st.subheader("Holiday Impact Analysis")
            # holiday impact
            output = show_holidays_impact(df, sector)
            # loop over the output and display the results
            for key, value in output.items():
                if key != "plot":
                    st.write(f"#### {key}")
                    st.code(value)
                else:
                    st.plotly_chart(value)

        elif "Lockdown Impact" in options:
            st.subheader("Lockdown Impact Analysis")
            # lockdown impact analysis
            output = analyze_sales_correlation_with_lockdown(df, sector)
            # loop over the output and display the results
            for key, value in output.items():
                if key != "plot":
                    st.code(value)
                else:
                    st.plotly_chart(value)

        elif "ARIMA Model & State-Space Model" in options:
            arima_model_analysis()
    # General Analysis
    else:
        st.subheader("Overview of the Dataset")

        st.write("#### Total Sales over Time")
        fig = plot_total_sales_over_time_interactive(df)
        st.plotly_chart(fig)

        # write two tables next to each other
        col1, col2 = st.columns(2)

        col1.write("Sorted Sectors by number of Sales")
        col1.write(get_sales_number_by_sector(df))

        col2.write("Total Sales by Sector")
        col2.write(get_total_sales_by_sector(df))

        st.write(
            f"##### Percentage of 6 top Sectors by Sales: {get_top_6_sectors_sales_percentage(df)}%"
        )

        st.write("### Dataset Summary")
        st.write(df.describe())

        arima_model_analysis()

else:
    st.info("Please upload a dataset to start.")
