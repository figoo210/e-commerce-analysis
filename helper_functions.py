from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from tabulate import tabulate
from scipy import stats
from IPython.display import display, HTML
from scipy.stats import ttest_ind
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf


# Generate a detailed sample CSV
def generate_sample_csv():
    sample_data = {
        "Date": ["4/13/2015", "4/14/2015", "4/15/2015"],
        "Total Sales": [681.04, 750.25, 820.15],
        "Sector": ["Calcio", "Fitness", "Pesca"],
        "Day": [13, 14, 15],
        "Month": [4, 4, 4],
        "Year": [2015, 2015, 2015],
        "Day_of_Week": [0, 1, 2],
        "Week_of_Year": [16, 16, 16],
        "Is_Weekend": [0, 0, 0],
        "Season": ["Spring", "Spring", "Spring"],
        "Total_Sales_Lag1": [None, 681.04, 750.25],
        "Total_Sales_Lag7": [None, None, None],
        "Total_Sales_Rolling7": [None, None, None],
        "Holiday": ["No", "No", "No"],
        "Is_Holiday": [0, 0, 0],
        "Is_Pre_Holiday": [False, False, False],
        "GDP": [1.65536e12, 1.65536e12, 1.65536e12],
        "Inflation": [0.0387904, 0.0387904, 0.0387904],
        "Unemployment": [11.315, 11.315, 11.315],
        "GDP_Normalized": [0.379854, 0.379854, 0.379854],
        "Inflation_Normalized": [0.021165, 0.021165, 0.021165],
        "Unemployment_Normalized": [1.0, 1.0, 1.0],
        "GDP_YoY": [None, None, None],
        "Inflation_YoY": [None, None, None],
        "Unemployment_YoY": [None, None, None],
        "Lockdown Status": [0, 0, 0],
    }
    sample_df = pd.DataFrame(sample_data)
    return sample_df.to_csv(index=False).encode("utf-8")


def get_sales_number_by_sector(df):
    return (
        df.groupby("Sector", as_index=False)
        .count()
        .sort_values("Date", ascending=False)
        .rename(columns={"Date": "Frequency"})
        .reset_index(drop=True)
        .sort_values("Frequency", ascending=False)[["Sector", "Frequency"]]
    )


def get_total_sales_by_sector(df):
    return (
        df.groupby("Sector", as_index=False)
        .sum("Total Sales")
        .sort_values("Total Sales", ascending=False)
        .rename(columns={"Total Sales": "Total Sales (€)"})
        .reset_index(drop=True)[["Sector", "Total Sales (€)"]]
        .round(2)
    )


def get_top_6_sectors_sales_percentage(df):
    return (
        df.groupby("Sector", as_index=False)
        .sum("Total Sales")
        .round(2)
        .sort_values("Total Sales", ascending=False)
        .head(6)["Total Sales"]
        .sum()
        / df["Total Sales"].sum()
        * 100
    ).round(2)


def plot_total_sales_over_time_interactive(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["Total Sales"], mode="lines", name="Total Sales")
    )
    fig.update_layout(
        title="Total Sales over Time",
        xaxis_title="Date",
        yaxis_title="Total Sales (€)",
    )
    # Find minimum and maximum sales and their corresponding dates
    min_sales = df["Total Sales"].min().round(2)
    max_sales = df["Total Sales"].max().round(2)
    min_date = df.loc[df["Total Sales"] == min_sales, "Date"].iloc[0]
    max_date = df.loc[df["Total Sales"] == max_sales, "Date"].iloc[0]

    # Add annotations for minimum and maximum values
    fig.add_annotation(
        x=min_date,
        y=min_sales,
        text=f"€{min_sales}",  # Format the text
        showarrow=False,
        ax=0,  # Arrow x-offset
        ay=-40,  # Arrow y-offset
        yshift=-10,
        bgcolor="lightyellow",
        font=dict(size=12, color="black"),
    )
    fig.add_annotation(
        x=max_date,
        y=max_sales,
        text=f"€{max_sales}",  # Format the text
        showarrow=False,
        ax=0,
        ay=40,
        yshift=10,
        bgcolor="purple",
        font=dict(size=12, color="white"),
    )
    return fig


def plot_sector_frequency_interactive(df, freq="ME"):
    """
    Plot the frequency of sales (number of sales events) per specified frequency
    (e.g., weekly, bi-weekly, monthly, quarterly) for the given sector dataframe.

    Parameters:
    - df: DataFrame containing the sales data
    - freq: Frequency for resampling (e.g., 'W' for weekly, '2W' for bi-weekly, 'ME' for monthly, 'Q' for quarterly)
    """
    sector_name = df["Sector"].iloc[0]

    # Ensure the Date column is in datetime format
    df.loc[:, "Date"] = pd.to_datetime(df["Date"])

    # Resample data to the specified frequency and count the number of sales
    sales_frequency = df.resample(freq, on="Date").size().reset_index(name="Frequency")

    # Create the trace for the frequency of sales events over time (using the specified frequency)
    trace = go.Scatter(
        x=sales_frequency["Date"],  # Use the resampled dates
        y=sales_frequency["Frequency"],
        mode="lines+markers",
        name=f"Sales Frequency - {sector_name}",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Frequency of Sales: %{y}<extra></extra>",
    )

    # Layout for the plot
    layout = go.Layout(
        title=f"Frequency of Sales per {freq} for the Sector {sector_name}",
        xaxis=dict(
            title=f"{freq} Date", tickformat="%Y-%m-%d"
        ),  # Format x-axis ticks for the specified frequency
        yaxis=dict(title="Frequency of Sales"),
        showlegend=False,  # No legend needed since there is only one trace
        hovermode="closest",
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=[trace], layout=layout)

    # Find minimum and maximum frequency and their corresponding dates
    min_freq = sales_frequency["Frequency"].min()
    max_freq = sales_frequency["Frequency"].max()
    min_date = sales_frequency.loc[
        sales_frequency["Frequency"] == min_freq, "Date"
    ].iloc[0]
    max_date = sales_frequency.loc[
        sales_frequency["Frequency"] == max_freq, "Date"
    ].iloc[0]

    # Add annotations for minimum and maximum values
    fig.add_annotation(
        x=min_date,
        y=min_freq,
        text=f"{min_freq}",
        showarrow=False,
        ax=0,  # Arrow x-offset
        ay=-40,  # Arrow y-offset
        yshift=-10,
        bgcolor="lightyellow",
        font=dict(size=10, color="black"),
    )
    fig.add_annotation(
        x=max_date,
        y=max_freq,
        text=f"{max_freq}",
        showarrow=False,
        ax=0,
        ay=40,
        yshift=10,
        bgcolor="purple",
        font=dict(size=10, color="white"),
    )

    return fig


def plot_sector_sales_interactive(df):
    """
    Plot total sales per month for the given sector dataframe, with each year as a separate line.
    This version uses Plotly for an interactive chart.
    """
    sector_name = df["Sector"].iloc[0]
    # Group by Year and Month, summing total sales
    df["Month"] = df["Date"].dt.month  # Extract month from Date
    # Changed here to sum only 'Total Sales'
    sales_data = (
        df.groupby(["Year", "Month"], as_index=False)["Total Sales"].sum().reset_index()
    )

    # Pivot data for plotting
    sales_pivot = sales_data.pivot(
        index="Month", columns="Year", values="Total Sales"
    ).fillna(0)

    # Create traces for each year
    traces = []
    for year in sales_pivot.columns:
        traces.append(
            go.Scatter(
                x=sales_pivot.index,
                y=sales_pivot[year],
                mode="lines+markers",  # Show both lines and markers
                name=f"Year {year}",
                hovertemplate="Month: %{x}<br>Total Sales: %{y}<extra></extra>",  # Custom hover text
            )
        )

    # Layout for the plot
    layout = go.Layout(
        title=f"Total Sales per Month for the Sector {sector_name}",
        xaxis=dict(title="Months", tickmode="linear", tickvals=list(range(1, 13))),
        yaxis=dict(title="Total Sales"),
        showlegend=True,
        hovermode="closest",  # Hover over closest point
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=traces, layout=layout)

    # Find minimum and maximum sales and their corresponding months
    min_sales = sales_pivot.min().min()  # Minimum across all years and months
    max_sales = sales_pivot.max().max()  # Maximum across all years and months
    min_month = (
        sales_pivot[sales_pivot.values == min_sales].stack().idxmin()[0]
    )  # Month of minimum sales
    max_month = (
        sales_pivot[sales_pivot.values == max_sales].stack().idxmax()[0]
    )  # Month of maximum sales

    # Add annotations for minimum and maximum values
    fig.add_annotation(
        x=min_month,
        y=min_sales,
        text=f"€{min_sales:.2f}",
        showarrow=False,
        ax=0,  # Arrow x-offset
        ay=-40,  # Arrow y-offset
        yshift=-10,
        bgcolor="lightyellow",
        font=dict(size=10, color="black"),
    )
    fig.add_annotation(
        x=max_month,
        y=max_sales,
        text=f"€{max_sales:.2f}",
        showarrow=False,
        ax=0,
        ay=40,
        yshift=10,
        bgcolor="purple",
        font=dict(size=10, color="white"),
    )

    return fig


def plot_sector_monthly_sales_interactive(df):
    """
    Plot total sales per month for the given sector dataframe,
    with monthly sales data (sum of sales per month) and rounded total sales on the line points.
    """
    sector_name = df["Sector"].iloc[0]

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Resample data to monthly frequency and sum sales, then round the sales
    monthly_sales = df.resample("ME", on="Date")["Total Sales"].sum().reset_index()

    # Round the 'Total Sales' to the nearest integer
    monthly_sales["Total Sales"] = monthly_sales["Total Sales"].round()

    # Create the trace for total sales over time (monthly)
    trace = go.Scatter(
        x=monthly_sales["Date"],  # Use month-end dates
        y=monthly_sales["Total Sales"],
        mode="lines+markers",
        name=f"Total Sales - {sector_name}",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Total Sales: %{y}<extra></extra>",
    )

    # Layout for the plot
    layout = go.Layout(
        title=f"Total Sales per Month for the Sector {sector_name}",
        xaxis=dict(
            title="Month-End Date", tickformat="%Y-%m-%d"
        ),  # Format x-axis ticks for monthly data
        yaxis=dict(title="Total Sales"),
        showlegend=False,  # No legend needed since there is only one trace
        hovermode="closest",
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=[trace], layout=layout)

    # Find minimum and maximum sales and their corresponding dates
    min_sales = monthly_sales["Total Sales"].min()
    max_sales = monthly_sales["Total Sales"].max()
    min_date = monthly_sales.loc[
        monthly_sales["Total Sales"] == min_sales, "Date"
    ].iloc[0]
    max_date = monthly_sales.loc[
        monthly_sales["Total Sales"] == max_sales, "Date"
    ].iloc[0]

    # Add annotations for minimum and maximum values
    fig.add_annotation(
        x=min_date,
        y=min_sales,
        text=f"€{min_sales}",  # Format the text
        showarrow=False,
        ax=0,  # Arrow x-offset
        ay=-40,  # Arrow y-offset
        yshift=-10,
        bgcolor="lightyellow",
        font=dict(size=10, color="black"),
    )
    fig.add_annotation(
        x=max_date,
        y=max_sales,
        text=f"€{max_sales}",  # Format the text
        showarrow=False,
        ax=0,
        ay=40,
        yshift=10,
        bgcolor="purple",
        font=dict(size=10, color="white"),
    )

    return fig


def plot_sector_sales_by_frequency_interactive(df, freq="W"):
    """
    Plot total sales per specified frequency (e.g., weekly, bi-weekly, etc.)
    for the given sector dataframe, with rounded total sales on the line points.

    Parameters:
    - df: DataFrame containing the sales data
    - freq: Frequency for resampling (e.g., 'W' for weekly, '2W' for bi-weekly, 'Q' for quarterly)
    """
    sector_name = df["Sector"].iloc[0]

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Resample data to the specified frequency and sum sales, then round the sales
    sales_resampled = df.resample(freq, on="Date")["Total Sales"].sum().reset_index()

    # Round the 'Total Sales' to the nearest integer
    sales_resampled["Total Sales"] = sales_resampled["Total Sales"].round()

    # Create the trace for total sales over time (using the specified frequency)
    trace = go.Scatter(
        x=sales_resampled["Date"],  # Use the resampled dates
        y=sales_resampled["Total Sales"],
        mode="lines+markers",
        name=f"Total Sales - {sector_name}",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Total Sales: %{y}<extra></extra>",
    )

    # Layout for the plot
    layout = go.Layout(
        title=f"Total Sales per {freq} for the Sector {sector_name}",
        xaxis=dict(
            title=f"{freq} Date", tickformat="%Y-%m-%d"
        ),  # Format x-axis ticks for the specified frequency
        yaxis=dict(title="Total Sales"),
        showlegend=False,  # No legend needed since there is only one trace
        hovermode="closest",
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=[trace], layout=layout)
    return fig


def plot_sector_daily_sales_interactive(df):
    """
    Plot total sales per day for the given sector dataframe,
    with daily sales data and rounded total sales on the line points.
    """
    sector_name = df["Sector"].iloc[0]

    # Ensure the Date column is in datetime format
    df["Date"] = pd.to_datetime(df["Date"])

    # Round the 'Total Sales' to the nearest integer
    df["Total Sales"] = df["Total Sales"].round()

    # Group by date to sum the sales per day (in case there are multiple entries for the same day)
    daily_sales = df.groupby("Date")["Total Sales"].sum().reset_index()

    # Create the trace for total sales over time
    trace = go.Scatter(
        x=daily_sales["Date"],  # Use all days
        y=daily_sales["Total Sales"],
        mode="lines+markers",
        name=f"Total Sales - {sector_name}",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Total Sales: %{y}<extra></extra>",
    )

    # Layout for the plot
    layout = go.Layout(
        title=f"Total Sales per Day for the Sector {sector_name}",
        xaxis=dict(
            title="Date", tickformat="%Y-%m-%d"
        ),  # Format x-axis ticks for daily data
        yaxis=dict(title="Total Sales"),
        showlegend=False,  # No legend needed since there is only one trace
        hovermode="closest",
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=[trace], layout=layout)

    # Find minimum and maximum sales and their corresponding dates
    min_sales = daily_sales["Total Sales"].min()
    max_sales = daily_sales["Total Sales"].max()
    min_date = daily_sales.loc[daily_sales["Total Sales"] == min_sales, "Date"].iloc[0]
    max_date = daily_sales.loc[daily_sales["Total Sales"] == max_sales, "Date"].iloc[0]

    # Add annotations for minimum and maximum values
    fig.add_annotation(
        x=min_date,
        y=min_sales,
        text=f"€{min_sales}",
        showarrow=False,
        ax=0,  # Arrow x-offset
        ay=-40,  # Arrow y-offset
        yshift=-10,
        bgcolor="lightyellow",
        font=dict(size=10, color="black"),
    )
    fig.add_annotation(
        x=max_date,
        y=max_sales,
        text=f"€{max_sales}",
        showarrow=False,
        ax=0,
        ay=40,
        yshift=10,
        bgcolor="purple",
        font=dict(size=10, color="white"),
    )

    return fig


def plot_interactive_decomposition(sector_df):
    """
    Perform seasonal decomposition of the 'Total Sales' column in the given sector dataframe
    and plot the decomposition components interactively using Plotly.

    Parameters:
    - sector_df: DataFrame containing the time series data with a 'Date' column and 'Total Sales' column
    """
    # Ensure Date is in datetime format
    sector_df["Date"] = pd.to_datetime(sector_df["Date"])

    # Set the Date column as the index for time series analysis
    sector_data = sector_df.set_index("Date")

    # Resample the data to monthly frequency (you can adjust frequency if needed)
    monthly_data = sector_data["Total Sales"].resample("ME").sum()

    # Decompose the time series into trend, seasonal, and residual components
    decomposition = seasonal_decompose(monthly_data, model="additive")

    # Create traces for each decomposed component
    trend_trace = go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend,
        mode="lines",
        name="Trend",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Trend: %{y}<extra></extra>",
    )

    seasonal_trace = go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal,
        mode="lines",
        name="Seasonal",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Seasonal: %{y}<extra></extra>",
    )

    residual_trace = go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid,
        mode="lines",
        name="Residual",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Residual: %{y}<extra></extra>",
    )

    # Create layout for the plot
    layout = go.Layout(
        title=f'Time Series Decomposition for Sector: {sector_df["Sector"].iloc[0]}',
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value"),
        showlegend=True,
        hovermode="closest",
        autosize=True,
    )

    # Create figure and plot
    fig = go.Figure(data=[trend_trace, seasonal_trace, residual_trace], layout=layout)

    # Add annotations for min/max values for each trace
    for trace, component in zip(fig.data, ["trend", "seasonal", "resid"]):
        component_data = getattr(decomposition, component)
        min_val = component_data.min()
        max_val = component_data.max()
        min_idx = component_data.idxmin()
        max_idx = component_data.idxmax()

        fig.add_annotation(
            x=min_idx,
            y=min_val,
            text=f"€{min_val:.2f}",
            showarrow=False,
            yshift=-10,
            bgcolor="lightyellow",
            font=dict(size=8, color="black"),
        )
        fig.add_annotation(
            x=max_idx,
            y=max_val,
            text=f"€{max_val:.2f}",
            showarrow=False,
            yshift=10,
            bgcolor="purple",
            font=dict(size=8, color="white"),
        )

    return fig


def plot_interactive_average_sales_by_season(sector_df):
    """
    Plot interactive average sales by season for the given sector dataframe.

    Parameters:
    - sector_df: DataFrame containing the sales data with 'Season' and 'Total Sales' columns.
    """
    # Group by season and calculate average sales
    seasonal_sales = sector_df.groupby("Season")["Total Sales"].mean()

    # Create an interactive bar chart using Plotly
    fig = go.Figure(
        data=[
            go.Bar(
                x=seasonal_sales.index,
                y=seasonal_sales.values,
                marker_color="green",
                hovertemplate="Season: %{x}<br>Average Sales: %{y}<extra></extra>",
                text=seasonal_sales.values.round(2),  # Display values on bars
                texttemplate="€%{text:.2f}",  # Format as money (€)
                textposition="outside",  # Position text outside bars
            )
        ]
    )

    # Customize the layout
    fig.update_layout(
        title=f'Average Sales by Season for {sector_df["Sector"].iloc[0]} Sector',
        xaxis_title="Season",
        yaxis_title="Average Sales",
        xaxis=dict(tickmode="array", tickvals=seasonal_sales.index),
        showlegend=False,
        hovermode="closest",
        autosize=True,
    )

    # Show the interactive plot
    return fig


def plot_interactive_correlation_matrix(sector_df):
    """
    Plot an interactive correlation matrix for 'Total Sales' with economic indicators
    like GDP, Inflation, and Unemployment from the given sector dataframe.

    Parameters:
    - sector_df: DataFrame containing the economic indicators and 'Total Sales' column.
    """
    # Select relevant columns and drop rows with missing values
    economic_data = sector_df[
        ["Total Sales", "GDP", "Inflation", "Unemployment"]
    ].dropna()

    # Calculate the correlation matrix
    correlation = economic_data.corr()

    # Create an interactive heatmap using Plotly with annotations
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.columns,
            colorscale="RdBu",
            colorbar=dict(title="Correlation Coefficient"),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z}<extra></extra>",
            text=correlation.values.round(2),  # Add text for annotation
            texttemplate="%{text}",  # Display the text directly
            textfont={"size": 10},  # Adjust font size as needed
        )
    )

    # Customize the layout
    fig.update_layout(
        title="Correlation of Sales with Economic Indicators",
        xaxis_title="Economic Indicators",
        yaxis_title="Economic Indicators",
        autosize=True,
        xaxis=dict(tickmode="array", tickvals=correlation.columns),
        yaxis=dict(tickmode="array", tickvals=correlation.columns),
        showlegend=False,
    )

    # Show the interactive plot
    return fig


def analyze_holiday_impact_with_chart(sector_df, sector_name="Sector"):
    """
    Analyzes the impact of holidays on total sales and creates an interactive bar chart
    for average sales by holiday, displaying the highest, lowest, and "No" values as annotations on top of the bars.

    Parameters:
    - sector_df: DataFrame containing the sales data for a specific sector, including
                  'Total Sales', 'Holiday', 'Is_Holiday', 'Is_Pre_Holiday'.
    - sector_name: Name of the sector (default: "Sector").

    Returns:
    - A dictionary containing the correlation results for holidays, pre-holidays, and the holiday column.
    - Displays an interactive bar chart for average sales by holiday.
    """
    # Ensure 'Date' is the index and is of datetime type
    sector_df = sector_df.set_index("Date")
    sector_df.index = pd.to_datetime(sector_df.index)

    # Ensure the 'Is_Holiday' and 'Is_Pre_Holiday' columns are boolean
    sector_df["Is_Holiday"] = sector_df["Is_Holiday"].astype(bool)
    sector_df["Is_Pre_Holiday"] = sector_df["Is_Pre_Holiday"].astype(bool)

    # Calculate correlation with 'Total Sales' for each holiday-related column
    correlation_holiday = (
        sector_df[["Total Sales", "Is_Holiday"]].corr().loc["Total Sales", "Is_Holiday"]
    )
    correlation_pre_holiday = (
        sector_df[["Total Sales", "Is_Pre_Holiday"]]
        .corr()
        .loc["Total Sales", "Is_Pre_Holiday"]
    )

    # Calculate average sales and observation count for each specific holiday
    holiday_impact = {}
    holiday_count = {}
    if "Holiday" in sector_df.columns:
        unique_holidays = sector_df["Holiday"].dropna().unique()
        for holiday in unique_holidays:
            holiday_sales = (
                sector_df[sector_df["Holiday"] == holiday]["Total Sales"]
                .mean()
                .round(2)
            )
            count = sector_df[sector_df["Holiday"] == holiday].shape[0]
            holiday_impact[holiday] = holiday_sales
            holiday_count[holiday] = count

    # Prepare the results
    results = {
        "Correlation with Holiday (Is_Holiday)": correlation_holiday,
        "Correlation with Pre-Holiday (Is_Pre_Holiday)": correlation_pre_holiday,
        "Holiday Impact on Sales": holiday_impact,
    }

    # Create a DataFrame for visualization
    holiday_sales_df = pd.DataFrame(
        {
            "Holiday": list(holiday_impact.keys()),
            "Average Sales": list(holiday_impact.values()),
            "Count": [holiday_count[h] for h in holiday_impact.keys()],
        }
    )
    holiday_sales_df = holiday_sales_df.sort_values(by="Average Sales", ascending=False)

    # Find the holidays with the highest and lowest sales for annotation
    top_holiday = holiday_sales_df.iloc[0]
    bottom_holiday = holiday_sales_df.iloc[-1]
    no_holiday = (
        holiday_sales_df[holiday_sales_df["Holiday"] == "No"]
        if "No" in holiday_sales_df["Holiday"].values
        else None
    )

    # Plot interactive bar chart
    fig = px.bar(
        holiday_sales_df,
        x="Holiday",
        y="Average Sales",
        title=f"Average Sales by Holiday for {sector_name}",
        labels={"Average Sales": "Average Sales (€)", "Holiday": "Holiday"},
        color="Average Sales",
        color_continuous_scale="sunsetdark",
        hover_data={"Holiday": True, "Average Sales": ":.2f", "Count": True},
    )
    fig.update_layout(
        xaxis_title="Holiday",
        yaxis_title="Average Sales (€)",
        xaxis_tickangle=45,
        height=600,
        hoverlabel=dict(font_size=12, font_family="Arial"),
    )

    # Add annotations for the highest and lowest holidays
    fig.add_annotation(
        x=top_holiday["Holiday"],
        y=top_holiday["Average Sales"],
        text=f"€{top_holiday['Average Sales']}",
        showarrow=False,
        font=dict(size=11, color="white"),
        bgcolor="purple",
        yshift=12,  # Position the text slightly above the bar
    )

    fig.add_annotation(
        x=bottom_holiday["Holiday"],
        y=bottom_holiday["Average Sales"],
        text=f"€{bottom_holiday['Average Sales']}",
        showarrow=False,
        font=dict(size=11, color="black"),
        bgcolor="lightyellow",
        yshift=12,  # Position the text slightly above the bar
    )

    # Add annotation for "No" holiday if exists
    if no_holiday is not None:
        fig.add_annotation(
            x=no_holiday["Holiday"].iloc[0],
            y=no_holiday["Average Sales"].iloc[0],
            text=f"€{no_holiday['Average Sales'].iloc[0]}",
            showarrow=False,
            font=dict(size=11, color="white"),
            bgcolor="red",
            yshift=12,  # Position the text slightly above the bar
        )

    return results, fig


def show_holidays_impact(df, sector_name):
    sector_df = df[df["Sector"] == sector_name].copy()
    sector_df["Date"] = pd.to_datetime(sector_df["Date"])

    output = {}

    results, fig = analyze_holiday_impact_with_chart(sector_df)

    output["plot"] = fig

    correlation_data = [
        [
            "Correlation with Holiday (Is_Holiday)",
            results["Correlation with Holiday (Is_Holiday)"],
        ],
        [
            "Correlation with Pre-Holiday (Is_Pre_Holiday)",
            results["Correlation with Pre-Holiday (Is_Pre_Holiday)"],
        ],
    ]
    output["Correlation Data"] = tabulate(
        correlation_data, headers=["Metric", "Value"], tablefmt="grid"
    )

    # Separate holiday impact data for holidays and non-holidays
    holiday_data = [
        [holiday, sales]
        for holiday, sales in results["Holiday Impact on Sales"].items()
        if holiday != "No"
    ]
    non_holiday_data = [
        [holiday, sales]
        for holiday, sales in results["Holiday Impact on Sales"].items()
        if holiday == "No"
    ]

    # Sort holiday data by average sales
    holiday_data.sort(key=lambda x: x[1], reverse=True)

    output["Holiday Data"] = tabulate(
        holiday_data, headers=["Holiday", "Average Sales"], tablefmt="grid"
    )

    output["Non-Holiday Data"] = tabulate(
        non_holiday_data, headers=["Holiday", "Average Sales"], tablefmt="grid"
    )

    return output


def analyze_sales_correlation_with_lockdown(df, sector_name):
    """
    Analyzes the correlation between Total Sales and Lockdown, Transitional periods,
    with detailed statistics and an interactive chart showing the correlation.

    Parameters:
    - sector_df (pd.DataFrame): The sector dataframe containing 'Date' and 'Total Sales' columns.

    Returns:
    - dict: A dictionary containing detailed results like correlation values, sales comparisons, and statistical tests.
    - Interactive chart for sales correlation with lockdown and transitional periods.
    """
    sector_df = df[df["Sector"] == sector_name].copy()

    if "Date" not in sector_df.columns or "Total Sales" not in sector_df.columns:
        raise ValueError("The sector_df must contain 'Date' and 'Total Sales' columns.")

    output = {}

    # Ensure 'Date' is datetime
    sector_df["Date"] = pd.to_datetime(sector_df["Date"])

    # Define the lockdown and transitional periods
    lockdown_periods = [
        (pd.Timestamp("2020-03-09"), pd.Timestamp("2020-05-17")),
        (pd.Timestamp("2020-11-06"), pd.Timestamp("2021-04-25")),
    ]
    transitional_period = (pd.Timestamp("2020-05-18"), pd.Timestamp("2020-11-06"))

    # Add Lockdown Status column
    sector_df["Lockdown Status"] = 0  # Default to non-lockdown
    for start, end in lockdown_periods:
        sector_df.loc[
            (sector_df["Date"] >= start) & (sector_df["Date"] <= end), "Lockdown Status"
        ] = 1

    # Add Transitional Period column
    sector_df["Is_Transitional"] = 0  # Default to non-transitional
    sector_df.loc[
        (sector_df["Date"] >= transitional_period[0])
        & (sector_df["Date"] <= transitional_period[1]),
        "Is_Transitional",
    ] = 1

    # Sales comparison for Lockdown, Transitional, and Non-Lockdown periods
    lockdown_sales = sector_df[sector_df["Lockdown Status"] == 1]["Total Sales"]
    transitional_sales = sector_df[sector_df["Is_Transitional"] == 1]["Total Sales"]
    non_lockdown_sales = sector_df[sector_df["Lockdown Status"] == 0]["Total Sales"]

    # Calculate mean and median for each period
    sales_stats = {
        "Lockdown": {"mean": lockdown_sales.mean(), "median": lockdown_sales.median()},
        "Transitional": {
            "mean": transitional_sales.mean(),
            "median": transitional_sales.median(),
        },
        "Non-Lockdown": {
            "mean": non_lockdown_sales.mean(),
            "median": non_lockdown_sales.median(),
        },
    }

    # Correlation calculation between sales and Lockdown/Transitional periods
    correlation_lockdown_sales = (
        sector_df[["Total Sales", "Lockdown Status"]].corr().iloc[0, 1]
    )
    correlation_transitional_sales = (
        sector_df[["Total Sales", "Is_Transitional"]].corr().iloc[0, 1]
    )

    # Statistical test (t-test) between Lockdown and Non-Lockdown periods
    t_stat_lockdown, p_val_lockdown = ttest_ind(
        lockdown_sales, non_lockdown_sales, nan_policy="omit"
    )

    # Statistical test (t-test) between Transitional and Non-Lockdown periods
    t_stat_transitional, p_val_transitional = ttest_ind(
        transitional_sales, non_lockdown_sales, nan_policy="omit"
    )

    # Prepare results for tabulation with improved formatting and grouping
    results_table = [
        ["Correlation with Lockdown Status", f"{correlation_lockdown_sales:.3f}"],
        [
            "Correlation with Transitional Period",
            f"{correlation_transitional_sales:.3f}",
        ],
        ["", ""],  # Empty row for visual separation
        [
            "Period",
            "Mean Sales (€)",
            "Median Sales (€)",
            "% Change from Non-Lockdown (Mean)",
        ],
        [
            "Lockdown",
            f"{sales_stats['Lockdown']['mean']:,.2f}",
            f"{sales_stats['Lockdown']['median']:,.2f}",
            f"{(sales_stats['Lockdown']['mean'] - sales_stats['Non-Lockdown']['mean']) / sales_stats['Non-Lockdown']['mean'] * 100:.2f}%",
        ],
        [
            "Transitional",
            f"{sales_stats['Transitional']['mean']:,.2f}",
            f"{sales_stats['Transitional']['median']:,.2f}",
            f"{(sales_stats['Transitional']['mean'] - sales_stats['Non-Lockdown']['mean']) / sales_stats['Non-Lockdown']['mean'] * 100:.2f}%",
        ],
        [
            "Non-Lockdown",
            f"{sales_stats['Non-Lockdown']['mean']:,.2f}",
            f"{sales_stats['Non-Lockdown']['median']:,.2f}",
            "-",
        ],
        ["", ""],  # Empty row for visual separation
        [
            "T-test (Lockdown vs Non-Lockdown)",
            f"t-statistic: {t_stat_lockdown:.3f}, p-value: {p_val_lockdown:.3f}",
        ],
        [
            "T-test (Transitional vs Non-Lockdown)",
            f"t-statistic: {t_stat_transitional:.3f}, p-value: {p_val_transitional:.3f}",
        ],
    ]

    output["results"] = tabulate(results_table, headers="firstrow", tablefmt="grid")

    # Prepare data for the grouped bar chart
    chart_data = {
        "Period": [
            "Lockdown",
            "Transitional",
            "Non-Lockdown",
            "Lockdown",
            "Transitional",
            "Non-Lockdown",
        ],  # Include Non-Lockdown
        "Metric": [
            "Mean Sales",
            "Mean Sales",
            "Mean Sales",
            "Median Sales",
            "Median Sales",
            "Median Sales",
        ],
        "Value": [
            sales_stats["Lockdown"]["mean"],
            sales_stats["Transitional"]["mean"],
            sales_stats["Non-Lockdown"]["mean"],
            sales_stats["Lockdown"]["median"],
            sales_stats["Transitional"]["median"],
            sales_stats["Non-Lockdown"]["median"],
        ],
    }
    chart_df = pd.DataFrame(chart_data)

    # Create the grouped bar chart with values displayed on top of the bars
    fig = px.bar(
        chart_df,
        x="Period",
        y="Value",
        color="Metric",
        barmode="group",  # Group bars by metric
        title=f"Sales Comparison: Lockdown, Transitional, and Non-Lockdown Periods for {sector_name}",
        labels={"Value": "Sales (€)"},
        color_discrete_sequence=px.colors.qualitative.Set1,  # Use a simple color palette
        text="Value",  # Add values on top of bars
    )

    # Adjust layout for clarity and enhance the visual appearance
    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Sales (€)",
        xaxis_tickangle=45,
        height=600,
        title_x=0.5,
        legend_title="Metric",  # Add a title to the legend
        template="plotly_white",  # A cleaner background
    )

    # Update bar text appearance
    fig.update_traces(
        texttemplate="€%{text:.2f}",
        textposition="inside",
        textfont=dict(size=12, color="black"),
    )

    output["plot"] = fig

    return output


def compare_sales_lockdowns(df, sector_name):
    """
    Compare sales between the first and second lockdown periods for the Pesca sector.
    This function will return statistics and perform a t-test on the sales between the two lockdown periods.

    Parameters:
    - sector_df: DataFrame containing the sales data for the sector, including the lockdown periods.

    Returns:
    - A dictionary containing the comparison results.
    """
    print(f"\n Lockdown Periods Comparing for {sector_name} Sector:")
    print("\n")

    # Filter Pesca sector data
    pesca_data = df[df["Sector"] == sector_name].copy()
    pesca_data.set_index("Date", inplace=True)

    # Define the lockdown periods (modify the exact dates as needed)
    lockdown_1_start = "2020-03-09"
    lockdown_1_end = "2020-05-17"
    lockdown_2_start = "2020-11-06"
    lockdown_2_end = "2020-12-03"

    # Filter data for first and second lockdowns
    lockdown_1_data = pesca_data[
        (pesca_data.index >= lockdown_1_start) & (pesca_data.index <= lockdown_1_end)
    ]
    lockdown_2_data = pesca_data[
        (pesca_data.index >= lockdown_2_start) & (pesca_data.index <= lockdown_2_end)
    ]

    # Calculate sales statistics for both lockdowns
    lockdown_1_mean = lockdown_1_data["Total Sales"].mean()
    lockdown_1_median = lockdown_1_data["Total Sales"].median()
    lockdown_2_mean = lockdown_2_data["Total Sales"].mean()
    lockdown_2_median = lockdown_2_data["Total Sales"].median()

    # Perform t-test between first and second lockdown periods
    t_stat, p_value = stats.ttest_ind(
        lockdown_1_data["Total Sales"],
        lockdown_2_data["Total Sales"],
        nan_policy="omit",
    )

    # Prepare results dictionary
    results = {
        "Sales Statistics": {
            "First Lockdown": {"mean": lockdown_1_mean, "median": lockdown_1_median},
            "Second Lockdown": {"mean": lockdown_2_mean, "median": lockdown_2_median},
        },
        "T-test between First and Second Lockdown": {
            "t-statistic": t_stat,
            "p-value": p_value,
        },
        "Sales Comparison": {
            "First Lockdown Sales": lockdown_1_data["Total Sales"].sum(),
            "Second Lockdown Sales": lockdown_2_data["Total Sales"].sum(),
            "Difference in Sales": lockdown_2_data["Total Sales"].sum()
            - lockdown_1_data["Total Sales"].sum(),
        },
    }

    # Prepare results for tabulation
    sales_stats_table = [
        ["Metric", "First Lockdown", "Second Lockdown"],
        ["Mean Sales", f"{lockdown_1_mean:.2f}", f"{lockdown_2_mean:.2f}"],
        ["Median Sales", f"{lockdown_1_median:.2f}", f"{lockdown_2_median:.2f}"],
        [
            "Total Sales",
            f"{results['Sales Comparison']['First Lockdown Sales']:.2f}",
            f"{results['Sales Comparison']['Second Lockdown Sales']:.2f}",
        ],
    ]

    ttest_table = [
        [
            "T-statistic",
            results["T-test between First and Second Lockdown"]["t-statistic"],
        ],
        ["P-value", results["T-test between First and Second Lockdown"]["p-value"]],
    ]

    # Print tabulated results
    print(tabulate(sales_stats_table, headers="firstrow", tablefmt="grid"))
    print("\n")
    print(tabulate(ttest_table, headers="firstrow", tablefmt="grid"))
    print("\n")

    # Create interactive bar chart with plotly.express
    fig = px.bar(
        x=["First Lockdown", "Second Lockdown"],
        y=[lockdown_1_mean, lockdown_2_mean],
        title="Sales Comparison Between First and Second Lockdowns (Pesca Sector)",
        labels={"x": "Lockdown Period", "y": "Average Sales (€)"},
        color=["red", "blue"],  # Custom colors
        text=[
            f"€{lockdown_1_mean:.2f}",
            f"€{lockdown_2_mean:.2f}",
        ],  # Add values on top
    )
    fig.update_traces(textposition="outside")  # Position values outside bars
    # Hide the legend
    fig.update_layout(showlegend=False)
    return fig


def analyze_sector_data(sector_df):
    """
    Analyze the sales data for a given sector dataframe and return various metrics including:
    - Monthly total sales
    - Trend, seasonal, and residual components from time series decomposition
    - Average sales by season
    - Correlation with economic indicators (GDP, Inflation, Unemployment)

    Parameters:
    - sector_df: DataFrame containing the sales data for a specific sector with columns
                  'Date', 'Sector', 'Total Sales', 'GDP', 'Inflation', 'Unemployment', 'Season'

    Returns:
    - A dictionary containing the analysis results
    """

    # Ensure 'Date' is the index and is of datetime type
    sector_df = sector_df.set_index("Date")
    sector_df.index = pd.to_datetime(sector_df.index)

    # Resample to monthly data (sum of total sales per month)
    sector_monthly = sector_df["Total Sales"].resample("ME").sum()

    # 1. Monthly total sales trend
    monthly_sales = sector_monthly.mean()

    # 2. Decomposition of the time series (additive model)
    decomposition = seasonal_decompose(sector_monthly, model="additive")
    trend = decomposition.trend.dropna().mean()  # Average trend component
    seasonal = decomposition.seasonal.dropna().mean()  # Average seasonal component
    residual = decomposition.resid.dropna().mean()  # Average residual component

    # 3. Average Sales by Season
    seasonal_sales = sector_df.groupby("Season")["Total Sales"].mean()

    # 4. Correlation with Economic Indicators (GDP, Inflation, Unemployment)
    economic_data = sector_df[
        ["Total Sales", "GDP", "Inflation", "Unemployment"]
    ].dropna()
    correlation = economic_data.corr()

    # Organize the results into a dictionary and format sales numbers as money
    results = {
        "Monthly Total Sales Trend": f"€{monthly_sales:,.2f}",  # Format as money
        "Time Series Decomposition": {
            "Trend": f"€{trend:,.2f}",  # Format as money
            "Seasonal": f"€{seasonal:,.2f}",  # Format as money
            "Residual": f"€{residual:,.2f}",  # Format as money
        },
        "Average Sales by Season": seasonal_sales.apply(
            lambda x: f"€{x:,.2f}"
        ),  # Format as money
        "Correlation with Economic Indicators": correlation,
    }

    return results


def sector_explore(df, sector_name):
    """
    Explore the sales data for a specific sector.

    Parameters:
    - sector_name (str): The name of the sector to explore.

    Returns:
    - None
    """
    sector_df = df[df["Sector"] == sector_name].copy()
    output = {}

    # Calculate and display summary numbers
    total_sales = sector_df["Total Sales"].sum()
    average_sales = sector_df["Total Sales"].mean()
    num_sales_events = len(sector_df)
    # Calculate percentage of sector sales compared to all sales
    percentage_of_total_sales = (total_sales / df["Total Sales"].sum()) * 100
    # Prepare data for the table
    table_data = [
        ["Total Sales (€)", f"{total_sales:,.2f}"],
        ["Average Sales per Event (€)", f"{average_sales:,.2f}"],
        ["Number of Sales Events", num_sales_events],
        ["Percentage of Total Sales (%)", f"{percentage_of_total_sales:.2f}"],
    ]

    output[f"Summary Numbers for {sector_name} Sector:"] = tabulate(
        table_data, headers=["Metric", "Value"], tablefmt="fancy_grid"
    )

    # Analyze the sector data and display the results
    results = analyze_sector_data(sector_df)

    output["Monthly Total Sales Trend"] = tabulate(
        [["Monthly Total Sales Trend", results["Monthly Total Sales Trend"]]],
        headers=["Metric", "Value"],
        tablefmt="fancy_grid",
    )

    decomposition_table = []
    for component, value in results["Time Series Decomposition"].items():
        decomposition_table.append([component, value])
    output["Decomposition Components"] = tabulate(
        decomposition_table, headers=["Component", "Value"], tablefmt="fancy_grid"
    )

    seasonal_sales_table = []
    for season, sales in results["Average Sales by Season"].items():
        seasonal_sales_table.append([season, sales])
    output["Average Sales by Season"] = tabulate(
        seasonal_sales_table,
        headers=["Season", "Average Sales"],
        tablefmt="fancy_grid",
    )

    # Create a DataFrame for correlation for easier tabulate usage
    correlation_df = pd.DataFrame(results["Correlation with Economic Indicators"])
    # You might need to adjust the column names here based on your output
    correlation_df = correlation_df.reset_index().rename(columns={"index": "Indicator"})
    output["Correlation with Economic Indicators"] = tabulate(
        correlation_df, headers="keys", tablefmt="fancy_grid"
    )

    # Plots
    output["plots"] = {}

    output["plots"]["plot_sector_frequency_interactive"] = (
        plot_sector_frequency_interactive(sector_df, freq="ME")
    )
    output["plots"]["plot_sector_monthly_sales_interactive"] = (
        plot_sector_monthly_sales_interactive(sector_df)
    )
    output["plots"]["plot_sector_daily_sales_interactive"] = (
        plot_sector_daily_sales_interactive(sector_df)
    )
    output["plots"]["plot_interactive_decomposition"] = plot_interactive_decomposition(
        sector_df
    )
    output["plots"]["plot_interactive_average_sales_by_season"] = (
        plot_interactive_average_sales_by_season(sector_df)
    )
    output["plots"]["plot_interactive_correlation_matrix"] = (
        plot_interactive_correlation_matrix(sector_df)
    )

    return output


# Enrich Pesca Sector
def enrich_pesca_sector(sector_df):
    output = {}
    # Adding World Fishing Championship (August)
    sector_df["World_Fishing_Championship"] = sector_df["Date"].apply(
        lambda x: 1 if (x.month == 8) else 0
    )

    # Adding National Fishing Day (June 18)
    sector_df["National_Fishing_Day"] = sector_df["Date"].apply(
        lambda x: 1 if (x.month == 6 and x.day == 18) else 0
    )

    # Filter for Pesca sector
    pesca_data = sector_df.copy()

    # Calculate average sales on World Fishing Championship days vs non-event days
    pesca_world_fishing = pesca_data[pesca_data["World_Fishing_Championship"] == 1][
        "Total Sales"
    ]
    pesca_non_world_fishing = pesca_data[pesca_data["World_Fishing_Championship"] == 0][
        "Total Sales"
    ]

    # Calculate average sales on National Fishing Day vs non-event days
    pesca_national_fishing = pesca_data[pesca_data["National_Fishing_Day"] == 1][
        "Total Sales"
    ]
    pesca_non_national_fishing = pesca_data[pesca_data["National_Fishing_Day"] == 0][
        "Total Sales"
    ]

    # Perform t-tests
    from scipy import stats

    # World Fishing Championship t-test
    t_stat_wf, p_val_wf = stats.ttest_ind(pesca_world_fishing, pesca_non_world_fishing)

    # National Fishing Day t-test
    t_stat_nfd, p_val_nfd = stats.ttest_ind(
        pesca_national_fishing, pesca_non_national_fishing
    )

    # Prepare data for tabulate
    results_table = [
        [
            "World Fishing Championship",
            pesca_world_fishing.mean(),
            pesca_non_world_fishing.mean(),
            t_stat_wf,
            p_val_wf,
        ],
        [
            "National Fishing Day",
            pesca_national_fishing.mean(),
            pesca_non_national_fishing.mean(),
            t_stat_nfd,
            p_val_nfd,
        ],
    ]

    # Define headers for the table
    headers = [
        "Event",
        "Average Sales (Event)",
        "Average Sales (Non-Event)",
        "t-statistic",
        "p-value",
    ]
    output["results"] = tabulate(results_table, headers=headers, tablefmt="grid")

    # Prepare data for the interactive chart
    chart_data = pd.DataFrame(
        {
            "Event": [
                "World Fishing Championship",
                "World Fishing Championship",
                "National Fishing Day",
                "National Fishing Day",
            ],
            "Type": ["Event", "Non-Event", "Event", "Non-Event"],
            "Average Sales": [
                pesca_world_fishing.mean(),
                pesca_non_world_fishing.mean(),
                pesca_national_fishing.mean(),
                pesca_non_national_fishing.mean(),
            ],
            "t-statistic": [t_stat_wf, t_stat_wf, t_stat_nfd, t_stat_nfd],
            "p-value": [p_val_wf, p_val_wf, p_val_nfd, p_val_nfd],
        }
    )

    # Create the grouped bar chart
    fig = px.bar(
        chart_data,
        x="Event",
        y="Average Sales",
        color="Type",
        barmode="group",
        title="Average Sales for Events vs. Non-Events",
        text="Average Sales",  # Display the average sales on the bars
        hover_data={"t-statistic": True, "p-value": True, "Average Sales": ":.2f"},
    )

    # Add layout details
    fig.update_traces(texttemplate="€%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Event",
        yaxis_title="Average Sales (€)",
        legend_title="Type",
        template="plotly_white",
        height=600,
    )

    output["plot"] = fig

    return output


# Enrich Calcio Sector
def enrich_calcio_sector(sector_df):
    # Define Serie A matchdays
    serie_a_start_year = 2015
    serie_a_end_year = 2024

    # Generate all matchdays
    serie_a_matches = set()

    for year in range(serie_a_start_year, serie_a_end_year + 1):
        start_date = datetime(year, 8, 20)  # Approx. Serie A start
        end_date = datetime(year + 1, 5, 20)  # Approx. Serie A end

        match_date = start_date
        while match_date <= end_date:
            if match_date.weekday() in [5, 6]:  # Weekend matches (Saturday, Sunday)
                serie_a_matches.add(match_date)
            elif match_date.weekday() in [
                1,
                2,
            ]:  # Occasional midweek matches (Tuesday, Wednesday)
                serie_a_matches.add(match_date)
            match_date += timedelta(days=1)  # Iterate day by day

    # Convert matchdays to a sorted list
    serie_a_dates = pd.to_datetime(sorted(serie_a_matches))

    # Mark Serie A matchdays in sector_df
    sector_df["Serie_A_Match"] = sector_df["Date"].isin(serie_a_dates).astype(int)

    # Optimized function to generate matchdays
    def generate_matchdays_optimized(start_date, end_date, weekdays):
        all_dates = pd.date_range(start=start_date, end=end_date)
        matchdays = all_dates[
            (
                all_dates.month.isin([2, 3, 4, 5, 6, 9, 10, 11, 12])
            )  # Exclude January & July & August
            & (all_dates.weekday.isin(weekdays))  # Filter by weekday
        ]
        return matchdays.tolist()

    # Function to combine matchdays with finals
    def get_matchdays(start, end, weekdays, finals=None):
        matchdays = generate_matchdays_optimized(start, end, weekdays)
        if finals:
            matchdays.extend(pd.to_datetime(finals))  # Include final matchdays
        return matchdays

    # Champions League dates
    ucl_group_stage_dates = get_matchdays(
        datetime(2015, 9, 15), datetime(2024, 12, 10), [1, 2]
    )
    ucl_knockout_dates = get_matchdays(
        datetime(2016, 2, 16), datetime(2024, 5, 8), [1, 2]
    )
    ucl_final_dates = [
        datetime(2016, 5, 28),
        datetime(2017, 6, 3),
        datetime(2018, 5, 26),
        datetime(2019, 6, 1),
        datetime(2020, 8, 23),
        datetime(2021, 5, 29),
        datetime(2022, 5, 28),
        datetime(2023, 6, 10),
    ]

    # Europa League dates
    uel_group_stage_dates = get_matchdays(
        datetime(2015, 9, 17), datetime(2024, 12, 12), [3]
    )
    uel_knockout_dates = get_matchdays(
        datetime(2016, 2, 18), datetime(2024, 5, 10), [3]
    )
    uel_final_dates = [
        datetime(2016, 5, 18),
        datetime(2017, 5, 24),
        datetime(2018, 5, 16),
        datetime(2019, 5, 29),
        datetime(2020, 8, 21),
        datetime(2021, 5, 26),
        datetime(2022, 5, 18),
        datetime(2023, 5, 31),
    ]

    # Combine all dates
    ucl_dates = pd.to_datetime(
        ucl_group_stage_dates + ucl_knockout_dates + ucl_final_dates
    )
    uel_dates = pd.to_datetime(
        uel_group_stage_dates + uel_knockout_dates + uel_final_dates
    )

    # Add to DataFrame
    sector_df["Champions_League_Match"] = sector_df["Date"].isin(ucl_dates).astype(int)
    sector_df["Europa_League_Match"] = sector_df["Date"].isin(uel_dates).astype(int)

    # Define Euro and World Cup periods
    event_periods = {
        "Euro": [
            (datetime(2016, 6, 10), datetime(2016, 7, 10)),  # Euro 2016
            (
                datetime(2021, 6, 11),
                datetime(2021, 7, 11),
            ),  # Euro 2020 (played in 2021)
        ],
        "World Cup": [
            (datetime(2018, 6, 14), datetime(2018, 7, 15)),  # World Cup 2018
            (datetime(2022, 11, 20), datetime(2022, 12, 18)),  # World Cup 2022
        ],
    }

    # Generate all event dates
    def generate_event_dates(event_periods):
        event_dates = [
            pd.date_range(start, end).to_list()
            for periods in event_periods.values()
            for start, end in periods
        ]
        return [date for sublist in event_dates for date in sublist]

    # Get all dates for Euros and World Cups
    euro_wc_dates = pd.to_datetime(generate_event_dates(event_periods))

    # Add new column for Euro and World Cup events
    sector_df["Euro_WC_Event"] = sector_df["Date"].isin(euro_wc_dates).astype(int)

    return analyze_football_event_impact(sector_df)


def analyze_football_event_impact(sector_df):
    # Ensure that the date column is set as datetime
    sector_df["Date"] = pd.to_datetime(sector_df["Date"])

    # Create a new dataframe with relevant columns for the analysis
    event_columns = [
        "Serie_A_Match",
        "Champions_League_Match",
        "Europa_League_Match",
        "Euro_WC_Event",
    ]
    sector_events = sector_df[["Date", "Total Sales"] + event_columns].copy()

    # Group the data by the event status (1 for event, 0 for no event)
    event_sales = {}
    for event in event_columns:
        event_sales[event] = sector_events.groupby(event)["Total Sales"].mean()

    # Calculate the correlation between event columns and sales
    correlation_results = {}
    for event in event_columns:
        correlation = sector_events[[event, "Total Sales"]].corr().iloc[0, 1]
        correlation_results[event] = correlation
    # Prepare data for tabulation
    correlation_table = [
        [event, correlation] for event, correlation in correlation_results.items()
    ]

    # Define headers for the table
    headers = ["Event", "Correlation with Total Sales"]

    results = tabulate(correlation_table, headers=headers, tablefmt="grid")

    # Create Interactive Pie Charts using Plotly
    # Specify 'specs' to create subplots suitable for pie charts
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=event_columns,
        specs=[
            [{"type": "domain"}, {"type": "domain"}],
            [{"type": "domain"}, {"type": "domain"}],
        ],
    )

    for i, event in enumerate(event_columns):
        # Get the sales during event and non-event
        sales_event = event_sales[event][1]  # Sales during event
        sales_non_event = event_sales[event][0]  # Sales during non-event

        # Pie chart data
        labels = ["Event Sales", "Non-Event Sales"]
        sizes = [sales_event, sales_non_event]
        colors = ["#66b3ff", "#99ff99"]

        # Adding a pie chart trace
        fig.add_trace(
            go.Pie(
                labels=labels,
                values=sizes,
                name=event,
                hole=0.3,
                marker=dict(colors=colors),
                hoverinfo="label+percent+value",
            ),
            row=(i // 2) + 1,
            col=(i % 2) + 1,
        )

    fig.update_layout(
        title_text="Sales Distribution During Events vs Non-Events",
        showlegend=False,
        height=800,
        title_x=0.5,
        title_y=0.98,
    )

    # Return correlation results for further insights
    return results, fig


def analyze_arima_relationships_interactive(
    df, sector_name="", p=1, d=1, q=0, verbose=False
):
    """
    Perform ARIMA-based time series analysis with interactive plots using Plotly.

    Parameters:
    - df: DataFrame containing the sales data with columns 'Date', 'Sector', 'Total Sales'
    - sector_name: Name of the sector to analyze
    - p, d, q: ARIMA model parameters
    - verbose: If True, prints progress updates and model summary

    Returns:
    - A dictionary containing analysis results, including residuals, trends, and seasonality insights.
    """
    try:
        output = {}
        # 1. Validate input data
        if (
            "Date" not in df.columns
            or "Sector" not in df.columns
            or "Total Sales" not in df.columns
        ):
            raise ValueError(
                "DataFrame must contain 'Date', 'Sector', and 'Total Sales' columns."
            )

        if sector_name == "":
            sector_df = df.copy()
        else:
            sector_df = df[df["Sector"] == sector_name].copy()
        if sector_df.empty:
            raise ValueError(f"No data found for sector: {sector_name}")

        sector_df["Date"] = pd.to_datetime(sector_df["Date"])
        sector_df = sector_df.sort_values("Date").set_index("Date")
        sales_data = sector_df["Total Sales"]

        # 2. Fit the ARIMA model
        if verbose:
            print(f"Fitting ARIMA({p}, {d}, {q}) model for sector: {sector_name}")

        model = ARIMA(sales_data, order=(p, d, q))
        model_fit = model.fit()

        if verbose:
            print(model_fit.summary())

        # 3. Extract and analyze residuals
        residuals = model_fit.resid
        residual_stats = {
            "Mean": np.mean(residuals),
            "Variance": np.var(residuals),
            "Min": np.min(residuals),
            "Max": np.max(residuals),
            "Skewness": residuals.skew(),
            "Kurtosis": residuals.kurtosis(),
        }
        # 3.1. Display results using Tabulate
        results_table = [
            ["ARIMA Order", f"({p}, {d}, {q})"],
            ["AIC", model_fit.aic],
            ["BIC", model_fit.bic],
            ["Residual Mean", residual_stats["Mean"]],
            ["Residual Variance", residual_stats["Variance"]],
            ["Residual Min", residual_stats["Min"]],
            ["Residual Max", residual_stats["Max"]],
            ["Residual Skewness", residual_stats["Skewness"]],
            ["Residual Kurtosis", residual_stats["Kurtosis"]],
        ]
        output["ARIMA Analysis Results"] = tabulate(
            results_table, headers=["Metric", "Value"], tablefmt="grid"
        )

        # 4. Interactive Plotly Visualization
        fig = go.Figure()

        # Add original data trace
        fig.add_trace(
            go.Scatter(
                x=sales_data.index,
                y=sales_data,
                mode="lines+markers",
                name="Original Data",
                line=dict(color="blue"),
                marker=dict(size=6),
            )
        )

        # Add fitted values trace
        fig.add_trace(
            go.Scatter(
                x=sales_data.index,
                y=model_fit.fittedvalues,
                mode="lines",
                name="Fitted Values",
                line=dict(color="orange", dash="dash"),
            )
        )

        # Add residuals trace
        fig.add_trace(
            go.Scatter(
                x=residuals.index,
                y=residuals,
                mode="lines",
                name="Residuals",
                line=dict(color="red"),
            )
        )

        # Customize layout
        fig.update_layout(
            title=f"{sector_name} ARIMA({p}, {d}, {q}) Analysis",
            xaxis_title="Date",
            yaxis_title="Total Sales",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            height=600,
        )

        # Show interactive plot
        output["plot"] = fig

        # 5. Store results
        results = {
            "Sector": sector_name,
            "Model": f"ARIMA({p}, {d}, {q})",
            "Residuals": residuals,
            "Fitted Values": model_fit.fittedvalues,
            "Summary": model_fit.summary(),
        }
        output["results"] = results

        return output

    except Exception as e:
        if verbose:
            print(f"Error analyzing sector {sector_name}: {e}")
        return None


def analyze_state_space(
    df, sector_name="", p=1, d=1, q=1, seasonal_order=None, verbose=True
):
    """
    Perform time series analysis using a state-space model and return tabulated results with an interactive plot.

    Parameters:
    - df: DataFrame containing 'Date', 'Sector', and 'Total Sales' columns
    - sector_name: The sector name to analyze
    - p, d, q: Non-seasonal ARIMA parameters for the state-space model
    - seasonal_order: Tuple (P, D, Q, S) for seasonal components (optional)
    - verbose: If True, prints detailed output and model summary

    Returns:
    - results: A dictionary containing model metrics, residual statistics, and the interactive plot
    """
    try:
        output = {}
        # Validate input
        if (
            "Date" not in df.columns
            or "Sector" not in df.columns
            or "Total Sales" not in df.columns
        ):
            raise ValueError(
                "The DataFrame must contain 'Date', 'Sector', and 'Total Sales' columns."
            )

        if sector_name == "":
            sector_df = df.copy()
        else:
            sector_df = df[df["Sector"] == sector_name].copy()
        if sector_df.empty:
            raise ValueError(f"No data found for sector: {sector_name}")

        # Prepare the data
        sector_df["Date"] = pd.to_datetime(sector_df["Date"])
        sector_df = sector_df.sort_values("Date").set_index("Date")
        sales_data = sector_df["Total Sales"]

        # Fit state-space model
        if verbose:
            print(f"Fitting state-space model for sector: {sector_name}")
        model = SARIMAX(
            sales_data,
            order=(p, d, q),
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False)

        if verbose:
            print(model_fit.summary())

        # Extract residual statistics
        residuals = model_fit.resid
        residual_stats = {
            "Mean": np.mean(residuals),
            "Variance": np.var(residuals),
            "Min": np.min(residuals),
            "Max": np.max(residuals),
            "Skewness": residuals.skew(),
            "Kurtosis": residuals.kurtosis(),
        }

        # Display results using Tabulate
        results_table = [
            ["ARIMA Order", f"({p}, {d}, {q})"],
            ["Seasonal Order", f"{seasonal_order}" if seasonal_order else "None"],
            ["AIC", model_fit.aic],
            ["BIC", model_fit.bic],
            ["Residual Mean", residual_stats["Mean"]],
            ["Residual Variance", residual_stats["Variance"]],
            ["Residual Min", residual_stats["Min"]],
            ["Residual Max", residual_stats["Max"]],
            ["Residual Skewness", residual_stats["Skewness"]],
            ["Residual Kurtosis", residual_stats["Kurtosis"]],
        ]
        output["State-Space Model Results"] = tabulate(
            results_table, headers=["Metric", "Value"], tablefmt="grid"
        )

        # Interactive plot
        fitted_values = model_fit.fittedvalues
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sales_data.index, y=sales_data, mode="lines", name="Actual Sales"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fitted_values.index,
                y=fitted_values,
                mode="lines",
                name="Fitted Values",
            )
        )
        fig.update_layout(
            title=f"{sector_name} State-Space Model Analysis",
            xaxis_title="Date",
            yaxis_title="Total Sales",
            template="plotly_white",
        )
        output["plot"] = fig

        # Store results
        results = {
            "Sector": sector_name,
            "Model": f"ARIMA({p}, {d}, {q})",
            "Seasonal Order": seasonal_order,
            "Residuals": residuals,
            "Summary": model_fit.summary(),
            "Statistics": residual_stats,
            "Plot": fig,
        }

        output["results"] = results

        return output

    except Exception as e:
        if verbose:
            print(f"Error analyzing sector {sector_name}: {e}")
        return None
