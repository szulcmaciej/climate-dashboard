import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from datetime import datetime

from data_loader import get_north_atlantic_sst_data, get_antarctic_sea_ice_extent_data
from matplotlib import cm
import matplotlib


def prepare_sst_figure():
    df = get_north_atlantic_sst_data()

    # Separate year records from other categories
    df_years = df[df['name'].str.isdigit()]

    # Expand data column to have a row per temperature value for year records
    df_years = df_years.explode('data').reset_index(drop=True)

    # Construct dates assuming the data starts from the 1st day of the year for year records
    df_years['date'] = pd.to_datetime(df_years['name']) + pd.to_timedelta(df_years.groupby('name').cumcount(), unit='D')

    # Replace 'null' (or None) values with np.nan for proper handling in pandas
    df_years['data'] = df_years['data'].replace({None: np.nan})

    # Concatenate year and other records while keeping 'name' and 'data' columns
    df = df_years

    # Get the current year
    current_year = datetime.now().year

    # Create day and month columns
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month

    # Initialize the figure
    fig = go.Figure()

    # Get the unique years
    years = df['name'].unique()

    # Get a color map
    cmap = cm.get_cmap('RdBu_r', len(years))
    cmap = cm.get_cmap('rainbow', len(years))
    cmap = cm.get_cmap('plasma', len(years))
    # cmap = cm.get_cmap('Wistia', len(years))

    # Add a line for each year
    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i)[:3])
        if int(year) == current_year:
            fig.add_trace(go.Scatter(x=df[df['name'] == year]['day_of_year'], y=df[df['name'] == year]['data'],
                                     mode='lines', name=year, line=dict(color='red', width=3)))
        else:
            fig.add_trace(go.Scatter(x=df[df['name'] == year]['day_of_year'], y=df[df['name'] == year]['data'],
                                     mode='lines', name=year, line=dict(color=color), opacity=0.3))

    # Set layout properties
    fig.update_layout(title='North Atlantic Sea Surface Temperature', xaxis_title='Day of Year', yaxis_title='Temperature [C]',
                      width=900, height=600, legend={'traceorder': 'reversed'})

    return fig


def prepare_sie_figure():
    df_ice = get_antarctic_sea_ice_extent_data()

    # Convert 'Year', 'Month', 'Day' to a datetime format
    df_ice['Date'] = pd.to_datetime(df_ice[['Year', 'Month', 'Day']])

    # Create a new column for the formatted date string
    df_ice['Date_str'] = df_ice['Date'].dt.strftime('%d-%m-%Y')

    # Convert date to day of the year
    df_ice['Day_of_Year'] = df_ice['Date'].dt.dayofyear

    # Get the unique years
    years = df_ice['Year'].unique()

    # Get the current year
    current_year = datetime.now().year

    # Get a color map
    cmap = cm.get_cmap('RdBu', len(years))
    cmap = cm.get_cmap('plasma', len(years))

    # Initialize the figure
    fig = go.Figure()

    # Add a line for each year
    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i)[:3])
        year_data = df_ice[df_ice['Year'] == year]
        if year == current_year:
            fig.add_trace(go.Scatter(x=year_data['Day_of_Year'], y=year_data['Extent'],
                                     mode='lines', name=str(year), line=dict(color='red', width=3),
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>Extent</b>: %{y:.2f}<br>'
                                     '<extra></extra>',
                                     customdata=year_data['Date_str']))
        else:
            fig.add_trace(go.Scatter(x=year_data['Day_of_Year'], y=year_data['Extent'],
                                     mode='lines', name=str(year), line=dict(color=color), opacity=0.3,
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>Extent</b>: %{y:.2f}<br>'
                                     '<extra></extra>',
                                     customdata=year_data['Date_str']))

    # Set layout properties
    fig.update_layout(title='Antarctic Sea Ice Extent', xaxis_title='Day of Year',
                      yaxis_title='Ice Extent (10^6 sq km)',
                      width=900, height=600, legend={'traceorder': 'reversed'})
    return fig


def sst_anomaly_fig():
    # same as sst_fig but showing anomaly instead of absolute value of SST (mean subtracted)
    avg_from_year = 1982
    avg_to_year = 2011

    df = get_north_atlantic_sst_data()

    # Separate year records from other categories
    df_years = df[df['name'].str.isdigit()]
    df_other = df[~df['name'].str.isdigit()]

    # Expand data column to have a row per temperature value for year records
    df_years = df_years.explode('data').reset_index(drop=True)
    df_other = df_other.explode('data').reset_index(drop=True)

    # Construct dates assuming the data starts from the 1st day of the year for year records
    df_years['date'] = pd.to_datetime(df_years['name']) + pd.to_timedelta(df_years.groupby('name').cumcount(), unit='D')

    df_years['day_of_year'] = df_years['date'].dt.dayofyear

    # Replace 'null' (or None) values with np.nan for proper handling in pandas
    df_years['data'] = df_years['data'].replace({None: np.nan})

    # calculate anomaly (subtract 1982-2011 mean)
    day_of_year_mean_1982_2011 = df_years[df_years['date'].dt.year.isin(range(avg_from_year, avg_to_year))].groupby(df_years['date'].dt.dayofyear)['data'].mean()
    day_of_year_mean_1982_2011 = day_of_year_mean_1982_2011.reset_index()
    day_of_year_mean_1982_2011.columns = ['day_of_year', 'mean']
    df_years = df_years.merge(day_of_year_mean_1982_2011, on='day_of_year', suffixes=('', '_mean'))
    df_years['data'] = df_years['data'] - df_years['mean']




    # Concatenate year and other records while keeping 'name' and 'data' columns
    df = df_years


    # Get the current year
    current_year = datetime.now().year

    # Create day and month columns
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month

    # Initialize the figure
    fig = go.Figure()

    # Get the unique years
    years = df['name'].unique()

    # Get a color map
    cmap = cm.get_cmap('RdBu_r', len(years))
    cmap = cm.get_cmap('rainbow', len(years))
    cmap = cm.get_cmap('plasma', len(years))
    # cmap = cm.get_cmap('Wistia', len(years))

    # Add a line for each year
    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i)[:3])
        if int(year) == current_year:
            fig.add_trace(go.Scatter(x=df[df['name'] == year]['day_of_year'], y=df[df['name'] == year]['data'],
                                     mode='lines', name=year, line=dict(color='red', width=3)))
        else:
            fig.add_trace(go.Scatter(x=df[df['name'] == year]['day_of_year'], y=df[df['name'] == year]['data'],
                                     mode='lines', name=year, line=dict(color=color), opacity=0.3))

        # Set layout properties
    fig.update_layout(title='North Atlantic Sea Surface Temperature Anomaly', xaxis_title='Day of Year',
                      yaxis_title=f'Temperature difference from {avg_from_year}-{avg_to_year} mean [C]',
                      width=900, height=600, legend={'traceorder': 'reversed'})

    fig.add_shape(
        type="line", line=dict(dash="dash"),
        x0=0, x1=1, xref="paper",
        y0=0, y1=0, yref="y",
        line_color="lightgrey"
    )
    fig.add_annotation(
        text=f"{avg_from_year}-{avg_to_year} mean",
        xref="paper", yref="y",
        x=0.1, y=0.04,  # Coordinates to place the start of the text
        showarrow=False,  # Ensures no arrow is attached to the text
        font=dict(
            size=12,
            # color="green"
        ),
    )

    return fig


def sie_anomaly_fig():
    df = get_antarctic_sea_ice_extent_data()

    # Convert 'Year', 'Month', 'Day' to a datetime format
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Create a new column for the formatted date string
    df['date_str'] = df['date'].dt.strftime('%d-%m-%Y')

    # Convert date to day of the year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Calculate anomaly (subtract 1982-2011 mean)
    day_of_year_mean_1982_2011 = df[df['date'].dt.year.isin(range(2000, 2022))].groupby(df['date'].dt.dayofyear)['Extent'].mean()
    day_of_year_mean_1982_2011 = day_of_year_mean_1982_2011.reset_index()

    day_of_year_mean_1982_2011.columns = ['day_of_year', 'mean']
    df = df.merge(day_of_year_mean_1982_2011, on='day_of_year', suffixes=('', '_mean'))

    df['Extent_diff'] = df['Extent'] - df['mean']
    df = df.sort_values('Year')
    df = df.sort_values('day_of_year')

    # Get the unique years
    years = df['Year'].unique()

    # Get the current year
    current_year = datetime.now().year

    # Get a color map
    cmap = cm.get_cmap('plasma', len(years))

    # Initialize the figure
    fig = go.Figure()

    # Add a line for each year
    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i)[:3])
        year_data = df[df['Year'] == year]
        if year == current_year:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'], y=year_data['Extent_diff'],
                                    mode='lines', name=str(year), line=dict(color='red', width=3),
                                    hovertemplate=
                                    '<b>Date</b>: %{customdata}<br>' +
                                    '<b>Extent</b>: %{y:.2f}<br>'
                                    '<extra></extra>',
                                    customdata=year_data['date_str']))
        else:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'], y=year_data['Extent_diff'],
                                     mode='lines', name=str(year), line=dict(color=color), opacity=0.3,
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>Extent</b>: %{y:.2f}<br>'
                                     '<extra></extra>',
                                     customdata=year_data['date_str']))

    # Set layout properties
    fig.update_layout(title='Antarctic Sea Ice Extent Anomaly', xaxis_title='Day of Year',
                      yaxis_title='Ice Extent difference from 1982-2011 mean (10^6 sq km)',
                      width=900, height=600, legend={'traceorder': 'reversed'})
    return fig


def sie_anomaly_perc_fig():
    df = get_antarctic_sea_ice_extent_data()
    df = df.drop(columns=['Source Data', 'Missing'])

    # Convert 'Year', 'Month', 'Day' to a datetime format
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

    # Create a new column for the formatted date string
    df['date_str'] = df['date'].dt.strftime('%d-%m-%Y')

    # Convert date to day of the year
    df['day_of_year'] = df['date'].dt.dayofyear

    # Calculate anomaly (subtract 1982-2011 mean)
    day_of_year_mean_1982_2011 = df[df['date'].dt.year.isin(range(2000, 2022))].groupby(df['date'].dt.dayofyear)['Extent'].mean()
    day_of_year_mean_1982_2011 = day_of_year_mean_1982_2011.reset_index()

    day_of_year_mean_1982_2011.columns = ['day_of_year', 'mean']
    df = df.merge(day_of_year_mean_1982_2011, on='day_of_year', suffixes=('', '_mean'))

    df['Extent_diff_perc'] = (df['Extent'] - df['mean']) / df['mean'] * 100
    df = df.sort_values('Year')
    df = df.sort_values('day_of_year')

    # Get the unique years
    years = df['Year'].unique()

    # Get the current year
    current_year = datetime.now().year

    # Get a color map
    cmap = cm.get_cmap('plasma', len(years))

    # Initialize the figure
    fig = go.Figure()

    # Add a line for each year
    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i)[:3])
        year_data = df[df['Year'] == year]
        if year == current_year:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'], y=year_data['Extent_diff_perc'],
                                    mode='lines', name=str(year), line=dict(color='red', width=3),
                                    hovertemplate=
                                    '<b>Date</b>: %{customdata}<br>' +
                                    '<b>Extent diff</b>: %{y:.2f}%<br>'
                                    '<extra></extra>',
                                    customdata=year_data['date_str']))
        else:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'], y=year_data['Extent_diff_perc'],
                                     mode='lines', name=str(year), line=dict(color=color), opacity=0.3,
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>Extent diff</b>: %{y:.2f}%<br>'
                                     '<extra></extra>',
                                     customdata=year_data['date_str']))

    # Set layout properties
    fig.update_layout(title='Antarctic Sea Ice Extent Anomaly', xaxis_title='Day of Year',
                      yaxis_title='Ice Extent difference from 1982-2011 mean (10^6 sq km)',
                      width=900, height=600, legend={'traceorder': 'reversed'})
    return fig


# Display the figure
st.header('North Atlantic Sea Surface Temperature')
sst_tab1, sst_tab2 = st.tabs(["SST", "SST Anomaly"])
with sst_tab1:
    st.plotly_chart(prepare_sst_figure())
with sst_tab2:
    st.plotly_chart(sst_anomaly_fig())

st.header('Antarctic Sea Ice Extent')
sie_tab1, sie_tab2, sie_tab3 = st.tabs(["SIE", "SIE Anomaly", "SIE Anomaly %"])
with sie_tab1:
    st.plotly_chart(prepare_sie_figure())
with sie_tab2:
    st.plotly_chart(sie_anomaly_fig())
with sie_tab3:
    st.plotly_chart(sie_anomaly_perc_fig())
