import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

import plotly.graph_objects as go
import matplotlib


@st.cache_data(ttl=timedelta(hours=6))
def get_antarctic_sea_ice_extent_data():
    url = 'https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv'
    df = pd.read_csv(url, skipinitialspace=True, skiprows=[1])
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['value'] = df['Extent']
    df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df[['date', 'day_of_year', 'value', 'date_formatted']]
    return df


@st.cache_data(ttl=timedelta(hours=6))
def get_north_atlantic_sst_data():
    url = 'https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json'
    df = pd.read_json(url)
    df = df[df['name'].str.isnumeric()].set_index('name').data.apply(pd.Series).stack().reset_index(name='value')
    df.columns = ['Year', 'Day', 'value']
    df['Year'] = df['Year'].astype(int)
    df['Day'] = df['Day'].astype(int) + 1  # To correct zero-based indexing
    df['date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Day'].astype(str), format='%Y-%j')
    df['day_of_year'] = df['Day']
    df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df[['date', 'day_of_year', 'value', 'date_formatted']]
    return df


def interpolate_missing_dates(df):
    min_year = df['date'].dt.year.min()
    max_year = df['date'].dt.year.max()
    full_date_range = pd.date_range(start=f'{min_year}-01-01', end=f'{max_year}-12-31')
    df_full_range = pd.DataFrame(full_date_range, columns=['date'])
    df = pd.merge(df_full_range, df, on='date', how='left')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['value'] = df['value'].interpolate(method='linear', limit_area='inside')
    df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
    return df


def calculate_anomalies(df, start_year, end_year):
    df = df.copy()
    # Filter the data to include only the years in the specified range
    df_range = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]

    # Calculate the multi-year average for each day of the year
    averages = df_range.groupby(df_range['date'].dt.dayofyear)['value'].mean()

    # Calculate anomalies by subtracting the multi-year average from the daily values
    df['value'] = df.apply(lambda row: row['value'] - averages[row['day_of_year']] if row['day_of_year'] in averages.index else np.NaN, axis=1)

    return df


def prepare_figure(df, title, yaxis_title, y_hover_format):
    current_year = datetime.now().year
    fig = go.Figure()
    years = df['date'].dt.year.unique()
    cmap = matplotlib.colormaps.get_cmap('plasma')

    for i, year in enumerate(years):
        color = matplotlib.colors.rgb2hex(cmap(i / len(years)))
        year_data = df[df['date'].dt.year == year]
        if year == current_year:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'],
                                     y=year_data['value'],
                                     mode='lines',
                                     name=str(year),
                                     line=dict(color='red', width=3),
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>' + yaxis_title + '</b>: %{y:.2f}' + y_hover_format + '<br>'
                                                                                               '<extra></extra>',
                                     customdata=year_data['date_formatted']))
        else:
            fig.add_trace(go.Scatter(x=year_data['day_of_year'],
                                     y=year_data['value'],
                                     mode='lines',
                                     name=str(year),
                                     line=dict(color=color),
                                     opacity=0.3,
                                     hovertemplate=
                                     '<b>Date</b>: %{customdata}<br>' +
                                     '<b>' + yaxis_title + '</b>: %{y:.2f}' + y_hover_format + '<br>'
                                                                                               '<extra></extra>',
                                     customdata=year_data['date_formatted']))

    fig.update_layout(title=title,
                      xaxis_title='Day of Year',
                      yaxis_title=yaxis_title,
                      legend={'traceorder': 'reversed'})
    return fig


def main():
    st.title('Climate Change Dashboard')
    st.write('AKA the "I\'m not a climate scientist but I play one on the internet" dashboard')
    st.write('AKA is it hot in here or is it just me? 🤔')

    sea_ice_df = get_antarctic_sea_ice_extent_data()
    sst_df = get_north_atlantic_sst_data()
    sea_ice_df = interpolate_missing_dates(sea_ice_df)
    sst_df = interpolate_missing_dates(sst_df)

    sie_fig = prepare_figure(sea_ice_df, 'Antarctic Sea Ice Extent', 'Ice Extent (10^6 sq km)', '')
    sst_fig = prepare_figure(sst_df, 'North Atlantic Sea Surface Temperature', 'Temperature [C]', ' C')

    st.header('North Atlantic Sea Surface Temperature')
    sst_tab1, sst_tab2 = st.tabs(["SST", "SST Anomaly"])
    with sst_tab1:
        st.plotly_chart(sst_fig, use_container_width=True)
    with sst_tab2:
        sst_min_year = int(sst_df['date'].dt.year.min())
        sst_max_year = int(sst_df['date'].dt.year.max())
        sst_avg_year_range_min, sst_avg_year_range_max = st.slider(
            'Select the year range to calculate the multi-year average for anomaly calculation baseline', sst_min_year,
            sst_max_year,
            (sst_min_year, sst_max_year - 1),
            key='sst_avg_year_range')
        sst_df_anomalies = calculate_anomalies(sst_df, sst_avg_year_range_min, sst_avg_year_range_max)
        sst_anomalies_fig = prepare_figure(sst_df_anomalies, 'North Atlantic Sea Surface Temperature Anomalies',
                                           'Temperature Anomaly [C]', ' C')
        st.plotly_chart(sst_anomalies_fig, use_container_width=True)

    st.header('Antarctic Sea Ice Extent')
    sie_tab1, sie_tab2 = st.tabs(["SIE", "SIE Anomaly"])
    with sie_tab1:
        st.plotly_chart(sie_fig, use_container_width=True)
    with sie_tab2:
        sie_min_year = int(sea_ice_df['date'].dt.year.min())
        sie_max_year = int(sea_ice_df['date'].dt.year.max())
        sie_avg_year_range_min, sie_avg_year_range_max = st.slider(
            'Select the year range to calculate the multi-year average for anomaly calculation baseline', sie_min_year, sie_max_year,
            (sie_min_year, sie_max_year - 1),
            key='sie_avg_year_range')
        sea_ice_df_anomalies = calculate_anomalies(sea_ice_df, sie_avg_year_range_min, sie_avg_year_range_max)
        sie_anomalies_fig = prepare_figure(sea_ice_df_anomalies, 'Antarctic Sea Ice Extent Anomalies',
                                           'Ice Extent Anomaly (10^6 sq km)', '')
        st.plotly_chart(sie_anomalies_fig, use_container_width=True)

    st.write('This dashboard was created to explore the effects of climate change on the North Atlantic and Antarctic regions.')
    st.write('The data used in this dashboard was obtained from the following sources:')
    st.write('Antarctic Sea Ice Extent: https://nsidc.org/data/g02135')
    st.write('North Atlantic Sea Surface Temperature: https://climatereanalyzer.org/clim/sst_daily')
    st.write('The code used to create this dashboard can be found at https://github.com/szulcmaciej/climate-dashboard')


if __name__ == '__main__':
    main()
