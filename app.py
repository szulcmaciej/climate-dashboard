import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

import plotly.graph_objects as go
import matplotlib


@st.cache_data(ttl=timedelta(hours=1))
def get_antarctic_sea_ice_extent_data():
    url = 'https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv'
    df = pd.read_csv(url, skipinitialspace=True, skiprows=[1])
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['value'] = df['Extent']
    df['date_formatted'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df[['date', 'day_of_year', 'value', 'date_formatted']]
    return df


@st.cache_data(ttl=timedelta(hours=1))
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


class DataSource:
    def __init__(self, url, title, y_axis_title, y_axis_unit):
        self.url = url
        self.title = title
        self.y_axis_title = y_axis_title
        self.y_axis_unit = y_axis_unit
        self.df = None
        self.fetch_data()
        self.interpolate_missing_dates()
        self.generate_layout()

    def fetch_data(self):
        raise NotImplementedError

    def interpolate_missing_dates(self):
        min_year = self.df['date'].dt.year.min()
        max_year = self.df['date'].dt.year.max()
        full_date_range = pd.date_range(start=f'{min_year}-01-01', end=f'{max_year}-12-31')
        df_full_range = pd.DataFrame(full_date_range, columns=['date'])
        self.df = pd.merge(df_full_range, self.df, on='date', how='left')
        self.df['day_of_year'] = self.df['date'].dt.dayofyear
        self.df['value'] = self.df['value'].interpolate(method='linear', limit_area='inside')
        self.df['date_formatted'] = self.df['date'].dt.strftime('%Y-%m-%d')

    def calculate_anomalies(self, start_year, end_year):
        df_range = self.df[(self.df['date'].dt.year >= start_year) & (self.df['date'].dt.year <= end_year)]
        averages = df_range.groupby(df_range['date'].dt.dayofyear)['value'].mean()
        self.df['value'] = self.df.apply(
            lambda row: row['value'] - averages[row['day_of_year']] if row['day_of_year'] in averages.index else np.NaN,
            axis=1)

    def prepare_figure(self, title, yaxis_title):
        current_year = datetime.now().year
        fig = go.Figure()
        years = self.df['date'].dt.year.unique()
        cmap = matplotlib.cm.get_cmap('plasma')

        for i, year in enumerate(years):
            color = matplotlib.colors.rgb2hex(cmap(i / len(years)))
            year_data = self.df[self.df['date'].dt.year == year]
            if year == current_year:
                fig.add_trace(go.Scatter(x=year_data['day_of_year'],
                                         y=year_data['value'],
                                         mode='lines',
                                         name=str(year),
                                         line=dict(color='red', width=3),
                                         hovertemplate=
                                         '<b>Date</b>: %{customdata}<br>' +
                                         '<b>' + yaxis_title + '</b>: %{y:.2f}' + '<br>' +
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
                                         '<b>' + yaxis_title + '</b>: %{y:.2f}' + '<br>' +
                                         '<extra></extra>',
                                         customdata=year_data['date_formatted']))

        fig.update_layout(title=title,
                          xaxis_title='Day of Year',
                          yaxis_title=yaxis_title,
                          legend={'traceorder': 'reversed'})
        return fig

    def generate_layout(self):
        st.header(self.title)
        tab1, tab2 = st.tabs([self.title, f"{self.title} Anomaly"])
        with tab1:
            full_y_axis_title = f'{self.y_axis_title} ({self.y_axis_unit})'
            fig = self.prepare_figure(self.title, full_y_axis_title)
            st.plotly_chart(fig, use_container_width=True)
        with tab2:
            min_year = int(self.df['date'].dt.year.min())
            max_year = int(self.df['date'].dt.year.max())
            avg_year_range_min, avg_year_range_max = st.slider(
                'Select the year range to calculate the multi-year baseline for anomaly calculation', min_year,
                max_year,
                (1991, 2020),
                key=f'{self.title}_avg_year_range')
            self.calculate_anomalies(avg_year_range_min, avg_year_range_max)
            full_y_axis_title = f'{self.y_axis_title} Anomaly ({self.y_axis_unit})'
            anomalies_fig = self.prepare_figure(f'{self.title} Anomalies', full_y_axis_title)
            st.plotly_chart(anomalies_fig, use_container_width=True)


class AntarcticSeaIceExtent(DataSource):
    def __init__(self):
        super().__init__('https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv',
                         'Antarctic Sea Ice Extent',
                         'Antarctic SIE', 'million square kilometers')

    def fetch_data(self):
        self.df = get_antarctic_sea_ice_extent_data()

    def generate_layout(self):
        super().generate_layout()
        with st.expander('Data description'):
            st.write('This data represents the daily sea ice extent in the Antarctic region. '
                     'The data is obtained from the National Snow and Ice Data Center (NSIDC).')
            st.write('The data used in this section was obtained from the following source:')
            st.write('https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv')
            st.write("Here's more info about the data: https://nsidc.org/data/g02135")


class NorthAtlanticSST(DataSource):
    def __init__(self):
        super().__init__('https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json',
                         'North Atlantic Sea Surface Temperature',
                         'North Atlantic SST', '°C')

    def fetch_data(self):
        self.df = get_north_atlantic_sst_data()

    def generate_layout(self):
        super().generate_layout()
        with st.expander('Data description'):
            st.write('The data used in this section was obtained from the following source:')
            st.write('https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json')
            st.write("Here's more info about the data: https://climatereanalyzer.org/clim/sst_daily")


def main():
    st.set_page_config(page_title='Toasty Times', page_icon=':fire:')
    st.title('🔥 Toasty Times 🔥')
    st.write('AKA the "I\'m not a climate scientist but I play one on the internet" dashboard')

    NorthAtlanticSST()
    AntarcticSeaIceExtent()

    st.header('About')
    st.write('Thanks for checking this out!')
    st.write(
        'This dashboard was heavily inspired by the charts posted on Twitter by prof. Eliot Jacobson [@EliotJacobson](https://twitter.com/EliotJacobson).')
    st.write(
        'I just wanted to learn how to use Streamlit and it seemed like a fun project to try to recreate some of the plots from his profile.')
    st.write('The code used to create this dashboard can be found at https://github.com/szulcmaciej/climate-dashboard')
    st.write(
        'If you have any suggestions or feedback, please create an issue on GitHub or reach out to me on Twitter [@nerdy_surfer](https://twitter.com/nerdy_surfer)')
    st.image(
        'https://static01.nyt.com/images/2016/08/05/us/05onfire1_xp/05onfire1_xp-superJumbo-v2.jpg?quality=75&auto=webp',
        caption='KC Green')


if __name__ == '__main__':
    main()
