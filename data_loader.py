import pandas as pd
import streamlit as st


@st.cache_data
def get_north_atlantic_sst_data():
    df = pd.read_json('https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json')
    return df


@st.cache_data
def get_antarctic_sea_ice_extent_data():
    return pd.read_csv('https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/data/S_seaice_extent_daily_v3.0.csv',
                       skipinitialspace=True, skiprows=[1])
