import pandas as pd
import numpy as np
import streamlit as st
the_usual_suspects= (TabError,KeyError,AttributeError,UnboundLocalError,TypeError,ValueError,ZeroDivisionError)
from pandas.api.types import (is_categorical_dtype,is_datetime64_any_dtype,is_numeric_dtype,is_object_dtype,)



def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            else:
                user_text_input = right.text_input(
                    f"Search in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input.lower())]

    return df
def string_to_list(input_string):
    # Split the string by commas and convert each element to an integer
    index_list = [int(num) if num != '7a' else str(num) for num in input_string.split('/') ]
    return index_list

#@st.cache_data
def cache_pumpstroubles_table():
    url = 'pump_trouble.xlsx'
    
    return pd.read_excel(url,'symptoms',index_col=[0]),pd.read_excel(url,'causes',index_col=[0]),pd.read_excel(url,'all_pumps',usecols=[0,1,2,3])

centri_pump_symptoms_df, centri_pump_causes_df, all_pumps_df = cache_pumpstroubles_table()
df_options = st.selectbox('Select your table',['Pump Clinic - Centrifugal pumps (extensive possible causes)','Pump Clinic - Centrifugal pumps',
                                                      'Pump Clinic - Reciprocating pumps','Pump Clinic - Reciprocating (Piston) pumps'], key = 'dframes_options2dd')

if df_options == 'Pump Clinic - Centrifugal pumps (extensive possible causes)':
    df_options = st.multiselect('Select your pump symptoms!',centri_pump_symptoms_df['Symptoms'], key = 'df_1option_options2dd')
    try:
        mask_1 = centri_pump_symptoms_df['Symptoms'].apply(lambda x: x in df_options)
        symptoms_df = centri_pump_symptoms_df[mask_1].reset_index()
        index_groups = []
        
        for i in symptoms_df['Possible Causes']:
            index_groups.append(string_to_list(i))
        #st.write(index_groups[0][0])
        #st.write(centri_pump_causes_df.loc[['7a']])
        df_lists=[]
        for i,j in zip(index_groups,df_options):
            df_i = centri_pump_causes_df.loc[i]
            df_i['Symptoms'] = j 
            cols = df_i.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df_i = df_i[cols]
            df_lists.append(df_i)
            
        concated_dfs = pd.concat(df_lists)
        st.dataframe(filter_dataframe(concated_dfs))
    except the_usual_suspects: pass
if df_options == 'Pump Clinic - Centrifugal pumps':
    mask = (all_pumps_df['pump type'] == 'Centrifugal Pump')
    df = all_pumps_df[mask]
    st.dataframe(filter_dataframe(df))
if df_options == 'Pump Clinic - Reciprocating pumps':
    mask = (all_pumps_df['pump type'] == 'Reciprocating Pump')
    df = all_pumps_df[mask]
    st.dataframe(filter_dataframe(df))
if df_options == 'Pump Clinic - Reciprocating (Piston) pumps':
    mask = (all_pumps_df['pump type'] == 'Reciprocating (Piston) pump')
    df = all_pumps_df[mask]
    st.dataframe(filter_dataframe(df))
    #except TabError: pass
#st.write(centri_pump_symptoms_df, centri_pump_causes_df, all_pumps_df)
