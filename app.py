
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('../models/rf_model.pkl')
features = joblib.load('../models/features.pkl')
dummies = joblib.load('../models/dummies_dict.pkl')

def prediction(online_order,book_table,location,rest_type,cuisines,approx_cost_for_two_people,listed_in_type,listed_in_city):
    df = pd.DataFrame(columns=features)
    df.at[0,'online_order'] = online_order
    df.at[0,'book_table'] = book_table
    df.at[0,'location'] = location
    df.at[0,'rest_type'] = rest_type
    df.at[0,'cuisines'] = cuisines
    df.at[0,'approx_cost_for_two_people'] = approx_cost_for_two_people
    df.at[0,'listed_in_type'] = listed_in_type
    df.at[0,'listed_in_city'] = listed_in_city
    result = model.predict(df)
    return result[0]

def main():
    st.title("Bangalore New Restaurants expectation if it is going to be good or not")
    online_order = st.selectbox("Online order (if avilable selct 1, if not 0)",[0,1])
    book_table = st.selectbox("book table (if avilable selct 1, if not 0)",[0,1])
    location = st.selectbox("Select Location", dummies['location'])
    rest_type = st.selectbox("Select rest_type", dummies['rest_type'])
    cuisines = st.selectbox("Select cuisines", dummies['cuisines'])
    approx_cost_for_two_people = st.slider("Select approx_cost_for_two_people", min_value = 0 , max_value =7000, value = 500, step=1)
    listed_in_type = st.selectbox("Select listed_in_type", dummies['listed_in_type'])
    listed_in_city = st.selectbox("Select listed_in_city", dummies['listed_in_city'])
    
    if st.button("Predict"):
        result = prediction(online_order,book_table,location,rest_type,cuisines,approx_cost_for_two_people,listed_in_type,listed_in_city)
        result_list = ["Restaurant may success Baad","Restaurant will success Good"]
        st.text(result_list[result])
        
        
main()
