import streamlit as st
import pandas as pd
import joblib

st.title("job change prediction")
df = pd.read_csv('/content/train.csv')
#input fields
#st selectbox
#st number input

city = st.selectbox('city',pd.unique(df['city']))
city_development_index = st.number_input('city_development_index')

gender = st.selectbox('Gender',df['gender'].unique())
relevent_experience = st.selectbox('Relevent Experience',df['relevent_experience'].unique())
enrolled_university = st.selectbox('Enrolled University',df['enrolled_university'].unique())
education_level = st.selectbox('Education Level',df['education_level'].unique())
major_discipline = st.selectbox('Major Discipline',df['major_discipline'].unique())
experience = st.selectbox('Experience',df['experience'].unique())
company_size = st.selectbox('Company Size',df['company_size'].unique())
company_type = st.selectbox('Company Type',df['company_type'].unique())
last_new_job = st.selectbox('Last New Job',df['last_new_job'].unique())
training_hours = st.number_input('Training Hours')

inputs = {
    
          "city":city,
          "city_development_index":city_development_index,
          "gender":gender,
          "relevent_experience":relevent_experience,
          "enrolled_university":enrolled_university,
          "education_level":education_level,
          "major_discipline":major_discipline,
          "experience":experience,
          "company_size":company_size,
          "company_type":company_type,
          "last_new_job":last_new_job,
          "training_hours":training_hours
}

if st.button('predict'):
  model = joblib.load('jobchange_pipeline_model.pkl')
  X_input = pd.DataFrame([inputs])
  prediction = model.predict(X_input)
  st.write(prediction)
