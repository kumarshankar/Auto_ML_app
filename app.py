import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import load_diabetes, load_boston
import streamlit as st 
from model import build_model

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Regression ML App', layout='wide')


#---------------------------------#
st.write("""
# The Random Forest Regressor Learning App
In this implementation, the ML model is trained using the user adjusted hyperparameters and user input dataset.
""")

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    st.sidebar.markdown("""[Example CSV input file](https://raw.githubusercontent.com/kumarshankar/Auto_ML_app/main/sample.csv)""")

params = dict()
# Sidebar - Specify parameter settings
with st.sidebar.header('2. Set Parameters'):
    params['split_size'] = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

with st.sidebar.subheader('Learning Parameters'):
    params['parameter_n_estimators'] = st.sidebar.slider('Number of estimators (n_estimators) in forest', 0, 1000, 100, 100)
    params['parameter_max_features'] = st.sidebar.select_slider('Max features (max_features)', options=['auto', 'sqrt', 'log2'])
    params['parameter_min_samples_split'] = st.sidebar.slider('Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
    params['parameter_min_samples_leaf'] = st.sidebar.slider('Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

with st.sidebar.subheader('General Parameters'):
    params['parameter_random_state'] = st.sidebar.slider('Seed number (random_state)', 0, 1000, 42, 1)
    params['parameter_criterion'] = st.sidebar.select_slider('Performance measure (criterion)', options=['mse', 'mae'])
    params['parameter_bootstrap'] = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)', options=[True, False])
    params['parameter_oob_score'] = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
    params['parameter_n_jobs'] = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])


def show_data_params(model_dict, df):
    st.subheader('1 Data')
    st.markdown('**1 a) Dataset top 10 rows**')
    st.write(df.head(10))
    st.markdown('**1 b) Data split to train and validation set**')
    st.write('Training set data shape')
    st.info(model_dict['data_params']['train_shape'])
    st.write('Test set data shape')
    st.info(model_dict['data_params']['test_shape'])
    st.markdown('**1 c) Variable details**:')
    st.write('X variable')
    st.info(model_dict['data_params']['X variable'])
    st.write('Y variable')
    st.info(model_dict['data_params']['Y variable'])


def show_model_metrics(model_dict):
    st.subheader('2. Model Performance')
    st.markdown('**2 a) Training set metrics**')    
    
    st.write('Coefficient of determination ($R^2$):')
    st.info(model_dict['metrics']['train_R2'])
    
    st.write('Mean Squared Error or Mean Absolute Error (based on param selected in sidebar):')
    st.info(model_dict['metrics']['train_MSE'])

    st.markdown('**2 b) Test set metrics**')
    st.write('Coefficient of determination ($R^2$):')
    st.info(model_dict['metrics']['test_R2'])

    st.write('Mean Squared Error or Mean Absolute Error (based on param selected in sidebar):')
    st.info(model_dict['metrics']['train_MSE'])

    st.subheader('3. Model Parameters')
    st.write(model_dict['model'].get_params())



if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    model_dict = build_model(df, params)
    show_data_params(model_dict, df)
    show_model_metrics(model_dict)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Example Dataset'):

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat([X,Y], axis=1 )
        model_dict = build_model(df, params)
        show_data_params(model_dict, df)
        show_model_metrics(model_dict)