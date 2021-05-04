from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#---------------------------------#
# Model Building builds only 
def build_model(df, params):
    X = df.iloc[:,:-1] # Use all columns except for the last column
    Y = df.iloc[:,-1] #Use last column as the response variable
    # train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-params['split_size'])/100) 

    model = get_rf_regressor_model(params)
    model.fit(X_train, Y_train)
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
  
    # return dict of model & data params
    return {'model': model,
            'data_params': {'train_shape': X_train.shape,
                       'test_shape': X_test.shape,
                       'X variable': list(X.columns),
                       'Y variable': Y.name},
            'metrics': {'train_R2': r2_score(Y_train, Y_pred_train),
                        'train_MSE': mean_squared_error(Y_train, Y_pred_train),
                        'test_R2': r2_score(Y_test, Y_pred_test),
                        'test_MSE': mean_squared_error(Y_test, Y_pred_test)}}


def get_rf_regressor_model(params):
    rf = RandomForestRegressor(n_estimators=params['parameter_n_estimators'],
        random_state=params['parameter_random_state'],
        max_features=params['parameter_max_features'],
        criterion=params['parameter_criterion'],
        min_samples_split=params['parameter_min_samples_split'],
        min_samples_leaf=params['parameter_min_samples_leaf'],
        bootstrap=params['parameter_bootstrap'],
        oob_score=params['parameter_oob_score'],
        n_jobs=params['parameter_n_jobs'])
    return rf

