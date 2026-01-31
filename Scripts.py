
# En este fichero ponemos en funciones las principales celdas que vimos en el notebook de MLFlow.
# Utilizaremos varias funciones en el fichero de FastAPI.

# Comenzamos con las librerías necesarias.

import pandas as pd
import numpy as np
from sklearn import preprocessing # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import matplotlib.pyplot as plt # type: ignore


import warnings
warnings.filterwarnings('ignore')


# La primera función lleva a cabo el preprocesado de un dataset independiente de airbnb.

def Preprocesado(df):
    fill_mode = ['Host Total Listings Count', 'Bathrooms', 'Bedrooms', 'Beds']

    fill_mean = ['Host Response Rate',
             'Cleaning Fee',
             'Review Scores Rating',
             'Review Scores Accuracy',
             'Review Scores Cleanliness',
             'Review Scores Checkin',
             'Review Scores Communication',
             'Review Scores Location',
             'Review Scores Value',
             'Reviews per Month']

    date_change = ['Host Since', 'First Review', 'Last Review']

    objects_null = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    categorical = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    minimo = 10

    cols_less_minimo = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                    'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    dict_more_minimo = {}

    for col in cols_less_minimo:
        s = df[col].value_counts()
        dict_more_minimo[col] = (s[s>minimo].index).to_list()


    def check_minimo (x, col):

        if (x != 'Desconocido') and (x not in dict_more_minimo[col]):
          x = 'Otros'

        return x

    Lat_avg = df['Latitude'].mean()
    Long_avg = df['Longitude'].mean()


    # Eliminación de outliers
    df = df.loc[df['Extra People'] < 50]
    df = df.loc[df['Minimum Nights'] < 100]
    df = df.loc[df['Maximum Nights'] < 1200]
    df = df.loc[df['Price'] < 400]

    # Eliminación de algunas columnas
    df = df.drop(['Host Listings Count', 'Weekly Price', 'Monthly Price', 'Security Deposit'], axis=1)

    # Imputación en columnas numéricas
    for col in fill_mode:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in fill_mean:
        df[col].fillna(df[col].mean(), inplace=True)

    # Transformación de columnas de fechas
    for col in date_change:
        df[col] = df[col].str[0:7]

    # Imputación en columnas string
    for col in objects_null:
        df[col].fillna('Desconocido', inplace=True)

    # Agrupación de registros poco comunes en columnas string
    for col in cols_less_minimo:
        df[col] = df[col].apply(check_minimo, args = [col])

    # Encoding columnas string
    mean_map = {}
    for c in categorical:
        mean = df.groupby(c)['Price'].mean()
        df[c] = df[c].map(mean)
        mean_map[c] = mean

    # Definición de nuevas columnas
    df['Distance_Center'] = ( (df['Latitude'] - Lat_avg)**2 + (df['Longitude'] - Long_avg)**2 ) ** (0.5)

    return df
    

# En esta función hacemos el preprocesado de los datasets de train y test, que no son independientes.

def Preprocesado_train_test(df, df_test):
    fill_mode = ['Host Total Listings Count', 'Bathrooms', 'Bedrooms', 'Beds']

    fill_mean = ['Host Response Rate',
             'Cleaning Fee',
             'Review Scores Rating',
             'Review Scores Accuracy',
             'Review Scores Cleanliness',
             'Review Scores Checkin',
             'Review Scores Communication',
             'Review Scores Location',
             'Review Scores Value',
             'Reviews per Month']

    date_change = ['Host Since', 'First Review', 'Last Review']

    objects_null = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    categorical = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    minimo = 10

    cols_less_minimo = ['Host Since', 'Host Response Time', 'District', 'Room Type',
                    'Amenities',
                'First Review',
                'Last Review',
                'Zipcode',
                'Neighbourhood',
                'Host Verifications',
                'Features',
                'Property Type',
                'Bed Type',
                'Calendar Updated',
                'Cancellation Policy']

    dict_more_minimo = {}

    for col in cols_less_minimo:
        s = df[col].value_counts()
        dict_more_minimo[col] = (s[s>minimo].index).to_list()


    def check_minimo (x, col):

        if (x != 'Desconocido') and (x not in dict_more_minimo[col]):
          x = 'Otros'

        return x

    Lat_avg = df['Latitude'].mean()
    Long_avg = df['Longitude'].mean()

    # airbnb_train = pd.read_csv('airbnb_train.csv', sep=';', decimal='.')

    # Eliminación de outliers
    df = df.loc[df['Extra People'] < 50]
    df = df.loc[df['Minimum Nights'] < 100]
    df = df.loc[df['Maximum Nights'] < 1200]
    df = df.loc[df['Price'] < 400]

    # Eliminación de algunas columnas
    df = df.drop(['Host Listings Count', 'Weekly Price', 'Monthly Price', 'Security Deposit'], axis=1)

    # Imputación en columnas numéricas
    for col in fill_mode:
        df[col].fillna(df[col].mode()[0], inplace=True)

    for col in fill_mean:
        df[col].fillna(df[col].mean(), inplace=True)

    # Transformación de columnas de fechas
    for col in date_change:
        df[col] = df[col].str[0:7]

    # Imputación en columnas string
    for col in objects_null:
        df[col].fillna('Desconocido', inplace=True)

    # Agrupación de registros poco comunes en columnas string
    for col in cols_less_minimo:
        df[col] = df[col].apply(check_minimo, args = [col])

    # Encoding columnas string
    mean_map = {}
    for c in categorical:
        mean = df.groupby(c)['Price'].mean()
        df[c] = df[c].map(mean)
        mean_map[c] = mean

    # Definición de nuevas columnas
    df['Distance_Center'] = ( (df['Latitude'] - Lat_avg)**2 + (df['Longitude'] - Long_avg)**2 ) ** (0.5)


    df_test = df_test.loc[df_test['Extra People'] < 50]
    df_test = df_test.loc[df_test['Minimum Nights'] < 100]
    df_test = df_test.loc[df_test['Maximum Nights'] < 1200]
    df_test = df_test.loc[df_test['Price'] < 400]

    df_test = df_test.drop(['Host Listings Count', 'Weekly Price', 'Monthly Price', 'Security Deposit'], axis=1)


    for col in fill_mode:
        df_test[col].fillna(df[col].mode()[0], inplace=True)

    for col in fill_mean:
        df_test[col].fillna(df[col].mean(), inplace=True)

    for col in date_change:
        df_test[col] = df_test[col].str[0:7]

    for col in objects_null:
        df_test[col].fillna('Desconocido', inplace=True)

    for col in cols_less_minimo:
        df_test[col] = df_test[col].apply(check_minimo, args = [col])

    for c in categorical:
        df_test[c] = df_test[c].map(mean_map[c])

    df_test['Distance_Center'] = ( (df_test['Latitude'] - Lat_avg)**2 + (df_test['Longitude'] - Long_avg)**2 ) ** (0.5)

    return df, df_test


# En esta función extraemos las columnas target (Price) y hacemos la normalización

def Normalizacion(df_train, df_test):

    X_train = df_train.drop(['Price'], axis=1).values
    y_train = df_train['Price'].values

    X_test = df_test.drop(['Price'], axis=1).values
    y_test = df_test['Price'].values

    scaler = preprocessing.StandardScaler().fit(X_train)
    XtrainScaled = scaler.transform(X_train)

    XtestScaled = scaler.transform(X_test)

    return XtrainScaled, y_train, XtestScaled, y_test


# En esta función determinamos el algoritmo de Random Forest para una profundidas y estimadores dados.
# La función devuelve las métricas de R2 y RMSE tanto para train como para test.

def random_forest_regressor(depth, estimators, XtrainScaled, y_train, XtestScaled, y_test):

    randomForest = RandomForestRegressor(max_depth = depth, n_estimators = estimators, max_features='sqrt').fit(XtrainScaled, y_train)
    
    r2_train = round(randomForest.score(XtrainScaled, y_train), 3)
    r2_test = round(randomForest.score(XtestScaled, y_test), 3)

    ytrainRF = randomForest.predict(XtrainScaled)
    ytestRF  = randomForest.predict(XtestScaled)
    
    mseTrainModelRF = mean_squared_error(y_train, ytrainRF)
    mseTestModelRF = mean_squared_error(y_test, ytestRF)

    rmse_train = round(mseTrainModelRF ** 0.5, 1)
    rmse_test = round(mseTestModelRF ** 0.5, 1)

    return r2_train, r2_test, rmse_train, rmse_test


# En esta función hacemos un GridSerach del algoritmo de Random Forest, en función de un rango de profundidades y estimadores.

def grid_search_RF(X_train, y_train, depth_min, depth_max, estimators, cv):

    maxDepth = range(depth_min, depth_max + 1)
    tuned_parameters = {'max_depth': maxDepth}

    grid = GridSearchCV(RandomForestRegressor(random_state = 0, n_estimators = estimators, max_features = 'sqrt'), param_grid = tuned_parameters,cv = cv, verbose=2)
    grid.fit(X_train, y_train)

    print("best mean cross-validation score: {:.3f}".format(grid.best_score_))
    print("best parameters: {}".format(grid.best_params_))

    scores = np.array(grid.cv_results_['mean_test_score'])
    plt.plot(maxDepth,scores,'-o')
    plt.xlabel('max_depth')
    plt.ylabel('5-fold R2')

    plt.show()