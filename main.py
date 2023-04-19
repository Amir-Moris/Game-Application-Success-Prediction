from collections import Counter

import PolynomialRegression as PolynomialRegression
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def cleanData(df):
    df.dropna(how='any', inplace=True)


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def preProcessing(X):
    # removing + symbol in Age Rating column
    X['Age Rating'] = X['Age Rating'].map(lambda x: x.rstrip('+').lstrip(' '))

    # split month column from date
    X['Original Release Date Month'] = pd.to_datetime(X['Original Release Date']).dt.month
    X['Current Version Release Date Month'] = pd.to_datetime(X['Current Version Release Date']).dt.month

    # split year column from date
    X['Original Release Date Year'] = pd.to_datetime(X['Original Release Date']).dt.year
    X['Current Version Release Date Year'] = pd.to_datetime(X['Current Version Release Date']).dt.year

    # split day column from date
    X['Original Release Date Day'] = pd.to_datetime(X['Original Release Date']).dt.day
    X['Current Version Release Date Day'] = pd.to_datetime(X['Current Version Release Date']).dt.day

    # state reason for each column
    X = X.drop(
        ['URL', 'Name', 'ID', 'Subtitle', 'Icon URL', 'In-app Purchases', 'Description', 'Developer', 'Primary Genre',
         'Original Release Date', 'Current Version Release Date'], axis=1)

    # encoding
    X = splitMultipleData(X, 'Languages', ', ', 100)
    X = splitMultipleData(X, 'Genres', ', ', 100)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])

    # scaling
    Scaled_X = featureScaling(X, -1, 1)
    X = pd.DataFrame.from_records(Scaled_X, columns=X.columns)

    # dropping these 2 columns as the have the same values in every row
    X = X.drop(['Games', 'Strategy'], axis=1)
    return X


def correlation(corrData):
    corr = corrData.corr()
    top_feature = corr.index[abs(corr['Average User Rating']) >= 0.05]

    plt.subplots(figsize=(12, 8))
    top_corr = corrData[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    top_feature_Data = corrData[top_feature]
    return top_feature


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def splitMultipleData(df, column, spliter, numberOfColumns):
    # get the top languages
    top_languages = df[column].str.split(spliter, expand=True).stack().value_counts().head(
        numberOfColumns).index.tolist()

    # create a new DataFrame with columns for each language
    df_encoded = pd.DataFrame(columns=top_languages)

    # encode each row
    for i, row in df.iterrows():
        languages = row[column]
        if pd.isna(languages):
            encoding = [0] * len(top_languages)
        else:
            languages = languages.split(spliter)
            encoding = [1 if language in languages else 0 for language in top_languages]
        df_encoded.loc[i] = encoding

    # merge the original dataframe with the encoded dataframe
    df_final = pd.concat([df, df_encoded], axis=1)

    # drop the original 'Languages' column
    df_final.drop(columns=[column], inplace=True)

    return df_final


def linearReggressionModel(X_train, Y_train, X_test, Y_test):
    model = linear_model.LinearRegression()
    X_train = np.expand_dims(X_train, axis=1)
    Y_train = np.expand_dims(Y_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    Y_test = np.expand_dims(Y_test, axis=1)
    model.fit(X_train, Y_train)  # Fit method is used for fitting your training data into the model
    Y_Predict = model.predict(X_test)
    print('Mean Square Error', metrics.mean_squared_error(Y_test, Y_Predict))
    print('r2_score', r2_score(Y_test, Y_Predict))
    return


def polynomialRegression(X_train, Y_train, X_test, Y_test, degree):
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    # predicting on training data-set
    ypred = poly_model.predict(poly_features.transform(X_test))

    # predicting on test data-set
    print('Mean Square Error', metrics.mean_squared_error(Y_test, ypred))
    print('r2_score', r2_score(Y_test, ypred))
    return


np.seterr(invalid='ignore')
FilePath = 'games-regression-dataset.csv'
df = pd.read_csv(FilePath)
X = df.iloc[:, :-1]
Y = df['Average User Rating']

X = preProcessing(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

correlationDataSet = X_train.copy()
correlationDataSet['Average User Rating'] = Y_train
top_features = correlation(correlationDataSet)

new_x_train = X_train[top_features]
new_x_test = X_test[top_features]
first_column_name = new_x_train.columns[0]

linearReggressionModel(new_x_train[first_column_name], Y_train, new_x_test[first_column_name], Y_test)
polynomialRegression(new_x_train.iloc[:, 0:2], Y_train, new_x_test.iloc[:, 0:2], Y_test, 3)

X.to_csv(r'newGameDataset.csv', index=False)

