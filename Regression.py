import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pandas.plotting import scatter_matrix
from sklearn import linear_model, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split



def FeatureEncoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def XPreprocessing(X):
    # removing + symbol in Age Rating column
    X['Age Rating'] = X['Age Rating'].map(lambda x: x.rstrip('+').lstrip(' '))

    # split month column from date
    X['Original Release Date Month'] = pd.to_datetime(X['Original Release Date'], format='%d/%m/%Y').dt.month
    X['Current Version Release Date Month'] = pd.to_datetime(X['Current Version Release Date'],
                                                             format='%d/%m/%Y').dt.month

    # split year column from date
    X['Original Release Date Year'] = pd.to_datetime(X['Original Release Date'], format='%d/%m/%Y').dt.year
    X['Current Version Release Date Year'] = pd.to_datetime(X['Current Version Release Date'],
                                                            format='%d/%m/%Y').dt.year

    # split day column from date
    X['Original Release Date Day'] = pd.to_datetime(X['Original Release Date'], format='%d/%m/%Y').dt.day
    X['Current Version Release Date Day'] = pd.to_datetime(X['Current Version Release Date'], format='%d/%m/%Y').dt.day

    # state reason for each column
    X = X.drop(
        ['URL', 'Name', 'ID', 'Subtitle', 'Icon URL', 'In-app Purchases', 'Description', 'Developer', 'Primary Genre',
         'Original Release Date', 'Current Version Release Date', 'com'], axis=1)

    return X


def Preprocessing(X):
    # encoding
    X = SplitMultipleData(X, 'Languages', ', ', languages_list)
    X = SplitMultipleData(X, 'Genres', ', ', genres_list)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])

    # scaling
    Scaled_X = FeatureScaling(X, -1, 1)
    X = pd.DataFrame.from_records(Scaled_X, columns=X.columns)

    # filling columns as the have the NaN values in every row
    X = X.fillna(0)
    return X


def Correlation(corrData):
    corr = corrData.corr()
    top_feature = corr.index[abs(corr['Average User Rating']) >= 0.05]
    plt.subplots(figsize=(12, 8))
    top_corr = corrData[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    top_feature = top_feature.delete(-1)
    top_feature_Data = corrData[top_feature]
    return top_feature


def FeatureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def SplitMultipleData(df, column, spliter, splitted_list):
    # create a new DataFrame with columns for each language
    df_encoded = pd.DataFrame(columns=splitted_list)

    # encode each row
    for i, row in df.iterrows():
        languages = row[column]
        if pd.isna(languages):
            encoding = [0] * len(splitted_list)
        else:
            languages = languages.split(spliter)
            encoding = [1 if language in languages else 0 for language in splitted_list]
        df_encoded.loc[i] = encoding

    # merge the original dataframe with the encoded dataframe
    df_final = pd.concat([df, df_encoded], axis=1)

    # drop the original 'Languages' column
    df_final.drop(columns=[column], inplace=True)

    return df_final


def GenerateMultipleDataList(df_column, splitter):
    # get the top languages
    splitted_list = df_column.str.split(splitter, expand=True).stack().value_counts().head(
        100).index.tolist()
    return splitted_list


def LinearRegressionModel(X_train, Y_train, X_test, Y_test, first_column_name):
    model = linear_model.LinearRegression()
    X_train = np.expand_dims(X_train, axis=1)
    Y_train = np.expand_dims(Y_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    Y_test = np.expand_dims(Y_test, axis=1)

    model.fit(X_train, Y_train)  # Fit method is used for fitting your training data into the model
    Y_Predict_test = model.predict(X_test)
    Y_Predict_train = model.predict(X_train)

    plt.scatter(X_test, Y_test)
    plt.xlabel(first_column_name, fontsize=20)
    plt.ylabel('Average User Rating', fontsize=20)
    plt.plot(X_test, Y_Predict_test, color='red', linewidth=3)
    plt.title('Linear Regression')
    plt.show()

    print('Mean Square Error For Linear Regression : ', metrics.mean_squared_error(Y_test, Y_Predict_test))
    print('R2 score For Linear Regression  : ', r2_score(Y_test, Y_Predict_test))
    return


def PolynomialRegressionModel(X_train, Y_train, X_test, Y_test, degree, columns_list):
    poly_features = PolynomialFeatures(degree=degree)
    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)

    # fit the transformed features to Linear Regression
    poly_model = linear_model.LinearRegression()
    poly_model.fit(X_train_poly, Y_train)

    # predicting on training data-set
    ypred_train = poly_model.predict(poly_features.transform(X_train))

    # predicting on test data-set
    ypred_test = poly_model.predict(poly_features.transform(X_test))

    # calculate and print metrics
    mse = metrics.mean_squared_error(Y_test, ypred_test)
    r2 = r2_score(Y_test, ypred_test)
    print('Mean Square Error For Polynomial Regression : ', mse)
    print('R2 score For Polynomial Regression : ', r2)

    # plot the regression line
    for col in columns_list:
        plt.scatter(X_train[col], Y_train, alpha=0.5)
        plt.plot(X_train[col], ypred_train, linestyle='', marker='.', lw=0.1)

    plt.title('Polynomial Regression')
    plt.xlabel('Features')
    plt.ylabel('Average Rate')
    plt.show()

    return


def LassoRegressionModel(Alpha, X_train, X_test, Y_train, Y_test, feature_names):
    # create a Lasso model
    lasso = Lasso(alpha=Alpha)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    # fit the model to the data
    lasso.fit(X_train, Y_train)
    y_pred = lasso.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, y_pred)
    r2 = metrics.r2_score(Y_test, y_pred)
    print('Mean Square Error For Lasso Regression : ', mse)
    print('R2 score For Lasso Regression : ', r2)

    # plot predicted vs true values for multiple features
    fig, axs = plt.subplots(1, len(feature_names), figsize=(15, 10))
    plt.title('Lasso Regression')

    for i, ax in enumerate(axs.flat):
        ax.scatter(X_test[:, i], Y_test, color='blue', label='True values')
        ax.scatter(X_test[:, i], y_pred, color='red', label='Predicted values')
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel('Target')
        ax.legend()

    plt.show()


def DataAnalysis(df):
    print(df.head())

    # Display number of rows and columns
    print('The dataset has {} rows and {} columns.'.format(df.shape[0], df.shape[1]))

    # Display the mean value of predicted column
    avg_rating = df['Average User Rating'].mean()
    print("Average user rating:", avg_rating)

    # Display the most common genre
    most_common_genre = df['Primary Genre'].mode()[0]
    print("Most common primary genre:", most_common_genre)

    # sum of Missing values in Dataset
    print(df.info())
    print("Sum of Null Values: \n", df.isnull().sum())

    # Visualize most popular gaming genre
    print(df['Genres'].value_counts().head())
    plt.figure(figsize=(10, 7))
    df.Genres.value_counts().head().sort_values().plot(kind='barh', color=list('rgbkymc'))

    # Visualize most Top 5 Developer
    plt.figure(figsize=(15, 7))
    df.Developer.value_counts().iloc[:5].plot(kind='pie', ylabel='')
    plt.title("Top 5 Game Developer", fontsize=(20))
    plt.show()

    # Visualize how features relate and affect each other
    df = df.drop(['ID'], axis=1)
    scatter_matrix(df, figsize=(10, 10))
    plt.show()
    return


def ExtractFeature(df, no_max_features):
    # Create a TF-IDF vectorized
    vectorized = TfidfVectorizer(stop_words="english", max_features=no_max_features)

    # Fit the vectorized to the "description" column
    X_text = vectorized.fit_transform(df["Description"])
    X_text = X_text.toarray()

    # Create a new DataFrame to represent the extracted features
    df_features = pd.DataFrame(X_text, columns=vectorized.get_feature_names_out())

    # Concatenate the original dataset with the new DataFrame
    df = pd.concat([df_features, df], axis=1)
    return df


np.seterr(invalid='ignore')
FilePath = 'games-regression-dataset.csv'
df = pd.read_csv(FilePath)

# data_analysis(df)
df = ExtractFeature(df, 15)

X = df.drop(['Average User Rating'], axis=1)
Y = df['Average User Rating']
X = XPreprocessing(X)

languages_list = GenerateMultipleDataList(X['Languages'], ', ')
genres_list = GenerateMultipleDataList(X['Genres'], ', ')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

X_train = Preprocessing(X_train)
X_test = Preprocessing(X_test)

correlationDataSet = X_train.copy()
correlationDataSet['Average User Rating'] = Y_train
top_features = Correlation(correlationDataSet)

X_train = X_train[top_features]
X_test = X_test[top_features]

LinearRegressionModel(X_train['Current Version Release Date Year'], Y_train,
                      X_test['Current Version Release Date Year'], Y_test, 'u2022')

columns_list = ['Original Release Date Year', 'Current Version Release Date Year', 'u2022']
X_train = X_train[columns_list]
X_test = X_test[columns_list]
PolynomialRegressionModel(X_train.iloc[:, :], Y_train, X_test.iloc[:, :], Y_test, 3, columns_list)

LassoRegressionModel(0.001, X_train, X_test, Y_train, Y_test, columns_list)

X.to_csv(r'newGameDataset.csv', index=False)