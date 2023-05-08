import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pandas.plotting import scatter_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


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


def data_analysis(df):
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


def DecisionTreeModel(X_train, Y_train, X_test, Y_test):
    # Define the parameter grid
    param_grid = {
        'criterion': ['gini'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2],
        'min_samples_leaf': [3],
        'max_features': ['sqrt']
    }

    # Create a decision tree classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Create a grid search object
    grid_search = GridSearchCV(dt_classifier, param_grid=param_grid, cv=5)

    # Fit the grid search object to the data
    grid_search.fit(X_train, Y_train)
    dt_classifier.fit(X_train, Y_train)
    # Print the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters for Decision Tree Classifier: ", best_params)
    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Evaluate the best model
    print("Decision Tree Classifier accuracy: ", accuracy_score(Y_test, y_pred))


def KNNModel(X_train, Y_train, X_test, Y_test):
    # Define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['distance'],
        'metric': ['euclidean']
    }

    # Create a K-Nearest Neighbors classifier
    KNN_classifier = KNeighborsClassifier()

    # Create a grid search object
    grid_search = GridSearchCV(KNN_classifier, param_grid=param_grid, cv=5)

    # Fit the grid search object to the data
    grid_search.fit(X_train, Y_train)
    KNN_classifier.fit(X_train, Y_train)
    # Print the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters for K-Nearest Neighbors Classifier: ", best_params)
    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Evaluate the best model
    print("K-Nearest Neighbors Classifier accuracy: ", accuracy_score(Y_test, y_pred))


def SVMModel(X_train, Y_train, X_test, Y_test):
    # Define the parameter grid
    param_grid = {
        'C': [1.0],
        'kernel': ['linear', 'poly', 'sigmoid'],
        'gamma': ['scale']
    }

    # Create a Support Vector Machines classifier
    SVM_classifier = SVC()

    # Create a grid search object
    grid_search = GridSearchCV(SVM_classifier, param_grid=param_grid, cv=5)

    # Fit the grid search object to the data
    grid_search.fit(X_train, Y_train)
    SVM_classifier.fit(X_train, Y_train)
    # Print the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters for Support Vector Machines Classifier: ", best_params)
    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    # Evaluate the best model
    print("Support Vector Machines Classifier accuracy: ", accuracy_score(Y_test, y_pred))


np.seterr(invalid='ignore')

FilePath = 'games-classification-dataset.csv'
df = pd.read_csv(FilePath)

df = ExtractFeature(df, 15)

X = df.drop(['Rate'], axis=1)
Y = df['Rate']
X = XPreprocessing(X)

languages_list = GenerateMultipleDataList(X['Languages'], ', ')
genres_list = GenerateMultipleDataList(X['Genres'], ', ')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=False)

X_train = Preprocessing(X_train)
X_test = Preprocessing(X_test)

lr = LogisticRegression(max_iter=100, solver="liblinear")
rfe = RFE(lr, n_features_to_select=10)
rfe.fit(X_train, Y_train)
top_features = X_train.columns[rfe.support_]

# evaluate Decision Tree Classifier
DecisionTreeModel(X_train[top_features], Y_train, X_test[top_features], Y_test)

# evaluate KNN Classifier
KNNModel(X_train[top_features], Y_train, X_test[top_features], Y_test)

# evaluate SVM Classifier
SVMModel(X_train[top_features], Y_train, X_test[top_features], Y_test)
