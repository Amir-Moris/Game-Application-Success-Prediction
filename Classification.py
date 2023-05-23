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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm

import seaborn as sns
from matplotlib.colors import ListedColormap
import random
import time
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance


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


def generate_random_color():
    # Generate random values for the RGB components
    red = random.uniform(0, 1)
    green = random.uniform(0, 1)
    blue = random.uniform(0, 1)

    # Return the RGB values as a tuple
    return (red, green, blue)


def BarCharForEachModel(importances, X, title):
    print()
    # Create bar chart
    plt.bar(range(X.shape[1]), importances)
    plt.xticks(range(X.shape[1]), X.columns.tolist(), rotation='vertical')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title(title)
    # plt.ylim([0, 0.25])
    plt.show()
    print()


def DecisionTreeModel(X_train, Y_train, X_test, Y_test):
    # Define the parameter grid
    param_grid = {
        'criterion': ['gini'],
        'max_depth': [None, 5, 10, 15],
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
    # get the best hyperparameters
    best_params = grid_search.best_params_

    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Visualize the decision tree model
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_tree(best_model, ax=ax, feature_names=X_train.columns, class_names=Y_train.unique(),
              filled=True, rounded=True, max_depth=best_params['max_depth'])
    plt.title("Decision Tree Model")
    plt.show()

    # Visualize Bar Char
    # BarCharForEachModel(best_model.feature_importances_, X_train, 'Decision Tree Feature Importances')

    # Evaluate the best model
    print("Best hyperparameters for Decision Tree Classifier: ", best_params)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Decision Tree Classifier accuracy: ", accuracy, "\n")

    startTrain = time.time()
    prediction = best_model.predict(X_train)
    endTrain = time.time()
    total_training_time = endTrain - startTrain

    start_test = time.time()
    prediction = best_model.predict(X_test)
    end_test = time.time()
    total_testing_time = end_test - start_test

    BarCharParameters = [accuracy, total_training_time, total_testing_time]
    return BarCharParameters


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
    # get the best hyperparameters
    best_params = grid_search.best_params_

    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, Y_train)

    y_pred = best_model.predict(X_test)

    # Visualize the KNN model
    if X_train.shape[1] == 2:
        # If the data is already 2-dimensional, use the scatter plot with random colors
        colors = [generate_random_color() for _ in range(len(Y_train))]
        plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, edgecolors='k')
        plt.title("K-Nearest Neighbors Scatter Plot")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    else:
        if X_train.shape[1] >= 50:
            # If the data has high dimensionality, use PCA for dimensionality reduction
            reducer = PCA(n_components=2)
        else:
            # If the data has relatively low dimensionality, use t-SNE for dimensionality reduction
            reducer = TSNE(n_components=2)

    X_train_reduced = reducer.fit_transform(X_train)

    # Generate random colors for the scatter plot
    colors = [generate_random_color() for _ in range(len(Y_train))]

    # Plot the reduced data using scatter plot with random colors
    plt.scatter(X_train_reduced[:, 0], X_train_reduced[:, 1], c=colors, edgecolors='k')
    plt.title("KNN Scatter Plot (Reduced Dimensionality)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # Visualize Bar Char
    perm_importance = permutation_importance(best_model, X_test, Y_test, n_repeats=10, random_state=42)
    # Get feature importance scores
    importance_scores = perm_importance.importances_mean
    BarCharForEachModel(importance_scores, X_train, 'KNN Feature Importances')

    # Evaluate the best model
    print("Best hyperparameters for K-Nearest Neighbors Classifier: ", best_params)
    accuracy = accuracy_score(Y_test, y_pred)
    print("K-Nearest Neighbors Classifier accuracy: ", accuracy, "\n")

    startTrain = time.time()
    prediction = best_model.predict(X_train)
    endTrain = time.time()
    total_training_time = endTrain - startTrain

    start_test = time.time()
    prediction = best_model.predict(X_test)
    end_test = time.time()
    total_testing_time = end_test - start_test

    BarCharParameters = [accuracy, total_training_time, total_testing_time]
    return BarCharParameters


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
    # get the best hyperparameters
    best_params = grid_search.best_params_

    # Predict on the test data using the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    startTrain = time.time()
    prediction = best_model.predict(X_train)
    endTrain = time.time()
    total_training_time = endTrain - startTrain

    start_test = time.time()
    prediction = best_model.predict(X_test)
    end_test = time.time()
    total_testing_time = end_test - start_test

    # Visualize the SVM model
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)
    # tsne = TSNE(n_components=2)
    # X_tsne = tsne.fit_transform(X_train)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(Y_train)

    # Fit the SVM classifier on reduced data
    best_model.fit(X_pca, y_train_encoded)  # or X_tsne if using t-SNE
    plot_decision_regions(X_pca, y_train_encoded, best_model)

    # Visualize Bar Char

    # Evaluate the best model
    print("Best hyperparameters for Support Vector Machines Classifier: ", best_params)
    accuracy = accuracy_score(Y_test, y_pred)
    print("Support Vector Machines Classifier accuracy: ", accuracy, "\n")

    BarCharParameters = [accuracy, total_training_time, total_testing_time]
    return BarCharParameters


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Create a mesh grid of feature values
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # Predict the class labels for each point in the mesh grid
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    # Plot the decision regions
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot the class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    color=cmap(idx),
                    marker=markers[idx],
                    label=cl)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title('SVM Model (Reduced Dimensionality)')

    plt.show()


def BarChar(title, values):
    # Create bar chart
    plt.bar(['Decision Tree', 'KNN', 'SVM'], values)
    plt.ylim([0.0, 1.0])  # rescale the y-axis to go from 0 to 1
    plt.xlabel('Models')
    plt.title(title)
    plt.show()
    print("\n")


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

BarCharAllParameters = []
# evaluate Decision Tree Classifier
BarCharAllParameters.append(DecisionTreeModel(X_train[top_features], Y_train, X_test[top_features], Y_test))

# evaluate KNN Classifier
BarCharAllParameters.append(KNNModel(X_train[top_features], Y_train, X_test[top_features], Y_test))

# evaluate SVM Classifier
BarCharAllParameters.append(SVMModel(X_train[top_features], Y_train, X_test[top_features], Y_test))

columnValues = [0.0, 0.0, 0.0]

for rowIndex in range(3):
    columnValues[rowIndex] = BarCharAllParameters[rowIndex][0]
BarChar('Accuracy', columnValues)

for rowIndex in range(3):
    columnValues[rowIndex] = BarCharAllParameters[rowIndex][1]
BarChar('Total Training Time', columnValues)

for rowIndex in range(3):
    columnValues[rowIndex] = BarCharAllParameters[rowIndex][2]
BarChar('Total Testing Time', columnValues)
