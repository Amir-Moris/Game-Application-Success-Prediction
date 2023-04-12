import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
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


def Amir(ages):
    newAges = []
    try:
        for i in range(len(ages)):
            if ages[i][-1] == '+':
                age_list = list(ages[i])
                age_list.pop()
                newAges.append(listToString(age_list))
        return newAges
    except:
        print("Warning")


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def preProcessing(FilePath):
    df = pd.read_csv(FilePath)
    X = df.iloc[:, :-1]
    Y = df['Average User Rating']

    X["Age Rating"] = Amir(X["Age Rating"])
    X['Original Release Date'] = pd.to_datetime(X['Original Release Date']).dt.month
    X['Current Version Release Date'] = pd.to_datetime(X['Current Version Release Date']).dt.day
    # df.dropna(how='any', inplace=True)
    # df.fillna('nan', inplace=True)

    X = X.drop(['URL', 'Name', 'Subtitle', 'Icon URL', 'In-app Purchases', 'Description', 'Genres'], axis=1)
    # encoding
    cols = ('Developer', 'Primary Genre')
    X = Feature_Encoder(X, cols)
    # scaling
    print(X.shape)
    # scaled_X = featureScaling(X, -1, 1)

    return
    # correlation
    corr = data.corr()
    top_feature = corr.index[abs(corr['Average User Rating']) > 0.02]
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()
    # print(df.isna().sum())
    return
    df['DataFrame Column'] = df['DataFrame Column'].replace(np.nan, 0)

    # print(df['Subtitle'])
    print(df.isna().sum())
    return


    print(df.isna().sum())

    print(data["Age Rating"])
    corr = data.corr()
    top_feature = corr.index[abs(corr['Average User Rating']) > 0.02]
    plt.subplots(figsize=(12, 8))
    top_corr = data[top_feature].corr()
    sns.heatmap(top_corr, annot=True)
    plt.show()

    return


#  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)


preProcessing('games-regression-dataset.csv')
