from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

    # removing + symbol in Age Rating column
    X["Age Rating"] = Amir(X["Age Rating"])

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
    X = splitMultipleData(X, 'Languages', ', ', 10)
    X = splitMultipleData(X, 'Genres', ', ', 15)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])

    # scaling
    Scaled_X = featureScaling(X, -1, 1)
    X = pd.DataFrame.from_records(Scaled_X, columns=X.columns)

    # dropping these 2 columns as the have the same values in every row
    X = X.drop(['Games', 'Strategy'], axis=1)

    X.to_csv(r'newGameDataset.csv', index=False)

    # correlation
    corrData = pd.DataFrame(X)
    corrData['Average User Rating'] = df['Average User Rating']

    corr = X.corr()
    top_feature = corr.index[abs(corr['Average User Rating']) >= 0.1]
    plt.subplots(figsize=(12, 8))
    top_corr = X[top_feature].corr()
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


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


def splitMultipleData(df, column, spliter, numberOfColumns):
    # get the top 10 languages
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


#  X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

np.seterr(invalid='ignore')
preProcessing('games-regression-dataset.csv')
