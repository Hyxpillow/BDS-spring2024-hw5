import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def do_naive_bayes():
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    train_acc = naive_bayes.score(X_train, y_train)
    test_acc = naive_bayes.score(X_test, y_test)
    print("Naive Bayes")
    print("Train acc:", train_acc)
    print("Test  acc:", test_acc)
    
