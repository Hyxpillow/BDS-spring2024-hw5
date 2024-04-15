from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def do_random_forest():
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)
    random_forest = RandomForestClassifier()
    random_forest.fit(X_train, y_train)
    train_acc = random_forest.score(X_train, y_train)
    test_acc = random_forest.score(X_test, y_test)
    print("Random Forest")
    print("Train acc:", train_acc)
    print("Test  acc:", test_acc)
    
    importances = random_forest.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots()
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    ytick = ["PC"+str(tick) for tick in np.array(X_train.columns)[indices]]
    ax.set_yticklabels(ytick)
    plt.title("Random Forest Feature importances")
    plt.show()
