from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

def do_gradient_boost():
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)
    gradient_boost = GradientBoostingClassifier()
    gradient_boost.fit(X_train, y_train)
    train_acc = gradient_boost.score(X_train, y_train)
    test_acc = gradient_boost.score(X_test, y_test)
    print("Gradient Boost")
    print("Train acc:", train_acc)
    print("Test  acc:", test_acc)
    
    importances = gradient_boost.feature_importances_
    indices = np.argsort(importances)
    fig, ax = plt.subplots()
    ax.barh(range(len(importances)), importances[indices])
    ax.set_yticks(range(len(importances)))
    ytick = ["PC"+str(tick) for tick in np.array(X_train.columns)[indices]]
    ax.set_yticklabels(ytick)
    plt.title("Gradient Boost Feature importances")
    plt.show()