import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt 

def do_gradient_boost(max_depth, max_features):
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)
    gradient_boost = GradientBoostingClassifier(max_depth = max_depth, max_features = max_features)
    gradient_boost.fit(X_train, y_train)
    acc = gradient_boost.score(X_test, y_test)
    return acc

def grid_search_gb():
    print("Gradient Boost Grid Searching... (about 20 seconds)")
    grid_space = {'max_depth':[3,5,10,None],
                  'max_features':[1,3,5,7,10],}
    
    for max_depth in grid_space["max_depth"]:
        x_axis = []
        y_axis = []
        for max_features in grid_space["max_features"]:
            acc = do_gradient_boost(max_depth, max_features)
            x_axis.append(max_features)
            y_axis.append(acc)
        plt.plot(x_axis, y_axis, label="max_depth="+str(max_depth))
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    plt.title("Grid Search - Gradient Boost")
    plt.legend()
    plt.grid()
    plt.show()
