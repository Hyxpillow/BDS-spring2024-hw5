import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 

def do_random_forest(max_depth, max_features):
    raw_df = pd.read_csv("./preprocessing/reduction.csv")
    X = raw_df.drop(['Status'], axis=1)
    y = raw_df['Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle=False)
    random_forest = RandomForestClassifier(max_depth = max_depth, max_features = max_features)
    random_forest.fit(X_train, y_train)
    test_acc = random_forest.score(X_test, y_test)
    return test_acc

def grid_search_rf():
    print("Random Forest Grid Searching... (about 10 seconds)")
    grid_space = {'max_depth':[3,5,10,None],
                  'max_features':[1,3,5,7,10],}
    
    for max_depth in grid_space["max_depth"]:
        x_axis = []
        y_axis = []
        for max_features in grid_space["max_features"]:
            acc = do_random_forest(max_depth, max_features)
            x_axis.append(max_features)
            y_axis.append(acc)
        plt.plot(x_axis, y_axis, label="max_depth="+str(max_depth))
    plt.xlabel("max_features")
    plt.ylabel("accuracy")
    plt.title("Grid Search - Random Forest")
    plt.legend()
    plt.grid()
    plt.show()

