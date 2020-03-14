import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def cv_model (X, y, report=True, seed=31, C=1, n_estimators=100, max_depth=5, 
                gamma=2, n_neighbors=5, scoring='accuracy', n_splits=10):
    
    # load model and parms
    models = []
    from sklearn.linear_model import LogisticRegression
    models.append(('LR', LogisticRegression(C=C)))
    
    from sklearn.neighbors import KNeighborsClassifier
    models.append(('KNN', KNeighborsClassifier(n_neighbors=n_neighbors)))
    
    from sklearn.svm import SVC
    models.append(('SVM', SVC(gamma=gamma, C=C)))
    
    from sklearn.ensemble import RandomForestClassifier
    models.append(('RF', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)))
    
    from sklearn.ensemble import AdaBoostClassifier
    models.append(('AB', AdaBoostClassifier()))
    
    
    # model validation
    from sklearn.model_selection import KFold, cross_val_score 
    results, names = [], []
    
    for name, model in models:
        kfold = KFold(n_splits=n_splits, random_state=seed)
        cv = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
        results.append(cv)
        names.append(name)
    

    # plot results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()
    
    return models
    
from sklearn.datasets import make_moons
data = pd.DataFrame(make_moons(noise=0.6)[0], index=make_moons(noise=0.3)[1]).reset_index()
X, y = data[[0, 1]], data[['index']]

# set X and y vars
#X_label, y_label = [], []
#X, y = df[X_label], df[y_label]
          
# fit the model
model = cv_model(X, y)

model[3][1].fit(X, y).predict(X)
