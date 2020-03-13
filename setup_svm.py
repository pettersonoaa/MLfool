import numpy as np, pandas as pd, matplotlib.pyplot as plt

def fit_svm (X, y, report=True, test_size=0.2, random_state=31, gamma=2):
  
    # train and test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # load model and parms
    from sklearn.svm import SVC
    parms = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    train_acc = test_acc = model = np.empty(len(parms))
    max_score_index = 0

    # get best parms
    for i, k in enumerate(parms):
        model = SVC(C=k, gamma=gamma).fit(X_train, y_train)
        train_acc[i], test_acc[i] = model.score(X_train, y_train), model.score(X_test, y_test)
        if (test_acc[max_score_index] < test_acc[i]) or (i == 0):
            max_score_index = i
            best_model = model

    # show report and plots
    if report:
        plt.plot(train_acc, label='Train')
        plt.plot(test_acc, label='Test')
        plt.legend()
        plt.show()

        y_pred = best_model.predict(X_test)
        y_prob = y_pred#best_model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        fpr, tpr, threshholds = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC: '+str(auc))
        plt.show()

    return best_model

#load dataset
df = pd.read_csv('xxx.csv')

# set X and y vars
X_label, y_label = [], []
X, y = df[X_label], df[y_label]
          
# fit the model
fit_svm(X, y, X_label, y_label)
