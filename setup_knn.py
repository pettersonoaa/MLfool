import numpy as np, pandas as pd, matplotlib.pyplot as plt

def fit_knn (X, y, X_label, y_label, max_k=10, report=True):
  
  # train and test split
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31)

  # load model and parms
  from sklearn.neighbors import KNeighborsClassifier
  neighbors = np.arange(1, max_k)
  train_acc = test_acc = knn = np.empty(len(neighbors))
  max_score = max_score_index = 0

  # get best k
  for i, k in enumerate(neighbors):
    knn[i] = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    train_acc[i], test_acc[i] = knn.score(X_train, y_train), knn.score(X_test, y_test)
    if max_score < test_acc[i]:
      max_score, max_score_index = test_acc[i], i

  # show report and plots
  if report:
    plt.plot(neighbors, train_acc, label='Train')
    plt.plot(neighbors, test_acc, label='Test')
    plt.legend()
    plt.show()

    y_pred = knn[max_score_index].predict(X_test)
    y_prob = knn[max_score_index].predict_proba(X_test)[:, 1]

    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
    print(confusion_matrix(t_test, y_pred))
    print(classification_report(y_test, y_pred)

    fpr, tpr, threshholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC: '+str(auc))
    plt.show()
  
  return knn[max_score_index]

#load dataset
df = pd.read_csv('xxx.csv')

# set X and y vars
X_label, y_label = [], []
X, y = df[X_label], df[y_label]
          
# fit the model
fit_knn(X, y, X_label, y_label)
