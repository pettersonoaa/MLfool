https://towardsdatascience.com/choosing-a-scikit-learn-linear-regression-algorithm-dd96b48105f5
https://github.com/qlanners/scikit-learn_disect/blob/master/all_model_tester.ipynb


#Import non-sklearn packages
import numpy as np
import time

#import sklearn auxillary packages
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

#import sklearn regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.isotonic import IsotonicRegression

#set the dataset parameters
sample_number = 100
feature_number = 1
test_set_perc = 0.3
noise = 0
scale = True
effective_rank = 1
n_informative = 1
random_state = 1
#shifts output labels into a quadratic structure (rather than linear)
make_quadratic = False

#print the model coeficients (not all models have this method, 
#so may have to set to False if certain models are being tested)
print_coef = False

#create a dataset to use for regression problem
x, y = make_regression(n_samples=sample_number, n_features=feature_number, noise=noise, n_informative=n_informative, 
                       effective_rank=effective_rank, random_state=random_state)
if make_quadratic:
  y = y**2
  
#split dataset into train and test sets
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_set_perc)

# scale the data if desired
if scale:
    print('Scaling Data')
    print('')
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    print('Data Scaled')
    print('')
 
#Dictionary of all models. All models intialized with no args. Can modify any of them to test various args.
#IsotonicRegression does not abide by the same fit() function and thus must be tested seperately
models = {'LinearRegression': LinearRegression(),
          'Ridge': Ridge(),
          'Lasso': Lasso(),
          'ElasticNet': ElasticNet(),
          'Lars': Lars(),
          'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
          'BayesianRidge': BayesianRidge(),
          'ARDRegression': ARDRegression(),
          'SGDRegressor': SGDRegressor(),
          'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
          'RANSACRegressor': RANSACRegressor(),
          'TheilSenRegressor': TheilSenRegressor(),
          'HuberRegressor': HuberRegressor(),
          'DecisionTreeRegressor': DecisionTreeRegressor(),
          'GaussianProcessRegressor': GaussianProcessRegressor(),
          'MLPRegressor': MLPRegressor(),
          'KNeighborsRegressor': KNeighborsRegressor(),
          'RadiusNeighborsRegressor': RadiusNeighborsRegressor(),
          'SVR': SVR(gamma='scale'),
          'NuSVR': NuSVR(gamma='scale'),
          'LinearSVR': LinearSVR(),
          'KernelRidge': KernelRidge()
         }

for key, model in models.items():
    begin = time.time()
    model.fit(train_x,train_y)
    print(key + ' Train time: ' + str((time.time() - begin)/60) + " minutes")
    preds = model.predict(test_x)
    mse = mean_squared_error(test_y,preds)
    r2 = r2_score(test_y,preds)
    scores = cross_val_score(model, train_x, train_y, cv=5)
    print(key + ' MSE: ' + str(mse))
    print(key + ' R2 ' + str(r2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    if print_coef:
      print('Coefficients:')
      print(model.coef_)
    print('')
