import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR

import seaborn as sns

import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

credits = preprocessing.load_tmdb_credits("dataset/tmdb_5000_credits.csv")
movies = preprocessing.load_tmdb_movies("dataset/tmdb_5000_movies.csv")
meta = pd.read_csv("dataset/movie_metadata.csv")
movies = preprocessing.preProcess(movies, meta, credits)

# # Remove all nominal features
movies = movies.drop(["genres", "homepage", "id", "keywords", "original_language", "original_title", "overview",
                      "production_companies", "production_countries", "spoken_languages", "status", "tagline",
                      "title",
                      "release_date", 'revenue', 'vote_average', 'vote_count'], axis=1)

# Substitue NaN values with 0
#movies = movies.fillna(0)

my_imputer = SimpleImputer()
X2 = my_imputer.fit_transform(movies[['runtime']])
movies['runtime'] = X2


# print(movies.isnull().sum())
#corrmat = movies.corr()
#hm = sns.heatmap(corrmat, annot=True, cmap='coolwarm')
#plt.show()
#plt.close()


y = movies['popularity']
movies = movies.drop(['popularity'], axis=1)


#print(movies.head())

# Select the random indexes for the test set and
x_train, x_test, y_train, y_test = train_test_split(
    movies, y, test_size=0.30, random_state=1234)

# #############################################################################
# Normalize (min-max-scaler)
xmins = x_train.min()
xmaxs = x_train.max()
ymins = y_train.min()
ymaxs = y_train.max()

x_train -= xmins
x_train /= xmaxs

x_test -= xmins
x_test /= xmaxs

y_train -= ymins
y_train /= ymaxs

y_test -= ymins
y_test /= ymaxs

#print(y_test.describe())
#print(y_train.describe())

# #############################################################################
# Define model

best_mse_rbf = 100
best_mse_lin = 100
best_mse_poly = 100

best_c_rbf = 10
best_c_lin = 10
best_c_poly = 10

best_rbf = ""
best_lin = ""
best_poly = ""

for c in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:

    svr_rbf = SVR(kernel='rbf', C=c, gamma='auto')
    svr_lin = SVR(kernel='linear', C=c, gamma='auto')
    svr_poly_2 = SVR(kernel='poly', C=c, degree=2, gamma='auto')
    #svr_poly_3 = SVR(kernel='poly', C=c, degree=3, gamma='auto')

    svr_rbf.fit(x_train, y_train)
    y_rbf = svr_rbf.predict(x_test)

    mse_rbf = mean_squared_error(y_test, y_rbf)

    svr_lin.fit(x_train, y_train)
    y_lin = svr_lin.predict(x_test)

    mse_lin = mean_squared_error(y_test, y_lin)

    svr_poly_2.fit(x_train, y_train)
    y_poly = svr_poly_2.predict(x_test)

    mse_poly = mean_squared_error(y_test, y_poly)

    if mse_poly < best_mse_poly:
        best_mse_poly = mse_poly
        best_c_poly = c
        best_poly = svr_poly_2

    if mse_rbf < best_mse_rbf:
        best_mse_rbf = mse_rbf
        best_c_rbf = c
        best_rbf = svr_rbf

    if mse_lin < best_mse_lin:
        best_mse_lin = mse_lin
        best_c_lin = c
        best_lin = svr_lin

plt.scatter(y_test, y_test, s=30, label="Real popularity")
plt.scatter(y_test, best_poly.predict(x_test), s=30, c='r', label="Predicted popularity")
plt.ylabel("POLY popularity")
plt.legend(loc=2)
plt.show()

plt.scatter(y_test, y_test, s=30, label="Real popularity")
plt.scatter(y_test, best_lin.predict(x_test), s=30, c='r', label="Predicted popularity")
plt.ylabel("LIN popularity")
plt.legend(loc=2)
plt.show()

plt.scatter(y_test, y_test, s=30, label="Real popularity")
plt.scatter(y_test, best_rbf.predict(x_test), s=30, c='r', label="Predicted popularity")
plt.ylabel(" RBF popularity")
plt.legend(loc=2)
plt.show()


print("\nBest SVM LINEAR, mse: " + str(best_mse_lin) + " with C = " + str(best_c_lin))
print("Best SVM POLYNOMIAL, mse: " + str(best_mse_poly) + " with C = " + str(best_c_poly))
print("Best SVM RBF, mse: " + str(best_mse_rbf) + " with C = " + str(best_c_rbf) + "\n")


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


print("Linear")
print("Mean squared error: %.6f"
      % mean_squared_error(y_test, y_lin))
print("Root mean squared error: %.6f"
      % rmse(y_test, y_lin))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_lin))

plt.plot(np.arange(len(y_test)), y_test, color='red', label='Real data')
plt.plot(np.arange(len(y_test)), best_lin.predict(x_test), color='blue', label='Predicted data')
plt.title('Linear prediction')
plt.legend()
plt.show()
plt.close()


print("\nPolynomial n=2")
print("Mean squared error: %.6f"
      % mean_absolute_error(y_test, y_poly))
print("Root mean squared error: %.6f"
      % rmse(y_test, y_poly))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_poly))

plt.plot(np.arange(len(y_test)), y_test, color='red', label='Real data')
plt.plot(np.arange(len(y_test)), best_poly.predict(x_test), color='blue', label='Predicted data')
plt.title('Polynomial prediction')
plt.legend()
plt.show()
plt.close()


print("\nRBF")
print("Mean squared error: %.6f"
      % mean_squared_error(y_test, y_rbf))
print("Root mean squared error: %.6f"
      % rmse(y_test, y_rbf))
print("Mean absolute error: %.6f"
      % mean_absolute_error(y_test, y_rbf))

plt.plot(np.arange(len(y_test)), y_test, color='red', label='Real data')
plt.plot(np.arange(len(y_test)), best_rbf.predict(x_test), color='blue', label='Predicted data')
plt.title('RBF Prediction')
plt.legend()
plt.show()
plt.close()


#############################################################################
# Validate using cross-validation
# See https://scikit-learn.org/stable/modules/model_evaluation.html
scores = cross_val_score(svr_rbf, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE RBF: {scores}")
scores = cross_val_score(svr_lin, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE Linear: {scores}")
cores = cross_val_score(svr_poly_2, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=2): {scores}")
scores = cross_val_score(svr_poly_3, movies, y, cv=10, scoring='neg_mean_squared_error')
print(f"MSE POLYNOMIAL (DEGREE=3): {scores}")
