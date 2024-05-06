# Multiple-Linear-Regression

A very simple python program to implement Multiple Linear Regression using the LinearRegression class from sklearn.linear_model library.

The program also does Backward Elimination to determine the best independent variables to fit into the regressor object of the LinearRegression class.

The program uses the statsmodels.formula.api library to get the P values of the independent variables. The variables with P values greater than the significant value ( which was set to 0.05 ) are removed. The process is continued till variables with the lowest P values are selected are fitted into the regressor ( the new dataset of independent variables are called X_Optimal ).

X_Optimal is again split into training set and test set using the test_train_split function from sklearn.model_selection.

The regressor is fitted with the X_Optimal_Train and Y_Train variables and the prediction for Y_Test ( the dependent varibale) is done using the regressor.predict(X_Optimal_Test)

# Logistic-Regression

A very simple Logistic Regression classifier implemented in python. The sklearn.linear_model library is used to import the LogisticRegression class. A classifier object of that class was created and fitted with the X_Train and Y_Train varibles.

A confusion matrix was implemented to test the prediction accuracy of the classifier. The confusion matrix function was imported from sklearn.metrics


# Random-Forest-Classifier

A very simple Random Forest Classifier implemented in python. The sklearn.ensemble library was used to import the RandomForestClassifier class. The object of the class was created. The following arguments was passed initally to the object:

 - n_estimators = 10
 - criterion = 'entropy'

The inital model was only given 10 decision tree, which resulted in a total of 10 incorrect prediction. Once the model was fitted with more the decision trees the number of incorrect prediction grew less.

It was found that a the optimal number of decision trees for this models to predict the answers was 200 decision trees. Hence the n_estimator argument was given a final value of 200.

Anything more that 200 will result in over-fitting and will lead further incorrect prediction.

# Support-Vector-Machine

A simple implementation of a (linear) Support Vector Machine model in python. The classifier is an object of the SVC class which was imported from sklearn.svm library.

the linear kernel type was choosen since this was a linear SVM classifier model. 


# KMeans-Clustering

A simple K-Means Clustering model implemented in python. The class KMeans is imported from sklearn.cluster library. In order to find the optimal number of cluster for the dataset, the model was provided with different numbers of cluster ranging from 1 to 10. The 'k-means++' method to passed to the init argument to avoid the Random Initialization Trap. The max_iter and the n_init were passed with their default values.

The WCSS ( or Within Cluster Sum of Squares ) was caluated and plotted to find the optimal number of clusters. The "Elbow Method" was used to find the optimal number of clusters. 

Once the optimal number of clusters were found the model was reinitalised with the n_cluster arguments begin passed with the optimal number of clusters found using the "Elbow Method".

Finally, the clusters were visualised using scatter plot. 
