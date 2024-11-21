import numpy as np


# by default it sets k=5 for folds  Xtrain and yTrain are the training data
'''
X => should be a numpy array of shape (n_samples, n_features) 
y => should be a numpy array of shape (n_samples,).
model => should be  an instance of a class that has fit and predict methods, such as LinearRegression or RidgeRegression.
seed => is an integer that sets the random seed for reproducibility.


'''
def kFoldCv(X, y,model, k=5,seed=1):

    np.random.seed(seed)
    # Shuffling the data indices
    indices = np.arange(len(y))
    np.random.shuffle(indices)  
    folds = np.array_split(indices, k)
    
    mseScores = []
    # it will loop through the folds
    for i in range(k):
        # split the data into training and validation sets
        valIndices = folds[i]
        trainIndices = np.hstack([folds[j] for j in range(k) if j != i])
        Xtrain, yTrain = X[trainIndices], y[trainIndices]
        Xval, yVal = X[valIndices], y[valIndices]
        # Fit the regression model on the training data
        model.fit(Xtrain, yTrain)
        # Predict on the validation set and calculate the MSE
        yPred = model.predict(Xval)
        mse = np.mean((yVal - yPred) ** 2)
        mseScores.append(mse)
    return mseScores, np.mean(mseScores)

'''
X => should be a numpy array of shape (n_samples, n_features)
y => should be a numpy array of shape (n_samples,).
model => should be  an instance of a class that has fit and predict methods, such as LinearRegression or RidgeRegression.
numIterations => is an integer that sets the number of bootstrap iterations.
seed => is an integer that sets the random seed for reproducibility.
'''
# Bootstrapping function
def bootstrap(X, y,model, numIterations=100, seed=1):
    np.random.seed(seed)
    n = len(y)
    mse_scores = []
    # Loop through the number of iterations
    for _ in range(numIterations):
         # Sample with replacement from the data
        indices = np.random.choice(np.arange(n), size=n, replace=True) 
        Xsample, ySample = X[indices], y[indices]
        # Train the regression model on the sampled data
        model.fit(Xsample, ySample)
        yPred = model.predict(Xsample)
        mse = np.mean((ySample - yPred) ** 2)
        mse_scores.append(mse)
    
    return mse_scores, np.mean(mse_scores)


