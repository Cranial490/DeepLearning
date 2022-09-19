import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return A@B - C

def problem_1c (A, B, C):
    return A*B - np.transpose(C)

def problem_1d (x, y):
    return np.transpose(x)@y

def problem_1e (A, x):
    return np.linalg.solve(A,x)

def problem_1f (A, x):
    return np.transpose(np.linalg.solve(np.transpose(A),np.transpose(x)))

def problem_1g (A, i):
    indices = list(np.arange(0,len(A),2))
    return np.sum(A[i][indices])

def problem_1h (A, c, d):
    return np.mean(A[np.nonzero((A>=c)&(A<=d))])

def problem_1i (A, k):
    e_vals, e_vecs = np.linalg.eig(A)
    indices = np.argsort(e_vals,k)[-k:]
    return e_vecs[indices]

def problem_1j (x, k, m, s):
    n = x.shape[0]
    mean = x + m*np.ones(x.shape)
    cov = s*np.ones((n,n))
    return np.random.multivariate_normal(mean,cov,(n,k))

def problem_1k (A):
    indices = np.random.permutation(A.shape[0])
    return A[indices]


def problem_1l (x):
    mean = np.mean(x)
    std = np.std(x)
    return (x-mean)/std

def problem_1m (x, k):   
    return np.repeat(x, k, axis=1)

def problem_1n (X):
    m = X.shape[0]
    n = X.shape[1]
    x1 = np.reshape(np.repeat(X,m,axis=0),(m,m,n))
    x2 = np.swapaxes(x1, 0, 1)
    return np.sum((x1-x2)**2,axis=2)

def f_MSE(X,w,y):
    h = X@w
    loss = np.sum((h-y)**2)
    return loss/(2*X.shape[1])


def linear_regression (X_tr, y_tr):
    features = X_tr.shape[1]
    w = np.random.normal(-1,1,(1,features))
    # if delta(MSE) wrt w == 0:
    w = np.transpose((np.transpose(y_tr)@X_tr)@np.linalg.inv(np.transpose(X_tr)@X_tr))
    return w

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, ytr)
    # Report fMSE cost on the training and testing data (separately)
    training_loss = f_MSE(X_tr,w,ytr)
    testing_loss = f_MSE(X_te,w,yte)
    return training_loss,testing_loss
tr_loss, te_loss = train_age_regressor()

