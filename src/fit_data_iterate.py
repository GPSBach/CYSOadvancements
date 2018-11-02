'''
Fits data in an iterative loop using gridsearchCV and an additional CV step
'''

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pickle as pkl
from collections import defaultdict

# number of variables to be excluded for logistic regression
# these must be the first two variables in the feature matrix
logit_num = 6

# global inputs
# names of models to be fitted in the loop
model_names = ['l1','l2','rf','gb','linsvc','rbfsvc']

'''
Model Building
'''

# Pipeline dictionary
pipelines = {
    'l1' : make_pipeline(StandardScaler(), LogisticRegression( penalty = 'l1', random_state=125)),
    'l2' : make_pipeline(StandardScaler(), LogisticRegression( penalty = 'l2', random_state=125)),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier(random_state=125)),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=125)),
    'linsvc' : make_pipeline(StandardScaler(), SVC(random_state=125,probability=True)),
    'rbfsvc' : make_pipeline(StandardScaler(), SVC(random_state=125,probability=True))
}

# Logistic Regression hyperparameters
l1_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-2, 1e1, 500)
}

l2_hyperparameters = {
    'logisticregression__C' : np.linspace(1e-2, 1e1, 500)
}

# Random Forest hyperparameters
rf_hyperparameters = {
    'randomforestclassifier__n_estimators': [100, 300, 500],
    'randomforestclassifier__max_features': ['auto', 'sqrt', 0.33],
    'randomforestclassifier__max_depth': [1, 2, 3, 4, 5]
}

# Boosted Tree hyperparameters
gb_hyperparameters = {
    'gradientboostingclassifier__n_estimators': [100, 300, 500],
    'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.5, 1],
    'gradientboostingclassifier__max_depth': [1, 2, 3, 4, 5]
}

linsvc_hyperparameters = {
    'svc__C' : [1e-5, 1e-3, 1e-1, 1e1],
    'svc__kernel' : ['linear']
}

rbfsvc_hyperparameters = {
    'svc__C': [1e-5, 1e-3, 1e-1, 1e1],
    'svc__gamma' : [1e-5, 1e-3, 1e-1, 1e1],
    'svc__kernel' : ['rbf']
}
# Create hyperparameters dictionary
hyperparameters = {
    'l1' : l1_hyperparameters, 
    'l2' : l2_hyperparameters,
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'linsvc' : linsvc_hyperparameters,
    'rbfsvc' : rbfsvc_hyperparameters
}
# Create data pointing dictionary
datapointers = {
    'l1' : 'logistic',
    'l2' : 'logistic',
    'rf' : 'not_logistic',
    'gb' : 'not_logistic',
    'linsvc' : 'not logistic',
    'rbfsvc' : 'not logistic'
}

def model_scoring_auc(X_in, y_in, model, datapointer, logit_num):
	''' scores model using AUC metric
	X_in - feature matrix
	y_in - obs matrix
	model - name of model
	datapointer - dictionary of logistic/non-logistic terms
	logit_num - number of features to be excluded for logistic
	'''

    if datapointer == 'logistic':
        pred = model.predict_proba(X_in[:,logit_num:])
    else:
        pred = model.predict_proba(X_in)
    # Get just the prediction for the positive class (1)
    pred = [p[1] for p in pred]
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_in, pred)
    # Calculate AUROC
    return auc(fpr, tpr)


def model_fitting(X, y, logit_num, model_names, pipelines, hyperparameters, datapointers, randstate,stratcolumn):
    '''fits model over hyperparameter space
    X - feature matrix
	y - obs vector
	logit_num - number of features to be excluded for logistic
	model_names - list of models to be included
	pipelines - dictionary of model pipelines inlcuding scaling
	hyperparameters - dictionary of hyperparameters dictionaries
	datapointer - dictionary of logistic/non-logistic terms
	randstate - random number for Monte Carlo
	stratcolumn - vector (stratcolumn.shape = y.shape) for data stratification in CV
	'''

    # Create empty dictionary called fitted_models
    fitted_models = {}
    fitted_scores = {}
    
    # split data for CV testing
    
    #this works:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=randstate,stratify=stratcolumn)




    # Loop through model pipelines, tuning each one and saving it to fitted_models
    for name in model_names:
        # Create cross-validation object from pipeline and hyperparameters
        model = GridSearchCV(pipelines[name], hyperparameters[name], scoring = 'neg_log_loss', cv=10, refit=True)

        # Fit model on X_train, y_train
        if datapointers[name] == 'logistic':
            model.fit(X_train[:,logit_num:], y_train)  
        else:
            model.fit(X_train, y_train)
        # Store model in fitted_models[name] 
        fitted_models[name] = model
        
        # store scores in fitted_scores[name]
        train_score = model_scoring_auc(X_train, y_train, model, datapointers[name],logit_num)
        test_score = model_scoring_auc(X_test, y_test, model, datapointers[name],logit_num)
        fitted_scores[name] = [train_score,test_score]
            
    return fitted_models, fitted_scores

'''
Model Fitting
'''

# read in and organize data
df = pd.read_csv('../data/fitting_data.csv')
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df = df.fillna(0)
y = df.pop('music').values
stratification_columns = df.pop('stratification_column').values
X = df.values

# set up dictionary to contain fitted models
models_iterate = {}
scores_iterate = {}

# loop over model pipelines i number of times
for i in range(1000):
    models_iterate[i], scores_iterate[i] = model_fitting(X,y,logit_num,model_names,pipelines,hyperparameters,datapointers,i+12000,stratification_columns)
    if i%10 == 0:
        print('step',i,'model',scores_iterate[i])

'''
model saving
'''
with open('../data/XXX_models.pkl', 'wb') as picklefile:
    pkl.dump(models_iterate, picklefile)
with open('../data/XXX_models_scores.pkl', 'wb') as picklefile:
    pkl.dump(scores_iterate, picklefile)

 
