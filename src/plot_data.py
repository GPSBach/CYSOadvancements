'''
Plotting functions
'''

import pandas as pd
import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import roc_curve, auc
matplotlib.rcParams['font.family'] = 'Arial'

# load data
with open("../data/pp_1000_models.pkl", 'rb') as picklefile:
    models = pkl.load(picklefile)

# turn coefficients into DF
def make_coeffiecients(models):
	'''
	turns coefficients into a dataframe
	column is feature
	row is model run #
	should have done this with pd.melt
	'''

	# list of named variables
	uni_public = []
	acceptance_rate = []
	male = []
	caucasian = []
	african_american = []
	latino = []
	asian = []
	chi_school = []
	chi_home = []
	financial_aid = []
	median_income = []
	private_hs = []
	chipub_hs = []

	# append relevant values to each list
	for x in range(1000):
	    chipub_hs.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][0])
	    private_hs.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][1])
	    uni_public.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][2])
	    acceptance_rate.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][3])
	    male.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][4])
	    caucasian.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][5])
	    african_american.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][6])
	    latino.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][7])
	    asian.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][8])
	    chi_school.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][9])
	    chi_home.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][10])
	    financial_aid.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][11])
	    median_income.append(models[x]['l2'].best_estimator_.named_steps['logisticregression'].coef_[0][12])
	    
	
	# turn lists into dataframe   
	coefficients = pd.DataFrame({'Public University' : uni_public,
	                            'Private University' : np.multiply(-1,uni_public),
	                            'University Acceptance Rate' : acceptance_rate, 
	                            'City of Chicago - School' : chi_school,
	                            'Suburban Chicago - School' : np.multiply(-1,chi_school),
	                            'School outside City of Chicago' : np.multiply(-1,chi_school),
	                            'City of Chicago - Home' : chi_home,
	                            'Home outside City of Chicago' : np.multiply(-1,chi_home),
	                            'CYSO Financial Aid' : financial_aid,
	                            'Home Median Income' : median_income,
	                            'Male' : male,
	                            'Female' : np.multiply(-1,male),
	                            'African American' : african_american,
	                            'Caucasian' :  caucasian,
	                            'Latinx' : latino,
	                            'Asian' : asian,
	                            'Public HS' : np.multiply(-1,private_hs),
	                            'Private HS' : private_hs,
	                            'Chicago Public School' : chipub_hs,
	                            'Suburban Public School' : np.multiply(-1,chipub_hs)
	                            })   
    
    
    # optional save
	#coefficients.to_csv('../data/model_coefficients.csv')

	#return data
	return coefficients
    
# make boxplot
def make_boxplot()
	'''
	boxplots
	'''

	#sns.set(rc={'figure.figsize':(10,5)})
	coefficients_plot = coefficients.mean().values
	coefficients_std = coefficients.std().values
	features_plot = coefficients.columns
	coefficients_inds = coefficients_plot.argsort()
	sorted_coefficients = coefficients_plot[coefficients_inds[::-1]]
	sorted_std = coefficients_std[coefficients_inds[::-1]]
	sorted_features = features_plot[coefficients_inds[::-1]]
	plot_colors = ['gainsboro','gainsboro','firebrick','firebrick',
	               'firebrick','firebrick','gainsboro','darkgrey',
	               'darkgrey','darkgrey','darkgrey','darkgrey',
	               'firebrick','royalblue']
	plt.subplots(figsize=(10,10))
	plt.axvline(x=0,c='k',linewidth=0.5)
	ax = sns.boxplot(data=coefficients,
	                orient = 'h',
	                order = sorted_features,
	                showfliers=False,
	                color='gainsboro')#,
	                #whis=0)
	plt.axvline(x=0,c='k',linewidth=0.5)
	#plt.style.use('ggplot')
	space_string = '                                                         '
	ax.set_xlabel(r'$\leftarrow$'+'non-music major'+space_string+'music major'+r'$\rightarrow$'+'\nLogistic Coefficient for Scaled Data',fontsize=18)
	#ax.set_xlabel('Logistic Coefficient for \n Scaled Data')
	#ax.set_ylabel('Student Attribute',fontsize=18)
	plt.title('All Student Attributes',fontsize=18)
	plt.tick_params(labelsize=15)
	#ax.get_xaxis().set_ticks([])
	fig = ax.get_figure()
	fig.savefig('../plots/allfactors_withwhisker_highlights.pdf',bbox_inches='tight')

def make_ROC()
'''
ROC Plot
'''

	df = pd.read_csv('../data/fitting_data.csv')
	df.drop(['Unnamed: 0'],axis=1,inplace=True)
	df = df.fillna(0)
	y = df.pop('music').values
	stratification_columns = df.pop('stratification_column').values
	X = df.values

	logit_num = 6

	#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,stratify=X[:,9])

	datapointers = {
	    'l1' : 'logistic',
	    'l2' : 'logistic',
	    'rf' : 'not_logistic',
	    'gb' : 'not_logistic',
	    'linsvc' : 'not logistic',
	    'rbfsvc' : 'not logistic'
	}


	# Initialize figure
	fig = plt.figure(figsize=(8,8))
	plt.title('Receiver Operating Characteristic',fontsize=20)

	name = 'l2'

	for x in range(1000):
	    if datapointers[name] == 'logistic':
	        pred = models[x][name].predict_proba(X[:,logit_num:])
	    else:
	        pred = models[x][name].predict_proba(X)
	    pred = [p[1] for p in pred]
	    fpr, tpr, thresholds = roc_curve(y, pred)

	    
	    # Plot ROC curve
	    plt.plot(fpr, tpr, c='silver',linewidth=1,alpha=0.2)
	    

	pred = models[52][name].predict_proba(X[:,logit_num:])
	pred = [p[1] for p in pred]
	fpr, tpr, thresholds = roc_curve(y, pred)
	plt.plot(fpr, tpr, label='model suite', c='silver',linewidth=3)    

	plotnum = 55
	pred = models[plotnum][name].predict_proba(X[:,logit_num:])
	pred = [p[1] for p in pred]
	fpr, tpr, thresholds = roc_curve(y, pred)
	plt.plot(fpr, tpr, label = 'AUC = {:0.2f}'.format(scores[plotnum]['l2'][1]), c='royalblue',linewidth=4)

	# Diagonal 45 degree line
	plt.plot([0,1],[0,1],'k--',linewidth=3)
	plt.legend(loc='lower right',fontsize=16)

	# Axes limits and labels
	plt.xlim([-0.1, 1.1])
	plt.ylim([-0.1, 1.1])
	plt.ylabel('True Positive Rate',fontsize=18)
	plt.xlabel('False Positive Rate',fontsize=18)
	plt.tick_params(labelsize=16)
	plt.grid(True)
	plt.show()
	fig.savefig('../plots/ROC_pp.pdf',bbox_inches='tight')
