
# This function was created by Lindsey Berlin to assist with printing un-logged target results for linear models.
# Thanks Lindsey you da bomb


# For the function to work while imported we need to import the necessary packages here too
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

#!/usr/bin/env python
# coding: utf-8

# In[1]:


def print_metrics(y_tr, tr_pred, y_te, te_pred, log=False):
    '''
    Prints the R2 Score, Mean Absolute Error and Root Mean Squared Error
    Will unlog to get MAE & RMSE in terms of the original target if log=True
    
    Inputs:
        y_tr: array-like or pandas series
            Actual target values for the train set
        tr_pred: array-like or pandas series
            Predicted target values for the train set
        y_te: array-like or pandas series
            Actual target values for the test set
        te_pred: array-like or pandas series
            Predicted target values for the test set
        log: boolean
            Toggles whether the target values have been logged or not
            If True, assumes all other arguments passed into the function have been logged
            
    Outputs:
        None, just prints the metrics
    '''
    # Unlogging all variables if you set log=True
    if log == True:
        # Please note - if you used log to log the variables, change this to exp
        y_tr_unlog = np.exp(y_tr)
        tr_pred_unlog = np.exp(tr_pred)
        y_te_unlog = np.exp(y_te)
        te_pred_unlog = np.exp(te_pred)
    
    # Printing train scores
    print("Training Scores")
    print("-"*10)
    print(f"R2: {r2_score(y_tr, tr_pred):.4f}") # R2 should not be done on unlogged values
    if log == True:
        print(f"RMSE: {mean_squared_error(y_tr_unlog, tr_pred_unlog, squared=False):.4f}")
        print(f"MAE: {mean_absolute_error(y_tr_unlog, tr_pred_unlog):.4f}")
    else:
        print(f"RMSE: {mean_squared_error(y_tr, tr_pred, squared=False):.4f}")
        print(f"MAE: {mean_absolute_error(y_tr, tr_pred):.4f}")
    
    print("\n"+"*"*10)
    
    # Printing test scores
    print("Testing Scores")
    print("-"*10)
    print(f"R2: {r2_score(y_te, te_pred):.4f}") # R2 should not be done on unlogged values
    if log == True:
        print(f"RMSE: {mean_squared_error(y_te_unlog, te_pred_unlog, squared=False):.4f}")
        print(f"MAE: {mean_absolute_error(y_te_unlog, te_pred_unlog):.4f}")
    else:
        print(f"RMSE: {mean_squared_error(y_te, te_pred, squared=False):.4f}")
        print(f"MAE: {mean_absolute_error(y_te, te_pred):.4f}")
