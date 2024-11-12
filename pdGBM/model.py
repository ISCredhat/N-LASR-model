# import os
# os.chdir('pdGBM')

from pdGBM.get_config import base_path
from numpy import absolute
import xgboost as xgb

from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
import graphviz

features = pd.read_pickle(base_path + 'dataFrames/features')
targets = pd.read_pickle(base_path + 'dataFrames/targets')['60']

start_date = '2016-11-01'
end_date = '2016-12-01'
X = features.loc[start_date: end_date]
y = targets.loc[start_date: end_date]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

# data = X_train
# labels = pd.DataFrame(features.columns)
# dtrain = xgb.DMatrix(data, label=labels)

param = {
    'verbosity': 2,  # [default=1]
    # Verbosity of printing messages. Valid values are 0 (silent), 1 (warning), 2 (info), 3 (debug).
    # Sometimes XGBoost tries to change configurations based on heuristics, which is displayed as warning message.
    # If there’s unexpected behaviour, please try to increase value of verbosity.
    'nthread': 7,  # [default to maximum number of threads available if not set]
    # Number of parallel threads used to run XGBoost. When choosing it,
    # please keep thread contention and hyperthreading in mind.
    'learning_rate': 0.1,  # [default=0.3, alias: learning_rate]
    # Step size shrinkage used in update to prevent overfitting.
    # After each boosting step, we can directly get the weights of new features, and 'eta' shrinks the feature
    # weights to make the boosting process more conservative.
    'gamma': 0,  # [default=0, alias: min_split_loss]
    # Minimum loss reduction required to make a further partition on a leaf node of the tree.
    # The larger 'gamma' is, the more conservative the algorithm will be.
    # Note that a tree where no splits were made might still contain a single terminal node with a non-zero score.
    'max_depth': 5,  # [default=6]
    # Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    # 0 indicates no limit on depth. Beware that XGBoost aggressively consumes memory when training a deep tree.
    # 'exact' tree method requires non-zero value.
    'min_child_weight': 1,  # [default=1]
    # Minimum sum of instance weight (hessian) needed in a child.
    # If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight,
    # then the building process will give up further partitioning. In linear regression task,
    # this simply corresponds to minimum number of instances needed to be in each node.
    # The larger min_child_weight is, the more conservative the algorithm will be.
    'max_delta_step': 0,  # [default=0]
    # Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint.
    # If it is set to a positive value, it can help making the update step more conservative.
    # Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    # Set it to value of 1-10 might help control the update.
    'subsample': 0.7,  # [default=1]
    # Subsample ratio of the training instances.
    # Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees.
    # and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
    'scale_pos_weight': 1,  # [default=1]
    # Control the balance of positive and negative weights, useful for unbalanced classes.
    # A typical value to consider: sum(negative instances) / sum(positive instances). See Parameters Tuning for more discussion. Also, see Higgs Kaggle competition demo for examples: R, py1, py2, py3.
    'sampling_method': 'uniform',  # [default= uniform]
    # The method to use to sample the training instances.
    # - uniform: each training instance has an equal probability of being selected. 
    #   Typically, set subsample >= 0.5 for good results.
    # - gradient_based: the selection probability for each training instance is proportional to the regularized 
    #   absolute value of gradients (more specifically, ). subsample may be set to as low as 0.1 without loss of 
    #   model accuracy. Note that this sampling method is only supported when tree_method is set to hist and the 
    #   device is cuda; other tree methods only support uniform sampling.
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
    # This is a family of parameters for subsampling of columns.
    # All colsample_by* parameters have a range of (0, 1], the default value of 1, and specify the fraction of
    # columns to be subsampled.
    # - colsample_bytree is the subsample ratio of columns when constructing each tree.
    #   Subsampling occurs once for every tree constructed.
    # - colsample_bylevel is the subsample ratio of columns for each level.
    #   Subsampling occurs once for every new depth level reached in a tree. Columns are subsampled from the set of columns chosen for the current tree.
    # - colsample_bynode is the subsample ratio of columns for each node (split).
    #   Subsampling occurs once every time a new split is evaluated. Columns are subsampled from the set of
    #   columns chosen for the current level. This is not supported by the exact tree method.
    # - colsample_by* parameters work cumulatively.
    #   For instance, the combination {'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}
    #   with 64 features will leave 8 features to choose from at each split.
    'lambda': 1,  # [default=1, alias: reg_lambda]
    # L2 regularization term on weights. Increasing this value will make model more conservative.
    # Using the Python or the R package, one can set the feature_weights for DMatrix to define the probability
    # of each feature being selected when using column sampling. There’s a similar parameter for fit method in
    # sklearn interface.
    'alpha': 0,  # [default=0, alias: reg_alpha]
    # L1 regularization term on weights. Increasing this value will make model more conservative.
    'tree_method': 'auto',  # string [default= auto]
    # The tree construction algorithm used in XGBoost. See description in the reference paper and Tree Methods.
    # Choices: auto, exact, approx, hist, this is a combination of commonly used updaters. For other updaters like refresh, set the parameter updater directly.
    # - auto: Same as the hist tree method.
    # - exact: Exact greedy algorithm. Enumerates all split candidates.
    # - approx: Approximate greedy algorithm using quantile sketch and gradient histogram.
    # - hist: Faster histogram optimized approximate greedy algorithm.
    'objective': 'reg:squarederror',  # [default=reg:squarederror]
    # - reg: squarederror: regression with squared loss.
    # - reg:squaredlogerror: regression with squared log loss
    # - reg:logistic: logistic regression, output probability
    # - reg:pseudohubererror: regression with Pseudo Huber loss, a twice differentiable alternative to absolute loss.
    # - reg:absoluteerror: Regression with L1 error. When tree model is used, leaf value is refreshed after
    #   tree construction. If used in distributed training, the leaf value is calculated as the mean value from all
    #   workers, which is not guaranteed to be optimal.
    # - reg:quantileerror: Quantile loss, also known as pinball loss. See later sections for its parameter and
    #   Quantile Regression for a worked example.
    # and others!
    'base_score': 0.5,
    # The initial prediction score of all instances, global bias
    # The parameter is automatically estimated for selected objectives before training.
    # To disable the estimation, specify a real number argument.
    # If base_margin is supplied, base_score will not be added.
    # For sufficient number of iterations, changing this value will not have too much effect.
    'seed': 0,  # [default=0]
    # Random number seed. In the R package, if not specified, instead of defaulting to seed ‘zero’,
    # will take a random seed through R’s own RNG engine.
}

# param['nthread'] = 4
# param['eval_metric'] = 'auc'

# You can also specify multiple eval metrics:
# param['eval_metric'] = ['auc', 'ams@0']

d_train = xgb.DMatrix(X_train, label=y_train)
d_test = xgb.DMatrix(X_test, label=y_test)

# specify validations set to watch performance
watchlist = [(d_train, "train"), (d_test, "eval")]

# number of boosting rounds
num_round = 2
bst = xgb.train(param, d_train, num_boost_round=num_round, evals=watchlist)

# num_round = 1000
# bst = xgb.train(param, D_train, num_round, evals=evallist)


xgb.plot_importance(bst)

xgb.plot_tree(bst, num_trees=2)
plt.show()

# dump model
bst.dump_model('dump.raw.txt')
# dump model with feature map
bst.dump_model('dump.raw.txt', 'featmap.txt')

# A saved model can be loaded as follows:
bst = xgb.Booster({'nthread': 7})  # init model
bst.load_model('model.bin')  # load model data

# ret = xgb.cv(params, dTrain, nfold=5, metrics={'rmse'}, seed=0,
#              callbacks=[xgb.callback.print_evaluation(show_stdv=True), xgb.callback.early_stop(3)])

# model = XGBRegressor(n_estimators=10, max_depth=3, eta=0.1, subsample=0.7, colsample_bytree=0.8)
m = XGBModel(params)
model = XGBRegressor(m)

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
t = time.time()
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print(f'Time taken: {time.time() - t:.6f} seconds')

# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()))

# fit model
# bst.fit(X_train, y_train)
# # make predictions
# preds = bst.predict(X_test)

# need to remove data from end of the day where the return would roll into the next day
# need to check that tech features or predictions are not using the current or past data
