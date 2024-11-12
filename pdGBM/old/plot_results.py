import numpy as np
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle

fileName = 'spy100k_1'
trials = pickle.load(open(fileName + '-trials.pkl', 'rb'))
#best = pickle.load(open(fileName + '-best.pkl', 'rb'))
#model_space = pickle.load(open(fileName + '-space.pkl', 'rb'))

parameters = ['max_depth', 'n_estimators', 'eta', 'num_boost_round']
cols = len(parameters)
f, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    xs, ys = zip(*sorted(zip(xs, ys)))
    ys = np.array(ys)
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.5, c=cmap(float(i) / len(parameters)))
    axes[i].set_title(val)
    # axes[i].set_ylim([0.9,1.0])
f.savefig(fileName + '-grid.png')
plt.show()
input("Press Enter to continue...")

# run the prediction performance for the chosen parameters on the training/val set and explore feature importance

# Choose a relatively high learning rate.
# Generally a learning rate of 0.1 works but somewhere between 0.05 to 0.3 should work for different problems.
# Determine the optimum number of trees for this learning rate.
# XGBoost has a very useful function called as “cv” which performs cross-validation at each boosting iteration and thus
# returns the optimum number of trees required.
# Tune tree-specific parameters ( max_depth, min_child_weight, gamma, subsample, colsample_bytree) for decided learning
# rate and number of trees. Note that we can choose different parameters to define a tree and I’ll take up an example here.
# Tune regularization parameters (lambda, alpha) for xgboost which can help reduce model complexity and enhance performance.
# Lower the learning rate and decide the optimal parameters .




# xgb.plot_importance(model)
    # xgb.plot_tree(bst, num_trees=2)
    # xgb.to_graphviz(bst, num_trees=2)
    # plt.show()
    #
    # # grid search
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    # param_grid = dict(learning_rate=learning_rate)
    # # n_estimators = [100, 200, 300, 400, 500]
    # # max_depth = [2, 4, 6, 8]
    # # subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    # # colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    # # param_grid = dict(subsample=subsample)
    # # param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
    # # param_grid = dict(colsample_bytree=colsample_bytree)
    # # param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    #
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    # grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=1, cv=kfold, verbose=1, refit=True)
    # grid_result = grid_search.fit(X, Y)
    #
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))
    #
    # # plot feature importance
    # fig = plt.figure(figsize=(10, 8))
    # sub1 = fig.add_subplot(211)
    # xgb.plot_importance(grid_result.best_estimator_, ax=sub1)
    #
    # # plot - if there was a single param_grid of learning rate
    # # plt.figure(2)
    # # plt.errorbar(learning_rate, means, yerr=stds)
    # # plt.title("XGBoost learning_rate vs Log Loss")
    # # plt.xlabel('learning_rate')
    # # plt.ylabel('Log Loss')
    # # plt.savefig('learning_rate.png')
    #
    # # plot learn rate with num estimators
    #
    # # lengths = [len(v) for v in grid_result.param_grid.values()]
    # # means2 = np.reshape(means, lengths)
    # #
    # # sub2 = fig.add_subplot(212)
    # # scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
    # # for i, value in enumerate(learning_rate):
    # #     sub2.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    # # sub2.legend()
    # # sub2.set_xlabel('n_estimators')
    # # sub2.set_ylabel('Log Loss')
    # # # plt.savefig('n_estimators_vs_learning_rate.png')
    #
    # plt.show()