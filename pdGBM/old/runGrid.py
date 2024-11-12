import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import pickle
from time import time

dTrain = []
counter = 0


def xgbcv(p):
    global counter
    counter += 1
    tic = time()
    params = {'max_depth': p['max_depth'], 'eta': p['eta'],
              'n_estimators': p['n_estimators'], 'num_boost_round': p['num_boost_round'],
              'silent': True, 'objective': 'reg:linear', 'gamma': 0, 'min_child_weight': 1,
              'max_delta_step': 0, 'subsample': 1, 'colsample_bytree': 1, 'colsample_bylevel': 1, 'reg_alpha': 0,
              'reg_lambda': 1, 'scale_pos_weight': 1, 'base_score': 0.5, 'seed': 0, 'missing': None}
    # 'nthread': -1,
    ret = xgb.cv(params, dTrain, nfold=10, metrics='rmse', seed=0,
                 # callbacks=[xgb.callback.early_stop(3)])
                 callbacks=[xgb.callback.print_evaluation(show_stdv=True), xgb.callback.early_stop(3)],
                 feval=evalerror)

    testErrorMean = ret['test-rmse-mean'].mean()
    print('RMSE: ' + str(testErrorMean) + ' count: ' + str(counter) + ' time(s): ' + str(time() - tic))
    return {'loss': testErrorMean, 'count': counter, 'time': time() - tic, 'status': STATUS_OK}


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)


def run_trials(model_space, trials_step, max_trials):
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open(fileName + '-trials.pkl', "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn=xgbcv, space=model_space, algo=tpe.suggest, max_evals=max_trials, trials=trials)
    print('best:', best)
    print('best trial error/count:', trials.best_trial['result']['loss'], ' / ',
          trials.best_trial['result']['count'])
    # for trial in trials.trials:
    #     print(trial)
    pickle.dump(trials, open(fileName + '-trials.pkl', 'wb'))
    pickle.dump(best, open(fileName + '-best.pkl', 'wb'))
    pickle.dump(model_space, open(fileName + '-space.pkl', 'wb'))


if __name__ == '__main__':
    fileName = 'spy100k_1'
    # dTrain = xgb.DMatrix('../IB4m/' + fileName + '.data')
    # dTrain.save_binary('../IB4m/' + fileName + '.buffer')
    dTrain = xgb.DMatrix('../IB4m/' + fileName + '.buffer')

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    model_space = {
        'max_depth': hp.choice('max_depth', [1, 2, 3, 5, 10]),  # 3-10
        'n_estimators': hp.choice('n_estimators', [10]),
        # 'min_child_weight': 1 # Used to control over-fitting; too high values can lead to under-fitting.
        'eta': hp.choice('eta', [0.6]),  # 0.01-0.2 default 0.3
        'num_boost_round': hp.choice('num_boost_round', [20])
    }

    while True:
        run_trials(model_space, trials_step, max_trials)