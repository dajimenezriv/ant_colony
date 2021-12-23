from acs import AntColonySystem

from os import listdir
from contextlib import contextmanager
import threading
import _thread
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys


#########
# TIMER #
#########

@contextmanager
def time_limit(seconds):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt as e:
        print(e)
        print('Time out')
    finally:
        timer.cancel()


##################
# TEST ALL FILES #
##################

def test(data, output):
    for file in listdir(data):
        input_file = f'{data}{file}'
        output_file = f'{output}{file}'
        start(input_file, output_file)


def start(input_file, output_file):
    n_ants = 70
    alpha = 0.6837113997747161
    beta = 0.5845333163118296
    p = 0.4998281808780217
    iter = 65

    print(f'############# {input_file} #############')
    print(f'n_ants={n_ants}')
    print(f'alpha={alpha}')
    print(f'beta={beta}')
    print(f'p={p}')
    print(f'iter={iter}')

    acs = AntColonySystem(input_file, n_ants, alpha, beta, p, iter, verbose=1)
    acs.execute()
    acs.output(output_file)

    print('#######################')


##########################
# TUNING HYPERPARAMETERS #
##########################

# [n_ants, alpha, beta, p]
def training(input_file, params_dir, params=[70, 0.65, 0.6, 0.5], iter=50, gamma=0.06, timeout=True):
    best_params=[]
    res = {}
    for pos, value in enumerate(params):
        possibilities = []

        # param possibilities
        values = [
            value - 2 * gamma * value, 
            value - gamma * value, 
            value, 
            value + gamma * value, 
            value + 2 * gamma * value]
        
        for value in values:
            possibilities.append(list(np.concatenate([params[:pos], [value], params[pos+1:]])))

        # execute acs
        param_res = []
        for possibility in possibilities:
            n_ants, alpha, beta, p = possibility
            n_ants = int(round(n_ants, 0))

            print(f'############# {input_file} #############')
            print(f'n_ants={n_ants}')
            print(f'alpha={alpha}')
            print(f'beta={beta}')
            print(f'p={p}')
            print(f'iter={iter}')

            acs = AntColonySystem(input_file, n_ants, alpha, beta, p, iter, verbose=1)
            if timeout:
                with time_limit(300):
                    acs.execute()
            else:
                acs.execute()

            param_res.append(acs.best_weight)
            res[f'{n_ants}-{alpha}-{beta}-{p}-{iter}'] = acs.best_weight

            print('#######################')

        # update params
        scaler = MinMaxScaler((-len(values), 0))
        m = -scaler.fit_transform(np.argsort(param_res).reshape(-1, 1))
        best_param = np.dot(values, m) / sum(m)
        best_params.append(best_param[0])

    best_res = dict(sorted(res.items(), key=lambda item: item[1]))

    # output the params of all iterations sorted by weight
    with open(f'{params_dir}best_params-{input_file.split("/")[1]}', 'w') as f:
        f.write(f'############# BEST PARAMS FOR {input_file} #############\n')
        for k, v in best_res.items():
            f.write(f'WEIGHT: {v}\n')
            n_ants, alpha, beta, p, iter = k.split('-')
            f.write(f'n_ants={n_ants}\n')
            f.write(f'alpha={alpha}\n')
            f.write(f'beta={beta}\n')
            f.write(f'p={p}\n')
            f.write(f'iter={iter}\n\n')

    # best params for that file
    print(f'\n############# BEST FARAMS FOR {input_file} #############')
    n_ants, alpha, beta, p = best_params
    print(best_params)
    print(f'n_ants={n_ants}')
    print(f'alpha={alpha}')
    print(f'beta={beta}')
    print(f'p={p}')
    print(f'iter={iter}\n')

    return best_params


if __name__ == '__main__':
    tuning_dir = 'tune_test_set/'
    input_dir = 'data/'
    output_dir = 'out/'
    params_dir = 'params/'

    file = '0024.txt'
    #test(input_dir, output_dir)
    #start(f'{input_dir}{file}', f'{output_dir}{file}')
    #training(f'{input_dir}{file}', params_dir)

    #sys.exit(0)

    file = 'inst01.txt'
    params=[70, 0.6837113997747161, 0.5845333163118296, 0.4998281808780217]
    for file in listdir(tuning_dir):
        params = training(f'{tuning_dir}{file}', params_dir, params)
