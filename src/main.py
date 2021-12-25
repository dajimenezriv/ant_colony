from acs import AntColonySystem

from os import listdir
from contextlib import contextmanager
import threading
import _thread
import numpy as np
import itertools
import random


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

def test(data, output, conf):
    for file in listdir(data):
        input_file = f'{data}{file}'
        output_file = f'{output}{file}'
        start(input_file, output_file, conf)


def start(input_file, output_file, conf):
    n_ants, alpha, beta, p = conf
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

def tuning(input_dir, conf, option=1, n=30, gamma=0.04, time=300, iter=50):
    if option == 1:
        confs = get_random_confs(n)
    else:
        confs = get_param_options(n, conf, gamma)

    scores = [0] * n

    for file in listdir(input_dir):
        input_file = f'{input_dir}{file}'

        weights = []
        for i, conf in enumerate(confs):
            n_ants, alpha, beta, p = conf
            n_ants = int(round(n_ants, 0))

            print(f'############# {input_file} | conf {i} #############')
            print(f'n_ants={n_ants}')
            print(f'alpha={alpha}')
            print(f'beta={beta}')
            print(f'p={p}')
            print(f'iter={iter}')

            acs = AntColonySystem(input_file, n_ants, alpha, beta, p, iter, verbose=1)
            with time_limit(time):
                acs.execute()
                
            weights.append(acs.best_weight)
        
        # for the min_weight we give it the highest score (n)
        # for the max_weight we give it the lowest score (1)
        file_scores = n - np.array(np.argsort(weights))
        scores = [a+b for a, b in zip(scores, file_scores)]
    
    print(f'scores: {scores}')
    best_conf = confs[np.argmax(scores)]
    print(f'best_conf: {best_conf}')

    return best_conf


def get_random_confs(n):
    '''
    Generate all possible confs and pick n random confs
    '''

    # create all possible confs
    n_ants = list(range(20, 100, 10))
    alpha = [x / 10.0 for x in range(1, 10)]
    beta = [x / 10.0 for x in range(1, 10)]
    p = [x / 10.0 for x in range(1, 10)]
    params = [n_ants, alpha, beta, p]
    all_confs = list(itertools.product(*params))
    print(f'n_ants: {len(n_ants)}. alpha: {len(alpha)}. beta: {len(beta)}. p: {len(p)}.')
    print(f'all_confs: {len(all_confs)}')

    # select n random confs
    random.seed(0)
    n_confs = random.sample(range(0, len(all_confs)), n)
    confs = list(map(all_confs.__getitem__, n_confs))
    for conf in confs:
        print(conf)
    return confs


def get_param_options(n, conf, gamma):
    '''
    Generate all confs from small variations of a previous conf
    Select n random confs
    '''
    
    params = []
    for value in conf:
        # param possibilities
        values = [
            value - 2 * gamma * value, 
            value - gamma * value, 
            value, 
            value + gamma * value, 
            value + 2 * gamma * value]

        params.append(values)

    all_confs = list(itertools.product(*params))
    print(f'confs: {len(all_confs)}')

    # select n random confs
    random.seed(0)
    n_confs = random.sample(range(0, len(all_confs)), n)
    confs = list(map(all_confs.__getitem__, n_confs))
    for conf in confs:
        print(conf)
    return confs


########
# MAIN #
########


if __name__ == '__main__':
    tuning_dir = 'tune_test_set/'
    input_dir = 'data/'
    output_dir = 'out/'

    best_conf = (70, 0.6837113997747161, 0.5845333163118296, 0.4998281808780217)

    file = '0011.txt'
    #test(input_dir, output_dir, best_conf)
    #start(f'{input_dir}{file}', f'{output_dir}{file}', best_conf)
    #tuning(input_dir, best_conf, option=1)

    file = 'inst01.txt'
    tuning(tuning_dir, best_conf, option=2)
    