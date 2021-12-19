
from acs import AntColonySystem

import numpy as np
from os import listdir
from contextlib import contextmanager
import threading
import _thread


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
    n_ants = 20
    alpha = 0.45
    beta = 0.65
    p = 0.7
    iter = 100

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

def training(input_file, params, timeout=True):
    params = {}

    for n_ants in [20]:
        for alpha in [0.3, 0.5, 0.7]:
            for beta in [0.3, 0.5, 0.7]:
                for p in [0.6, 0.7, 0.8, 0.9]:
                    for iter in [100]:
                        print('#######################')
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
                        weight = acs.best_weight
                        params[f'{n_ants}-{alpha}-{beta}-{p}-{iter}'] = weight

                        print('#######################')

    best_params = {k: v for k, v in sorted(params.items(), key=lambda item: item[1])}   

    # output the params of all iterations sorted by weight
    with open(f'{params}best_params-{input_file.split("/")[1]}', 'w') as f:
        f.write(f'############# BEST PARAMS FOR {input_file} #############\n')
        for k, v in best_params.items():
            f.write(f'WEIGHT: {v}\n')
            n_ants, alpha, beta, p, iter = k.split('-')
            f.write(f'n_ants={n_ants}\n')
            f.write(f'alpha={alpha}\n')
            f.write(f'beta={beta}\n')
            f.write(f'p={p}\n')
            f.write(f'iter={iter}\n\n')


if __name__ == '__main__':
    tuning = 'tune_test_set/'
    data = 'data/'
    output = 'out/'
    params = 'params/'

    file = '0024.txt'
    #test(data, output)
    #start(f'{data}{file}', f'{output}{file}')
    #training(f'{data}{file}', f'{output}{file}', params)

    file = 'inst01.txt'
    training(f'{tuning}{file}', f'{output}{file}', params, True)
