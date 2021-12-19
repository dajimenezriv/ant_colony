
from acs import AntColonySystem

import numpy as np
from os import listdir
from contextlib import contextmanager
import threading
import _thread


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

    acs = AntColonySystem(input_file, n_ants, alpha, beta, p, iter)
    acs.execute()
    acs.output(output_file)

    print('#######################')


def training(input_file, output_file, params, timeout=True):
    best_weight = np.inf
    best_n_ants = None
    best_alpha = None
    best_beta = None
    best_p = None
    best_iter = None

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

                        acs = AntColonySystem(input_file, n_ants, alpha, beta, p, iter)
                        if timeout:
                            with time_limit(300):
                                acs.execute()
                        else:
                            acs.execute()
                        weight = acs.best_weight
                        params[f'{n_ants}-{alpha}-{beta}-{p}-{iter}'] = weight
                        if weight < best_weight:
                            best_weight = weight
                            best_n_ants = n_ants
                            best_alpha = alpha
                            best_beta = beta
                            best_p = p
                            best_iter = iter

                        print('#######################')

    best_params = {k: v for k, v in sorted(params.items(), key=lambda item: item[1])}   

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

    print(f'############# BEST PARAMS FOR {input_file} #############')
    print(f'best_weight={best_weight}')
    print(f'best_n_ants={best_n_ants}')
    print(f'best_alpha={best_alpha}')
    print(f'best_beta={best_beta}')
    print(f'best_p={best_p}')
    print(f'best_iter={best_iter}')

    acs = AntColonySystem(input_file, best_n_ants, best_alpha, best_beta, best_p, best_iter)
    print(f'new_weight={acs.execute()}')
    acs.output(output_file)

    print('#######################')


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
