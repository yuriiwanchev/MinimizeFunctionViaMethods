import random
import math
import time

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


def f_ot_x(x_w, *args) -> float:
    """

    :param x_w:
    :return:
    """
    # n = int
    # c = list[float]
    # k = int
    n, c, k = args

    r = 0.0
    x = list[float]
    x = x_w
    for i in range(0, n):
        r = r + (x[i] - c[i]) ** 2
    return 0.1 * r - math.cos(k * (r ** (1 / 2)))

def f(x, *args) -> float:
    n, c, k = args
    r = (x - c) ** 2
    return 0.1 * r - np.cos(k * (r ** (1 / 2)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    t1 = np.arange(-25.0, 25.0, 0.1)
    tt = (1, 0, 5)
    fff = f(t1, *tt)

    plt.figure()
    plt.subplot(111)
    plt.grid(True)
    plt.plot(t1, f(t1, *tt), 'k')
    plt.show()

    n = 2
    c = [5., 3.]
    k = 5

    params = (n, c, k)

    r_min, r_max = -100.0, 100.0
    # define the bounds on the search

    bounds = []
    x_start = []
    for i in range(n):
        bounds.append([r_min, r_max])
        x_start.append(random.uniform(r_min, r_max))
    x0 = np.array(x_start)

    opt = {'maxiter': 10000, 'adaptive': True}

    start_time = time.process_time_ns()
    res = optimize.dual_annealing(func=f_ot_x, bounds=bounds, args=params, x0=x0, maxiter=opt['maxiter'], no_local_search=True, visit=2.9)
    print("--- %s milliseconds ---" % ((time.process_time_ns() - start_time) / 10 ** 6))
    print(res['x'])
    print(res['fun'])

    start_time_b = time.process_time_ns()
    res_b = optimize.brute(func=f_ot_x, ranges=bounds, args=params, workers=4, Ns=100)
    print("--- %s milliseconds ---" % ((time.process_time_ns() - start_time_b) / 10 ** 6))
    print(res_b)
    print(f_ot_x(res_b, *params))

    opt = {'maxiter': 1000, 'adaptive': True}
    start_time_m = time.process_time_ns()
    res_m = optimize.minimize(fun=f_ot_x, args=params, x0=x0, bounds=bounds, method='Nelder-Mead', options=opt)
    print("--- %s milliseconds ---" % ((time.process_time_ns() - start_time_m) / 10 ** 6))
    print(res_m['x'])
    # print(f_ot_x(res_m['x'], *params))
    print(res_m['fun'])

