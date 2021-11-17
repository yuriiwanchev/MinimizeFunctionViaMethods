from flask import Flask, render_template, request, url_for, flash, redirect
import glob

import random
import math
import time

from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['SECRET_KEY'] = '1234567'

RESULT_NAME = dict()


@app.route('/', methods=('GET', 'POST'))
def index():
    """
    Start page to generate machines with configurations or start creating the solution
    """

    if request.method == 'POST':
        if "solution" in request.form:
            f = open("temp.txt", "w")
            f.write(request.form['input_n'] + '\n')
            f.write(request.form['input_c'] + '\n')
            f.write(request.form['input_k'])
            f.close()

            n = int(request.form['input_n'])
            k = int(request.form['input_k'])
            c = []
            for item in request.form['input_c'].split(" "):
                c.append(float(item))

            options = {'maxiter_da': 100,
                       'maxiter_brute': 10,
                       'maxiter_min': 100}
            res = get_optimum(n, c, k, options)

            return render_template('result.html', res=res, options=options)
        elif "restart" in request.form:
            f = open("temp.txt", "r")
            n = int(f.readline())
            c = []
            for item in f.readline().split(" "):
                c.append(float(item))
            k = int(f.readline())
            f.close()

            options = {'maxiter_da': int(request.form['input_n_da']),
                       'maxiter_brute': int(request.form['input_n_b']),
                       'maxiter_min': int(request.form['input_n_m'])}
            res = get_optimum(n, c, k, options)

            return render_template('result.html', res=res, options=options)
        elif "plot" in request.form:
            f = open("temp.txt", "r")
            n = int(f.readline())
            c = []
            for item in f.readline().split(" "):
                c.append(float(item))
            k = int(f.readline())
            f.close()

            options = {'maxiter_da': int(request.form['input_n_da']),
                       'maxiter_brute': int(request.form['input_n_b']),
                       'maxiter_min': int(request.form['input_n_m'])}
            res = get_optimum(n, c, k, options)

            if n == 1:
                t1 = np.arange(c[0]-25.0/k, c[0]+25.0/k, 0.1)
                tt = (n, c[0], k)

                plt.figure()
                plt.subplot(111)
                plt.grid(True)
                plt.plot(t1, f_x_1(t1, *tt), 'k')
                plt.show()
            elif n == 2:
                tt = (n, c, k)

                # define range for input
                r_min, r_max = -25.0/k, 25.0/k
                # sample input range uniformly at 0.1 increments
                xaxis = np.arange(c[0] + r_min, c[0] + r_max, 0.1)
                yaxis = np.arange(c[1] + r_min, c[1] + r_max, 0.1)
                # create a mesh from the axis
                x, y = np.meshgrid(xaxis, yaxis)
                # compute targets
                results = f_x_2(x, y, *tt)
                # create a surface plot with the jet color scheme
                figure = plt.figure()
                axis = figure.gca(projection='3d')
                axis.plot_surface(x, y, results, cmap='jet')
                # show the plot
                plt.show()
            return render_template('result.html', res=res, options=options)

    return render_template('index.html')


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


def f_x_1(x, *args) -> float:
    n, c, k = args
    r = (x - c) ** 2
    return 0.1 * r - np.cos(k * (r ** (1 / 2)))

def f_x_2(x, y, *args) -> float:
    n, c, k = args
    r = ((x - c[0]) ** 2) + ((y - c[1]) ** 2)
    return 0.1 * r - np.cos(k * (r ** (1 / 2)))


def get_optimum(n, c, k, options):
    params = (n, c, k)

    r_min, r_max = -25.0/k, 25.0/k
    # define the bounds on the search

    bounds = []
    x_start = []
    for i in range(n):
        bounds.append([r_min, r_max])
        x_start.append(random.uniform(r_min, r_max))
    x0 = np.array(x_start)

    start_time = time.process_time_ns()
    res = optimize.dual_annealing(func=f_ot_x, bounds=bounds, args=params, x0=x0, maxiter=options['maxiter_da'],
                                  no_local_search=True, visit=2.9)
    t_da = (time.process_time_ns() - start_time) / 10 ** 6
    print("--- %s milliseconds ---" % t_da)
    print(res['x'])
    print(res['fun'])
    print(res['message'])

    start_time_b = time.process_time_ns()
    res_b = optimize.brute(func=f_ot_x, ranges=bounds, args=params, workers=4, Ns=options['maxiter_brute'])
    t_b = (time.process_time_ns() - start_time_b) / 10 ** 6
    print("--- %s milliseconds ---" % t_b)
    print(res_b)
    print(f_ot_x(res_b, *params))

    opt = {'maxiter': options['maxiter_min'], 'adaptive': n > 2}
    start_time_m = time.process_time_ns()
    res_m = optimize.minimize(fun=f_ot_x, args=params, x0=x0, bounds=bounds, method='Nelder-Mead', options=opt)
    t_m = (time.process_time_ns() - start_time_m) / 10 ** 6
    print("--- %s milliseconds ---" % t_m)
    print(res_m['x'])
    print(f_ot_x(res_m['x'], *params))

    times = {'da': t_da,
             'brute': t_b,
             'min': t_m}

    return {'res_da': res,
            'res_brute': res_b,
            'res_brute_f': f_ot_x(res_b, *params),
            'res_m': res_m,
            'times': times}


if __name__ == '__main__':
    app.run()