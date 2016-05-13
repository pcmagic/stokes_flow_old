# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 22:45:14 2015

@author: Kedar


"""


# This function solves a system Ax = b using the congujate gradient method.
def CG_method(A, b):
    """
    This function solves a system Ax = b using the Conjugate Gradient method.
    The Conjugate Gradient method works best when A is symmetric and positive
    definite.
    
    
    Inputs: A, b

    Outputs: x, plot of convergence
    """

    import numpy as np
    import math
    import time
    from matplotlib import pylab

    # convergence criteria: 2-norm of the residual is less than
    eps = 1e-10

    # number of rows in b
    n = len(b)

    # intial guess for the solution vector, x[0]
    x = []
    x.append(np.zeros(n))

    # intital residual vector, r[0]
    r = []
    r.append(b - np.dot(A, x[0]))

    # list initializations
    rho = []  # starts at index 0
    p = [float('nan')]  # starts at index 1
    beta = []  # starts at index 0
    alpha = [float('nan')]  # starts at index 1
    r_norm = [np.linalg.norm(r[0])]  # starts at index 0

    # stopping criterion (maximum iterations)
    max_iter = 100

    # for plotting
    pylab.ion()  # turn on interactive mode first
    pylab.figure()

    for i in range(1, max_iter + 1):

        # magnitude squared of previous residual vector, rho[i-1]
        rho.append(np.dot(r[i - 1], r[i - 1]))

        # comptue the scalar improvment this step, beta[i-1], 
        # and the vector search direction, p[i]
        if i == 1:
            beta.append(float('nan'))  # for consistent indexing
            p.append(r[0])
        else:
            beta.append(rho[i - 1] / rho[i - 2])
            p.append(r[i - 1] + beta[i - 1] * p[i - 1])

        # define vector shorthand term q_i
        q_i = np.dot(A, p[i])

        # define scalar step length alpha[i]
        alpha.append(rho[i - 1] / np.dot(p[i], q_i))

        # update the solution vector, x[i]
        x.append(x[i - 1] + np.dot(alpha[i], p[i]))

        # update the residual vector, r[i]
        r.append(r[i - 1] - np.dot(alpha[i], q_i))

        # compute the 2-norm of the new residual vector, r[i]
        r_norm.append(np.linalg.norm(r[i]))

        # compute the orders of magnitude the residual has fallen
        orders_fallen = math.log10(r_norm[0]) - math.log10(r_norm[i])

        # print the progress to the screen
        print
        "( iteration:", i, ") ||r|| = %.10f (%.2f orders of magnitude)" \
        % (r_norm[i], orders_fallen)

        # plot the convergence to the screen
        pylab.semilogy(range(i + 1), r_norm, 'ko-')
        # ax = pylab.gca()
        # ax.set_aspect('equal')
        pylab.rc('text', usetex=True)  # for using latex
        pylab.rc('font', family='serif')  # setting font
        pylab.xlabel('iteration')
        pylab.ylabel(r'$\|r\|$')
        pylab.title('Conjugate Gradient Method')
        pylab.draw()
        time.sleep(.01)

        # check for convergence
        if r_norm[i] < eps:
            break
        else:

            if i == max_iter:
                print
                "The problem has not converged."
                print
                "The maximum number of iterations has been reached."
                print
                "If the problem appears to be converging, consider \
                               increasing the maximum number of iterations in line 52 \
                               of iterative_methods.py"
            continue

    # pylab interactive mode off (this keeps the plot from closing)
    pylab.ioff()
    pylab.show()

    return x[i]


# This function returns the transpose of a matrix when given a list of lists
def transpose(A):
    """
    This function returns the transpose of a given matrix A.
    
    Input: A
    
    Output: transpose of A
    """

    # recover the matrix dimensions
    n_rows = len(A)
    n_columns = len(A[0])

    # initialize space for the transpose matrix
    A_transpose = [[float('nan')] * n_rows for k in range(n_columns)]

    # iterate through the rows and columns
    for i in range(n_rows):
        for j in range(n_columns):
            A_transpose[j][i] = A[i][j]

    # return the transposed matrix
    return A_transpose


# This function solves a system Ax = b using the bicongujate gradient method.
def BCG_method(A, b):
    """
    This function solves a system Ax = b using the BiConjugate Gradient method.
    The BiConjugate Gradient method works for nonsymmetric matrices A. It does
    this by replacing the orthogonal sequence of residuals (produced during 
    the standard Conjugate Gradient method) with two mutually orthogonal 
    sequences, at the price of no longer providing a minimization. For 
    symmetric, positive definite systems the method delivers the same results 
    as the Conjugate Gradient method, but at twice the cost per iteration.
    
    
    Inputs: A, b

    Outputs: x, plot of convergence
    """

    import numpy as np
    import math
    import time
    from matplotlib import pylab

    # convergence criteria: 2-norm of the residual is less than
    eps = 1e-10

    # number of rows in b
    n = len(b)

    # intial guess for the solution vector, x[0]
    x = []
    x.append(np.zeros(n))

    # intital bi-orthogonal residual vectors, r[0] and r_tilde[0]
    r = []
    r_tilde = []
    r.append(b - np.dot(A, x[0]))
    r_tilde.append(r[0])  # r_tilde[0] = r[0]

    # list initializations
    rho = []  # starts at index 0
    p = [float('nan')]  # starts at index 1
    p_tilde = [float('nan')]  # starts at index 1
    beta = []  # starts at index 0
    alpha = [float('nan')]  # starts at index 1
    r_norm = [np.linalg.norm(r[0])]  # starts at index 0

    # print the starting residual norm to the screen
    print
    "\n\t Solution Computed Using BiConjugate Gradient Method \n"
    print
    "||r_0|| = ", r_norm[0]

    # stopping criterion (maximum iterations)
    max_iter = 100

    # for plotting
    pylab.ion()  # turn on interactive mode first
    pylab.figure()

    for i in range(1, max_iter + 1):

        # dot the two previous residuals vector, rho[i-1]
        rho.append(np.dot(r[i - 1], r_tilde[i - 1]))

        # make sure this dot product is not equal to zero
        if rho[i - 1] == 0:
            print
            "The Biconjugate Gradient method is quitting in order to \
                             prevent a divide-by-zero error"
            import sys
            sys.exit()

        # comptue the scalar improvment this step, beta[i-1], 
        # and the vector search directions, p[i] and p_tilde[0]
        if i == 1:
            beta.append(float('nan'))  # for consistent indexing
            p.append(r[0])
            p_tilde.append(r_tilde[0])
        else:
            beta.append(rho[i - 1] / rho[i - 2])
            p.append(r[i - 1] + beta[i - 1] * p[i - 1])
            p_tilde.append(r_tilde[i - 1] + beta[i - 1] * p_tilde[i - 1])

        # define vector shorthand terms q_i and q_tilde_i
        q_i = np.dot(A, p[i])
        q_tilde_i = np.dot(transpose(A), p_tilde[i])

        # define scalar step length alpha[i]
        alpha.append(rho[i - 1] / np.dot(p_tilde[i], q_i))

        # update the solution vector, x[i]
        x.append(x[i - 1] + np.dot(alpha[i], p[i]))

        # update the two residual vectors, r[i] and r_tilde[i]
        r.append(r[i - 1] - np.dot(alpha[i], q_i))
        r_tilde.append(r_tilde[i - 1] - np.dot(alpha[i], q_tilde_i))

        # compute the 2-norm of the new residual vector, r[i]
        r_norm.append(np.linalg.norm(r[i]))

        # compute the orders of magnitude the residual has fallen
        orders_fallen = math.log10(r_norm[0]) - math.log10(r_norm[i])

        # print the progress to the screen
        print
        "( iteration:", i, ") ||r|| = %.10f (%.2f orders of magnitude)" \
        % (r_norm[i], orders_fallen)

        # plot the convergence to the screen
        pylab.semilogy(range(i + 1), r_norm, 'ko-')
        # ax = pylab.gca()
        # ax.set_aspect('equal')
        pylab.rc('text', usetex=True)  # for using latex
        pylab.rc('font', family='serif')  # setting font
        pylab.xlabel('iteration')
        pylab.ylabel(r'$\|r\|$')
        pylab.title('BiConjugate Gradient Method')
        pylab.draw()
        time.sleep(.01)

        # check for convergence
        if r_norm[i] < eps:

            # print the solution to the screen
            print
            "\n BiConjugate Gradient Method has converged."
            print
            "  -No. of iterations: ", i
            print
            "  -Solution: x = ", x[i]

            break

        else:

            if i == max_iter:
                print
                "The problem has not converged."
                print
                "The maximum number of iterations has been reached."
                print
                "If the problem appears to be converging, consider \
                               increasing the maximum number of iterations in line 209 \
                               of iterative_methods.py"
            continue

    # pylab interactive mode off (this keeps the plot from closing)
    pylab.ioff()
    pylab.show()

    return x[i]
