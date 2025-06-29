"""
Created on Mon Jun  9 23:01:19 2025

Code originally sourced from:
    https://github.com/pranshupant/2D_LES_CFD

Implementation is detailed in guo_joseph_pant_report.pdf

Original Authors:
    @pranshupant - Pranshu Pant
    @joejoseph007 - Joe Joseph
    @jguo4

Updated Code:
    @rhann-09 - Reuben Hann

Changes from Original Code:
    * Cleaned up code a bit
    * Conglomerated the original code into single script to be ran in Spyder IDE rather than command line
    * Edited dependancy on custom yaml module to inbuilt python yaml module for input file
    * Included automated example of writing/reading yaml file for reading inputs
    * Sped up I/O
    * Setup Sphinx Docstrings

Next Steps:
    * Include setting of Smagorinsky Constant & gravity in input yaml file
    * Parallelise Run Script using the module Joblib Parallel
    * Port over to C++ and/or Cuda

"""

import numpy as np
import time
import copy
import os
import shutil
import yaml
import io
from numba import jit
from joblib import Parallel, delayed


# %% I/O Functions


def write_points(fileout, Points):
    """
    Write grid points to a specified file.

    :param fileout: The path to the output file.
    :type fileout: str
    :param Points: A 3D NumPy array containing the x and y coordinates of the grid points.
                   Shape should be (2, nx, ny).
    :type Points: numpy.ndarray
    """
    d = 6
    nx = len(Points[0, :, 0])
    ny = len(Points[0, 0, :])
    file = open(fileout, 'w')
    file.write('[%i,%i]\n' % (nx, ny))
    for j in range(ny):
        for i in range(nx):
            file.write('[%i,%i]:%s,%s' % (i, j, round(Points[0, i, j], d), round(Points[1, i, j], d)))
            if j < ny and i < nx:
                file.write('\n')


def read_points(filein):
    """
    Read grid points from a specified file.

    :param filein: The path to the input file.
    :type filein: str
    :returns: A 3D NumPy array containing the x and y coordinates of the grid points.
    :rtype: numpy.ndarray
    """
    with open(filein) as tsvfile:
        iterant = tsvfile.read().strip().split('\n')
        n = iterant[0].strip('[').strip(']').split(',')
        nx = int(n[0])
        ny = int(n[1])
        Points_read = np.zeros([2, nx, ny])
        for row in iterant[1:]:
            row = row.split(':')
            ind = row[0].strip('[').strip(']').split(',')
            xy = row[1].strip('(').strip(')').split(',')
            i = int(ind[0])
            j = int(ind[1])
            Points_read[0, i, j] = float(xy[0])
            Points_read[1, i, j] = float(xy[1])
    return Points_read


def deltas_write(Points_read):
    """
    Calculate grid cell sizes (dx and dy) from given grid points.

    :param Points_read: A 3D NumPy array containing the x and y coordinates of the grid points.
                        Shape should be (2, nx, ny).
    :type Points_read: numpy.ndarray
    :returns: A tuple containing two 2D NumPy arrays: dx (cell sizes in x-direction) and dy (cell sizes in y-direction).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nx = len(Points_read[0, :, 0])
    ny = len(Points_read[0, 0, :])
    dx = np.zeros([nx, ny])
    dy = np.zeros([nx, ny])
    for i in range(nx - 1):
        for j in range(ny - 1):
            dx[i, j] = Points_read[0, i + 1, j] - Points_read[0, i, j]
            dy[i, j] = Points_read[1, i, j + 1] - Points_read[1, i, j]
    dx[0, :] = copy.deepcopy(dx[1, :])
    dx[-1, :] = copy.deepcopy(dx[-2, :])
    dx[:, 0] = copy.deepcopy(dx[:, 1])
    dx[:, -1] = copy.deepcopy(dx[:, -2])
    dy[:, 0] = copy.deepcopy(dy[:, 1])
    dy[:, -1] = copy.deepcopy(dy[:, -2])
    dy[0, :] = copy.deepcopy(dy[1, :])
    dy[-1, :] = copy.deepcopy(dy[-2, :])
    return dx, dy


def deltas_write_nonuni(nx, ny, xmax, ymax, dyf):
    """Calculate  the non-uniform grid cell sizes (dx and dy) using a Newton-Raphson method for expansion ratio in y-direction.

    :param nx: Number of grid points in the x-direction.
    :type nx: int
    :param ny: Number of grid points in the y-direction.
    :type ny: int
    :param xmax: Maximum x-coordinate of the domain.
    :type xmax: float
    :param ymax: Maximum y-coordinate of the domain.
    :type ymax: float
    :param dyf: Initial cell size in the y-direction.
    :type dyf: float
    :returns: A tuple containing two 2D NumPy arrays: dx (cell sizes in x-direction) and dy (cell sizes in y-direction).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    # Newton-Raphson method to find expansion ratio `e`
    def func(e):
        return dyf * e**ny - ymax * e + (ymax - dyf)

    def dfunc(e):
        return dyf * ny * e**(ny - 1) - ymax

    e = 10.0  # initial guess
    tol = 1e-6
    max_iter = 1000000
    for i in range(max_iter):
        f_val = func(e)
        df_val = dfunc(e)
        e_new = e - f_val / df_val
        if abs(e_new - e) < tol:
            e = e_new
            break
        e = e_new

    # Precompute dy only once per row
    j_indices = np.arange(ny)
    dy_row = dyf * e**j_indices

    dx = np.full((nx, ny), xmax / (nx - 1))
    dy = np.tile(dy_row, (nx, 1))

    return dx, dy


def write_scalar(fileout, Scalar):
    """
    Write a 2D scalar field to a specified file.

    :param fileout: The path to the output file.
    :type fileout: str
    :param Scalar: A 2D NumPy array representing the scalar field.
    :type Scalar: numpy.ndarray
    """
    d = 6
    nx = len(Scalar[:, 0])
    ny = len(Scalar[0, :])
    file = open(fileout, 'w')
    file.write('[%i,%i]\n' % (nx, ny))
    for j in range(ny):
        for i in range(nx):
            file.write('[%i,%i]:%s' % (i, j, round(Scalar[i, j], d)))
            if j < ny and i < nx:
                file.write('\n')


def read_scalar(filein, ch=0):
    """
    Read a 2D scalar field from a specified file.

    :param filein: The path to the input file.
    :type filein: str
    :param ch: Control parameter. If 0, returns a NumPy array. If 1, returns a dictionary.
               Defaults to 0.
    :type ch: int, optional
    :returns: A 2D NumPy array or a dictionary representing the scalar field.
    :rtype: numpy.ndarray or dict
    """
    with open(filein) as tsvfile:
        iterant = tsvfile.read().strip().split('\n')
        n = iterant[0].strip('[').strip(']').split(',')
        nx = int(n[0])
        ny = int(n[1])
        Scalar = np.zeros([nx, ny])
        for row in iterant[1:]:
            row = row.split(':')
            ind = row[0].strip('[').strip(']').split(',')
            xy = row[1].strip('(').strip(')').split(',')
            i = int(ind[0])
            j = int(ind[1])
            Scalar[i, j] = float(xy[0])
    if ch == 0:
        return Scalar
    elif ch == 1:
        Dict = {}
        for i in range(nx):
            for j in range(ny):
                Dict[(i, j)] = Scalar[i, j]
        return Dict


def write_all_scalar(P, T, U, V, phi, decimals, t='trash'):
    """
    Write all primary scalar fields (Pressure, Temperature, U-velocity, V-velocity, Pollutant)
    to separate files within a time-stamped directory.

    :param P: Pressure field (2D NumPy array).
    :type P: numpy.ndarray
    :param T: Temperature field (2D NumPy array).
    :type T: numpy.ndarray
    :param U: U-velocity field (2D NumPy array).
    :type U: numpy.ndarray
    :param V: V-velocity field (2D NumPy array).
    :type V: numpy.ndarray
    :param phi: Pollutant concentration field (2D NumPy array).
    :type phi: numpy.ndarray
    :param decimals: Number of decimal places for formatting the output.
    :type decimals: int
    :param t: Simulation time, used for creating the directory name. Defaults to 'trash'.
    :type t: float or str, optional
    """
    try:
        path_results = os.path.join(os.getcwd(), "Results", '%.6f' % t)

        os.makedirs(path_results)

        np.savetxt(os.path.join(path_results, 'P_tab.txt'), P, delimiter='\t', fmt=f'%.{decimals}f')
        np.savetxt(os.path.join(path_results, 'U_tab.txt'), U, delimiter='\t', fmt=f'%.{decimals}f')
        np.savetxt(os.path.join(path_results, 'V_tab.txt'), V, delimiter='\t', fmt=f'%.{decimals}f')
        np.savetxt(os.path.join(path_results, 'T_tab.txt'), T, delimiter='\t', fmt=f'%.{decimals}f')
        np.savetxt(os.path.join(path_results, 'phi_tab.txt'), phi, delimiter='\t', fmt=f'%.{decimals}f')

    finally:
        print('## WRITTEN : time=%.6f' % t)


def read_all_scalar(ch=0):
    """
    Read all primary scalar fields (Pressure, Temperature, U-velocity, V-velocity, Pollutant)
    from files within a specified time-step directory.

    :param ch: Time step directory name (e.g., '0' for initial conditions). Defaults to 0.
    :type ch: int or str, optional
    :returns: A list containing the loaded scalar fields in the order: P, T, U, V, phi.
    :rtype: list
    """
    files = os.listdir(os.path.join(os.getcwd(), "Results", str(ch)))
    files.sort()
    # print(files)
    X = []

    for paths in files:
        try:
            path = os.path.join(os.getcwd(), "Results", "0", paths)
            X.append(np.loadtxt(path))
        finally:
            continue
    return X


def read_delta(ch=0):
    """
    Read the Dx and Dy grid cell size files from the Constants directory.

    :param ch: Control parameter for read_scalar function. Defaults to 0.
    :type ch: int, optional
    :returns: A tuple containing two 2D NumPy arrays: Dx and Dy.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    path_constant = os.path.join(os.getcwd(), "Constants")
    Dx = read_scalar(os.path.join(path_constant, 'Dx.txt'), ch)
    Dy = read_scalar(os.path.join(path_constant, 'Dy.txt'), ch)
    return Dx, Dy


# %% Initialise Functions

def isfloat(value):
    """
    Checks if a given value can be converted to a float.

    :param value: The value to check.
    :type value: any
    :returns: True if the value can be converted to a float, False otherwise.
    :rtype: bool
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def initialise(config_file):
    """
    Initialises the simulation environment by setting up directories, reading configuration,
    generating grid points, and writing initial scalar fields.

    :param config_file: The path to the YAML configuration file.
    :type config_file: str
    """
    path_results = os.path.join(os.getcwd(), "Results")
    path_zero = os.path.join(path_results, "0")
    path_constant = os.path.join(os.getcwd(), "Constants")

    # Read yaml file
    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    constants_dict = data["constants"]
    N = constants_dict["N"]
    M = constants_dict["M"]
    xmax = constants_dict["xmax"]
    ymax = constants_dict["ymax"]
    atm = constants_dict["atm"]
    T_ref = constants_dict["T_ref"]
    dyf = constants_dict["dyf"]
    decimals = constants_dict["decimals"]

    # Make Constants and Results folders
    if os.path.exists(path_results):
        shutil.rmtree(path_results)
        print('Removed Old Results')
    os.mkdir(path_results)
    print('Created Results')

    if os.path.exists(path_zero):
        shutil.rmtree(path_zero)
        print('Removed Time-Step 0')
    os.mkdir(path_zero)
    print('Created Time-Step 0')

    if os.path.exists(path_constant):
        shutil.rmtree(path_constant)
        print('Removed Old Constants')
    os.mkdir(path_constant)
    print('Created Constants')

    nx = N + 1
    ny = M + 1

    x = np.linspace(0, xmax, nx)
    y = np.linspace(0, ymax, ny)

    X, Y = np.meshgrid(x, y)

    dx, dy = deltas_write_nonuni(nx, ny, xmax, ymax, dyf)

    Points = np.zeros([2, nx, ny])
    Points[0, 1:, :] = np.cumsum(dx[:-1, :], axis=0)
    Points[1, :, 1:] = np.cumsum(dy[:, :-1], axis=1)

    write_points(os.path.join(path_constant, 'Points.txt'), Points)
    write_scalar(os.path.join(path_constant, 'Dx.txt'), dx)
    write_scalar(os.path.join(path_constant, 'Dy.txt'), dy)

    P = np.zeros([nx + 1, ny + 1]) * atm
    write_scalar(os.path.join(path_zero, 'P.txt'), P)

    U = np.zeros([nx + 1, ny + 1])
    U[0, :] = 1
    write_scalar(os.path.join(path_zero, 'U.txt'), U)

    V = np.zeros([nx + 1, ny + 1])
    write_scalar(os.path.join(path_zero, 'V.txt'), V)

    T = np.ones([nx + 1, ny + 1]) * T_ref
    T[150:152, 100:102] = 375
    write_scalar(os.path.join(path_zero, 'T.txt'), T)

    phi = np.zeros([nx + 1, ny + 1])
    phi[150:152, 100:102] = 40
    write_scalar(os.path.join(path_zero, 'phi.txt'), phi)

    np.savetxt(os.path.join(path_zero, 'P_tab.txt'), P, delimiter='\t', fmt=f'%.{decimals}f')
    np.savetxt(os.path.join(path_zero, 'U_tab.txt'), U, delimiter='\t', fmt=f'%.{decimals}f')
    np.savetxt(os.path.join(path_zero, 'V_tab.txt'), V, delimiter='\t', fmt=f'%.{decimals}f')
    np.savetxt(os.path.join(path_zero, 'T_tab.txt'), T, delimiter='\t', fmt=f'%.{decimals}f')
    np.savetxt(os.path.join(path_zero, 'phi_tab.txt'), phi, delimiter='\t', fmt=f'%.{decimals}f')


# %% Main Functions


def timer(t1):
    """
    Calculates the elapsed time since a given timestamp.

    :param t1: The initial timestamp.
    :type t1: float
    :returns: The current time.
    :rtype: float
    """
    # print(time.time()-t1)
    return time.time()


@jit(nopython=True, fastmath=True, cache=True)
def transport(T, u, v, dt, dx, dy, alpha, shift_begin, mesh_dir_N):
    """
    Solves the transport equation for a scalar field T using finite difference method.

    This function is JIT compiled using Numba for performance.

    :param T: The scalar field (e.g., temperature, pollutant) to be transported.
    :type T: numpy.ndarray
    :param u: The U-velocity field.
    :type u: numpy.ndarray
    :param v: The V-velocity field.
    :type v: numpy.ndarray
    :param dt: The time step size.
    :type dt: float
    :param dx: The grid cell sizes in the x-direction.
    :type dx: numpy.ndarray
    :param dy: The grid cell sizes in the y-direction.
    :type dy: numpy.ndarray
    :param alpha: The diffusion coefficient.
    :type alpha: float
    :returns: The updated scalar field after one time step.
    :rtype: numpy.ndarray
    """
    nx = T.shape[0]-1
    ny = T.shape[1]-1

    T_ = np.zeros(T.shape)
    T_[:, :] = T[:, :]

    def RHS(alpha, T, dx, dy, i, j, shift_begin, mesh_dir_N):
        """
        Helper function to calculate the right-hand side (diffusion terms) of the transport equation.

        :param alpha: The diffusion coefficient.
        :type alpha: float
        :param T: The scalar field.
        :type T: numpy.ndarray
        :param dx: The grid cell sizes in the x-direction.
        :type dx: numpy.ndarray
        :param dy: The grid cell sizes in the y-direction.
        :type dy: numpy.ndarray
        :param i: Current x-index.
        :type i: int
        :param j: Current y-index.
        :type j: int
        :returns: The calculated RHS value.
        :rtype: float
        """
        if mesh_dir_N:
            rx_top = dx[(i + shift_begin, j)] - dx[(i - 1 + shift_begin, j)]
            rx_bot = dx[(i + shift_begin, j)] + dx[(i - 1 + shift_begin, j)]
            rx = rx_top/rx_bot

            ry_top = dy[(i + shift_begin, j)] - dy[(i + shift_begin, j - 1)]
            ry_bot = dy[(i + shift_begin, j)] + dy[(i + shift_begin, j - 1)]
            ry = ry_top / ry_bot

            T2_x = (((1 - rx) * T[(i + 1, j)]) - (2 * T[(i, j)]) + ((1 + rx) * T[(i - 1, j)])) / \
                ((dx[(i + shift_begin, j)]**2 + dx[(i - 1 + shift_begin, j)]**2) / 2)
            T2_y = (((1 - ry) * T[(i, j + 1)]) - (2 * T[(i, j)]) + ((1 + ry) * T[(i, j - 1)])) / \
                ((dy[(i + shift_begin, j)]**2 + dy[(i + shift_begin, j - 1)]**2) / 2)
        else:
            rx_top = dx[(i, j + shift_begin)] - dx[(i - 1, j + shift_begin)]
            rx_bot = dx[(i, j + shift_begin)] + dx[(i - 1, j + shift_begin)]
            rx = rx_top/rx_bot

            ry_top = dy[(i, j)] - dy[(i, j - 1 + shift_begin)]
            ry_bot = dy[(i, j)] + dy[(i, j - 1 + shift_begin)]
            ry = ry_top / ry_bot

            T2_x = (((1 - rx) * T[(i + 1, j)]) - (2 * T[(i, j)]) + ((1 + rx) * T[(i - 1, j)])) / \
                ((dx[(i, j + shift_begin)]**2 + dx[(i - 1, j + shift_begin)]**2) / 2)
            T2_y = (((1 - ry) * T[(i, j + 1)]) - (2 * T[(i, j)]) + ((1 + ry) * T[(i, j - 1)])) / \
                ((dy[(i, j + shift_begin)]**2 + dy[(i, j - 1 + shift_begin)]**2) / 2)

        rhs = alpha*(T2_x + T2_y)

        return rhs

    def Der_1(u, v, T, dx, dy, i, j, shift_begin, mesh_dir_N):
        """
        Helper function to calculate the first derivatives (convection terms) of the transport equation.

        :param u: The U-velocity field.
        :type u: numpy.ndarray
        :param v: The V-velocity field.
        :type v: numpy.ndarray
        :param T: The scalar field.
        :type T: numpy.ndarray
        :param dx: The grid cell sizes in the x-direction.
        :type dx: numpy.ndarray
        :param dy: The grid cell sizes in the y-direction.
        :type dy: numpy.ndarray
        :param i: Current x-index.
        :type i: int
        :param j: Current y-index.
        :type j: int
        :returns: A tuple containing the x and y convection terms (duTdx, dvTdy).
        :rtype: tuple(float, float)
        """
        if mesh_dir_N:
            x_1 = dx[(i + shift_begin, j)]
            x_0 = dx[(i - 1 + shift_begin, j)]

            y_1 = dy[(i + shift_begin, j)]
            y_0 = dy[(i + shift_begin, j - 1)]

            U = (0.5 * (u[i, j] + u[i - 1, j]))
            V = (0.5 * (v[i, j] + v[i, j - 1]))

            if U > 0:
                duTdx = (x_0)**(-1) * U * (T[i, j] - T[i - 1, j])
            else:
                duTdx = (x_1)**(-1) * U * (T[i + 1, j] - T[i, j])

            if V > 0:
                dvTdy = (y_0)**(-1) * V * (T[i, j] - T[i, j - 1])
            else:
                dvTdy = (y_1)**(-1) * V * (T[i, j + 1] - T[i, j])
        else:
            x_1 = dx[(i, j + shift_begin)]
            x_0 = dx[(i - 1, j + shift_begin)]

            y_1 = dy[(i, j + shift_begin)]
            y_0 = dy[(i, j - 1 + shift_begin)]

            U = (0.5 * (u[i, j] + u[i - 1, j]))
            V = (0.5 * (v[i, j] + v[i, j - 1]))

            if U > 0:
                duTdx = (x_0)**(-1) * U * (T[i, j] - T[i - 1, j])
            else:
                duTdx = (x_1)**(-1) * U * (T[i + 1, j] - T[i, j])

            if V > 0:
                dvTdy = (y_0)**(-1) * V * (T[i, j] - T[i, j - 1])
            else:
                dvTdy = (y_1)**(-1) * V * (T[i, j + 1] - T[i, j])

        return duTdx, dvTdy

    for i in range(1, nx):
        for j in range(1, ny):
            if i == 10 and j == 10:
                Q = 0
            else:
                Q = 0
            rhs = RHS(alpha, T, dx, dy, i, j, shift_begin, mesh_dir_N)
            duTdx, dvTdy = Der_1(u, v, T, dx, dy, i, j, shift_begin, mesh_dir_N)
            T_[(i, j)] = T[(i, j)] + (dt * (rhs - duTdx - dvTdy + Q))

    return T_


@jit(nopython=True, fastmath=True, cache=True)
def poisson(P, u, v, dt, dx, dy, rho, shift_begin, mesh_dir_N):
    """
    Solves the Poisson equation for pressure using an iterative solver.

    This function is JIT compiled using Numba for performance.

    :param P: The pressure field from the previous time step.
    :type P: numpy.ndarray
    :param u: The U-velocity field from the predictor step.
    :type u: numpy.ndarray
    :param v: The V-velocity field from the predictor step.
    :type v: numpy.ndarray
    :param dt: The time step size.
    :type dt: float
    :param dx: The grid cell sizes in the x-direction.
    :type dx: numpy.ndarray
    :param dy: The grid cell sizes in the y-direction.
    :type dy: numpy.ndarray
    :param rho: The fluid density.
    :type rho: float
    :returns: The updated pressure field.
    :rtype: numpy.ndarray
    """
    # u and v are from the "predictor step"
    # P comes from the previous time step
    nx = P.shape[0] - 1
    ny = P.shape[1] - 1

    def Frac(dx, dy, i, j, shift_begin, mesh_dir_N):
        """
        Helper function to calculate fractional coefficients for the Poisson equation.

        :param dx: The grid cell sizes in the x-direction.
        :type dx: numpy.ndarray
        :param dy: The grid cell sizes in the y-direction.
        :type dy: numpy.ndarray
        :param i: Current x-index.
        :type i: int
        :param j: Current y-index.
        :type j: int
        :returns: A tuple of fractional coefficients (frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n).
        :rtype: tuple(float, float, float, float, float, float)
        """
        if mesh_dir_N:
            rx_top = dx[(i + shift_begin, j)] - dx[(i - 1 + shift_begin, j)]
            rx_bot = dx[(i + shift_begin, j)] + dx[(i - 1 + shift_begin, j)]
            rx = rx_top/rx_bot

            ry_top = dy[(i + shift_begin, j)] - dy[(i + shift_begin, j - 1)]
            ry_bot = dy[(i + shift_begin, j)] + dy[(i + shift_begin, j - 1)]
            ry = ry_top / ry_bot

            frac_x = 4 / (dx[(i + shift_begin, j)]**2 + dx[(i - 1 + shift_begin, j)]**2)
            frac_y = 4 / (dy[(i + shift_begin, j)]**2 + dy[(i + shift_begin, j - 1)]**2)
            Rx_p = 2 * (1 - rx) / (dx[(i + shift_begin, j)]**2 + dx[(i - 1 + shift_begin, j)]**2)
            Rx_n = 2 * (1 + rx) / (dx[(i + shift_begin, j)]**2 + dx[(i - 1 + shift_begin, j)]**2)
            Ry_p = 2 * (1 - ry) / (dy[(i + shift_begin, j)]**2 + dy[(i + shift_begin, j - 1)]**2)
            Ry_n = 2 * (1 + ry) / (dy[(i + shift_begin, j)]**2 + dy[(i + shift_begin, j - 1)]**2)
        else:
            rx_top = dx[(i, j + shift_begin)] - dx[(i - 1, j + shift_begin)]
            rx_bot = dx[(i, j + shift_begin)] + dx[(i - 1, j + shift_begin)]
            rx = rx_top/rx_bot

            ry_top = dy[(i, j + shift_begin)] - dy[(i, j - 1 + shift_begin)]
            ry_bot = dy[(i, j + shift_begin)] + dy[(i, j - 1 + shift_begin)]
            ry = ry_top / ry_bot

            frac_x = 4 / (dx[(i, j + shift_begin)]**2 + dx[(i - 1, j + shift_begin)]**2)
            frac_y = 4 / (dy[(i, j + shift_begin)]**2 + dy[(i, j - 1 + shift_begin)]**2)
            Rx_p = 2 * (1 - rx) / (dx[(i, j + shift_begin)]**2 + dx[(i - 1, j + shift_begin)]**2)
            Rx_n = 2 * (1 + rx) / (dx[(i, j + shift_begin)]**2 + dx[(i - 1, j + shift_begin)]**2)
            Ry_p = 2 * (1 - ry) / (dy[(i, j + shift_begin)]**2 + dy[(i, j - 1 + shift_begin)]**2)
            Ry_n = 2 * (1 + ry) / (dy[(i, j + shift_begin)]**2 + dy[(i, j - 1 + shift_begin)]**2)
        return frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n

    def RHS(u, v, dx, dy, i, j, dt, rho, shift_begin, mesh_dir_N):
        """
        Helper function to calculate the right-hand side of the Poisson equation (divergence term).

        :param u: The U-velocity field.
        :type u: numpy.ndarray
        :param v: The V-velocity field.
        :type v: numpy.ndarray
        :param dx: The grid cell sizes in the x-direction.
        :type dx: numpy.ndarray
        :param dy: The grid cell sizes in the y-direction.
        :type dy: numpy.ndarray
        :param i: Current x-index.
        :type i: int
        :param j: Current y-index.
        :type j: int
        :param dt: The time step size.
        :type dt: float
        :param rho: The fluid density.
        :type rho: float
        :returns: The calculated RHS value.
        :rtype: float
        """
        if mesh_dir_N:
            U_ = (u[(i, j)] - u[(i - 1, j)]) / dx[(i - 1 + shift_begin, j)]
            V_ = (v[(i, j)] - v[(i, j - 1)]) / dy[(i + shift_begin, j - 1)]
        else:
            U_ = (u[(i, j)] - u[(i - 1, j)]) / dx[(i - 1, j + shift_begin)]
            V_ = (v[(i, j)] - v[(i, j - 1)]) / dy[(i, j - 1 + shift_begin)]

        rhs = rho / dt * (U_ + V_)
        return rhs

    Con = 1e-2
    err = 1
    k = 0
    temp = np.zeros(P.shape)
    P_ = np.zeros(P.shape)

    while (err > Con):
        temp[:, :] = P[:, :]
        k += 1
        for i in range(1, nx):
            for j in range(1, ny):
                frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n = Frac(dx, dy, i, j, shift_begin, mesh_dir_N)
                rhs = RHS(u, v, dx, dy, i, j, dt, rho, shift_begin, mesh_dir_N)
                P_[(i, j)] = (frac_x + frac_y)**(-1) * ((Rx_p * P[(i + 1, j)] + Rx_n *
                                                        P[(i - 1, j)]) + (Ry_p * P[(i, j + 1)] + Ry_n * P[(i, j - 1)]) - rhs)
        if k == 100000:  # Look into this
            print('not converged', err)
            break
        err = np.max(np.abs(P_ - temp))
    return P_


@jit(nopython=True, fastmath=True, cache=True)
def Adv(u, v, x, y, i, j, flag, shift_begin, mesh_dir_N):
    """
    Calculates the advection term for velocity components.

    This function is JIT compiled using Numba for performance.

    :param u: The U-velocity field.
    :type u: numpy.ndarray
    :param v: The V-velocity field.
    :type v: numpy.ndarray
    :param x: The grid cell sizes in the x-direction (same as dx).
    :type x: numpy.ndarray
    :param y: The grid cell sizes in the y-direction (same as dy).
    :type y: numpy.ndarray
    :param i: Current x-index.
    :type i: int
    :param j: Current y-index.
    :type j: int
    :param flag: Controls which velocity component's advection is calculated (0 for U, 1 for V).
    :type flag: int
    :returns: The calculated advection term.
    :rtype: float
    """
    if mesh_dir_N:
        x_1 = x[(i + shift_begin, j)]
        x_0 = x[(i - 1 + shift_begin, j)]

        y_1 = y[(i + shift_begin, j)]
        y_0 = y[(i + shift_begin, j - 1)]

        if flag == 0:
            U = u[i][j]
            V = 0.25 * (v[i - 1][j] + v[i][j] + v[i - 1][j + 1] + v[i][j + 1])

            if U > 0:
                u_1 = U * (u[i][j] - u[i - 1][j]) / (x_0)
            else:
                u_1 = U * (u[i + 1][j] - u[i][j]) / (x_1)

            if V > 0:
                u_2 = V * (u[i][j] - u[i][j - 1]) / (y_0)
            else:
                u_2 = V * (u[i][j + 1] - u[i][j]) / (y_1)

            return u_1 + u_2

        if flag == 1:
            U = 0.25 * (u[i][j - 1] + u[i][j] + u[i+1][j - 1] + u[i + 1][j])
            V = v[i][j]

            if V > 0:
                v_1 = V * (v[i][j] - v[i][j - 1]) / (y_0)
            else:
                v_1 = V * (v[i][j + 1] - v[i][j]) / (y_1)

            if U > 0:
                v_2 = U * (v[i][j] - v[i - 1][j]) / (x_0)
            else:
                v_2 = U * (v[i + 1][j] - v[i][j]) / (x_1)
    else:
        x_1 = x[(i, j + shift_begin)]
        x_0 = x[(i - 1, j + shift_begin)]

        y_1 = y[(i, j + shift_begin)]
        y_0 = y[(i, j - 1 + shift_begin)]

        if flag == 0:
            U = u[i][j]
            V = 0.25 * (v[i - 1][j] + v[i][j] + v[i - 1][j + 1] + v[i][j + 1])

            if U > 0:
                u_1 = U * (u[i][j] - u[i - 1][j]) / (x_0)
            else:
                u_1 = U * (u[i + 1][j] - u[i][j]) / (x_1)

            if V > 0:
                u_2 = V * (u[i][j] - u[i][j - 1]) / (y_0)
            else:
                u_2 = V * (u[i][j + 1] - u[i][j]) / (y_1)

            return u_1 + u_2

        if flag == 1:
            U = 0.25 * (u[i][j - 1] + u[i][j] + u[i+1][j - 1] + u[i + 1][j])
            V = v[i][j]

            if V > 0:
                v_1 = V * (v[i][j] - v[i][j - 1]) / (y_0)
            else:
                v_1 = V * (v[i][j + 1] - v[i][j]) / (y_1)

            if U > 0:
                v_2 = U * (v[i][j] - v[i - 1][j]) / (x_0)
            else:
                v_2 = U * (v[i + 1][j] - v[i][j]) / (x_1)

        return v_1 + v_2


@jit(nopython=True, fastmath=True, cache=True)
def Diff(u, x, y, i, j, shift_begin, mesh_dir_N):
    """
    Calculates the diffusion term for a scalar field.

    This function is JIT compiled using Numba for performance.

    :param u: The scalar field for which to calculate diffusion.
    :type u: numpy.ndarray
    :param x: The grid cell sizes in the x-direction (same as dx).
    :type x: numpy.ndarray
    :param y: The grid cell sizes in the y-direction (same as dy).
    :type y: numpy.ndarray
    :param i: Current x-index.
    :type i: int
    :param j: Current y-index.
    :type j: int
    :returns: The calculated diffusion term.
    :rtype: float
    """
    if mesh_dir_N:
        x_1 = x[(i + shift_begin, j)]
        x_0 = x[(i - 1 + shift_begin, j)]

        y_1 = y[(i + shift_begin, j)]
        y_0 = y[(i + shift_begin, j - 1)]

        r_x = (x_1 - x_0)/(x_0 + x_1)
        r_y = (y_1 - y_0)/(y_0 + y_1)

        u_1 = 2 * ((1 + r_x) * u[i-1][j] - 2 * u[i][j] + (1 - r_x) * u[i + 1][j]) / (x_0**2 + x_1**2)
        u_2 = 2 * ((1 + r_y) * u[i][j - 1] - 2 * u[i][j] + (1 - r_y) * u[i][j + 1]) / (y_0**2 + y_1**2)
    else:
        x_1 = x[(i, j + shift_begin)]
        x_0 = x[(i - 1, j + shift_begin)]

        y_1 = y[(i, j + shift_begin)]
        y_0 = y[(i, j - 1 + shift_begin)]

        r_x = (x_1 - x_0)/(x_0 + x_1)
        r_y = (y_1 - y_0)/(y_0 + y_1)

        u_1 = 2 * ((1 + r_x) * u[i-1][j] - 2 * u[i][j] + (1 - r_x) * u[i + 1][j]) / (x_0**2 + x_1**2)
        u_2 = 2 * ((1 + r_y) * u[i][j - 1] - 2 * u[i][j] + (1 - r_y) * u[i][j + 1]) / (y_0**2 + y_1**2)
    return u_1 + u_2


@jit(nopython=True, fastmath=True, cache=True)
def predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta, shift_begin, mesh_dir_N):
    """
    Performs the predictor step for velocity components in the SIMPLE algorithm.
    Calculates provisional velocities considering advection, diffusion, and buoyancy.

    This function is JIT compiled using Numba for performance.

    :param x: The grid cell sizes in the x-direction (same as dx).
    :type x: numpy.ndarray
    :param y: The grid cell sizes in the y-direction (same as dy).
    :type y: numpy.ndarray
    :param u: The current U-velocity field.
    :type u: numpy.ndarray
    :param v: The current V-velocity field.
    :type v: numpy.ndarray
    :param T: The temperature field.
    :type T: numpy.ndarray
    :param dt: The time step size.
    :type dt: float
    :param T_ref: Reference temperature for buoyancy calculation.
    :type T_ref: float
    :param rho: Fluid density.
    :type rho: float
    :param g: Gravitational acceleration.
    :type g: float
    :param nu: Kinematic viscosity.
    :type nu: float
    :param beta: Thermal expansion coefficient.
    :type beta: float
    :returns: A tuple containing the provisional U and V velocity fields (u_, v_).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    # Main Predictor Loop
    nx = T.shape[0]
    ny = T.shape[1]
    Cs = 0.2
    delta_g = 0.00125

    u_ = np.zeros((u.shape[0], u.shape[1]))
    v_ = np.zeros((v.shape[0], v.shape[1]))
    u_[:, :] = u[:, :]
    v_[:, :] = v[:, :]

    if mesh_dir_N:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                S = 0.5 * ((u[i][j + 1] - u[i][j]) / y[i + shift_begin, j]) + ((v[i + 1][j] - v[i][j]) / x[i + shift_begin, j])
                nut = ((Cs*delta_g)**2)*S
                nut = min(1e-4, abs(nut))
                Nu = nu + nut
                if nut > 1e-3:
                    print(nut)

                u_[i][j] = u[i][j] + dt * (Nu * (Diff(u, x, y, i, j, shift_begin, mesh_dir_N)) - Adv(u, v, x, y, i, j, 0, shift_begin, mesh_dir_N))
                v_[i][j] = v[i][j] + dt * (Nu * (Diff(v, x, y, i, j, shift_begin, mesh_dir_N)) - Adv(u, v, x, y, i, j, 1, shift_begin, mesh_dir_N) +
                                           rho * g * beta * (0.5 * (T[i][j] + T[i][j + 1]) - T_ref))  # Add Bousinessq Terms
    else:
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                S = 0.5 * ((u[i][j + 1] - u[i][j]) / y[i, j + shift_begin]) + ((v[i + 1][j] - v[i][j]) / x[i, j + shift_begin])
                nut = ((Cs*delta_g)**2)*S
                nut = min(1e-4, abs(nut))
                Nu = nu + nut
                if nut > 1e-3:
                    print(nut)

                u_[i][j] = u[i][j] + dt * (Nu * (Diff(u, x, y, i, j, shift_begin, mesh_dir_N)) - Adv(u, v, x, y, i, j, 0, shift_begin, mesh_dir_N))
                v_[i][j] = v[i][j] + dt * (Nu * (Diff(v, x, y, i, j, shift_begin, mesh_dir_N)) - Adv(u, v, x, y, i, j, 1, shift_begin, mesh_dir_N) +
                                           rho * g * beta * (0.5 * (T[i][j] + T[i][j + 1]) - T_ref))  # Add Bousinessq Terms

    return u_, v_


@jit(nopython=True, fastmath=True, cache=True)
def corrector(x, y, u, v, p, dt, rho, shift_begin, mesh_dir_N):
    """
    Performs the corrector step in the SIMPLE algorithm, adjusting velocities
    based on the updated pressure field to satisfy continuity.

    This function is JIT compiled using Numba for performance.

    :param x: The grid cell sizes in the x-direction (same as dx).
    :type x: numpy.ndarray
    :param y: The grid cell sizes in the y-direction (same as dy).
    :type y: numpy.ndarray
    :param u: The provisional U-velocity field from the predictor step.
    :type u: numpy.ndarray
    :param v: The provisional V-velocity field from the predictor step.
    :type v: numpy.ndarray
    :param p: The corrected pressure field from the Poisson solver.
    :type p: numpy.ndarray
    :param dt: The time step size.
    :type dt: float
    :param rho: Fluid density.
    :type rho: float
    :returns: A tuple containing the corrected U and V velocity fields (u_, v_).
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """

    nx = p.shape[0]
    ny = p.shape[1]

    u_ = np.zeros((u.shape[0], u.shape[1]))
    v_ = np.zeros((v.shape[0], v.shape[1]))
    u_[:, :] = u[:, :]
    v_[:, :] = v[:, :]

    if mesh_dir_N:
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                x_1 = x[(i + shift_begin, j)]
                # x_0 = x[(i - 1, j)]
                y_1 = y[(i + shift_begin, j)]
                # y_0 = y[(i, j - 1)]

                u_[i][j] = u[i][j] - (dt / rho) * (p[i + 1][j] - p[i][j]) / (x_1)
                v_[i][j] = v[i][j] - (dt / rho) * (p[i][j + 1] - p[i][j]) / (y_1)
    else:
        for i in range(1, nx-1):
            for j in range(1, ny-1):

                x_1 = x[(i, j + shift_begin)]
                # x_0 = x[(i - 1, j)]
                y_1 = y[(i, j + shift_begin)]
                # y_0 = y[(i, j - 1)]

                u_[i][j] = u[i][j] - (dt / rho) * (p[i + 1][j] - p[i][j]) / (x_1)
                v_[i][j] = v[i][j] - (dt / rho) * (p[i][j + 1] - p[i][j]) / (y_1)

    return u_, v_


@jit(nopython=True, fastmath=True, cache=True)
def BC_update(u, v, p, T, phi):
    """
    Applies boundary conditions to velocity, pressure, temperature, and pollutant fields.

    This function is JIT compiled using Numba for performance.

    :param u: U-velocity field.
    :type u: numpy.ndarray
    :param v: V-velocity field.
    :type v: numpy.ndarray
    :param p: Pressure field.
    :type p: numpy.ndarray
    :param T: Temperature field.
    :type T: numpy.ndarray
    :param phi: Pollutant concentration field.
    :type phi: numpy.ndarray
    :returns: A tuple containing the updated u, v, p, T, and phi fields after applying boundary conditions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    nx = p.shape[0]-1
    ny = p.shape[1]-1

    # inlet
    v[0, :] = -v[1, :]

    p[0, :] = p[1, :]
    T[0, :] = T[1, :]
    phi[0, :] = phi[1, :]

    # bottom
    u[:, 0] = -u[:, 1]
    v[:, 0] = 0.
    p[:, 0] = p[:, 1]
    T[:, 0] = T[:, 1]
    phi[:, 0] = phi[:, 1]

    # outlet
    u[nx - 1, :] = u[nx - 2, :]
    u[nx, :] = u[nx - 1, :]
    v[nx, :] = v[nx - 1, :]
    p[nx - 1, :] = p[nx, :]
    T[nx, :] = T[nx - 1, :]
    phi[nx, :] = phi[nx - 1, :]

    # top
    u[:, ny] = u[:, ny - 1]
    v[:, ny] = 0.
    v[:, ny - 1] = 0.
    p[:, ny] = p[:, ny - 1]
    T[:, ny] = T[:, ny - 1]
    phi[:, ny] = phi[:, ny - 1]

    T[150:152, 100:102] = 375
    phi[150:152, 100:102] = 40

    return u, v, p, T, phi


@jit(nopython=True, fastmath=True, cache=True)
def Building_BC(u, v, p, T, phi, Dim):
    """
    Applies boundary conditions for building obstacles. Sets velocities to zero
    and potentially adjusts scalar fields within or near building regions.

    This function is JIT compiled using Numba for performance.

    :param u: U-velocity field.
    :type u: numpy.ndarray
    :param v: V-velocity field.
    :type v: numpy.ndarray
    :param p: Pressure field.
    :type p: numpy.ndarray
    :param T: Temperature field.
    :type T: numpy.ndarray
    :param phi: Pollutant concentration field.
    :type phi: numpy.ndarray
    :param Dim: A list or tuple specifying the building dimensions [x_0, x_1, y_1],
                where x_0 and x_1 are x-indices, and y_1 is the y-index defining the top.
    :type Dim: list or tuple
    :returns: A tuple containing the updated u, v, p, T, and phi fields after applying building boundary conditions.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    k, k_, r = Dim
    # nx = p.shape[0] - 1
    # ny = p.shape[1] - 1

    # left
    u[k, :r] = 0.
    v[k + 1, :r] = -v[k, :r]
    p[k + 1, :r] = p[k, :r]
    T[k + 1, :r] = T[k, :r]
    phi[k + 1, :r] = phi[k, :r]

    # top
    u[k + 1:k_, r - 1] = -u[k + 1:k_, r]
    v[k + 1:k_, r - 1] = 0.
    p[k + 1:k_, r - 1] = p[k + 1:k_, r]
    T[k + 1:k_, r - 1] = T[k + 1:k_, r]
    phi[k + 1:k_, r - 1] = phi[k + 1:k_, r]

    # right
    u[k_ - 1, :r] = 0.
    v[k_ - 1, :r] = -v[k_, :r]
    p[k_ - 1, :r] = p[k_, :r]
    T[k_ - 1, :r] = T[k_, :r]
    phi[k_ - 1, :r] = phi[k_, :r]

    # inside
    u[k + 1:k_, :r - 1] = 0.
    v[k + 2:k_-1, :r - 1] = 0.
    p[k + 2:k_-1, :r - 1] = 0.
    T[k + 2:k_-1, :r - 1] = 325.
    phi[k + 2:k_ - 1, :r - 1] = 10.

    return u, v, p, T, phi


#@jit(nopython=True, fastmath=True, cache=True)
def parallel(x, y, u, v, T, dt, T_ref, rho, g, nu, beta, P, phi, alpha_T, alpha_pollutant, start_ends, i, mesh_dir_N):

    # gets points on array including bc
    shift_begin = start_ends[i, 0]
    shift_endin = start_ends[i, 1] + 1 # need to add 1 to include end point

    if mesh_dir_N:
        # gets out local array
        u = u[shift_begin:shift_endin, :]
        v = v[shift_begin:shift_endin, :]
        T = T[shift_begin:shift_endin, :]
        P = P[shift_begin:shift_endin, :]
        phi = phi[shift_begin:shift_endin, :]

    else:
        # gets out local array
        u = u[:, shift_begin:shift_endin]
        v = v[:, shift_begin:shift_endin:]
        T = T[:, shift_begin:shift_endin]
        P = P[:, shift_begin:shift_endin]
        phi = phi[:, shift_begin:shift_endin]

    u_, v_ = predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta, shift_begin, mesh_dir_N)

    p_new = poisson(P, u_, v_, dt, x, y, rho, shift_begin, mesh_dir_N)

    u_new, v_new = corrector(x, y, u_, v_, p_new, dt, rho, shift_begin, mesh_dir_N)

    T_new = transport(T, u, v, dt, x, y, alpha_T, shift_begin, mesh_dir_N)

    phi_new = transport(phi, u, v, dt, x, y, alpha_pollutant, shift_begin, mesh_dir_N)  # Pollutant Transport

    return [u_new, v_new, p_new, T_new, phi_new]

# %% Run Script


def main():
    """
    Main function to run the CFD simulation.

    This function orchestrates the entire simulation process, including:
    - Reading configuration from 'config.yaml'.
    - Initializing the simulation environment.
    - Iterating through time steps.
    - Performing predictor, Poisson solver, and corrector steps for fluid flow.
    - Solving transport equations for temperature and pollutant.
    - Applying boundary conditions, including those for buildings.
    - Writing results periodically.
    """

    # Read YAML file
    config_name = "config.yaml"
    config_path = os.path.join(os.getcwd(), config_name)

    with open(config_path, 'r') as stream:
        data = yaml.safe_load(stream)

    constants_dict = data["constants"]
    rho = constants_dict["rho"]
    T_ref = constants_dict["T_ref"]
    beta = constants_dict["beta"]
    nu = constants_dict["nu"]
    alpha_T = constants_dict["alpha_T"]
    alpha_pollutant = constants_dict["alpha_pollutant"]
    total_t = constants_dict["total_t"]
    dt = constants_dict["dt"]
    dx = constants_dict["dx"]
    g = constants_dict["g"]
    cfl = constants_dict["cfl"]
    decimals = constants_dict["decimals"]
    nproc = constants_dict["nproc"]
    mesh_dir_N = constants_dict["mesh_dir_N"]
    N = constants_dict["N"]
    M = constants_dict["M"]


    buildings_dict = data["buildings"]
    b_0 = buildings_dict['building_0']
    b_1 = buildings_dict['building_1']
    b_2 = buildings_dict['building_2']

    initialise(config_path)

    x, y = read_delta(0)
    P, T, u, v, phi = read_all_scalar(0)

    # Parallel indexing
    start_ends = np.zeros((nproc, 2))
    N_idx = np.tile(np.squeeze(np.array([range(1, N // nproc + 1)])), nproc)
    M_idx = np.tile(np.squeeze(np.array([range(1, M // nproc + 1)])), nproc)

    # add start and endings to include BC idx
    N_idx = np.append(0, N_idx)
    N_idx = np.append(N_idx, N // nproc + 1)
    M_idx = np.append(0, M_idx)
    M_idx = np.append(M_idx, M // nproc + 1)

    if mesh_dir_N:
        if (N) % nproc == 0:
            starts = np.append(0, np.squeeze(np.where(N_idx == (N // nproc)))[:-1])
            ends = np.append(np.squeeze(np.where(N_idx == 1))[1:], N + 1)
        else:
            raise Exception("need to have nproc as a factor of N")
    else:
        if (M) % nproc == 0:
            starts = np.append(0, np.squeeze(np.where(M_idx == (M // nproc)))[:-1])
            ends = np.append(np.squeeze(np.where(M_idx == 1))[1:], M + 1)
        else:
            raise Exception("need to have nproc as a factor of M")
    start_ends = np.stack((starts, ends)).T

    running = True
    t = 0.
    ite = 0.

    # while running:
    if running:
        ite += 1
        print(ite)
        if int(ite) % 100 == 0:
            print('\ntime=%.3f' % t)

            if t != 0:
                write_all_scalar(P, T, u, v, phi, decimals, t)

        idx = list(range(16))
        if mesh_dir_N:
            print("N")
            # parfor loop x and y can be kept const
            output_list = Parallel(n_jobs=nproc)(delayed(parallel)(x, y, u, v, T, dt, T_ref, rho, g, nu, beta, P, phi, alpha_T, alpha_pollutant, start_ends, j, mesh_dir_N) for j in idx)
            # build array back exluding BC

        else:
            print("M")
            # parfor loop x and y can be kept const
            output_list = Parallel(n_jobs=nproc)(delayed(parallel)(x, y, u, v, T, dt, T_ref, rho, g, nu, beta, P, phi, alpha_T, alpha_pollutant, start_ends, j, mesh_dir_N) for j in idx)
            # build array back exluding BC

        u_new = copy.deepcopy(u)
        v_new = copy.deepcopy(v)
        T_new = copy.deepcopy(T)
        p_new = copy.deepcopy(P)
        phi_new = copy.deepcopy(phi)

        for i in idx:

            u, v, P, T, phi = output_list[i]

            # gets points on array including bc
            shift_begin = start_ends[i, 0] + 1
            shift_endin = start_ends[i, 1] # need to add 1 to include end point

            if mesh_dir_N:
                # gets out local array
                u_new[shift_begin:shift_endin, :] = u
                v_new[shift_begin:shift_endin, :] = v
                T_new[shift_begin:shift_endin, :] = T
                p_new[shift_begin:shift_endin, :] = P
                phi_new[shift_begin:shift_endin, :] = phi

            else:
                # gets out local array
                u_new[:, shift_begin:shift_endin] = u
                v_new[:, shift_begin:shift_endin:] = v
                T_new[:, shift_begin:shift_endin] = T
                p_new[:, shift_begin:shift_endin] = P
                phi_new[:, shift_begin:shift_endin] = phi


        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(BC_update(u_new, v_new, p_new, T_new, phi_new))

        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new,
                                                                        v_new,
                                                                        p_new,
                                                                        T_new,
                                                                        phi_new,
                                                                        [b_0["x_0"], b_0["x_1"], b_0["y_1"]]))

        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new,
                                                                        v_new,
                                                                        p_new,
                                                                        T_new,
                                                                        phi_new,
                                                                        [b_1["x_0"], b_1["x_1"], b_1["y_1"]]))

        u_new, v_new, p_new, T_new, phi_new = copy.deepcopy(Building_BC(u_new,
                                                                        v_new,
                                                                        p_new,
                                                                        T_new,
                                                                        phi_new,
                                                                        [b_2["x_0"], b_2["x_1"], b_2["y_1"]]))

        u = copy.deepcopy(u_new)
        v = copy.deepcopy(v_new)

        P = copy.deepcopy(p_new)
        T = copy.deepcopy(T_new)
        phi = copy.deepcopy(phi_new)

        u_max = np.max(abs(u))
        v_max = np.max(abs(v))
        V = np.sqrt(u_max**2 + v_max**2)
        dt = (cfl*dx)/V
        t += dt

        if t >= total_t:
            write_all_scalar(P, T, u_new, v_new, phi, decimals, t)
            running = False


# %% Running from Spyder IDE


# Setting up config.yaml example
config = {
    'constants': {
        'rho': 1.225,
        'N': 1600,
        'M': 208,
        'xmax': 2,
        'ymax': 0.25,
        'atm': 1,
        'T_ref': 300,
        'beta': 3.333e-3,
        'nu': 1.569e-5,
        'alpha_T': 2.239e-5,
        'alpha_pollutant': 10,
        'total_t': 1,
        'dt': 0.00005,
        'dx': 0.00125,
        'dyf': 0.00025,
        'g': 2.,
        'cfl': 0.65,
        'decimals': 9,
        'nproc': 4,
        "mesh_dir_N":False
    },
    'buildings': {
        'building_0': {
            'x_0': 650,
            'x_1': 665,
            'y_0': 0,
            'y_1': 100
        },
        'building_1': {
            'x_0': 700,
            'x_1': 715,
            'y_0': 0,
            'y_1': 105
        },
        'building_2': {
            'x_0': 800,
            'x_1': 815,
            'y_0': 0,
            'y_1': 95
        }
    }
}

# Write YAML file example
with io.open('config.yaml', 'w', encoding='utf8') as outfile:
    yaml.dump(config, outfile, default_flow_style=False, allow_unicode=True)


# Read YAML file example
config_name = "config.yaml"
config_path = os.path.join(os.getcwd(), config_name)
with open(config_path, 'r') as stream:
    data = yaml.safe_load(stream)

# Run from Spyder Python IDE, not from terminal
main()
