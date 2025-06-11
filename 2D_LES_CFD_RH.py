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
    * Update Cntour & Animation to be operating system independant
    * Include setting of Smagorinsky Constant in input yaml file
    * Further Vectorise Main Script Functions
    * Parallelise Run Script using the module Joblib Parallel
    * Port over to C++ and/or Cuda

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


# %% Animation Functions


def Animation(Scalar_name, grid='no'):
    """
    Generate an animation of a specified scalar field over time.

    .. warning:: This function relies on `matplotlib.animation` which is commented out
                 in the original code. It might not function as expected without it.

    :param Scalar_name: The name of the scalar field to animate (e.g., 'U', 'V', 'P', 'T', 'phi').
    :type Scalar_name: str
    :param grid: Specifies whether to use grid-averaged values for plotting. Defaults to 'no'.
    :type grid: str, optional
    """
    def gridder(Scalar, Points, grid='no'):
        """
        Helper function to prepare scalar and point data for plotting based on grid option.

        :param Scalar: The scalar field data.
        :type Scalar: numpy.ndarray
        :param Points: The grid points data.
        :type Points: numpy.ndarray
        :param grid: Specifies whether to use grid-averaged values. Defaults to 'no'.
        :type grid: str, optional
        :returns: A tuple containing the processed scalar field and grid points.
        :rtype: tuple(numpy.ndarray, numpy.ndarray)
        """
        if grid == 'no':
            Points = copy.deepcopy(Point_average(Points))
            if Scalar_name == 'U':
                Scalar = copy.deepcopy(averaging(Scalar, 1))
            elif Scalar_name == 'V':
                Scalar = copy.deepcopy(averaging(Scalar, 0))
            else:
                Scalar = copy.deepcopy(Scalar[1:-1, 1:-1])
        else:
            if Scalar_name == 'U':
                Scalar = copy.deepcopy(averaging_grid(Scalar, 1))
            elif Scalar_name == 'V':
                Scalar = copy.deepcopy(averaging_grid(Scalar, 0))
            else:
                Scalar = copy.deepcopy(average_scalar(Scalar))
        return Scalar, Points

    times = []
    times.append('0\\')
    files = os.listdir('Results')
    counts = []
    for i in files:
        if isfloat(i):
            counts.append(float(i))
    counts.sort()
    k = 0
    for i in counts:
        if k % 10 == 0:
            times.append('Results\\' + '%.6f' % i)
        k += 1

    Scalars = [None] * len(times)

    for t in range(len(times)):
        Scalars[t] = read_scalar(times[t] + '\\' + Scalar_name + '.txt')

    # initialization function
    def init():
        # creating an empty plot/frame
        ax.collections = []
        # plt.ylabel(r'$y  \longrightarrow$')
        # plt.xlabel(r'$x  \longrightarrow$')
        ax.contour(Points[0], Points[1], np.zeros(Scalars[0].shape))

        # return ax

    # lists to store x and y axis points
    # animation function
    Points = read_points('Constant\\Points.txt')

    # i=5
    # Scalar,Point=gridder(Scalars[i],Points,grid)
    # print(Scalar,Point)
    # print(Scalar.shape,Point.shape)

    # fig=plt.figure()
    # plt.contourf(Point[0],Point[1],Scalar)
    # plt.show()
    # plt.close()
    # sys.exit()

    # fig = plt.figure()
    xmin, xmax = min(Points[0][:, 0]), max(Points[0][:, 0])
    ymin, ymax = np.min(Points[1][0, :]), np.max(Points[1][0, :])

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    # contour_opts = {'levels': np.linspace(-9, 9, 10), 'cmap':'RdBu', 'lw': 2}

    Scalar, Point = gridder(Scalars[0], Points, grid)
    # cax = ax.contourf(Point[0], Point[1], Scalar, cmap='coolwarm', vmin=np.min(Scalars), vmax=np.max(Scalars))
    # cmapp = 'gist_yarg'
    # cmapp = 'hot'

    # cax = ax.contourf(Point[0], Point[1], Scalar, 100, cmap=cmapp, vmin=np.min(Scalars), vmax=np.max(Scalars))

    def animate(i):
        ax.collections = []
        # cbar=[], cax=[]
        Scalar, Point = gridder(Scalars[i + 1], Points, grid)
        plt.gca().set_aspect('equal')
        print(i)
        # cax = ax.contourf(Point[0], Point[1], Scalar, 100, cmap=cmapp, vmin=np.min(Scalars), vmax=np.max(Scalars))
    # anim = animation.FuncAnimation(fig, animate, frames=len(times)-1, interval=1)
    # cbar   = fig.colorbar(cax,orientation='horizontal')

    plt.draw()
    plt.show()

    # save = {'bbox_inches=':'tight'}
    # anim.save('%s.gif'%Scalar_name,savefig_kwargs=save,dpi=600)
    # anim.save('%s.mp4'%Scalar_name,fps=10)

    plt.close()


# %% Contour Functions


def averaging(Scalar, axis=0):
    """
    Averages a scalar field along a specified axis (x or y).

    This function appears to perform averaging for staggered grids or specific interpolation needs.

    :param Scalar: The input 2D NumPy array representing the scalar field.
    :type Scalar: numpy.ndarray
    :param axis: The axis along which to perform averaging (0 for x-axis, 1 for y-axis). Defaults to 0.
    :type axis: int, optional
    :returns: The averaged 2D NumPy array.
    :rtype: numpy.ndarray
    """
    if axis == 0:  # x axis averaging
        nx = Scalar.shape[0] - 2
        ny = Scalar.shape[1] - 2
        Scalar1 = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i, j] = (Scalar[i + 1, j] + Scalar[i + 1, j + 1]) / 2
    if axis == 1:  # y axis averaging
        nx = Scalar.shape[0] - 2
        ny = Scalar.shape[1] - 2
        Scalar1 = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i, j] = (Scalar[i, j + 1] + Scalar[i + 1, j + 1]) / 2
    return Scalar1


def averaging_grid(Scalar, axis=0):
    """
    Averages a scalar field on a grid, likely for cell-centered representation from node data.

    :param Scalar: The input 2D NumPy array representing the scalar field.
    :type Scalar: numpy.ndarray
    :param axis: The axis along which to perform averaging (0 for x-axis, 1 for y-axis). Defaults to 0.
    :type axis: int, optional
    :returns: The averaged 2D NumPy array.
    :rtype: numpy.ndarray
    """
    if axis == 0:  # x axis averaging
        nx = Scalar.shape[0] - 1
        ny = Scalar.shape[1] - 1
        Scalar1 = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i, j] = (Scalar[i, j] + Scalar[i + 1, j]) / 2
    if axis == 1:  # y axis averaging
        nx = Scalar.shape[0]-1
        ny = Scalar.shape[1]-1
        Scalar1 = np.zeros([nx, ny])
        for i in range(nx):
            for j in range(ny):
                Scalar1[i, j] = (Scalar[i, j] + Scalar[i, j + 1]) / 2
    return Scalar1


def average_scalar(Scalar):
    """
    Average a scalar field over a 2x2 stencil to get a cell-centered value.

    :param Scalar: The input 2D NumPy array representing the scalar field.
    :type Scalar: numpy.ndarray
    :returns: The averaged 2D NumPy array (cell-centered).
    :rtype: numpy.ndarray
    """
    nx = Scalar.shape[0] - 1
    ny = Scalar.shape[1] - 1
    Scalar1 = np.zeros([nx, ny])
    for i in range(nx):
        for j in range(ny):
            Scalar1[i, j] = (Scalar[i, j] + Scalar[i + 1, j] + Scalar[i, j + 1] + Scalar[i + 1, j + 1]) / 4
    return Scalar1


def Point_average(Points):
    """
    Average grid points to get cell-centered coordinates.

    :param Points: A 3D NumPy array containing the x and y coordinates of the grid points.
                   Shape should be (2, nx, ny).
    :type Points: numpy.ndarray
    :returns: A 3D NumPy array of averaged (cell-centered) grid points.
    :rtype: numpy.ndarray
    """
    nx = Points.shape[1] - 1
    ny = Points.shape[2] - 1
    Points1 = np.zeros([2, nx, ny])
    for k in range(2):
        for i in range(nx):
            for j in range(ny):
                Points1[k, i, j] = (Points[k, i, j]
                                    + Points[k, i + 1, j]
                                    + Points[k, i, j + 1]
                                    + Points[k, i + 1, j + 1]) / 4
    return Points1


def plotting(Points, Scalar, Scalar_name, show='yes', P='no'):
    """
    Generate a contour plot of a scalar field.

    :param Points: A 3D NumPy array containing the x and y coordinates of the grid points.
                   Shape should be (2, nx, ny).
    :type Points: numpy.ndarray
    :param Scalar: The 2D NumPy array representing the scalar field to plot.
    :type Scalar: numpy.ndarray
    :param Scalar_name: The name of the scalar field (e.g., 'Temperature', 'Pressure') for the plot title.
    :type Scalar_name: str
    :param show: If 'yes', displays the plot. Otherwise, closes it without displaying. Defaults to 'yes'.
    :type show: str, optional
    :param P: If 'yes', prints the scalar array to console. Defaults to 'no'.
    :type P: str, optional
    """
    if Scalar_name == 'T':
        plt.contourf(Points[0], Points[1], Scalar, 200, cmap='coolwarm', vmin=300, vmax=350)
    else:
        plt.contourf(Points[0], Points[1], Scalar, 250, cmap='coolwarm')

    plt.title('Contour of ' + Scalar_name)
    plt.colorbar(orientation='horizontal')
    plt.gca().set_aspect('equal')
    xmin, xmax = np.min(Points[0]), np.max(Points[0])
    ymin, ymax = np.min(Points[1]), np.max(Points[1])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    if P == 'yes':
        print(Scalar)
    if show == 'yes':
        plt.show()
    else:
        plt.close()


def Contour(Scalar_name, time=-1, show='yes', P='no', grid='no'):  # scalar must be string of what you want to print
    """
    Loads and plots a contour of a specified scalar field at a given time step.

    :param Scalar_name: The name of the scalar field to plot (e.g., 'U', 'V', 'P', 'T', 'phi').
    :type Scalar_name: str
    :param time: The time step to load data from. -1 for the last time step, '0' for initial,
                 or a specific time value. Defaults to -1.
    :type time: int or str, optional
    :param show: If 'yes', displays the plot. Otherwise, closes it without displaying. Defaults to 'yes'.
    :type show: str, optional
    :param P: If 'yes', prints the scalar array to console. Defaults to 'no'.
    :type P: str, optional
    :param grid: Specifies whether to use grid-averaged values for plotting. Defaults to 'no'.
    :type grid: str, optional
    :returns: 0 if 'Results' directory is not found, otherwise plots the contour.
    :rtype: int or None
    """
    # find all times, hence end time
    times = []
    try:
        files = os.listdir('Results')
        for i in files:
            if isfloat(i):
                times.append(' Results\\' + i)
    except:
        print('Result file not there')
        return 0
    if time == -1:
        Scalar = read_scalar(times[time] + '\\' + Scalar_name + '.txt')
    elif time == '0':
        Scalar = read_scalar(str(time) + '\\' + Scalar_name + '.txt')
    else:
        Scalar = read_scalar('Results\\' + str(time) + '\\' + Scalar_name + '.txt')
    Points = read_points('Constant\\Points.txt')
    if grid == 'no':
        Points = copy.deepcopy(Point_average(Points))
        if Scalar_name == 'U':
            Scalar = copy.deepcopy(averaging(Scalar, 1))
        elif Scalar_name == 'V':
            Scalar = copy.deepcopy(averaging(Scalar, 0))
        else:
            Scalar = copy.deepcopy(Scalar[1:-1, 1:-1])
    else:
        if Scalar_name == 'U':
            Scalar = copy.deepcopy(averaging_grid(Scalar, 1))
        elif Scalar_name == 'V':
            Scalar = copy.deepcopy(averaging_grid(Scalar, 0))
        else:
            Scalar = copy.deepcopy(average_scalar(Scalar))

    plotting(Points, Scalar, Scalar_name, show, P)


def Streamlines(U_name, V_name, time=-1, show='yes', grid='no'):
    """
    Load U and V velocity fields and generates a streamline plot.

    :param U_name: The name of the U-velocity scalar file (e.g., 'U').
    :type U_name: str
    :param V_name: The name of the V-velocity scalar file (e.g., 'V').
    :type V_name: str
    :param time: The time step to load data from. -1 for the last time step, '0' for initial,
                 or a specific time value. Defaults to -1.
    :type time: int or str, optional
    :param show: If 'yes', displays the plot. Otherwise, closes it without displaying. Defaults to 'yes'.
    :type show: str, optional
    :param grid: Specifies whether to use grid-averaged values for plotting. Defaults to 'no'.
    :type grid: str, optional
    :returns: 0 if 'Results' directory is not found, otherwise plots the streamlines.
    :rtype: int or None
    """
    times = []
    try:
        files = os.listdir('Results')
        for i in files:
            if isfloat(i):
                times.append('Results\\' + i)
    except:
        print('Result file not there')
        return 0
    if time == -1:
        U = read_scalar(times[time] + '\\' + U_name + '.txt')
        V = read_scalar(times[time] + '\\' + V_name + '.txt')
    elif time == '0':
        U = read_scalar(str(time) + '\\' + U_name + '.txt')
        V = read_scalar(str(time) + '\\' + V_name + '.txt')
    else:
        U = read_scalar('Results\\' + str(time) + '\\' + U_name + '.txt')
        V = read_scalar('Results\\' + str(time) + '\\' + V_name + '.txt')
    Points = read_points('Constant\\Points.txt')
    if grid == 'no':
        Points = copy.deepcopy(Point_average(Points))
        U = copy.deepcopy(averaging(U, 1))
        V = copy.deepcopy(averaging(V, 0))
    else:
        U = copy.deepcopy(averaging_grid(U, 1))
        V = copy.deepcopy(averaging_grid(V, 0))

    speed = np.sqrt(U**2 + V**2).T
    lw = 1 * speed**0.3 / np.max(speed)

    plt.title('Streamlines')
    # plt.gca().set_aspect('equal')
    print(Points[0][:, 0], Points[1][0, :])
    print(Points.shape)
    print(U.shape)
    print(V.shape)
    np.savetxt('temp.txt', U, fmt='%0.3f')
    # stpoints = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    plt.streamplot(Points[0][:, 0], Points[1][0, :], U.T, V.T, color=speed, linewidth=lw, cmap='coolwarm', density=5)
    # plt.colorbar(orientation='horizontal')
    xmin, xmax = min(Points[0][:, 0]), max(Points[0][:, 0])
    ymin, ymax = np.min(Points[1][0, :]), np.max(Points[1][0, :])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.show()


def Quiver(U_name, V_name, time=-1, show='yes', grid='no'):
    """
    Load U and V velocity fields and generates a quiver (vector) plot.

    :param U_name: The name of the U-velocity scalar file (e.g., 'U').
    :type U_name: str
    :param V_name: The name of the V-velocity scalar file (e.g., 'V').
    :type V_name: str
    :param time: The time step to load data from. -1 for the last time step, '0' for initial,
                 or a specific time value. Defaults to -1.
    :type time: int or str, optional
    :param show: If 'yes', displays the plot. Otherwise, closes it without displaying. Defaults to 'yes'.
    :type show: str, optional
    :param grid: Specifies whether to use grid-averaged values for plotting. Defaults to 'no'.
    :type grid: str, optional
    """
    times = []
    files = os.listdir('Results')
    for i in files:
        if isfloat(i):
            times.append('Results\\'+i)
    if time == -1:
        U = read_scalar(times[time] + '\\' + U_name + '.txt')
        V = read_scalar(times[time] + '\\' + V_name + '.txt')
    elif time == '0':
        U = read_scalar(str(time) + '\\' + U_name + '.txt')
        V = read_scalar(str(time) + '\\' + V_name + '.txt')
    else:
        U = read_scalar('Results\\' + str(time) + '\\' + U_name + '.txt')
        V = read_scalar('Results\\' + str(time) + '\\' + V_name + '.txt')
    Points = read_points('Constant\\Points.txt')
    if grid == 'no':
        Points = copy.deepcopy(Point_average(Points))
        U = copy.deepcopy(averaging(U, 1))
        V = copy.deepcopy(averaging(V, 0))
    else:
        U = copy.deepcopy(averaging_grid(U, 1))
        V = copy.deepcopy(averaging_grid(V, 0))
    # speed = np.sqrt(U**2 + V**2).T
    # lw = 1 * speed**0.3 / np.max(speed)

    plt.title('Streamlines')
    plt.gca().set_aspect('equal')

    # print(Points[0][:,0].shape)
    # print(Points[1][0,:].shape)
    # sys.exit()

    plt.quiver([Points[0][:, 0], Points[1][0, :]], U.T, V.T)

    plt.colorbar(orientation='horizontal')
    xmin, xmax = min(Points[0][:, 0]), max(Points[0][:, 0])
    ymin, ymax = min(Points[1][0, :]), max(Points[1][0, :])
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.show()


def Grid_plot(show='no'):
    """
    Generate a plot of the computational grid.

    :param show: If 'no', saves the grid plot as 'grid.png'. If 'yes', displays the plot. Defaults to 'no'.
    :type show: str, optional
    :returns: 0 upon completion.
    :rtype: int
    """
    dx = read_scalar('Constant\\Dx.txt')
    dy = read_scalar('Constant\\Dy.txt')
    nx = dx.shape[0]
    ny = dx.shape[1]

    Points = np.zeros([2, nx, ny])
    for i in range(1, nx):
        for j in range(ny):
            Points[0, i, j] = Points[0, i - 1, j] + dx[i - 1, j]
    for i in range(nx):
        for j in range(1, ny):
            Points[1, i, j] = Points[1, i, j - 1] + dy[i, j - 1]
    for i in range(nx):
        for j in range(ny-1):
            x = [Points[0, i, j], Points[0, i, j + 1]]
            y = [Points[1, i, j], Points[1, i, j + 1]]
            plt.plot(x, y, c='k', lw=5.0/ny, zorder=6)

    for j in range(nx-1):
        for i in range(ny):
            x = [Points[0, j, i], Points[0, j + 1, i]]
            y = [Points[1, j, i], Points[1, j + 1, i]]
            plt.plot(x, y, c='k', lw=5.0/ny, zorder=6)

    if show == 'no':
        plt.gca().set_aspect('equal')
        plt.axis('off')
        plt.savefig('grid.png', dpi=1200, bbox_inches='tight')
        print('Grid saved !')
    elif show == 'yes':
        plt.gca().set_aspect('equal')
        plt.show()

    return 0


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


@jit(nopython=True)
def transport(T, u, v, dt, dx, dy, alpha):
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

    def RHS(alpha, T, dx, dy, i, j):
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
        rx_top = dx[(i, j)] - dx[(i - 1, j)]
        rx_bot = dx[(i, j)] + dx[(i - 1, j)]
        rx = rx_top/rx_bot

        ry_top = dy[(i, j)] - dy[(i, j - 1)]
        ry_bot = dy[(i, j)] + dy[(i, j - 1)]
        ry = ry_top / ry_bot

        T2_x = (((1 - rx) * T[(i + 1, j)]) - (2 * T[(i, j)]) + ((1 + rx) * T[(i - 1, j)])) / \
            ((dx[(i, j)]**2 + dx[(i - 1, j)]**2) / 2)
        T2_y = (((1 - ry) * T[(i, j + 1)]) - (2 * T[(i, j)]) + ((1 + ry) * T[(i, j - 1)])) / \
            ((dy[(i, j)]**2 + dy[(i, j - 1)]**2) / 2)

        rhs = alpha*(T2_x + T2_y)

        return rhs

    def Der_1(u, v, T, dx, dy, i, j):
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
        x_1 = dx[(i, j)]
        x_0 = dx[(i - 1, j)]

        y_1 = dy[(i, j)]
        y_0 = dy[(i, j - 1)]

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
            rhs = RHS(alpha, T, dx, dy, i, j)
            duTdx, dvTdy = Der_1(u, v, T, dx, dy, i, j)
            T_[(i, j)] = T[(i, j)] + (dt * (rhs - duTdx - dvTdy + Q))
    return T_


@jit(nopython=True)
def poisson(P, u, v, dt, dx, dy, rho):
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

    def Frac(dx, dy, i, j):
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
        rx_top = dx[(i, j)] - dx[(i - 1, j)]
        rx_bot = dx[(i, j)] + dx[(i - 1, j)]
        rx = rx_top/rx_bot

        ry_top = dy[(i, j)] - dy[(i, j - 1)]
        ry_bot = dy[(i, j)] + dy[(i, j - 1)]
        ry = ry_top / ry_bot

        frac_x = 4 / (dx[(i, j)]**2 + dx[(i - 1, j)]**2)
        frac_y = 4 / (dy[(i, j)]**2 + dy[(i, j - 1)]**2)
        Rx_p = 2 * (1 - rx) / (dx[(i, j)]**2 + dx[(i - 1, j)]**2)
        Rx_n = 2 * (1 + rx) / (dx[(i, j)]**2 + dx[(i - 1, j)]**2)
        Ry_p = 2 * (1 - ry) / (dy[(i, j)]**2 + dy[(i, j - 1)]**2)
        Ry_n = 2 * (1 + ry) / (dy[(i, j)]**2 + dy[(i, j - 1)]**2)
        return frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n

    def RHS(u, v, dx, dy, i, j, dt, rho):
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
        U_ = (u[(i, j)] - u[(i - 1, j)]) / dx[(i - 1, j)]
        V_ = (v[(i, j)] - v[(i, j - 1)]) / dy[(i, j - 1)]

        rhs = rho / dt * (U_ + V_)
        return rhs

    Con = 1e-2
    err = 1
    k = 0
    temp = np.zeros(P.shape)
    while (err > Con):
        temp[:, :] = P[:, :]
        k += 1
        for i in range(1, nx):
            for j in range(1, ny):
                frac_x, frac_y, Rx_p, Rx_n, Ry_p, Ry_n = Frac(dx, dy, i, j)
                rhs = RHS(u, v, dx, dy, i, j, dt, rho)
                P[(i, j)] = (frac_x + frac_y)**(-1) * ((Rx_p * P[(i + 1, j)] + Rx_n *
                                                        P[(i - 1, j)]) + (Ry_p * P[(i, j + 1)] + Ry_n * P[(i, j - 1)]) - rhs)
        if k == 100000:  # Look into this
            print('not converged', err)
            break
        err = np.max(np.abs(P - temp))
    return P


@jit(nopython=True)
def Adv(u, v, x, y, i, j, flag):
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
    x_1 = x[(i, j)]
    x_0 = x[(i - 1, j)]

    y_1 = y[(i, j)]
    y_0 = y[(i, j - 1)]

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


@jit(nopython=True)
def Diff(u, x, y, i, j):
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
    x_1 = x[(i, j)]
    x_0 = x[(i - 1, j)]

    y_1 = y[(i, j)]
    y_0 = y[(i, j - 1)]

    r_x = (x_1 - x_0)/(x_0 + x_1)
    r_y = (y_1 - y_0)/(y_0 + y_1)

    u_1 = 2 * ((1 + r_x) * u[i-1][j] - 2 * u[i][j] + (1 - r_x) * u[i + 1][j]) / (x_0**2 + x_1**2)
    u_2 = 2 * ((1 + r_y) * u[i][j - 1] - 2 * u[i][j] + (1 - r_y) * u[i][j + 1]) / (y_0**2 + y_1**2)

    return u_1 + u_2


@jit(nopython=True)
def predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta):
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

    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            S = 0.5 * ((u[i][j + 1] - u[i][j]) / y[i, j]) + ((v[i + 1][j] - v[i][j]) / x[i, j])
            nut = ((Cs*delta_g)**2)*S
            nut = min(1e-4, abs(nut))
            Nu = nu + nut
            if nut > 1e-3:
                print(nut)

            u_[i][j] = u[i][j] + dt * (Nu * (Diff(u, x, y, i, j)) - Adv(u, v, x, y, i, j, 0))
            v_[i][j] = v[i][j] + dt * (Nu * (Diff(v, x, y, i, j)) - Adv(u, v, x, y, i, j, 1) +
                                       rho * g * beta * (0.5 * (T[i][j] + T[i][j + 1]) - T_ref))  # Add Bousinessq Terms

    return u_, v_


@jit(nopython=True)
def corrector(x, y, u, v, p, dt, rho):
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

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            x_1 = x[(i, j)]
            # x_0 = x[(i - 1, j)]
            y_1 = y[(i, j)]
            # y_0 = y[(i, j - 1)]

            u_[i][j] = u[i][j] - (dt / rho) * (p[i + 1][j] - p[i][j]) / (x_1)
            v_[i][j] = v[i][j] - (dt / rho) * (p[i][j + 1] - p[i][j]) / (y_1)

    return u_, v_


@jit(nopython=True)
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


@jit(nopython=True)
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


# %% Run Script


# @numba.jit(nopython=True, parallel=True)
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
    - Generating final contour plots.
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

    buildings_dict = data["buildings"]
    b_0 = buildings_dict['building_0']
    b_1 = buildings_dict['building_1']
    b_2 = buildings_dict['building_2']

    initialise(config_path)

    x, y = read_delta(0)
    P, T, u, v, phi = read_all_scalar(0)

    running = True
    t = 0.
    ite = 0.
    while running:
        ite += 1
        print(ite)
        if int(ite) % 5 == 0:
            print('\ntime=%.3f' % t)

            if t != 0:
                write_all_scalar(P, T, u, v, phi, decimals, t)
            # sys.exit()
        t1 = time.time()

        u_, v_ = predictor(x, y, u, v, T, dt, T_ref, rho, g, nu, beta)
        t1 = timer(t1)

        p_new = poisson(P, u_, v_, dt, x, y, rho)
        t1 = timer(t1)

        u_new, v_new = corrector(x, y, u_, v_, p_new, dt, rho)
        t1 = timer(t1)

        T_new = transport(T, u, v, dt, x, y, alpha_T)

        phi_new = transport(phi, u, v, dt, x, y, alpha_pollutant)  # Pollutant Transport

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

    Contour('U', grid='yes')
    Contour('V', grid='yes')
    Contour('P', grid='yes')
    Contour('T', grid='yes')
    Contour('phi', grid='yes')

# %% Running from Spyder IDE


# Setting up config.yaml example
config = {
    'constants': {
        'rho': 1.225,
        'N': 1600,
        'M': 200,
        'xmax': 2,
        'ymax': 0.25,
        'atm': 1,
        'T_ref': 300,
        'beta': 3.333e-3,
        'nu': 1.569e-5,
        'alpha_T': 2.239e-5,
        'alpha_pollutant': 10,
        'total_t': 10,
        'dt': 0.00005,
        'dx': 0.00125,
        'dyf': 0.00025,
        'g': 2.,
        'cfl': 0.65,
        'decimals': 9
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
