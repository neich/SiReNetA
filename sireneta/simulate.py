# -*- coding: utf-8 -*-
# Copyright 2024, Gorka Zamora-López and Matthieu Gilson.
# Contact: gorka@zamora-lopez.xyz
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simulate the temporal solution x(t) of the nodes
=================================================

This module contains functions to run simulations of the different canonical
models on networks.

Discrete-time models
---------------------
Sim_DiscreteCascade (simDC)
    Simulates temporal evolution of the nodes for the discrete cascade.
Sim_RandomWalk (simRW)
    Simulates temporal evolution of the nodes for the random walks.

Continuous-time models
-----------------------
Sim_ContCascade (simCC)
    Simulates temporal evolution of the nodes for the continuous cascade.
Sim_LeakyCascade (simLC)
    Simulates temporal evolution of the nodes for the leaky-cascade model.
Sim_OrnsteinUhlenbeck (simMOU)
    Simulates temporal evolution for the multivariate Ornstein-Uhlenbeck.
Sim_ContDiffusion (simCD)
    Simulates temporal evolution of the nodes for the simple diffustion model.


**Reference and Citation**

1) G. Zamora-Lopez and M. Gilson "An integrative dynamical perspective for graph
theory and the analysis of complex networks" arXiv:2307.02449 (2023).
DOI: `https://doi.org/10.48550/arXiv.2307.02449
<https://doi.org/10.48550/arXiv.2307.02449>`_
"""
# TODO: Shall we call these functions as Sim_DiscreteCascade(), Sim_RandomWalk, etc?

# Standard libary imports

# Third party packages
import numpy as np
# Local imports from sireneta
from . import io_helpers



## DISCRETE-TIME CANONICAL MODELS #############################################
def Sim_DiscreteCascade(con, X0=1.0, tmax=10):
    """Simulates temporal evolution of the nodes for the discrete cascade.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = A x(t),

    If A is a positive definite connectivity  matrix, then the solutions
    xt grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    tmax : integer, optional
        The duration of the simulation, discrete time steps.

    Returns
    -------
    Xt : ndarray (2d) of shape (tmax+1,N)
        Time-courses of the N nodes.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Initialise the output array and enter the initial conditions
    nt = round(tmax) + 1
    Xt = np.zeros((nt,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nt):
        Xt[t] = np.dot(con, Xt[t-1])

    return Xt

def Sim_RandomWalk(con, X0=1.0, tmax=10):
    """Simulates temporal evolution of the nodes for the random walks.

    It returns the time-series of the nodes for the discrete cascade model

            x(t+1) = T x(t),

    where T is the transition probability matrix. The solutions are computed
    recursively iterating the equation above. If A is a positive definite
    connectivity matrix the solutions converge to x_i(inf) values proportional
    to the input degree of the nodes.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Number of walkers (agents) starting at each node. If scalar value is
        entered, 'X0 = c', all nodes are initialised with c walkers (same
        initial conditions). Default value, X0 = 1.0. If a 1d-array is entered,
        X0[i] walkers are initialised at each node.
    tmax : integer, optional
        The duration of the simulation, discrete time steps.

    Returns
    -------
    Xt : ndarray (2d) of shape (tmax+1,N)
        Time-courses of the N nodes. Xt[t,j] is the expected number of walker
        in node j at time t.
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Compute the transition probability matrix
    N = len(con)
    Tmat = con / con.sum(axis=0)    # Assumes Aij = 1 if j -> i
    if np.isnan(Tmat.min()):
        Tmat[np.isnan(Tmat)] = 0
    # Initialise the output array and enter the initial conditions
    nt = round(tmax) + 1
    Xt = np.zeros((nt,N), np.float64)
    Xt[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nt):
        Xt[t] = np.dot(Tmat, Xt[t-1])

    return Xt


## CONTINUOUS-TIME CANONICAL MODELS ###########################################
def Sim_ContCascade(con, X0=1.0, noise=None, tmax=10, timestep=0.01):
    """Simulates the temporal evolution of the nodes for the continuous cascade.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) model of nodes coupled via connectivity matrix A,
    with no local dynamics at the nodes.

            xdot = A x(t).

     If A is a positive definite connectivity  matrix, then the solutions
     xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is `nt = round(tmax*timestep) + 1`.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    if noise is not None and noise.dtype != np.float64:
        noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Initialise the output array
    nt = round(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), dtype=np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    if noise is None:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(con,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup
    else:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(con,Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * xcoup + noise[t]

    return Xdot

def Sim_LeakyCascade(con, X0=1.0, tau=1.0, tmax=10, timestep=0.01):
    """Simulates temporal evolution of the nodes for the leaky-cascade model.

    It solves the differential equation for the linear propagation model of
    nodes coupled via connectivity matrix A, and a dissipative term leaking
    a fraction of the activity.

            xdot(t) = - x(t) / tau + A x(t).

     With λmax being the largest eigenvalue of the (positive definite) matrix A,
     - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
     time and the solutions for all nodes converge to zero.
     - If tau = tau_max, all nodes converge to x_i(t) = 1.
     - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    tau : real value or ndarray (1d) of length N, optional
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[j] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`. Default `tau = 1.0` is probably too large
        for most real networks and will diverge. If so, enter a `tau` smaller
        than the spectral diameter (λ_max) of `con`.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is `nt = round(tmax*timestep) + 1`.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    tau = io_helpers.validate_tau(tau, N)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if X0.dtype != np.float64:      X0 = X0.astype(np.float64)
    if tau.dtype != np.float64:     tau = tau.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Conver the time-constants into decay rations
    alphas = 1./tau
    # Initialise the output array
    nt = round(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nt):
        Xpre = Xdot[t-1]
        # Calculate the input to nodes due to couplings
        xcoup = np.dot(con,Xpre)
        # Integration step
        Xdot[t] = Xpre + timestep * ( -alphas * Xpre + xcoup )

    return Xdot

def Sim_OrnsteinUhlenbeck(con, X0=1.0, tau=1.0, noise=1.0, tmax=10, timestep=0.01):
    """Simulates temporal evolution for the multivariate Ornstein-Uhlenbeck.

    It solves the differential equation for the linear propagation model of
    nodes coupled via connectivity matrix A, and a dissipative term leaking
    a fraction of the activity under Gaussian noise D added at every node.

            xdot(t) = - x(t) / tau + A x(t) + D(t)

     With λmax being the largest eigenvalue of the (positive definite) matrix A,
     - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
     time and the solutions for all nodes converge to zero.
     - If tau = tau_max, all nodes converge to x_i(t) = 1.
     - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    tau : real value or ndarray (1d) of length N, optional
        The decay time-constants of the nodes. If a scalar value is entered,
        `tau = c`, then all nodes will be assigned the same value `tau[j] = 2`
        (identical nodes). If an 1d-array is entered, each node i is assigned
        decay time-constant `tau[i]`. Default `tau = 1.0` is probably too large
        for most real networks and will diverge. If so, enter a `tau` smaller
        than the spectral diameter (λ_max) of `con`.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is `nt = round(tmax*timestep) + 1`.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    tau = io_helpers.validate_tau(tau, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:     con = con.astype(np.float64)
    if X0.dtype != np.float64:      X0 = X0.astype(np.float64)
    if tau.dtype != np.float64:     tau = tau.astype(np.float64)
    if noise.dtype != np.float64:   noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    # Convert the time-constants into decay rations
    alphas = 1./tau
    # Initialise the output array
    nt = round(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    for t in range(1,nt):
        Xpre = Xdot[t-1]
        # Calculate the input to nodes due to couplings
        xcoup = np.dot(con,Xpre)
        # Integration step
        Xdot[t] = Xpre + timestep * ( -alphas * Xpre + xcoup ) + noise[t]

    return Xdot

def Sim_ContDiffusion(con, X0=1.0, alpha=1.0, noise=None, tmax=10, timestep=0.01):
    """Simulates the temporal evolution of the nodes for the continuous diffusion.

    It solves the differential equation for the simplest possible linear
    dynamical (propagation) linear model of nodes coupled via diffusive coupling,
    with no local dynamics at the nodes.

        xdot_i = alpha * A_ij * (x_i - x_j) .

    In matrix form, this equation is represented as:

        xdot(t) = -D x(t) + A x(t) =  L x(t),

    where D is the diagonal matrix with the input degrees ink_i in the diagonal
    and L = -D + A is the graph Laplacian matrix. This equation is also known
    as the heat equation for graphs, where `alpha` is a thermal diffusivite
    parameter controlling for how fast heat propagates.

    Parameters
    ----------
    con : ndarray (2d) of shape (N,N).
        The connectivity matrix of the network.
    X0 : scalar or ndarray (1d) of length N, optional
        Initial values of the nodes at time t = 0. If scalar value is entered,
        `X0 = c`, all nodes are initialised as `X0[i] = c` (same initial conditions).
        Default value, `X0 = 1.0`. If a 1d-array is entered, each node i is
        assigned initial value `X0[i]`.
    alpha : scalar.
        Diffusivity or thermal diffusitivity parameter.
    noise : None, scalar or ndarray (2d) of shape (nt,N), optional
        Additive noise. If `noise = None` simulation is run without noise.
        If scalar `noise = c`, a Gaussian white noise, centered at zero and
        variance 'c' is applied to all nodes. Independent for each node.
        Also, `noise = arr` accepts ndarray (arr) of shape (nt, N) with noise
        signals precomputed by the user.
    tmax : scalar, optional
        Duration of the simulation, arbitrary time units.
    timestep : scalar, optional
        Time-step of the numerical integration.

    Returns
    -------
    Xdot : ndarray (2d) of shape (nt,N)
        Time-courses of the N nodes.

    NOTE
    ----
    Total number of integration steps (samples) is `nt = round(tmax*timestep) + 1`.
    - Simulation runs from t=0 to t=tmax.
    - Integration goes from it=0 to it=nt, with `Xdot[0] = X0`.
    - The sampled time points are `tpoints = np.arange(0,tmax+timestep,timestep)`
    """
    # 0) HANDLE AND CHECK THE INPUTS
    io_helpers.validate_con(con)
    N = len(con)
    X0 = io_helpers.validate_X0(X0, N)
    noise = io_helpers.validate_noise(noise, N,tmax,timestep)

    if tmax <= 0.0: raise ValueError("'tmax' must be positive")
    if timestep <= 0.0: raise ValueError( "'timestep' must be positive")
    if timestep >= tmax: raise ValueError("'timestep' must be smaller than 'tmax'")

    # Ensure all arrays are of same dtype (float64)
    if con.dtype != np.float64:    con = con.astype(np.float64)
    if X0.dtype != np.float64:     X0 = X0.astype(np.float64)
    alpha = np.float64(alpha)
    if noise is not None and noise.dtype != np.float64:
        noise = noise.astype(np.float64)

    # 1) PREPARE FOR THE SIMULATION
    ink = con.sum(axis=0)
    # Lmat = - ink * np.identity(N, dtype=np.float64) + con
    # Initialise the output array
    nt = round(tmax / timestep) + 1
    Xdot = np.zeros((nt, N), np.float64, order='C')
    # Enter the initial conditions
    Xdot[0] = X0

    # 2) RUN THE SIMULATION
    if noise is None:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(con,Xpre) - ink * Xpre
            # xcoup = np.dot(Lmat, Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * alpha * xcoup
    else:
        for t in range(1,nt):
            Xpre = Xdot[t-1]
            # Calculate the input to nodes due to couplings
            xcoup = np.dot(con,Xpre) - ink * Xpre
            # xcoup = np.dot(Lmat, Xpre)
            # Integration step
            Xdot[t] = Xpre + timestep * alpha * xcoup + noise[t]

    return Xdot



## DEFINE ALIASES FOR SHORTER NAMES ############################################
simDC = Sim_DiscreteCascade
simRW = Sim_RandomWalk
simCC = Sim_ContCascade
simLC = Sim_LeakyCascade
simMOU = Sim_OrnsteinUhlenbeck
simCD = Sim_ContDiffusion




##
