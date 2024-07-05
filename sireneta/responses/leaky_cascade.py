import logging

from sireneta import io_helpers
from sireneta.responses.base_response import BaseResponse
import numpy as np
import scipy

logger = logging.getLogger(__name__)



class LeakyCascade(BaseResponse):
    caselist = ['regressed', 'full', 'intrinsic']

    def __init__(self):
        super().__init__()
        self.case = 'regressed'
        self.normed = False
        self.S0 = 1.0
        self.tau =1.0
        self.tmax = 10
        self.timestep = 0.1

        self._responses = None

    @property
    def responses(self):
        return self._responses

    def configure(self, **kwargs):
        super().configure(**kwargs)
        for arg, value in kwargs.items():
            if hasattr(self, arg):
                setattr(self, arg, value)
            else:
                raise Exception(f"Class LeakyCascade does not have attribute {arg}")

        self.check_config()
        return self

    def check_config(self):
        super().check_config()
        super()._check_attrs('S0', 'tau', 'tmax', 'timestep', 'case')
        self.S0 = io_helpers.validate_S0(self.S0, self.N)
        self.tau = io_helpers.validate_tau(self.tau, self.N)
        if self.tmax <= 0.0: raise ValueError("'.tmax' must be positive")
        if self.timestep <= 0.0: raise ValueError( "'.timestep' must be positive")
        if self.timestep >= self.tmax: raise ValueError("'.timestep' must be smaller than '.tmax'")
        if self.S0.dtype != np.float64:      self.S0 = self.S0.astype(np.float64)
        if self.tau.dtype != np.float64:     self.tau = self.tau.astype(np.float64)
        if self.case not in self.caselist:
            raise ValueError( "Please enter one of accepted cases: %s" %str(caselist) )

    def Jacobian_LeakyCascade(self):
        """Calculates the Jacobian matrix for the leaky-cascade dynamical system.

        NOTE: This is the same as the Ornstein-Uhlenbeck process on a network.

        TODO: RETHINK THE NAME OF THIS FUNCTION. MERGE DIFFERENT JACOBIAN GENERATOR
        FUNCTIONS INTO A SINGLE FUNCTION !?

        Parameters
        ----------
        con : ndarray (2d) of shape (N,N).
            The connectivity matrix of the network.
        tau : real value or ndarray (1d) of length N.
            The decay time-constants of the nodes. If a scalar value is entered,
            `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
            (identical nodes). If an 1d-array is entered, each node i is assigned
            decay time-constant `tau[i]`.

        Returns
        -------
        jac : ndarray (2d) of shape (N,N)
            The Jacobian matrix for the MOU dynamical system.
        """

        # CALCULATE THE JACOBIAN MATRIX
        jac = -1.0 / self.tau * np.identity(self.N, dtype=np.float64) + self.con

        return jac

    def simulate(self):
        """Computes the pair-wise responses over time for the leaky-cascade model.

        TODO: DECIDE ABOUT THE 'normed' PARAMETER.

        Given a connectivity matrix A, where Aij represents the (weighted)
        connection from j to i, the response matrices Rij(t) encode the temporal
        response observed at node i due to a short stimulus applied on node j at
        time t=0.
        The leaky-cascade is the time-continuous and variable-continuous linear
        propagation model represented by the following differential equation:

                xdot(t) = - x(t) / tau + A x(t).

        where tau is a leakage time-constant for a dissipation of the flows through
        the nodes. This model is reminiscent of the multivariate Ornstein-Uhlenbeck
        process, when additive Gaussian white noise is included.
        Given λmax is the largest eigenvalue of the (positive definite) matrix A, then
        - if tau < tau_max = 1 / λmax, then the leakage term dominates in the long
        time and the solutions for all nodes converge to zero.
        - If tau = tau_max, all nodes converge to x_i(t) = 1.
        - And, if tau < tau_max, then time-courses xdot(t) grow exponentially fast.

        Parameters
        ----------
        con : ndarray (2d) of shape (N,N).
            The connectivity matrix of the network.
        self.S0 : scalar or ndarray (1d) of length N or ndarray of shape (N,N), optional
            Amplitude of the stimuli applied to nodes at time t = 0.
            If scalar value given, `self.S0 = c`, all nodes are initialised as `self.S0[i] = c`
            Default, `self.S0 = 1.0` represents a unit perturbation to all nodes.
            If a 1d-array is given, stimulus `self.S0[i]` is initially applied at node i.
        tau : real value or ndarray (1d) of length N, optional
            The decay time-constants of the nodes. If a scalar value is entered,
            `tau = c`, then all nodes will be assigned the same value `tau[i] = 2`
            (identical nodes). If an 1d-array is entered, each node i is assigned
            decay time-constant `tau[i]`. Default `tau = 1.0` is probably too large
            for most real networks and will diverge. If so, enter a `tau` smaller
            than the spectral diameter (λ_max) of `con`.
        self.tmax : scalar, optional
            Duration of the simulation, arbitrary time units.
        self.timestep : scalar, optional
            Temporal step (resolution) between consecutive calculations of responses.
        case : string (optional)
            - 'full' Computes the responses a given by the Green's function of the
            Jacobian of the system: e^{Jt} with J = - I/tau + A.
            - 'intrinsic' Computes the trivial responses due to the leakage through
            the nodes: e^{J0t} with J0 = I/tau. This represents a 'null' case where
            the network is empty (has no links) and the initial inputs passively
            leak through the nodes without propagating.
            - 'regressed' Computes the network responses due to the presence of the
            links: e^{Jt} - e^{J0t}. That is, the 'full' response minus the passive,
            'intrinsic' leakage.
        normed : boolean (optional)
            DEPRECATED. If True, normalises the tensor by a scaling factor, to make networks
            of different size comparable.

        Returns
        -------
        resp_matrices : ndarray (3d) of shape (self.tmax+1,N,N)
            Temporal evolution of the pair-wise responses. The first time point
            contains the matrix of inputs. Entries `resp_matrices[t,i,j]` represent
            the response of node i at time t, due to an initial perturbation on j.

        NOTE
        ----
        Simulation runs from t=0 to t=self.tmax, in sampled `self.timestep` apart. Thus,
        simulation steps go from it=0 to it=nt, where `nt = int(self.tmax*self.timestep) + 1`
        is the total number of time samples (number of response matrices calculated).
        Get the sampled time points as `tpoints = np.arange(0,self.tmax+self.timestep,self.timestep)`.
        """


        # 1) PREPARE FOR THE CALCULATIONS
        # Initialise the output array and enter the initial conditions
        self.nsteps = nt = int(self.tmax / self.timestep) + 1
        resp_matrices = np.zeros((nt, self.N, self.N), dtype=np.float64 )
        # Compute the Jacobian matrices
        jac = self.Jacobian_LeakyCascade()
        jacdiag = np.diagonal(jac)
        # Convert the stimuli into a matrix
        if self.S0.ndim in [0,1]:
            self.S0mat = self.S0 * np.identity(self.N, dtype=np.float64)

        if self.case == 'full':
            for it in range(nt):
                t = it * self.timestep
                # Calculate the Green's function at time t
                green_t = scipy.linalg.expm(jac * t)
                # Calculate the pair-wise responses at time t
                resp_matrices[it] = np.matmul( green_t, self.S0mat )

        elif self.case == 'intrinsic':
            for it in range(nt):
                t = it * self.timestep
                # Calculate the Green's function (of an empty graph) at time t
                greendiag_t = np.diag( np.exp(jacdiag * t) )
                # Calculate the pair-wise responses at time t
                resp_matrices[it] = np.matmul( greendiag_t, self.S0mat )

        elif self.case == 'regressed':
            for it in range(nt):
                t = it * self.timestep
                # Calculate the Green's function (of the full system) at time t
                green_t = scipy.linalg.expm(jac * t)
                # Calculate the Green's function (of an empty graph) at time t
                greendiag_t = np.diag( np.exp(jacdiag * t) )
                # Calculate the pair-wise responses at time t
                resp_matrices[it] = np.matmul( green_t - greendiag_t, self.S0mat )

        # 2.2) Normalise by the scaling factor
        if self.normed:
            scaling_factor = np.abs(1./jacdiag).sum()
            resp_matrices /= scaling_factor

        self._responses = resp_matrices
