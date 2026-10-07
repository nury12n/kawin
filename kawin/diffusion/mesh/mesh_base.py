'''
Defines a generic mesh that diffusion models can use. This will allow for
different coordinate systems or even different numerical schemes to be used
without changes to the diffusion models themselves

Basic idea:
    Fick's first law: J = -D*dy/dz
    Fick's second law: dc/dt = -dJ/dz

    The mesh will hold y (response variable) and z (spatial variable)
        A pair of y, z defines a node (e.g. y_i, z_i)
    The diffusion model will supply the diffusivity as D = f(y,z)

    In more general terms, we will define the flux as J_j = -sum(D_i * dy_i/dz)
    For single phase diffusion, this will account for the diffusion matrix
        J_k = -sum_j(D^n_kj * dx_j/dz)
    For homogenization model, this will account for both the homogenization function and ideal entropy contribution
        J_k = -M_K * dmu_k/dz + -eps*R*T*M_k/x_k * dx_k/dz
    Thus, the diffusion model will supply a pair of D and y
        E.g. for single phase diffusion - (D^n_k1, x_1), (D^n_k2, x_2), ...
             for homogenization         - (M_k, mu_k), (eps*R*T*M_k/x_k, x_k)

    Finally, since D is computed at the node but is to represent the middle of two nodes,
    we will have the diffusion model supply an averaging function for D:
        Arithmetic mean - D_avg = 1/2*(D_i + D_j)
        Geometric mean  - D_avg = sqrt(D_i * D_j)
        Harmonic mean   - D_avg = (1/2 * (D_i^-1 + D_j^-1))^-1

    TODO: I want an averaging function that can account for both log scale and negative numbers, which are two not so friendly terms
'''
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, Protocol
import numpy as np

DiffusionPair = namedtuple('DiffusionPair', ['diffusivity', 'response', 'averaging_function', 'at_node_function'], defaults=[None, None, None, None])

def no_change_at_node(x):
    '''
    Default function for at node diffusivity
    '''
    return x

def arithmetic_mean(x):
    '''
    Arithmetic mean - avg = 1/2*(x_i + x_j)
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    '''
    return np.average(x, axis=0)

def geometric_mean(x):
    '''
    Geometric mean - avg = sqrt(x_i * x_j)
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: this is only valid when x_i and x_j are the same sign
    '''
    return np.power(np.prod(x, axis=0), 1/len(x))

def log_mean(x):
    '''
    Log mean - avg = exp(1/2*(log(x_i)+log(x_j)))
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: this is only valid for positive x
    '''
    return np.exp(np.average(np.log(x), axis=0))

def harmonic_mean(x):
    '''
    Harmonic mean - avg = (1/2 * (x_i^-1 + x_j^-1))^-1
    Ds should be in the shape of (m x N x y)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        Average is taken along m
    Note: in the limit of x_i or x_j -> 0, the mean is 0. In practice, this is undefined, so we
          convert all inf/nan to 0
    '''
    avg = np.power(np.sum(np.power(x, -1), axis=0), -1)
    avg[~np.isfinite(avg)] = 0
    return avg

def _format_response(response_var: int|str):
    if isinstance(response_var, int):
        response_var = f'R{response_var}'
    return response_var

def _get_response_variables(responses: list[int|str]) -> tuple[int, list[str]]:
    '''
    This returns the number of response variables and names for each variable

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length
    '''
    if isinstance(responses, int):
        return responses, [_format_response(i) for i in range(responses)]
    else:
        return len(responses), list(responses)

def _get_response_index(response_var: int|str, responses: list[str]):
        '''
        Given the response variable, return the corresponding index
        '''
        return responses.index(response_var)

def _format_spatial(z):
    '''
    Make z into (N,r) array
    '''
    z = np.atleast_2d(z)
    if z.shape[0] == 1:
        return z.T
    else:
        return z

def diffusive_flux(D, r_high, r_low, dz):
    '''Flux = -D (r1 - r0) / dz'''
    return -D * (r_high - r_low) / dz

class ProfileFunction(Protocol):
    def __call__(self, z: np.ndarray) -> np.ndarray:
        '''
        Function that takes in a list of coordinates and returns compositions at
        the corresponding coordinates

        Parameters
        ----------
        z: np.ndarray
            Array of floats with shape of (N,d)
            where N is number of points and d is number of dimensions

        Returns
        -------
        np.ndarray of shape (N,r) where N is number of points and
        r is number of response variables

        Note that the number of response variables can be specific to the
        profile function. It does not have to correspond to the responses in
        the mesh as long as the ProfileBuilder knows what response variables
        the ProfileFunction corresponds to
        '''
        ...

class ProfileBuilder:
    '''
    Stores build steps to construct a response profile in a mesh
    Note that all steps are additive to the response profile

    Parameters
    ----------
    steps: list[tuple(callable, int|str | list[int|str])] (optional)
        List of build steps to initialize
    '''
    def __init__(self, steps = []):
        self.build_steps = []
        for step in steps:
            self.add_build_step(step[0], step[1])

    def add_build_step(self, func: ProfileFunction, response_var: int|str|list[int|str] = 0):
        '''
        Adds a build step to construct the initial profile

        Parameters
        ----------
        func: ProfileFunction
            Function that takes in a set of coordinates and returns the profile values
            at coordinates
        responseVar: int or str or list of int or str (optional)
            Response variable(s) that func will return
            This does not have to match exactly to the mesh variables, but only needs
            to be a subset
        '''
        self.build_steps.append((func, np.atleast_1d(response_var)))

class ConstantProfile(ProfileFunction):
    '''
    Constant value across profile
    '''
    def __init__(self, value):
        self.value = np.atleast_1d(value)

    def __call__(self, z):
        return np.squeeze(self.value[np.newaxis,:]*np.ones((_format_spatial(z).shape[0], self.value.shape[0])))

class DiracDeltaProfile(ProfileFunction):
    '''
    Zero at all values except at z
    Note: if a mesh does not have a node exactly at z, it will choose the closest node
    '''
    def __init__(self, z, value):
        self.value = np.atleast_1d(value)
        self.z = np.atleast_1d(z)

    def __call__(self, z):
        z = _format_spatial(z)
        y = np.zeros((z.shape[0], self.value.shape[0]))
        r = np.sqrt(np.sum((z-self.z[np.newaxis,:])**2, axis=1))
        nearest_index = np.argmin(r)
        y[nearest_index,:] = self.value
        return np.squeeze(y)

class GaussianProfile(ProfileFunction):
    '''
    Gaussian function centered around z with standard deviation of sigma
    Scaled to maxValue
    '''
    def __init__(self, z, sigma, max_value):
        self.max_value = np.atleast_1d(max_value)
        self.z = np.atleast_1d(z)

        if np.isscalar(sigma):
            self.sigma = sigma*np.ones(len(self.z))
        else:
            self.sigma = np.atleast_1d(sigma)

    def __call__(self, z):
        z = _format_spatial(z)
        r2 = (z-self.z[np.newaxis,:])**2
        y = np.exp(-np.sum(r2 / self.sigma[np.newaxis,:]**2, axis=1))
        # maximum value of y will be 1, so we can just multiply the max value
        y = y[:,np.newaxis]*self.max_value[np.newaxis,:]
        return np.squeeze(y)

class BoundedRectangleProfile(ProfileFunction):
    '''
    Defines rectangle with lower and upper corners
    innerValue corresponds to value inside the rectangle while
    outerValue corresponds to value outside the rectangle (defaults to 0)
    '''
    def __init__(self, lower_z, upper_z, inner_value, outer_value = 0):
        self.iv = np.atleast_1d(inner_value)
        if outer_value == 0:
            self.ov = np.zeros(self.iv.shape)
        else:
            self.ov = np.atleast_1d(outer_value)
        self.lz = np.atleast_1d(lower_z)
        self.uz = np.atleast_1d(upper_z)

    def __call__(self, z):
        z = _format_spatial(z)
        y = self.ov[np.newaxis,:]*np.ones((z.shape[0], self.ov.shape[0]))
        indices = (self.lz[0] < z[:,0]) & (z[:,0] <= self.uz[0])
        for i in range(1, len(self.lz)):
            indices &= (self.lz[i] < z[:,i]) & (z[:,i] <= self.uz[i])
        y[indices,:] = self.iv
        return np.squeeze(y)

class BoundedEllipseProfile(ProfileFunction):
    '''
    Defines ellipse centered at z with radii r
    innerValue corresponds to value inside the rectangle while
    outerValue corresponds to value outside the rectangle (defaults to 0)
    '''
    def __init__(self, z, r, inner_value, outer_value = 0):
        self.iv = np.atleast_1d(inner_value)
        self.ov = np.atleast_1d(outer_value)
        self.z = np.atleast_1d(z)
        if np.isscalar(r):
            self.r = r*np.ones(len(self.z))
        else:
            self.r = np.atleast_1d(r)

    def __call__(self, z):
        z = _format_spatial(z)
        y = self.ov[np.newaxis,:]*np.ones((z.shape[0], self.ov.shape[0]))
        r2 = np.sum((z-self.z[np.newaxis,:])**2 / self.r[np.newaxis,:]**2, axis=1)
        y[r2 < 1,:] = self.iv
        return y

class MeshBase(ABC):
    '''
    Abstract mesh class that defines basic methods to be used in a diffusion model

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length

    Attributes
    ----------
    num_responses: int
        Number of response variables
    responses: list[str]
        Names of response variables
    N: int
        Total number of nodes
    z: np.ndarray
        Values of spatial coordinates corresponding to y
        Assumed to be shape [N,dims]
    '''
    def __init__(self, responses, N, dims):
        self.num_responses, self.responses = _get_response_variables(responses)
        self.dims = dims
        self.N = N
        self.z = np.zeros((N, dims))
        self.profile = ProfileBuilder()

    def build_response_profile(self):
        y = np.zeros((self.N, self.num_responses))
        for step, response_vars in self.profile.build_steps:
            y_step = np.atleast_2d(step(self.z))
            if y_step.shape[0] == 1:
                y_step = y_step.T

            for i, rv in enumerate(response_vars):
                index = _get_response_index(rv, self.responses)
                y[:,index] += y_step[:,i]

        return y

    @abstractmethod
    def compute_dxdt(self, pairs: list[DiffusionPair]):
        '''
        Given list of diffusivity and response pairs, compute dX/dt from diffusion fluxes

        Parameters
        ----------
        pairs: list[DiffusionPair]
            Tuple will contain (diffusivity, response, averaging function, at node function)
            Note that the diffusivity and response will be in the form of [N,e] (corresponding
            to getDiffusivityCoordinates and getResponseCoordinates), which may not be the
            shape of self.y. This function is responsible for accounting for this difference
            and the shape of dxdt and y should be the same

        Returns
        -------
        dxdt: np.ndarray
            Must be same shape as y
        '''
        pass

    def get_response_coordinates(self, y):
        '''Returns y as [N,e] and z as [N,dims] arrays to compute response terms'''
        return y, self.z

    def get_diffusivity_coordinates(self, y):
        '''
        Returns y as [N,e] and z as [N,D] arrays to compute diffusivity terms
        We decouple this from the response coordinates to allow for different ways
        of computing D_i+1/2. These options may include:
            a) By computing D_i and D_i+1 at (y_i, z_i) and (y_i+1, z_i+1)
               and averaging (default implementation)
            b) By computing D_i+1/2 at (y_i+1/2, z_i+1/2)
        '''
        return self.get_response_coordinates(y)