from typing import Protocol

import numpy as np

from pycalphad import variables as v

from kawin.state_process import ModelState, DifferentialEquationProcess
from kawin.thermo.thermodynamics import Thermodynamics, ThermodynamicFunction, enumerate_conditions
from kawin.thermo.mobility import x_to_u_frac, expand_x_frac, u_to_x_frac, expand_u_frac, interstitials
from kawin.diffusion.mesh.mesh_base import MeshBase, ProfileBuilder, BoundaryCondition

# TODO: can we decare MobilityFunction and InterdiffusivityFunction as
# a function only of conditions. Ideally, I want the diffusion models to
# be independent of Thermodynamics.

class DiffusionState(ModelState):
    def __init__(self):
        super().__init__()
        self.u = None

    def x(self, all_elements):
        u_ext = expand_u_frac(self.u, all_elements, interstitials)
        return u_to_x_frac(u_ext, all_elements, interstitials)

class DiffusionModel(DifferentialEquationProcess):
    def __init__(self, mesh: MeshBase, elements: list[str], min_composition: float = 1e-8):
        super().__init__()
        self.mesh = mesh
        self.all_elements = elements
        self.ref_element = elements[0]
        self.elements = elements[1:]
        self.min_composition = min_composition

    def initialize_state(self, profile_builder: ProfileBuilder, bcs: BoundaryCondition = None, state: DiffusionState = None):
        if state is None:
            state = DiffusionState()
        state.u = self.mesh.build_response_profile(profile_builder, bcs)
        usum = np.sum(state.u, axis=1)
        if np.any(usum < 0) or np.any(usum > 1):
            raise Exception('Some compositions sum up to below 0 or above 1')
        scaled_min_comp = len(self.all_elements)*self.min_composition
        state.u[state.u > scaled_min_comp] = state.u[state.u > scaled_min_comp] - scaled_min_comp
        state.u[state.u < scaled_min_comp] = self.min_composition
        u_full = expand_x_frac(state.u, self.all_elements, self.ref_element)
        u_full = x_to_u_frac(u_full, self.all_elements, interstitials, return_usum=False)
        state.u = u_full[:,1:]
        return state

    def apply_dxdt(self, state: DiffusionState, dxdt: any, dt: float):
        state.u += dxdt*dt

    def finalize(self, state: DiffusionState):
        # clamp composition
        # Bound u-fraction to not be invalid values
        #   For substitutional - u-fraction = (0, 1)
        #   For interstitial   - u-fraction = (0, inf)
        for i, e in enumerate(self.elements):
            if e not in interstitials:
                state.u[:,i] = np.clip(state.u[:,i], self.min_composition, 1-self.min_composition)
            else:
                state.u[:,i] = np.clip(state.u[:,i], self.min_composition, None)

# -- Temperature protocol and some implementations --
class DiffusionModelTemperature(Protocol):
    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        ...

class IsothermalTemperature:
    def __init__(self, T: float):
        self.T = T

    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        return self.T*np.ones(z.shape[0])

class HeatTreatmentProfile:
    def __init__(self, times: list[float], temperatures: list[float]):
        self.times = times
        self.temperatures = temperatures

    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        return np.interp(time/3600, self.times, self.temperatures[1], self.temperatures[1][0], self.temperatures[1][-1]) * np.ones(z.shape[0])

# -- Hash function protocol for LookupTable implementation --
class HashFunction(Protocol):
    def __call__(conditions: dict[v.StateVariable, float], precision: int, *args, **kwargs) -> str:
        ...

def hash_conditions(conditions: dict[v.StateVariable, float], precision: int, *args, **kwargs) -> str:
    condition_vals = [float(np.round(conditions[key]*precision).astype(np.int32)) for key in sorted(conditions.keys(), key=str)]
    return hash(tuple(condition_vals))

class LookupTable:
    '''
    Wrapper around a thermodynamic function that stores a cache
    Implements a hash table that stores mobility, phases, phase fractions and
    chemical potentials for a given (composition, temperature) pair
    '''
    def __init__(
            self, function: ThermodynamicFunction,
            hash_function: HashFunction=hash_conditions,
            squeeze_function = lambda x: x if len(x)>1 else x[0],
            precision: int=4):
        self.cache = {}
        self.function = function
        self.hash_function = hash_function
        self.set_precision(precision)
        self.squeeze_function = squeeze_function

    def set_precision(self, s: int):
        '''
        Sets sensitivity of the hash table by significant digits

        For example, if a composition set is (0.5693, 0.2937) and s = 3, then
        the hash will be stored as (0.569, 0.294)

        Lower s values will give faster simulation times at the expense of accuracy

        Parameters
        ----------
        s : int
            Number of significant digits to keep for the hash table
        '''
        self.precision = np.power(10, int(s))

    def __call__(self, therm: Thermodynamics, conditions: dict[v.StateVariable, float], *args, **kwargs):
        @enumerate_conditions(self.squeeze_function)
        def query_lookup_table(therm: Thermodynamics, conditions: dict[v.StateVariable, float], *args, **kwargs):
            hash_value = self.hash_function(conditions, self.precision, *args, **kwargs)
            if hash_value in self.cache:
                val = self.cache[hash_value]
            else:
                val = self.function(therm, conditions, *args, **kwargs)
                self.cache[hash_value] = val
            return val
        return query_lookup_table(therm, conditions, *args, **kwargs)
