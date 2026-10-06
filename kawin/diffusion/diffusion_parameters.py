from collections import namedtuple
from typing import Callable, Protocol
from functools import wraps

import numpy as np

from pycalphad import variables as v

#from kawin.thermo.utils import _process_xT_arrays
#from kawin.thermo import GeneralThermodynamics
#from kawin.thermo.Mobility import mobility_from_composition_set, x_to_u_frac, interstitials

from kawin.thermo.thermodynamics import Thermodynamics, ThermodynamicFunction, get_eq, enumerate_conditions
from kawin.thermo.diffusivity import InterdiffusivityFunction

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

class DiffusionModelTemperature(Protocol):
    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        ...

class IsothermalTemperature:
    def __init__(self, T: float):
        self.T = T

    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        return self.T*np.ones(z.shape)

class HeatTreatmentProfile:
    def __init__(self, times: list[float], temperatures: list[float]):
        self.times = times
        self.temperatures = temperatures

    def __call__(self, time: float, z: np.ndarray) -> np.ndarray:
        return np.interp(time/3600, self.times, self.temperatures[1], self.temperatures[1][0], self.temperatures[1][-1]) * np.ones(z.shape)

# def _computeSingleMobility(therm: GeneralThermodynamics, x: np.array, T: np.array, unsortIndices: np.array, hashTable : HashTable = None) -> MobilityData:
#     '''
#     Gets mobility data for x and T (at single composition/temperature)

#     If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

#     Parameters
#     ----------
#     therm : GeneralThermodynamics object
#     x : float, list[float]
#         For binary, it must be a float
#         For ternary, it must be list[float]
#     T : float
#         Temperature(s)

#     Returns
#     -------
#     MobilityData
#     '''
#     mobility_data = None
#     if hashTable is not None:
#         mobility_data = hashTable.retrieveFromHashTable(x, T)

#     if mobility_data is None:
#         # Compute equilibrium
#         try:
#             wks = therm.getEq(x, T, 0, therm.phases)
#             chemical_potentials = np.squeeze(wks.eq.MU)[unsortIndices]
#             comp_sets = wks.get_composition_sets()
#             if len(comp_sets) == 0 or any(np.isnan(chemical_potentials)):
#                 raise ValueError("Equilibrium did not converge")
#         except Exception as e:
#             print(f'Error at {x}, {T}')
#             raise e

#         # Compute mobility and phase fractions
#         mob = -1*np.ones((len(comp_sets), len(therm.elements)-1))
#         phases, phase_fracs = zip(*[(cs.phase_record.phase_name, cs.NP) for cs in comp_sets])
#         phases = np.array(phases)
#         phase_fracs = np.array(phase_fracs, dtype=np.float64)
#         for p, cs in enumerate(comp_sets):
#             if therm.mobCallables.get(phases[p], None) is not None:
#                 mob[p,:] = mobility_from_composition_set(cs, therm.mobCallables[phases[p]], therm.mobility_correction)[unsortIndices]
#                 mob[p,:] *= x_to_u_frac(np.array(cs.X, dtype=np.float64)[unsortIndices], therm.elements[:-1], interstitials)

#         mobility_data = MobilityData(mobility = mob, phases = phases, phase_fractions = phase_fracs, chemical_potentials=chemical_potentials)

#         if hashTable is not None:
#             hashTable.addToHashTable(x, T, mobility_data)

#     return mobility_data

# def computeMobility(therm : GeneralThermodynamics, x, T, hashTable : HashTable = None):
#     '''
#     Gets mobility data for x and T where x and T is an array

#     If mobility does not exist for a phase or a phase is unstable, then the mobility is set to -1

#     Parameters
#     ----------
#     therm : GeneralThermodynamics object
#     x : float or array
#         For binary, it must be (1,) or (N,)
#         For ternary, it must be (1,e) or (N,e)
#     T : float or array
#         Must be (1,) or (N,)

#     Returns
#     -------
#     MobilityData
#     '''
#     x, T = _process_xT_arrays(x, T, therm.numElements == 2)

#     # we want to make sure that the mobility is sorted to be the same order as listed in therm
#     sortIndices = np.argsort(therm.elements[:-1])
#     unsortIndices = np.argsort(sortIndices)

#     mob = []
#     phases = []
#     phase_fracs = []
#     chemical_potentials = []
#     for i in range(len(x)):
#         mobility_data = _computeSingleMobility(therm, x[i], T[i], unsortIndices, hashTable)
#         mob.append(mobility_data.mobility)
#         phases.append(mobility_data.phases)
#         phase_fracs.append(mobility_data.phase_fractions)
#         chemical_potentials.append(mobility_data.chemical_potentials)
#     return MobilityData(mobility=mob, phases=phases, phase_fractions=phase_fracs, chemical_potentials=chemical_potentials)

class DiffusionConstraints:
    '''
    Constraints for numerical stability

    Attributes
    ----------
    minComposition : float
        Minimum composition that a node can take
    vonNeumannThreshold : float
        Factor to ensure numerical stability in the single phase diffusion model
        In theory, from von Neumann stability analysis for the heat equation, this can be 0.5
        But I will use 0.4 to be safe
    maxCompositionChange : float
        Max composition change a node can see per iteration in the homogenization model
        There is not an analagous method for von Neumann stability as we have for the single
        phase diffusion model, so this is a naive approach to numerical stability
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.minComposition = 1e-8
        self.vonNeumannThreshold = 0.4
        self.maxCompositionChange = 0.002