from typing import Protocol

import numpy as np

from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet

from kawin.thermo.thermodynamics import Thermodynamics, enumerate_conditions, _get_phase, get_local_eq
import kawin.thermo.mobility as mob_funcs

class InterdiffusitivityFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
            ref_element: str=None, phase: str=None, cache: dict[str, any]={}, **kwargs) -> np.array:
        ...

    @staticmethod
    def squeeze(values: list[np.array]):
        return np.squeeze(values)

class ChemicalDiffusitivityFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
            phase: str=None, cache: dict[str, any]={}, **kwargs) -> np.array:
        ...

    @staticmethod
    def squeeze(values: list[np.array]):
        return np.squeeze(values)

class TracerDiffusitivityFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
            phase: str=None, cache: dict[str, any]={}, **kwargs) -> np.array:
        ...

    @staticmethod
    def squeeze(values: list[np.array]):
        return np.squeeze(values)

@enumerate_conditions(InterdiffusitivityFunction.squeeze)
def compute_interdiffusivity(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    ref_element: str, phase: str=None, cache: dict[str, list[CompositionSet]]=None, **kwargs) -> np.array:
    """
    Gets interdiffusivity ((m-1) x (m-1) matrix) at unique composition and temperature

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    ref_element: str
        Reference element to interdiffusivity against
    phase: str
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    Interdiffusivity as a matrix (will return float in binary case)
        Binary: (N,)
        Multicomponent: (N,(m-1),(m-1))
        Where N is number of conditions, m is number of elements
    """
    phase = _get_phase(therm.phases, phase)
    if cache is None:
        cache = {}
    composition_sets = cache.get(f'{phase}_diff', None)
    result, composition_sets = get_local_eq(therm, conditions, [phase], composition_sets=composition_sets)
    cs_matrix = composition_sets[0]
    chemical_potentials = result.chemical_potentials

    # Get interdiffusivity from mobility or diffusivity models whichever is available
    # If both mobility and diffusivity models exist, then favor the mobility model
    if therm.mobility_callables.get(phase, None) is None:
        Dnkj = mob_funcs.interdiffusivity_from_diff(
            cs_matrix, ref_element, therm.diffusivity_callables[phase],
            diffusivity_correction=therm.mobility_correction, parameters=therm._parameters
            )
    else:
        Dnkj = mob_funcs.interdiffusivity(
            chemical_potentials, cs_matrix, ref_element, therm.mobility_callables[phase],
            mobility_correction=therm.mobility_correction,
            vacancy_poor_interstitial_sublattice=therm.vacancy_poor_interstitial_sublattice.get(phase, False),
            parameters=therm._parameters
            )

    cache[f'{phase}_diff'] = composition_sets
    return Dnkj

@enumerate_conditions(ChemicalDiffusitivityFunction.squeeze)
def compute_chemical_diffusivity(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    phase: str=None, cache: dict[str, list[CompositionSet]]=None, **kwargs) -> np.array:
    """
    Gets chemical diffusivity (m x m matrix) at unique composition and temperature

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    phase: str
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    Interdiffusivity as a matrix (N,m,m)
        Where N is number of conditions, m is number of elements
    """
    phase = _get_phase(therm.phases, phase)
    if cache is None:
        cache = {}
    composition_sets = cache.get(f'{phase}_diff', None)
    result, composition_sets = get_local_eq(therm, conditions, [phase], composition_sets=composition_sets)
    cs_matrix = composition_sets[0]
    chemical_potentials = result.chemical_potentials

    # Get interdiffusivity from mobility or diffusivity models whichever is available
    # If both mobility and diffusivity models exist, then favor the mobility model
    if therm.mobility_callables.get(phase, None) is None:
        raise ValueError("Computing chemical diffusivity matrix from DF and DQ is not supported")
    else:
        Dkj = mob_funcs.chemical_diffusivity(
            chemical_potentials, cs_matrix, therm.mobility_callables[phase],
            mobility_correction=therm.mobility_correction,
            vacancy_poor_interstitial_sublattice=therm.vacancy_poor_interstitial_sublattice.get(phase, False),
            parameters=therm._parameters
        )

    cache[f'{phase}_diff'] = composition_sets
    return Dkj

@enumerate_conditions(TracerDiffusitivityFunction.squeeze)
def compute_tracer_diffusivity(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    phase: str=None, cache: dict[str, list[CompositionSet]]={}, **kwargs) -> np.array:
    """
    Gets tracer diffusivity at unique composition and temperature

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    phase: str
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    Tracer diffusivity for all elements
    """
    phase = _get_phase(therm.phases, phase)
    if cache is None:
        cache = {}

    composition_sets = cache.get(f'{phase}_diff', None)
    result, composition_sets = get_local_eq(therm, conditions, [phase], composition_sets=composition_sets)
    cs_matrix = composition_sets[0]

    # Get interdiffusivity from mobility or diffusivity models whichever is available
    # If both mobility and diffusivity models exist, then favor the mobility model
    if therm.mobility_callables.get(phase, None) is None:
        #NOTE: This is not tested yet
        dtrace = mob_funcs.tracer_diffusivity_from_diff(
            cs_matrix, therm.diffusivity_callables[phase],
            diffusivity_correction=therm.mobility_correction, parameters=therm._parameters
            )
    else:
        dtrace = mob_funcs.tracer_diffusivity(
            cs_matrix, therm.mobility_callables[phase],
            mobility_correction=therm.mobility_correction, parameters=therm._parameters
            )

    cache[f'{phase}_diff'] = composition_sets
    return dtrace