from collections import namedtuple
from typing import Protocol

import numpy as np

from pycalphad import variables as v
from kawin.thermo.mobility import mobility_from_composition_set, x_to_u_frac, interstitials
from kawin.thermo.thermodynamics import Thermodynamics, get_eq, enumerate_conditions

MobilityData = namedtuple(
    'MobilityData',
    ['mobility', 'phases', 'phase_fractions', 'chemical_potentials']
    )

class AveragingFunction(Protocol):
    def __call__(self, mobility: np.array, phase_fracs: np.array) -> np.array:
        """
        Returns
            mobility array - (e,)
        """
        ...

class PostProcessFunction(Protocol):
    def __call__(mob_data: MobilityData) -> tuple[np.array, np.array]:
        """
        Returns
            mobility array - (p,e)
            phase fractions array - (p,)
        """
        ...

class MobilityFunction(Protocol):
    def __call__(therm: Thermodynamics, conditions: dict[v.StateVariable, float]) -> MobilityData:
        ...

class HomogenizationFunction(Protocol):
    def __call__(
            therm : Thermodynamics, conditions: dict[v.StateVariable, float],
            mobility_func: MobilityFunction,
            post_process_func: PostProcessFunction,
            avg_func: AveragingFunction,
            *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
            average mobility array - (N, e)
            chemical potential array - (N, e)
        """
        ...

def wiener_upper(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Upper wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(phase_fracs[:,np.newaxis], modified_mob), axis=0)
    return avg_mob

def wiener_lower(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Lower wiener bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    avg_mob = 1/np.sum(np.multiply(phase_fracs[:,np.newaxis], 1/(modified_mob)), axis=0)
    return avg_mob

def _labyrinth(mobility: np.array, phase_fracs: np.array, labyrinth_factor: float) -> np.array:
    '''
    Labyrinth mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(np.power(phase_fracs[:,np.newaxis], labyrinth_factor), modified_mob), axis=0)
    return avg_mob

def labyrinth(labyrinth_factor: float = 1):
    labyrinth_factor = np.clip(labyrinth_factor, 1, 2, dtype=np.float32)
    return lambda mobility, phase_fracs: _labyrinth(mobility, phase_fracs, labyrinth_factor)

def _hashin_shtrikman_general(mobility: np.array, phase_fracs: np.array, extreme_mob: np.array) -> np.array:
    '''
    General hashin shtrikman bounds

    Ak = f^phi / (1 / (gamma^phi - gamma^alpha) + 1/(3*gamma^alpha))
    gamma^* = gamma^alpha + Ak / (1 - Ak / (3*gamma^alpha))

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)
    extreme_mob : list of length (e,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    Ak = phase_fracs[:,np.newaxis] * (mobility - extreme_mob) * (3*extreme_mob) / (2*extreme_mob + mobility)
    Ak = np.sum(Ak, axis=0)
    avg_mob = extreme_mob + Ak / (1 - Ak / (3*extreme_mob))
    return avg_mob

def hashin_shtrikman_upper(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Upper hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    max_mob = np.amax(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, max_mob)

def hashin_shtrikman_lower(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Lower hashin shtrikman bounds for average mobility

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    (e,) mobility array - e is number of elements
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    min_mob = np.amin(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, min_mob)

def post_process_do_nothing(mob_data: MobilityData):
    return np.array(mob_data.mobility), np.array(mob_data.phase_fractions)

def _post_process_predefined_matrix_phase(mob_data: MobilityData, alpha_phase: str):
    '''
    User will supply a predefined "alpha" phase, which the mobility is taken
    from for all undefined mobility

    Note: this assumes the user will know that "alpha" is continuously stable
    across the diffusion couple
    '''
    mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
    alpha_idx = np.squeeze(np.where(phases == alpha_phase)[0])
    if len(alpha_idx.shape) > 0:
        raise ValueError(f'{alpha_phase} must be stable as a single phase.')
    for i in range(mobility.shape[0]):
        if mobility[i][0] == -1:
            mobility[i] = mobility[alpha_idx]
    return mobility, phase_fractions

def post_process_predefined_matrix_phase(phase: str):
    return lambda mob_data: _post_process_predefined_matrix_phase(mob_data, phase)

def post_process_majority_phase(mob_data: MobilityData):
    '''
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
    max_idx = np.argmax(phase_fractions)
    for i in range(mobility.shape[0]):
        if mobility[i][0] == -1:
            mobility[i] = mobility[max_idx]
    return mobility, phase_fractions

def _post_process_exclude_phases(mob_data: MobilityData, excluded_phases: list[str]):
    '''
    For all excluded phases, the mobility and phase fraction will be set to 0
    This assumes that user knows the excluded phases to be minor or that the
    mobility is unknown
    '''
    mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
    for i in range(mobility.shape[0]):
        if phases[i] in excluded_phases:
            mobility[i] = 0
            phase_fractions[i] = 0
    return mobility, phase_fractions

def post_process_exclude_phases(excluded_phases: str|list[str]):
    if isinstance(excluded_phases, str):
        excluded_phases = [excluded_phases]
    return lambda mob_data: _post_process_exclude_phases(mob_data, excluded_phases)

def compute_mobility(therm: Thermodynamics, conditions: dict[v.StateVariable, float]) -> MobilityData:
    # Compute equilibrium
    try:
        wks = get_eq(therm, conditions, therm.phases)
        chemical_potentials = np.squeeze(wks.eq.MU)
        comp_sets = wks.get_composition_sets()
        if len(comp_sets) == 0 or any(np.isnan(chemical_potentials)):
            raise ValueError("Equilibrium did not converge")
    except Exception as e:
        print(f'Error at {conditions}')
        raise e

    # Compute mobility and phase fractions
    mob = -1*np.ones((len(comp_sets), len(therm.nonvacant_elements)))
    phases, phase_fracs = zip(*[(cs.phase_record.phase_name, cs.NP) for cs in comp_sets])
    phases = np.array(phases)
    phase_fracs = np.array(phase_fracs, dtype=np.float64)
    for p, cs in enumerate(comp_sets):
        if therm.mobility_callables.get(phases[p], None) is not None:
            mob[p,:] = mobility_from_composition_set(cs, therm.mobility_callables[phases[p]], therm.mobility_correction)
            mob[p,:] *= x_to_u_frac(np.array(cs.X, dtype=np.float64), therm.nonvacant_elements, interstitials)

    return MobilityData(mobility = mob, phases = phases, phase_fractions = phase_fracs, chemical_potentials=chemical_potentials)

def squeeze_homogenization_results(values):
    return np.squeeze([vals[0] for vals in values]), np.squeeze([vals[1] for vals in values])

@enumerate_conditions(squeeze_homogenization_results)
def compute_homogenization_function(
        therm : Thermodynamics, conditions: dict[v.StateVariable, float],
        mobility_func: MobilityFunction = compute_mobility,
        post_process_func: PostProcessFunction = post_process_do_nothing,
        avg_func: AveragingFunction = hashin_shtrikman_lower):
    '''
    Compute homogenization function (defined by HomogenizationParameters) for list of x,T

    Parameters
    ----------
    therm : GeneralThermodynamics object
    x : float, list[float], list[list[float]]
        Compositions
        For binary, it must be either a float or list[float] (shape N) for multiple compositions
        For ternary, it must be either list[float] (shape e) or list[list[float]] (shape N x e) for multiple compositions
    T : float, list[float]
        Temperature(s)
        First dimensions must correspond to first dimension of x
    diffusion_parameters: DiffusionParameters

    Returns
    -------
    average mobility array - (N, e)
    chemical potential array - (N, e)

    Where N is size of (x,T), and e is number of elements
    '''
    mobility_data = mobility_func(therm, conditions)
    mob, phase_fracs = post_process_func(mobility_data)
    avg_mob = avg_func(mob, phase_fracs)
    return avg_mob, mobility_data.chemical_potentials