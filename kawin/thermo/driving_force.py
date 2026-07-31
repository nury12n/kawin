from typing import Protocol
from collections import namedtuple

import numpy as np

from pycalphad import calculate, variables as v
from pycalphad.core.composition_set import CompositionSet

from kawin.thermo.thermodynamics import Thermodynamics, enumerate_conditions, generate_model_subset, _full_comp_from_conditions
from kawin.thermo.thermodynamics import local_equilibrium, get_local_eq, _get_cs_eq
import kawin.thermo.mobility as mob_funcs

SampledPointsCache = namedtuple(
    'SampledPointsCache',
    ['temperature', 'pressure', 'samples', 'ordered_samples']
    )

DrivingForceOutput = namedtuple(
    'DrivingForceOutput',
    ['driving_force', 'x_beta']
)

class DrivingForceFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
            matrix_phase: str, prec_phase: str,
            local_phase_sampling_conditions: dict[v.StateVariable, float]=None, cache: dict[str, any]={}, **kwargs) -> DrivingForceOutput:
        ...

    @staticmethod
    def squeeze(values: list[DrivingForceOutput]) -> DrivingForceOutput:
        return DrivingForceOutput(
            driving_force=np.squeeze([val.driving_force for val in values]),
            x_beta=np.squeeze([val.x_beta for val in values])
        )


def _get_cs_for_df(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
        matrix_phase: str, prec_phase: str, cache: dict[str, any]) -> tuple[np.array, CompositionSet, CompositionSet]:
    """
    Wrapper for getting composition set from x and T by either global equilibrium or local from a cached composition set

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    chemical_potentials
    cs_matrix - composition set of matrix phase
    cs_precip - composition set of precipitate phase
    """
    eq_results = _get_cs_eq(therm, conditions, matrix_phase, prec_phase, cache)
    cache_id = f'{matrix_phase}_{prec_phase}'
    if eq_results is None:
        cache[cache_id] = None
        return None
    else:
        chemical_potentials, cs_matrix, cs_precip = eq_results
        if cs_matrix is None or cs_precip is None:
            cache[cache_id] = None
            return None
        cache[cache_id] = [cs_matrix, cs_precip]
        return chemical_potentials, cs_matrix, cs_precip


def _sample_precipitate_cs(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
        matrix_chem_pot: np.array, prec_phase: str,
        local_phase_sampling_conditions: dict[v.StateVariable, float], cache: dict[str, any]) -> tuple[float, CompositionSet]:
    """
    Gets samples for precipitate phase for use in sampling driving force method and returns driving force and precipitate composition

    This is also use in tangent driving force method for when equilibrium is not (yet) cached

    Steps
        1. Compute local equilibrium at x and T of only the matrix phase
        2. Sample precipitate phase
            If ordered contribution to matrix phase, then sample ordering contribution
            and remove points on the matrix free energy surface
        3. Compute energy difference between precipitate samples and chemical potential hyperplane
        4. Find sample that maximizes energy difference and return sample composition and driving force

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_chem_pot: np.array
    matrix_phase: str
    prec_phase: str
    local_phase_sampling_conditions: dict
        User supplied conditions to force sampling
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    driving force - max free energy difference
    precipitate composition set - corresponds to max driving force
    """
    orderTol = -1e-8
    state_cond = {v.GE: therm.g_offset, v.N: 1, v.P: conditions[v.P], v.T: conditions[v.T]}
    str_cond = {str(key): val for key,val in state_cond.items()}

    #Sample precipitate phase and get driving force differences at all points -------------------------------------------------------------------
    #Sample points of precipitate phase
    cache_id = f"{prec_phase}_samples"
    phases, sub_models = generate_model_subset(therm, [prec_phase])
    sample_data = cache.get(cache_id, None)

    if sample_data is None or sample_data.temperature != conditions[v.T] or sample_data.pressure != conditions[v.P]:
        prec_points = calculate(therm.dbf, therm.elements, phases[0],
                                pdens=therm.sampling_pdens, model=sub_models, output='GM',
                                phase_records=therm.phase_record_factory, conditions=local_phase_sampling_conditions,
                                to_xarray=False, **str_cond)
        ordered_points = None
        if therm.is_ordered_phase.get(prec_phase, False):
            ordered_points = calculate(therm.dbf, therm.elements, phases[0],
                                        pdens=therm.sampling_pdens, model=sub_models, output='OCM',
                                        phase_records=therm.phase_record_factory, to_xarray=False, **str_cond)
        cache[cache_id] = SampledPointsCache(temperature=conditions[v.T], pressure=conditions[v.P], samples=prec_points, ordered_samples=ordered_points)
    else:
        prec_points = sample_data.samples
        ordered_points = sample_data.ordered_samples

    #For phases at fixed composition, there will only be 1 set of site fractions
    #So we force composition and site fractions to be 2D
    prec_comp = np.atleast_2d(np.squeeze(prec_points.X))
    y = np.atleast_2d(np.squeeze(prec_points.Y))
    gm = np.atleast_1d(np.squeeze(prec_points.GM))
    mu = np.atleast_2d(matrix_chem_pot)

    #Difference between the chemical potential hyperplane and the free energy of the samples
    #   Chemical potential hyperplane is the free energy on the plane at the composition of the samples
    #The max driving force is the same as when the chemical potentials of the two phases are parallel
    mult = prec_comp * mu
    diff = np.sum(mult, axis=1) - np.squeeze(gm)

    #Find maximum driving force and corresponding composition -----------------------------------------------------------------------------------
    #For phases with order/disorder transition, a filter is applied such that it will only use points that are below the disordered energy surface
    if therm.is_ordered_phase.get(prec_phase, False):
        indices = np.squeeze(ordered_points.OCM) < orderTol
        diff = diff[indices]
        y = y[indices]

    dg = np.amax(diff)
    idx = np.argmax(diff)

    prec_cs = CompositionSet(therm.phase_record_factory[prec_phase])
    state_variables = np.array([state_cond[v.GE], state_cond[v.N], state_cond[v.P], state_cond[v.T]], dtype=np.float64)
    y = np.array(y[idx][:prec_cs.phase_record.phase_dof], dtype=np.float64)
    prec_cs.update(y, 1, state_variables)

    return dg, prec_cs

@enumerate_conditions(DrivingForceFunction.squeeze)
def compute_driving_force_sampling(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    matrix_phase: str, prec_phase: str,
    local_phase_sampling_conditions: dict[str, v.StateVariable]=None, cache: dict[str, any]=None, **kwargs) -> DrivingForceOutput:
    """
    Gets driving force for nucleation by sampling

    Steps
        1. Compute local equilibrium at x and T of only the matrix phase
        2. Sample precipitate phase
            If ordered contribution to matrix phase, then sample ordering contribution
            and remove points on the matrix free energy surface
        3. Compute energy difference between precipitate samples and chemical potential hyperplane
        4. Find sample that maximizes energy difference and return sample composition and driving force

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    local_phase_sampling_conditions: dict
        User supplied conditions to force sampling
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    if cache is None:
        cache = {}
    # Calculate equilibrium with only the parent phase
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Get precipitate composition set that maximizes driving force
    dg, prec_cs = _sample_precipitate_cs(therm, conditions, result.chemical_potentials, prec_phase, local_phase_sampling_conditions, cache)

    beta_x = np.array(prec_cs.X, dtype=np.float64)
    cache[f'{matrix_phase}_dg'] = matrix_cs
    # we add therm.g_offset to get consistent results with the other driving force functions
    return DrivingForceOutput(
        driving_force=np.squeeze(dg+therm.g_offset),
        x_beta=np.squeeze(beta_x)
    )

@enumerate_conditions(DrivingForceFunction.squeeze)
def compute_driving_force_approximate(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    matrix_phase: str, prec_phase: str,
    local_phase_sampling_conditions: dict[str, v.StateVariable]=None, cache: dict[str, any]=None, **kwargs) -> DrivingForceOutput:
    """
    Approximate method of driving force calculation
    Assumes equilibrium composition of precipitate phase

    Sampling method is used if driving force is negative

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Compute equilibrium at matrix composition and get chemical potential hyperplane
        4. Driving force is the difference between the free energy of the precipitate (from step 1)
            and the free energy on the chemical potential hyperplane (from step 3) at the precipitate composition

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    local_phase_sampling_conditions: dict
        User supplied conditions to force sampling
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    if cache is None:
        cache = {}
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    cs_results = _get_cs_for_df(therm, conditions, matrix_phase, prec_phase, cache)
    if cs_results is None:
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)
    chemical_potentials, cs_matrix, cs_precip = cs_results

    # Calculate equilibrium with only the parent phase
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    xp = np.array(cs_precip.X, dtype=np.float64)
    dg = np.sum(xp * result.chemical_potentials) - np.sum(xp * chemical_potentials)

    cache[f'{matrix_phase}_dg'] = matrix_cs
    return DrivingForceOutput(
        driving_force=np.squeeze(dg),
        x_beta=np.squeeze(xp)
    )

@enumerate_conditions(DrivingForceFunction.squeeze)
def compute_driving_force_curvature(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    matrix_phase: str, prec_phase: str,
    local_phase_sampling_conditions: dict[str, v.StateVariable]=None, cache: dict[str, any]=None,
    ref_element: str=None, **kwargs) -> DrivingForceOutput:
    """
    Gets driving force from curvature of free energy function
    Assumes small saturation

    Steps:
        1. Compute equilibrium and get composition sets for matrix and precipitate phase
        2. Check for 2 phases and that one phase is the matrix and other phase is precipitate
            If not, then resort to sampling method
        3. Get dmu/dx (free energy curvature)
        4. Compute (x_infty - x_matrix) * dmu/dx * (x_prec - x_matrix)^T
            This does a first (or second?) order approximation of the driving force based off the curvature at x_infty

    Sampling method is used if driving force is negative

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    local_phase_sampling_conditions: dict
        User supplied conditions to force sampling
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    if cache is None:
        cache = {}
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    cs_results = _get_cs_for_df(therm, conditions, matrix_phase, prec_phase, cache)
    # cs_results will be None if convergence error or two-phase region, fallback to sampling method
    if cs_results is None:
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)

    chemical_potentials, cs_matrix, cs_precip = cs_results

    x_full = _full_comp_from_conditions(conditions, therm.nonvacant_elements)
    x_matrix = np.array(cs_matrix.X, dtype=np.float64)
    x_precip = np.array(cs_precip.X, dtype=np.float64)
    # need reference for N-1 dependency with total derivatives in simplex space. For precipitation, we take the largest
    # composition assuming the alloy is in a corner
    if ref_element is None:
        ref_index = np.argmax(x_full)
    else:
        ref_index = therm.nonvacant_elements.index(ref_element)

    # If in two phase region, then get curvature of parent phase and use it to calculate driving force
    dmudx_parent = mob_funcs.total_dmudx(chemical_potentials, cs_matrix, cs_matrix.phase_record.nonvacant_elements[ref_index])
    xf = np.delete(x_full, ref_index)
    xm = np.delete(x_matrix, ref_index)
    xp = np.delete(x_precip, ref_index)
    xbar = np.array([xp - xm])

    xd = np.array([xf - xm])
    dg = np.matmul(xd, np.matmul(dmudx_parent, xbar.T))

    cache[f'{matrix_phase}_dg'] = matrix_cs
    return DrivingForceOutput(
        driving_force=np.squeeze(dg),
        x_beta=np.squeeze(x_precip)
    )

@enumerate_conditions(DrivingForceFunction.squeeze)
def compute_driving_force_tangent(
    therm: Thermodynamics, conditions: dict[v.StateVariable, float | list[float]],
    matrix_phase: str, prec_phase: str,
    local_phase_sampling_conditions: dict[str, v.StateVariable]=None, cache: dict[str, any]=None, **kwargs) -> DrivingForceOutput:
    """
    Gets driving force from parallel tangent calculation

    Steps
        1. Compute equilibrium to get composition sets (or used previous cached CS)
        2. Compute equilibrium of matrix phase at matrix composition
        3. Remove composition and extra free energy from conditions
        4. Add chemical potential for each component to conditions
        5. Compute equilibrium of precipitate phase with new conditions
            The calculated v.GE is the driving force

    This will work for positive and negative driving forces

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str
    local_phase_sampling_conditions: dict
        User supplied conditions to force sampling
    cache: dict|None
        If supplied, will cache composition sets for future equilibrium calculations

    Returns
    -------
    (driving force, precipitate composition)
    Driving force is positive if precipitate can form
    Precipitate composition will be None if driving force is negative
    """
    if cache is None:
        cache = {}
    # Calculate equilibrium with only the parent phase
    matrix_cs = cache.get(f'{matrix_phase}_dg', None)
    result, matrix_cs = get_local_eq(therm, conditions, [matrix_phase], composition_sets=matrix_cs)
    if any(np.isnan(result.chemical_potentials)):
        return None, None

    # Setup conditions, which include T, P, N and MU (for all elements)
    # The nonvacant elements list in the matrix phase record will be sorted
    mu_conds = {v.T: conditions[v.T], v.P: conditions.get(v.P, 101325), v.N: conditions.get(v.N, 1)}
    non_va_elements = matrix_cs[0].phase_record.nonvacant_elements
    mu_conds.update({v.MU(e): result.chemical_potentials[i] for i,e in enumerate(non_va_elements)})

    # If we do not have a precipitate composition set, then find one by sampling
    if cache.get(f'{prec_phase}_dg', None) is None:
        dg, prec_cs = _sample_precipitate_cs(therm, conditions, result.chemical_potentials, prec_phase, local_phase_sampling_conditions, cache)
        cache[f'{prec_phase}_dg'] = [prec_cs]

    #Solving for local equilibrium on precipitate
    #The fixed conditions are T, P and MU, so this should solve for precipitate composition and GE
    #   Rather than solving for parallel tangent where the driving force is the difference between the chemical potentials of matrix and precipitate phase
    #   This instead solves for the offset in the precipitate energy surface to make the precipitate lie on the chemical potential hyperplane of the matrix phase
    phases, sub_models = generate_model_subset(therm, [prec_phase])
    prec_eq_results, prec_cs = local_equilibrium(
        therm.dbf, therm.elements, phases,
        mu_conds, sub_models, therm.phase_record_factory,
        composition_sets=cache[f'{prec_phase}_dg'], pdens=therm.local_pdens
        )
    if any(np.isnan(prec_eq_results.chemical_potentials)):
        return None, None

    # NOTE: we assume that v.GE is the first state variable in alphabetical order
    dg = prec_eq_results.x[0]
    xb = np.array(prec_cs[0].X)

    #Check if precipitate composition at equilibrium is the matrix composition
    #This can occur in order/disordered models where the miscibility gap is small enough that the parallel tangent can only be found at the matrix composition
    #In this case, switch to sampling for the driving force
    #This still seems to be an improvement over approximate and curvature methods since this occurs after the driving force becomes negative
    mat_comps = np.array(matrix_cs[0].X, dtype=np.float64)
    if np.allclose(xb, mat_comps, 1e-6):
        cache[f'{prec_phase}_dg'] = None
        return compute_driving_force_sampling(therm, conditions, matrix_phase, prec_phase, local_phase_sampling_conditions, cache)

    cache[f'{prec_phase}_dg'] = prec_cs
    cache[f'{matrix_phase}_dg'] = matrix_cs
    return DrivingForceOutput(
        driving_force=np.squeeze(dg),
        x_beta=np.squeeze(xb)
    )