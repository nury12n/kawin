from typing import Protocol
from collections import namedtuple
import numpy as np

from pycalphad import variables as v

from kawin.constants import GAS_CONSTANT
from kawin.thermo.thermodynamics import Thermodynamics, get_eq, enumerate_conditions
from kawin.thermo.mobility import expand_u_frac, u_to_x_frac, x_to_u_frac, mobility_from_composition_set, interstitials
from kawin.diffusion.diffusion import DiffusionModel, DiffusionState, LookupTable, DiffusionModelTemperature, IsothermalTemperature
from kawin.diffusion.mesh.mesh_base import MeshBase, DiffusionPair, arithmetic_mean, log_mean, harmonic_mean

MobilityData = namedtuple(
    'MobilityData',
    ['mobility', 'phases', 'phase_fractions', 'chemical_potentials']
    )

class AveragingFunction(Protocol):
    """
    Averages mobility from set of phases

    Parameters
    ----------
    mobility : list of length (p, e)
    phase_fracs : list of length (p,)

    Returns
    -------
    mobility array - (e,)
    """
    def __call__(self, mobility: np.array, phase_fracs: np.array) -> np.array:
        ...

class PostProcessFunction(Protocol):
    """
    Post modifications to mobility. Generally, this is used to assume
    mobility values for phases that do not have their own mobility model

    Returns
    -------
    mobility array - (p,e)
    phase fractions array - (p,)
    """
    def __call__(mob_data: MobilityData) -> tuple[np.array, np.array]:
        ...

class MobilityFunction(Protocol):
    """
    Thermodynamic function computing mobility for all phases in equilibrium
    Phases that do not have a mobility model are expected to return a mobility of -1.
        These phases are expected to be modified by the PostProcessFunction
    """
    def __call__(therm: Thermodynamics, conditions: dict[v.StateVariable, float]) -> MobilityData:
        ...

class HomogenizationFunction(Protocol):
    """
    Computes homogenization term \Gamma for a set of phases in equilibrium

    Returns
        average mobility array - (N, e)
        chemical potential array - (N, e)
    """
    def __call__(
            therm : Thermodynamics, conditions: dict[v.StateVariable, float],
            mobility_func: MobilityFunction,
            post_process_func: PostProcessFunction,
            avg_func: AveragingFunction,
            *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        ...

def wiener_upper(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Implements AveragingFunction
    Upper wiener bounds for average mobility
    gamma^* = sum(f^alpha * gamma^alpha)
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    avg_mob = np.sum(np.multiply(phase_fracs[:,np.newaxis], modified_mob), axis=0)
    return avg_mob

def wiener_lower(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Implements AveragingFunction
    Lower wiener bounds for average mobility
    gamma^* = sum(f^alpha * (gamma^alpha)^-1)^-1
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    avg_mob = 1/np.sum(np.multiply(phase_fracs[:,np.newaxis], 1/(modified_mob)), axis=0)
    return avg_mob

def labyrinth(labyrinth_factor: float = 1):
    '''
    Implements AveragingFunction
    Labyrinth mobility
    gamma^* = sum((f^alpha)^lab * gamma^alpha)
    '''
    labyrinth_factor = np.clip(labyrinth_factor, 1, 2, dtype=np.float32)
    def _labyrinth(mobility: np.array, phase_fracs: np.array, labyrinth_factor: float = labyrinth_factor) -> np.array:
        modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
        avg_mob = np.sum(np.multiply(np.power(phase_fracs[:,np.newaxis], labyrinth_factor), modified_mob), axis=0)
        return avg_mob

    return _labyrinth

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
    Implements AveragingFunction
    Upper hashin shtrikman bounds for average mobility
    gamma^alpha pertains to phase with maximum mobility per element
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).tiny)
    max_mob = np.amax(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, max_mob)

def hashin_shtrikman_lower(mobility: np.array, phase_fracs: np.array) -> np.array:
    '''
    Implements AveragingFunction
    Lower hashin shtrikman bounds for average mobility
    gamma^alpha pertains to phase with minimum mobility per element
    '''
    modified_mob = np.where(mobility != -1, mobility, np.finfo(np.float64).max)
    min_mob = np.amin(modified_mob, axis=0)    # (p, e) -> (e,)
    return _hashin_shtrikman_general(modified_mob, phase_fracs, min_mob)

def post_process_do_nothing(mob_data: MobilityData):
    '''
    Implements PostProcessFunction
    '''
    return np.array(mob_data.mobility), np.array(mob_data.phase_fractions)

def post_process_predefined_matrix_phase(phase: str):
    '''
    Implements PostProcessFunction
    User will supply a predefined "alpha" phase, which the mobility is taken
    from for all undefined mobility

    Note: this assumes the user will know that "alpha" is continuously stable
    across the diffusion couple
    '''
    def _post_process_predefined_matrix_phase(mob_data: MobilityData, alpha_phase: str = phase):
        mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
        alpha_idx = np.squeeze(np.where(phases == alpha_phase)[0])
        if len(alpha_idx.shape) > 0:
            raise ValueError(f'{alpha_phase} must be stable as a single phase.')
        for i in range(mobility.shape[0]):
            if mobility[i][0] == -1:
                mobility[i] = mobility[alpha_idx]
        return mobility, phase_fractions

    return _post_process_predefined_matrix_phase

def post_process_majority_phase(mob_data: MobilityData):
    '''
    Implements PostProcessFunction
    Takes the majority phase and applies the mobility for all other phases
    with undefined mobility
    '''
    mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
    max_idx = np.argmax(phase_fractions)
    for i in range(mobility.shape[0]):
        if mobility[i][0] == -1:
            mobility[i] = mobility[max_idx]
    return mobility, phase_fractions

def post_process_exclude_phases(excluded_phases: str|list[str]):
    '''
    Implements PostProcessFunction
    For all excluded phases, the mobility and phase fraction will be set to 0
    This assumes that user knows the excluded phases to be minor or that the
    mobility is unknown
    '''
    if isinstance(excluded_phases, str):
        excluded_phases = [excluded_phases]
    def _post_process_exclude_phases(mob_data: MobilityData, excluded_phases: list[str] = excluded_phases):
        mobility, phases, phase_fractions = np.array(mob_data.mobility), np.array(mob_data.phases), np.array(mob_data.phase_fractions)
        for i in range(mobility.shape[0]):
            if phases[i] in excluded_phases:
                mobility[i] = 0
                phase_fractions[i] = 0
        return mobility, phase_fractions

    return _post_process_exclude_phases

def compute_mobility(therm: Thermodynamics, conditions: dict[v.StateVariable, float]) -> MobilityData:
    '''
    Implements MobilityFunction
    '''
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
    Implements HomogenizationFunction
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

def _homogenization_mean(Ds):
    '''
    Average homogenization flux
    Ds should be in the shape of (m x N x y x 2)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        2 - (transformation to volume fixed frame, homogenization term (\Gamma))
            transformation is in composition, so use arithmetic average
            homogenization term is in terms of mobility, so use log average (mobility is always positive)
    '''
    v = arithmetic_mean([D[...,0] for D in Ds])
    m = log_mean([D[...,1] for D in Ds])
    return v * m

def _ideal_mean(Ds):
    '''
    Average homogenization flux
    Ds should be in the shape of (m x N x y x 4)
        m - number of items to average over
        N - number of nodes
        y - number of responses
        4 - (transformation to volume fixed frame, temperature, homogenization term (\Gamma), inverse u term)
            transformation is in composition, so use arithmetic average
            temperature (+eps*R) is linear, so use arithmetic average
            homogenization term is in terms of mobility, so use log average
            inverse u term is inverse composition, so use harmonic mean (composition should always be > 0 based off constraints in diffusion model)
    '''
    v = arithmetic_mean([D[...,0] for D in Ds])
    t = arithmetic_mean([D[...,1] for D in Ds])
    m = log_mean([D[...,2] for D in Ds])
    inv_u = harmonic_mean([D[...,3] for D in Ds])
    #print(v.shape, t.shape, m.shape, invU.shape)
    return v * t * m * inv_u

def _at_node_product(Ds):
    return np.prod(Ds, axis=-1)

class HomogenizationModel(DiffusionModel):
    def __init__(
            self,
            mesh: MeshBase,
            elements: list[str],
            phases: list[str],
            therm: Thermodynamics,
            temperature: DiffusionModelTemperature | float,
            averaging_function: AveragingFunction = hashin_shtrikman_lower,
            post_process_function: PostProcessFunction = post_process_do_nothing,
            mobility_function: MobilityFunction = None,
            homogenization_function: HomogenizationFunction = compute_homogenization_function,
            eps: float = 0.05,
            *args, **kwargs):
        super().__init__(mesh, elements)

        # model parameters
        self.phases = phases
        self.therm = therm
        self.temperature = temperature
        if np.isscalar(self.temperature):
            self.temperature = IsothermalTemperature(temperature)
        self.conditions = kwargs.get("conditions", dict())
        self.averaging_function = averaging_function
        self.post_process_function = post_process_function
        self.mobility_function = mobility_function
        if self.mobility_function is None:
            self.mobility_function = LookupTable(compute_mobility)
        self.homogenization_function = homogenization_function
        self.eps = eps

        # utility
        self.sort_indices = np.argsort(np.argsort(self.all_elements))

        # constraints
        self.max_composition_change = 0.002

    def compute_pairs(self, state: DiffusionState) -> np.ndarray:
        '''
        Compute diffusivity-response pairs

        J_k = -\Gamma_k d\mu_k/dz - \eps*RT*\Gamma_k/u_k du_k/dz
        J^n_k = -sum(\delta_jk - u_k) J_j
        dx_k/dt = -dJ^n_k/dz

        For x_k, a pair would comprise of:
            (\delta_jk - u_k) \Gamma_j, \mu_k
            (\delta_jk - u_k) \eps*RT*\Gamma_k/u_k, u_k
        '''
        # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
        # TODO: this feels inefficient to convert u->x for mobility, then back to u for flux calculation
        u = expand_u_frac(state.u, self.all_elements, self.ref_element, interstitials)
        x = u_to_x_frac(u, self.all_elements, interstitials)[:,1:]

        y_res, z_res = self.mesh.get_response_coordinates(x)
        y_diff, z_diff = self.mesh.get_diffusivity_coordinates(x)

        # temp is (N,e)
        T_d = self.temperature(state.time, z_diff)
        T_r = self.temperature(state.time, z_res)

        # mob and mu are (N,e)
        conditions = {v.X(e): y_diff[:,self.elements.index(e)] for e in self.elements}
        conditions[v.T] = T_d
        conditions[v.P] = self.conditions.get(v.P, 101325)
        conditions[v.GE] = self.conditions.get(v.GE, 0)
        mob_d, mu_d = self.homogenization_function(
            self.therm, conditions,
            mobility_func=self.mobility_function,
            post_process_func=self.post_process_function,
            avg_func=self.averaging_function)

        conditions = {v.X(e): y_res[:,self.elements.index(e)] for e in self.elements}
        conditions[v.T] = T_r
        mob_r, mu_r = self.homogenization_function(
            self.therm, conditions,
            mobility_func=self.mobility_function,
            post_process_func=self.post_process_function,
            avg_func=self.averaging_function)


        # Full composition
        # x_full = (N,e+1), u_full = (N,e+1), u_term = (N,e+1,e+1)
        n_ele = len(self.all_elements)
        x_full_d = np.concatenate((1-np.sum(y_diff, axis=1)[:,np.newaxis], y_diff), axis=1)
        u_full_d = x_to_u_frac(x_full_d, self.all_elements, interstitials)
        # u_term is the (\delta_jk - u_k), which converts from a lattice fixed frame to a volume fixed frame
        u_term_d = (np.eye(n_ele)[np.newaxis,:,:] - u_full_d[:,:,np.newaxis])
        # When converting to a volume fixed frame, only the substitutional (or volume contributing) elements
        # contribute to the corrected flux. To account for interstitials, we replace the column
        # corresponding to the interstitial (which at this point is [u_A, u_B, 1-u_I, u_D] where I is interstital)
        # to be [0, 0, 1, 0]. So the column is 0 except for the interstitial row
        for i,e in enumerate(self.all_elements):
            if e in interstitials:
                u_term_d[:,:,i] = 0
                u_term_d[:,i,i] = 1

        x_r = np.concatenate((1-np.sum(y_res, axis=1)[:,np.newaxis], y_res), axis=1)
        u_r = x_to_u_frac(x_r, self.all_elements, interstitials)

        # mobility matrix is repeated among rows
        mob_d_matrix = np.repeat(mob_d[:,np.newaxis,:], n_ele, axis=1)
        # We'll group eps*R with T here
        T_matrix = np.tile(T_d[:,np.newaxis,np.newaxis], (1, n_ele, n_ele)) * self.eps * GAS_CONSTANT
        # inverse composition and response is repeated among rows
        inv_u_matrix = np.repeat(1/u_full_d[:,np.newaxis,:], n_ele, axis=1)
        mu_matrix = np.repeat(mu_r[:,np.newaxis,:], n_ele, axis=1)
        u_r_matrix = np.repeat(u_r[:,np.newaxis,:], n_ele, axis=1)

        pairs = []
        # Since volume fixed frame leads to 1 dependent component (which we take as the first)
        # we don't need to take the 1st row of mob_term and ideal_term
        # u and x are in the same order as self.all_elements but mu and mob are alphabetically sorted
        # proc_mat removes the reference element
        # sort_mat does the sorting for mu and mob then removes the reference element
        proc_mat = lambda x, i: x[:,1:,i]
        sort_mat = lambda x, i: x[:,self.sort_indices, self.sort_indices[i]][:,1:]
        for i in range(n_ele):
            # homogenization contribution - (vol transform * \Gamma) * dmu/dz
            #print(u_term_d.shape, mob_d_matrix.shape, proc_mat(u_term_d,i).shape, sort_mat(mob_d_matrix,i).shape)
            pairs.append(DiffusionPair(
                diffusivity=np.transpose(np.array([proc_mat(u_term_d,i), sort_mat(mob_d_matrix,i)]), axes=(1,2,0)),
                response=sort_mat(mu_matrix,i),
                averaging_function=_homogenization_mean,
                at_node_function=_at_node_product
            ))
            # ideal contribution - (vol transform * eps*R*T * \Gamma / u) * du/dz
            pairs.append(DiffusionPair(
                diffusivity=np.transpose(np.array([proc_mat(u_term_d,i), proc_mat(T_matrix, i), sort_mat(mob_d_matrix, i), proc_mat(inv_u_matrix, i)]), axes=(1,2,0)),
                response=proc_mat(u_r_matrix,i),
                averaging_function=_ideal_mean,
                at_node_function=_at_node_product
            ))
        return pairs

    def compute_max_dt(self, state: DiffusionState, dxdt: np.ndarray) -> float:
        return self.max_composition_change / np.amax(np.abs(dxdt[dxdt!=0]))