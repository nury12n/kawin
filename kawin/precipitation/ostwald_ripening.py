from collections import namedtuple
from typing import Protocol

import numpy as np
import scipy.special as ssp

from pycalphad import variables as v
from pycalphad.core.composition_set import CompositionSet

from kawin.thermo.thermodynamics import Thermodynamics, _full_comp_from_conditions, get_two_phase_eq, two_phase_cache_id, EquilibriumCS
from kawin.thermo.diffusivity import compute_interdiffusivity
from kawin.thermo.interfacial_composition import InterfacialCompositionFunction, compute_interfacial_composition_equilibrium
from kawin.thermo.driving_force import DrivingForceOutput, DrivingForceFunction, compute_driving_force_tangent
import kawin.thermo.mobility as mob_funcs

"""
Compute particle growth rates and interfacial composition for Ostwald Ripening processes
"""

InterfacialCompositionCache = namedtuple(
    'InterfacialCompositionCache',
    ['conditions', 'x_alpha', 'x_beta', 'x_eq_alpha', 'x_eq_beta']
    )

# mc = 1 / (dx_bar^T * inv(M) * dx_bar)
# dc = inv(Dnkj) * dx_bar / mc
# gba = inv(d2G_beta/dx2) * d2G_alpha/dx2
# x_eq_alpha = equilibrium composition of alpha phase
# x_eq_beta = equilibrium composition of beta phase
CurvatureOutput = namedtuple(
    'CurvatureOutput',
    ['dc', 'mc', 'gba', 'dg', 'beta', 'x_eq_alpha', 'x_eq_beta']
    )

# growth_rate = dR/dt
# x_alpha = interfacial composition of alpha phase
# x_beta = interfacial composition of beta phase
# x_eq_alpha = equilibrium composition of alpha phase
# x_eq_beta = equilibrium composition of beta phase
GrowthRateOutput = namedtuple(
    'GrowthRateOutput',
    ['growth_rate', 'x_alpha', 'x_beta', 'x_eq_alpha', 'x_eq_beta']
    )

class GrowthRateFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float],
            matrix_phase: str, prec_phase: str, ref_element: str,
            r: float | list[float], g_extra: float | list[float],
            cache: dict[str, any]=None, **kwargs) -> GrowthRateOutput:
        ...

def binary_composition_cache_id(matrix_phase, prec_phase):
    return f'bin_comp_{matrix_phase}_{prec_phase}'

def curvature_cache_id(matrix_phase, prec_phase):
    return f'curvature_{matrix_phase}_{prec_phase}'

def omega(lam):
    """
    Supersaturation as a function of lambda
    """
    # note: erfcx(x) = exp(x^2)*erfc(x)
    return 2*lam**2 - 2*np.sqrt(np.pi)*lam**3*ssp.erfcx(lam)

def lamda_low(omega):
    """
    Lambda when supersaturation approaches 0
    """
    return np.sqrt(omega/2)

def lambda_high(omega):
    """
    Lambda when supersaturation approaches 1
    """
    return np.sqrt(3/2*(1-omega))

class EffectiveDiffusionDistanceLookupTable:
    def __init__(self, n, lam_min, lam_max):
        self.lam = np.logspace(lam_min, lam_max, n)
        self.omega = omega(self.lam)
        self.eff_dist = self.omega/(2*self.lam**2)
        self.omega = np.concatenate(([0], self.omega, [1]))
        self.eff_dist = np.concatenate(([1], self.eff_dist, [0]))

def effective_diffusion_distance(omega):
    """
    Effective diffusion distance given supersaturation
    This gives a constant to be multiplied by the particle radius during growth rate

    The effective diffusion distance is given by
    eps = omega/2*lambda^2

    Where Q is the super saturation and lambda is given by
    omega = 2*lambda^2 - 2*lambda^3 * sqrt(pi) * exp(lambda^2) * erfc(lambda)

    We use a lookup table here for efficiency

    Parameters
    ----------
    omega: supersaturation
        (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition
    """
    return np.interp(omega, effective_diffusion_distance.table.omega, effective_diffusion_distance.table.eff_dist)

# function attributes just seem weird, but it effectively allows for "static" variables in functions
# this lets us make this function as a standalone without a class implementation
# values are selected here that give a max error of ~1e-5
#   this is based off sweeping the supersaturation from 0 to 1 at an interval of 1e-5
effective_diffusion_distance.table = EffectiveDiffusionDistanceLookupTable(n=1000, lam_min=-6, lam_max=1)

def compute_growth_rate_binary(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        matrix_phase: str, prec_phase: str, ref_element: str,
        r: float | list[float], g_extra: float | list[float],
        cache: dict[str, any]=None,
        interfacial_comp_function: InterfacialCompositionFunction=compute_interfacial_composition_equilibrium,
        **kwargs) -> GrowthRateOutput:
    """
    For binary system, there is only 1 equation to solve for growth rate, so we can
    obtain it directly from interfacial composition

    Omega (supersaturation) = (x - x_alpha) / (Vm_alpha/Vm_beta * x_beta - x_alpha)
    dR/dt = D/R * Omega / xi

    xi is correction factor for the effective diffusion distance from the
    radius at high supersaturations. During coarsening, supersaturation->0 and xi->1

    This function can also be modified:
      - Non-spherical particles: dR/dt = f(R) * dR_sph/dt
      - Effective diffusion: dR/dt = dR_ideal/dt / xi(Omega)
    """
    cache = {} if cache is None else cache
    cache_id = binary_composition_cache_id(matrix_phase, prec_phase)
    int_comp = cache.get(cache_id, None)
    if int_comp is None:
        conditions[v.GE] = g_extra
        x_comp = interfacial_comp_function(therm, conditions, matrix_phase, prec_phase, **kwargs)
        conditions[v.GE] = 0
        x_eq_comp = interfacial_comp_function(therm, conditions, matrix_phase, prec_phase, **kwargs)
        int_comp = InterfacialCompositionCache(
            conditions=conditions,
            x_alpha=x_comp.x_alpha,
            x_beta=x_comp.x_beta,
            x_eq_alpha=x_eq_comp.x_alpha,
            x_eq_beta=x_eq_comp.x_beta
            )
        cache[cache_id] = int_comp

    # interdiffusivity is same for each reference element in a binary system
    D = compute_interdiffusivity(therm, conditions, therm.nonvacant_elements[0], matrix_phase, **kwargs)

    ref_index = therm.nonvacant_elements.index(ref_element)
    x = _full_comp_from_conditions(conditions, therm.nonvacant_elements)
    xs = np.squeeze(np.delete(x, ref_index))
    xa = np.squeeze(np.delete(int_comp.x_alpha, ref_index, axis=1))
    xb = np.squeeze(np.delete(int_comp.x_beta, ref_index, axis=1))
    xeqa = np.squeeze(np.delete(int_comp.x_eq_alpha, ref_index))
    xeqb = np.squeeze(np.delete(int_comp.x_eq_beta, ref_index))

    omega = np.zeros(len(r))
    non_zero = xb > 0
    omega[non_zero] = (xs-xa[non_zero])/(xb[non_zero]-xa[non_zero])
    return GrowthRateOutput(
        growth_rate=D/r*omega/effective_diffusion_distance(omega),
        x_alpha=xa,
        x_beta=xb,
        x_eq_alpha=xeqa,
        x_eq_beta=xeqb
        )

def _search_for_two_phase_eq(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        matrix_phase: str, prec_phase: str, cache: dict[str, any],
        driving_force_output: DrivingForceOutput=None, driving_force_func: DrivingForceFunction=None, **kwargs):
    '''
    Given x and a search direction (which should correspond to the composition of precipitate from driving force calc)
    Iteratively search between x and search direction until two phase equilibria is found

    If two phase equilibria is not found, then return None
    else return chemical_potential, cs_matrix, cs_precip
    '''
    if driving_force_output is None:
        if driving_force_func is not None:
            driving_force_output = driving_force_func(therm, conditions, matrix_phase, prec_phase, {}, cache, **kwargs)
            if driving_force_output.x_beta is None:
                return None
        else:
            return None

    # fill composition
    defined_elements = {therm.nonvacant_elements.index(e): e for e in therm.nonvacant_elements if v.X(e) in conditions}
    conds = {key: val for key,val in conditions.items()}
    search_dir = driving_force_output.x_beta
    x_init = _full_comp_from_conditions(conditions, therm.nonvacant_elements)
    curr_x = 0.5*x_init + 0.5*search_dir
    for i, e in defined_elements.items():
        conds[v.X(e)] = curr_x[i]

    curr_it = 0
    # max_it is 15, which refers to a maximum of 6e-5 difference in test composition between the 14th and 15th iteration
    #    Which is probably more than enough to find a two-phase region
    max_it = 15
    while curr_it < max_it:
        eq_cs = get_two_phase_eq(therm, conds, matrix_phase, prec_phase, {})
        if eq_cs is None:
            return None
        # If matrix and precipitate are both stable, then return composition sets
        if eq_cs.cs_matrix is not None and eq_cs.cs_prec is not None:
            return eq_cs
        # If only matrix is stable, then move currX towards searchDir
        elif eq_cs.cs_prec is None:
            curr_x = 0.5*curr_x + 0.5*search_dir
        # If only precipitate is stable, then move currX towards x
        elif eq_cs.cs_matrix is None:
            curr_x = 0.5*curr_x + 0.5*x_init
        else:
            return None

        for i, e in defined_elements.items():
            conds[v.X(e)] = curr_x[i]

        curr_it += 1

    return None

def _growth_rate_from_curvature(x, r, g_extra, curvature: CurvatureOutput):
    x = np.atleast_1d(x)
    r = np.atleast_1d(r)
    g_extra = np.atleast_1d(g_extra)

    # Eq 28 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    # Rdiff is (dCbar^T * d2G/dx2 * C^inf - 2yVm/R) = (driving force - Gibbs Thompson energy)
    rdiff = (curvature.dg - g_extra)
    gr = (curvature.mc / r) * rdiff

    # Eq 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    # calpha and cbeta are shape (len(gExtra), len(elements)-1)
    xalpha = x[np.newaxis,:] - np.outer(rdiff, curvature.dc)

    # Eq 36 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    dca = xalpha - curvature.x_eq_alpha[np.newaxis,:]
    dcb = np.matmul(curvature.gba, dca.T).T
    xbeta = curvature.x_eq_beta[np.newaxis,:] + dcb

    xalpha = np.clip(xalpha, 0, 1)
    xbeta = np.clip(xbeta, 0, 1)

    return GrowthRateOutput(
        growth_rate=np.squeeze(gr),
        x_alpha=np.squeeze(xalpha),
        x_beta=np.squeeze(xbeta),
        x_eq_alpha=np.squeeze(curvature.x_eq_alpha),
        x_eq_beta=np.squeeze(curvature.x_eq_beta))

def _curvature_factor_from_eq(
        therm: Thermodynamics, x_full: list[float],
        matrix_phase: str, prec_phase: str, ref_element: str,
        eq_cs: EquilibriumCS,
        driving_force_output: DrivingForceOutput=None
        ) -> CurvatureOutput:
    if therm.mobility_callables.get(matrix_phase, None) is None:
        Dnkj, dmudx_matrix, inv_mob = mob_funcs.inverse_mobility_from_diffusivity(
            eq_cs.chemical_potentials, eq_cs.cs_matrix,
            ref_element, therm.diffusivity_callables[matrix_phase],
            diffusivity_correction=therm.mobility_correction, parameters=therm.parameters
            )

        #NOTE: This is note tested yet
        dtrace = mob_funcs.tracer_diffusivity_from_diff(
            eq_cs.cs_matrix, therm.diffusivity_callables[matrix_phase],
            diffusivity_correction=therm.mobility_correction, parameters=therm.parameters
            )
    else:
        Dnkj, dmudx_matrix, inv_mob = mob_funcs.inverse_mobility(
            eq_cs.chemical_potentials, eq_cs.cs_matrix,
            ref_element, therm.mobility_callables[matrix_phase],
            mobility_correction=therm.mobility_correction,
            vacancy_poor_interstitial_sublattice=therm.vacancy_poor_interstitial_sublattice.get(matrix_phase, False),
            parameters=therm.parameters
            )
        dtrace = mob_funcs.tracer_diffusivity(
            eq_cs.cs_matrix, therm.mobility_callables[matrix_phase],
            mobility_correction=therm.mobility_correction, parameters=therm.parameters)

    ref_index = therm.nonvacant_elements.index(ref_element)
    x = np.delete(x_full, ref_index)
    xm_full = np.array(eq_cs.cs_matrix.X)
    xm = np.delete(xm_full, ref_index)

    dmudx_prec = mob_funcs.total_dmudx(eq_cs.chemical_potentials, eq_cs.cs_prec, ref_element)
    xp_full = np.array(eq_cs.cs_prec.X)
    xp = np.delete(xp_full, ref_index)
    dxbar_full = np.array([xp_full - xm_full])
    dxbar = np.array([xp - xm])
    dxinf = np.array([x - xm])

    # Numerator term for Eq 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    # D^-1 * dCbar
    num = np.matmul(np.linalg.inv(Dnkj), dxbar.T).flatten()

    # Denominator term for Eq 28 and 31 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    # Cbar^T * M^-1 * Cbar
    #Denominator should be a scalar since its V * M * V^T
    den = np.matmul(dxbar, np.matmul(inv_mob, dxbar.T)).flatten()[0]

    # Matrix/precipitate curvature term for Eq 36 from Philippe and Voorhees, Acta Mat 61 (2013) 4237
    # (d2G_beta/dx2)^-1 * (d2G_alpha/dx2)
    if np.linalg.matrix_rank(dmudx_prec) == dmudx_prec.shape[0]:
        gba = np.matmul(np.linalg.inv(dmudx_prec), dmudx_matrix)
    else:
        gba = np.zeros(dmudx_prec.shape)

    # We compute the impingment rate here since we get tracer diffusivity
    # and equilibrium composition as we're computing the other terms
    beta_num = dxbar_full**2
    beta_den = dtrace * xm_full.flatten()
    bsum = np.sum(beta_num / beta_den)
    beta = 1 / bsum

    # driving force from curvature
    if driving_force_output is None:
        dg = np.matmul(dxbar, np.matmul(dmudx_matrix, dxinf.T))
    else:
        dg = driving_force_output.driving_force

    return CurvatureOutput(
        dc=num/den,
        mc=1/den,
        gba=gba,
        dg=dg,
        beta=beta,
        x_eq_alpha=xm,
        x_eq_beta=xp
        )

def compute_curvature(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        matrix_phase: str, prec_phase: str, ref_element: str,
        cache: dict[str, any]=None,
        driving_force_output: DrivingForceOutput=None,
        driving_force_func: DrivingForceFunction=None,
        **kwargs) -> CurvatureOutput:
    """
    Computes factors for computing growth rate:
        mc = 1 / (dx_bar^T * inv(M) * dx_bar)
        dc = inv(Dnkj) * dx_bar / mc
        gba = inv(d2G_beta/dx2) * d2G_alpha/dx2
        x_eq_alpha = equilibrium composition of alpha phase
        x_eq_beta = equilibrium composition of beta phase

    This function also has a couple extra steps to handle equilibrium failures and single
    phase equilibria
        1. Try to use cached composition sets to compute equilibria if available
        2. Use global equilibrium if cache is not available
        3. If in single phase, do bisection search between composition and precipitate composition to find two phase region
             Note: we assume the input composition is near the matrix composition. I say this is fine since
                   the curvature model would be invalid at high supersaturations
    """
    def _process_invalid_eq(calc_source, eq_cache, curvature_cache):
        if eq_cache is None:
            return None
        else:
            print(f'Warning: {calc_source} equilibrum was not able to be solved for, using results of previous calculation')
            return curvature_cache

    cache = {} if cache is None else cache
    eq_cache_id = two_phase_cache_id(matrix_phase, prec_phase)
    curv_cache_id = curvature_cache_id(matrix_phase, prec_phase)

    cs_eq = get_two_phase_eq(therm, conditions, matrix_phase, prec_phase, cache)

    if cs_eq is None:
        return _process_invalid_eq('cached', cache[eq_cache_id], cache[curv_cache_id])

    # If either the matrix of precipitate is unstable, then search for two-phase equilibria
    if cs_eq.cs_matrix is None or cs_eq.cs_prec is None:
        cache[eq_cache_id] = None
        cs_eq = _search_for_two_phase_eq(therm, conditions, matrix_phase, prec_phase, cache, driving_force_output, driving_force_func, **kwargs)
        if cs_eq is None:
            return _process_invalid_eq('search', cache[eq_cache_id], cache[curv_cache_id])

    cache[eq_cache_id] = cs_eq

    x_full = _full_comp_from_conditions(conditions, therm.nonvacant_elements)
    cache[curv_cache_id] = _curvature_factor_from_eq(
        therm, x_full, matrix_phase, prec_phase, ref_element,
        cs_eq, driving_force_output
    )
    return cache[curv_cache_id]


def compute_growth_rate_curvature(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        matrix_phase: str, prec_phase: str, ref_element: str,
        r: float | list[float], g_extra: float | list[float],
        cache: dict[str, any]=None,
        driving_force_output: DrivingForceOutput=None, driving_force_func: DrivingForceFunction=None, **kwargs) -> GrowthRateOutput:
    """
    Assume small supersaturations, the growth rate can be determined from the
    curvature of the Gibbs free energy. This works for any number of components

    Note that because we assume the system is at a low supersaturation, the
    effective diffusion distance that is applied in the binary system is
    not applied here since the low supersaturation assumption means that
    we also assume that the effective diffusion distance is near unity
    """
    cache = {} if cache is None else cache
    curv_factor = compute_curvature(
        therm, conditions, matrix_phase, prec_phase, ref_element,
        cache, driving_force_output, driving_force_func, **kwargs
        )
    if curv_factor is None:
        return None

    x_full = _full_comp_from_conditions(conditions, therm.nonvacant_elements)
    x = np.delete(x_full, therm.nonvacant_elements.index(ref_element))
    return _growth_rate_from_curvature(x, r, g_extra, curv_factor)