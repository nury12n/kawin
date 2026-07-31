from typing import Protocol
from collections import namedtuple

import numpy as np

from pycalphad import variables as v

from kawin.thermo.thermodynamics import Thermodynamics, enumerate_conditions, get_eq, _full_comp_from_conditions
from kawin.thermo.free_energy_hessian import total_dmudx

InterfacialCompositionOutput = namedtuple(
    'InterfacialCompositionOutput',
    ['x_alpha', 'x_beta']
)

class InterfacialCompositionFunction(Protocol):
    def __call__(
            therm: Thermodynamics, conditions: dict[v.StateVariable, float|list[float]],
            matrix_phase: str, prec_phase: str, **kwargs) -> InterfacialCompositionOutput:
        ...

    @staticmethod
    def squeeze(values: list[InterfacialCompositionOutput]) -> InterfacialCompositionOutput:
        return InterfacialCompositionOutput(
            x_alpha=np.squeeze([val.x_alpha for val in values]),
            x_beta=np.squeeze([val.x_beta for val in values])
        )

@enumerate_conditions(InterfacialCompositionFunction.squeeze)
def compute_interfacial_composition_equilibrium(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float|list[float]],
        matrix_phase: str, prec_phase: str, guess_conditions: dict[v.StateVariable, float]=None, **kwargs) -> InterfacialCompositionOutput:
    """
    Gets interfacial composition between matrix and precipitate phase. This performs
    an equilibrium calculation between the two phases

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str

    Returns
    -------
    matrix composition, precipitate composition
    """
    if guess_conditions is not None:
        for key, val in guess_conditions.items():
            conditions[key] = val
    conditions[v.GE] += therm.g_offset
    wks = get_eq(therm, conditions, [matrix_phase, prec_phase])
    cs_list = list(wks.get_composition_sets())
    cs_matrix = [cs for cs in cs_list if cs.phase_record.phase_name == matrix_phase]
    cs_prec = [cs for cs in cs_list if cs.phase_record.phase_name == prec_phase]
    if len(cs_matrix) != 1 or len(cs_prec) != 1:
        return InterfacialCompositionOutput(
            x_alpha=-np.ones(therm.num_elements, dtype=np.float64),
            x_beta=-np.ones(therm.num_elements, dtype=np.float64)
        )
    return InterfacialCompositionOutput(
        x_alpha=np.array(cs_matrix[0].X, dtype=np.float64),
        x_beta=np.array(cs_prec[0].X, dtype=np.float64)
    )

@enumerate_conditions(InterfacialCompositionFunction.squeeze)
def compute_interfacial_composition_curvature(
        therm: Thermodynamics, conditions: dict[v.StateVariable, float|list[float]],
        matrix_phase: str, prec_phase: str, ref_element: str=None, **kwargs) -> InterfacialCompositionOutput:
    """
    Gets interfacial composition between matrix and precipitate phase. This extrapolates
    from the matrix composition using the Gibbs free energy curvature, solving for x_m and x_p
    in the following:
        (x_m - x_m_eq) * dmu_m/dx * (x_p_eq - x_m_eq) = gExtra
        (x_m - x_m_eq) * dmu_m/dx = (x_p - x_peq) * dmu_p/dx

    Parameters
    ----------
    therm: Thermodynamics
    conditions: dict
    matrix_phase: str
    prec_phase: str

    Returns
    -------
    matrix composition, precipitate composition
    """
    if therm.num_elements > 2:
        raise ValueError("compute_interfacial_composition_equilibrium is only for binary systems")

    # we need gextra to extrapolate the Gibbs free energy curvate, but for
    # equilibrium, we compute assuming flat interface
    gextra = conditions[v.GE]
    conditions[v.GE] = therm.g_offset
    wks = get_eq(therm, conditions, [matrix_phase, prec_phase])
    chemical_potentials = np.squeeze(wks.eq.MU)

    cs_list = list(wks.get_composition_sets())
    cs_matrix = [cs for cs in cs_list if cs.phase_record.phase_name == matrix_phase]
    cs_prec = [cs for cs in cs_list if cs.phase_record.phase_name == prec_phase]
    if len(cs_matrix) != 1 or len(cs_prec) != 1:
        return InterfacialCompositionOutput(
            x_alpha=-np.ones(therm.num_elements, dtype=np.float64),
            x_beta=-np.ones(therm.num_elements, dtype=np.float64)
        )

    if ref_element is None:
        ref_index = np.argmax(_full_comp_from_conditions(conditions, therm.nonvacant_elements))
        ref_element = therm.nonvacant_elements[ref_index]
    else:
        ref_index = therm.nonvacant_elements.index(ref_element)

    dmudx_matrix = np.squeeze(total_dmudx(chemical_potentials, cs_matrix[0], ref_element))
    dmudx_prec = np.squeeze(total_dmudx(chemical_potentials, cs_prec[0], ref_element))

    x_matrix_eq = np.array(cs_matrix[0].X, dtype=np.float64)
    x_prec_eq = np.array(cs_prec[0].X, dtype=np.float64)
    x_matrix_eq = np.squeeze(np.delete(x_matrix_eq, ref_index))
    x_prec_eq = np.squeeze(np.delete(x_prec_eq, ref_index))

    #Compute composition of matrix and precipitate phase
    # x_meq, x_peq - matrix and precipitate composition at bulk equilibrum
    # x_m, x_p - matrix and precipitate composition for particle
    # (x_m - x_meq) * dmu/dx * (x_peq - x_meq) = gExtra
    # (x_m - x_meq) * dmu_m/dx = (x_p - x_peq) * dmu_p/dx
    # If dmu/dx is 0 (or undefined), then we use the equilibrium compositions
    if dmudx_matrix != 0:
        x_matrix = gextra / dmudx_matrix / (x_prec_eq - x_matrix_eq) + x_matrix_eq
    else:
        x_matrix = x_matrix_eq*np.ones(len(therm.nonvacant_elements))

    if dmudx_prec != 0:
        x_prec = dmudx_matrix * (x_matrix - x_matrix_eq) / dmudx_prec + x_prec_eq
    else:
        x_prec = x_prec_eq*np.ones(gextra.shape)

    xm = -np.ones(2)
    xp = -np.ones(2)
    xm[1-ref_index] = x_matrix
    xm[ref_index] = 1-x_matrix
    xp[1-ref_index] = x_prec
    xp[ref_index] = 1-x_prec
    return InterfacialCompositionOutput(
        x_alpha=np.clip(xm, 0, 1, dtype=np.float64),
        x_beta=np.clip(xp, 0, 1, dtype=np.float64)
    )