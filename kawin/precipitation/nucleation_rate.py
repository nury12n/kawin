from typing import Protocol
from collections import namedtuple
import functools

import numpy as np

from pycalphad import variables as v

from kawin.constants import AVOGADROS_NUMBER, BOLTZMANN_CONSTANT
from kawin.thermo.thermodynamics import Thermodynamics, enumerate_conditions, squeeze_named_tuple, squeeze_np_array
from kawin.thermo.diffusivity import TracerDiffusivityFunction, compute_tracer_diffusivity
from kawin.thermo.driving_force import DrivingForceFunction, DrivingForceOutput, compute_driving_force_tangent
from kawin.thermo.interfacial_composition import InterfacialCompositionFunction, InterfacialCompositionOutput, compute_interfacial_composition_equilibrium
from kawin.precipitation.precipitation_parameters import MatrixParameters, PrecipitateParameters
from kawin.precipitation.ostwald_ripening import compute_curvature, CurvatureOutput
from kawin.precipitation.parameters.nucleation import nucleation_rcrit, nucleation_gcrit_from_rcrit

VolumeDrivingForce = namedtuple('VolumeDrivingForce', [
    'dg_vol',
    'dg_chem',
    'x_beta'
])

NucleationBarrier = namedtuple('NucleationBarrer', [
    'r_crit', 'g_crit'
])

NucleationData = namedtuple('NucleationData', [
    'chemical_driving_force',
    'volumetric_driving_force',
    'g_crit',
    'r_crit',
    'nucleation_radius',
    'nucleation_rate',
    'precipitate_composition',
    'Z',
    'beta'
    ])

@enumerate_conditions(squeeze_named_tuple(VolumeDrivingForce))
def compute_volumetric_driving_force(
        therm: Thermodynamics,
        conditions: dict[v.StateVariable, float | list[float]],
        mat: MatrixParameters,
        prec: PrecipitateParameters,
        driving_force_function: DrivingForceFunction = compute_driving_force_tangent,
        aspect_ratio: float = 1,
        cache: dict[str, any] = None
        ):
    if cache is None:
        cache = {}
    output = driving_force_function(therm, conditions, mat.phase, prec.phase, cache=cache)
    dg_vol = output.driving_force / prec.volume.vm
    dg_vol -= prec.strain_energy.compute(mat.elastic_parameters, prec.shape.normal_radii(aspect_ratio))
    return VolumeDrivingForce(
        dg_vol = dg_vol,
        dg_chem = output.driving_force,
        x_beta = output.x_beta
    )

def compute_nucleation_barrier(
        volume_driving_force: VolumeDrivingForce, mat: MatrixParameters, prec: PrecipitateParameters,
        aspect_ratio: float = 1) -> NucleationBarrier:
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in NucleationBarrierParameters to account for grain boundary energy
    '''
    dg_vol = np.atleast_1d(volume_driving_force.dg_vol)
    indices = dg_vol > 0
    r_crit = np.zeros(dg_vol.shape)
    g_crit = np.zeros(dg_vol.shape)

    if not prec.nucleation_site.is_grain_boundary:
        r_crit_proposal = 2*prec.shape.thermo_factor(aspect_ratio)*prec.gamma / dg_vol[indices]
        r_crit[indices] = np.clip(r_crit_proposal, a_min=prec.r_min, a_max=None)
        g_crit[indices] = (4*np.pi/3) * prec.gamma * r_crit[indices]**2
    else:
        r_crit_proposal = nucleation_rcrit(prec.nucleation_site, dg_vol[indices], prec.gamma, mat.gb_energy)
        r_crit[indices] = np.clip(r_crit_proposal, a_min=prec.r_min, a_max=None)
        g_crit[indices] = nucleation_gcrit_from_rcrit(prec.nucleation_site, dg_vol[indices], r_crit[indices], prec.gamma, mat.gb_energy)
    r_crit[~indices] = np.nan
    g_crit[~indices] = np.nan
    return NucleationBarrier(
        r_crit=np.squeeze(r_crit),
        g_crit=np.squeeze(g_crit)
    )

def zeldovich(conditions: dict[v.StateVariable, float], r_crit: float, mat: MatrixParameters, prec: PrecipitateParameters) -> float:
    T = np.atleast_1d(conditions[v.T])
    r_crit = np.atleast_1d(r_crit)
    indices = np.isfinite(r_crit)
    z = np.zeros(r_crit.shape)
    z[~indices] = np.nan
    vol_factor = np.sqrt(3*prec.nucleation_site.volume_factor(prec.gamma, mat.gb_energy)/(4*np.pi))
    z[indices] = vol_factor*prec.volume.vm*np.sqrt(prec.gamma/(BOLTZMANN_CONSTANT*T[indices]))
    z[indices] /= (2*np.pi*AVOGADROS_NUMBER*r_crit[indices]**2)
    return z

class BetaFunction(Protocol):
    def __call__(
            self, therm: Thermodynamics, conditions: dict[v.StateVariable, float],
            r_crit: float, dg_vol: VolumeDrivingForce,
            mat: MatrixParameters, prec: PrecipitateParameters,
            cache: dict[str, any] = None, *args, **kwargs) -> float:
        ...

def beta_binary_simple(
        therm: Thermodynamics, conditions,
        r_crit: float, dg_vol: VolumeDrivingForce,
        mat: MatrixParameters, prec: PrecipitateParameters,
        cache: dict[v.StateVariable, any] = None,
        *args, **kwargs) -> float:
    cache = {} if cache is None else cache
    d_trace = compute_tracer_diffusivity(therm, conditions, mat.phase, cache)
    d = d_trace[therm.nonvacant_elements.index(mat.solutes[0])]
    x = conditions[v.X(mat.solutes[0])]
    beta = prec.nucleation_site.area_factor(prec.gamma, mat.gb_energy) * r_crit**2 * x * d / mat.volume.a**4
    return beta

def beta_binary_general_impl(
        interfacial_comp_function: InterfacialCompositionFunction,
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        r_crit: float, dg_vol: VolumeDrivingForce,
        mat: MatrixParameters, prec: PrecipitateParameters,
        cache: dict[str, any] = None,
        x_eq: InterfacialCompositionOutput = None,
        *args, **kwargs) -> float:
    cache = {} if cache is None else cache
    if x_eq is None:
        x_eq = interfacial_comp_function(therm, conditions, mat.phase, prec.phase, *args, **kwargs)
    D = compute_tracer_diffusivity(therm, conditions, mat.phase, cache)
    idx = therm.nonvacant_elements.index(mat.solutes[0])
    diff = (x_eq.x_beta[idx] - x_eq.x_alpha[idx])**2
    D_factor = diff / (x_eq.x_alpha[0]*D[0]) + diff / (x_eq.x_alpha[1]*D[1])
    beta = prec.nucleation_site.area_factor(prec.gamma, mat.gb_energy)*r_crit**2*(1/D_factor) / mat.volume.a**4
    return beta

def create_beta_binary_general(
        interfacial_comp_function: InterfacialCompositionFunction = compute_interfacial_composition_equilibrium
        ) -> BetaFunction:
    return functools.partial(beta_binary_general_impl, interfacial_comp_function)

def beta_multi_impl(
        driving_force_function: DrivingForceFunction,
        therm: Thermodynamics, conditions: dict[v.StateVariable, float],
        r_crit: float, dg_vol: VolumeDrivingForce,
        mat: MatrixParameters, prec: PrecipitateParameters,
        cache: dict[str, any] = None,
        *args, **kwargs) -> float:
    ref_element = list(set(therm.nonvacant_elements) - set(mat.solutes))[0]
    dg = DrivingForceOutput(driving_force=dg_vol.dg_chem, x_beta=dg_vol.x_beta)
    curv = compute_curvature(therm, conditions, mat.phase, prec.phase, ref_element, cache, driving_force_output=dg, driving_force_func=driving_force_function, *args, **kwargs)
    return curv.beta*prec.nucleation_site.area_factor(prec.gamma, mat.gb_energy)*r_crit**2 / mat.volume.a**4

def create_beta_multi(
        driving_force_function: DrivingForceFunction = compute_driving_force_tangent
        ) -> BetaFunction:
    return functools.partial(beta_multi_impl, driving_force_function)

def incubation_time(beta: float, z: float, theta: float = 2) -> float:
    beta = np.atleast_1d(beta)
    z = np.atleast_1d(z)
    tau = np.zeros(beta.shape)
    indices = np.isfinite(z)
    tau[~indices] = np.nan
    tau[indices] = 1/(theta*beta[indices]*z[indices]**2)

def nucleation_rate(z: float, beta: float, g_crit: float, conditions: dict[v.StateVariable, float], tau: float, time: float = np.inf) -> float:
    incubation_time = np.amin([np.exp(-tau/time), 1])
    nuc_rate = z*beta*np.exp(-g_crit/(BOLTZMANN_CONSTANT*conditions[v.T]))*incubation_time
    return nuc_rate

def nucleation_radius(conditions: dict[v.StateVariable, float], r_crit: float, prec: PrecipitateParameters) -> float:
    r_nuc = r_crit + 0.5*np.sqrt(BOLTZMANN_CONSTANT*conditions[v.T]/np.pi*prec.gamma)
    return r_nuc

@enumerate_conditions(squeeze_named_tuple(NucleationData))
def compute_steady_state_nucleation(
        therm: Thermodynamics,
        conditions: dict[v.StateVariable, float],
        mat: MatrixParameters,
        prec: PrecipitateParameters,
        dg_func: DrivingForceFunction = compute_driving_force_tangent,
        beta_func: BetaFunction = None,
        aspect_ratio: float = 1,
        cache: dict[str, any] = None,
        *args, **kwargs) -> NucleationData:
    if cache is None:
        cache = {}
    dg_vol = compute_volumetric_driving_force(therm, conditions, mat, prec, dg_func, aspect_ratio, cache)
    nuc_barrier = compute_nucleation_barrier(dg_vol, mat, prec, aspect_ratio)
    Z = zeldovich(conditions, nuc_barrier.r_crit, mat, prec)
    if beta_func is None:
        beta_func = create_beta_binary_general() if therm.num_elements == 2 else create_beta_multi()
    beta = beta_func(therm, conditions, nuc_barrier.r_crit, dg_vol, mat, prec, cache, *args, **kwargs)
    nuc_rate = Z*beta*np.exp(-nuc_barrier.g_crit/(BOLTZMANN_CONSTANT*conditions[v.T]))
    nuc_radius = nucleation_radius(conditions, nuc_barrier.r_crit, prec)
    return NucleationData(
        chemical_driving_force = dg_vol.dg_chem,
        volumetric_driving_force = dg_vol.dg_vol,
        g_crit = nuc_barrier.g_crit,
        r_crit = nuc_barrier.r_crit,
        nucleation_radius = nuc_radius,
        nucleation_rate = nuc_rate,
        precipitate_composition = dg_vol.x_beta,
        Z = Z,
        beta = beta
    )


# from collections import namedtuple

# import numpy as np

# from kawin.constants import AVOGADROS_NUMBER, BOLTZMANN_CONSTANT
# from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
# from kawin.thermo.utils import _process_xT_arrays
# from kawin.precipitation.PrecipitationParameters import MatrixParameters, PrecipitateParameters

# NucleationData = namedtuple('NucleationData', [
#     'x', 'T',
#     'nucleation_rate', 'chemical_driving_force', 'volumetric_driving_force', 'Rcrit', 'Gcrit',
#     'precipitate_composition', 'Z', 'beta', 'tau', 'nucleation_radius'])

# def volumetricDrivingForce(therm: GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, aspectRatio = 1, removeCache = False):
#     '''
#     Computes volumetric driving force (chemical DG / VM - strain energy)
#         Strain energy will always reduce the driving force since the precipitate will add strain to the matrix
#         In the case where the matrix is prestrained and the precipitate will relax the matrix, then the strain
#         energy is negative
#     '''
#     x, T = _process_xT_arrays(x, T, therm.numElements == 2)
#     chemDGs, betaComp = therm.getDrivingForce(x, T, precPhase=precipitate.phase, removeCache=removeCache)
#     volDGs = chemDGs / precipitate.volume.Vm
#     volDGs -= precipitate.strainEnergy.compute(precipitate.shapeFactor.description.normalRadii(aspectRatio))

#     return np.squeeze(chemDGs), np.squeeze(volDGs), np.squeeze(betaComp)

# def nucleationBarrier(volumeDrivingForce, precipitate : PrecipitateParameters, aspectRatio = 1):
#     '''
#     Critical Gibbs free energy and radius at the nucleation barrier
#     For bulk and dislocation nucleation
#         Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
#         Gcrit = (4*pi/3)*gamma*Rcrit^2
#     For grain boundary, edge or corner nucleation, critical G and R is computed to in NucleationBarrierParameters to account for grain boundary energy
#     '''
#     volumeDrivingForce = np.atleast_1d(volumeDrivingForce)
#     indices = volumeDrivingForce > 0

#     Rmin = precipitate.Rmin*np.ones(np.array(volumeDrivingForce).shape)
#     Rcrit = np.zeros(volumeDrivingForce.shape)
#     Gcrit = np.zeros(volumeDrivingForce.shape)

#     if not precipitate.nucleation.isGrainBoundaryNucleation:
#         RcritProposal = 2*precipitate.shapeFactor.description.thermoFactor(aspectRatio) * precipitate.gamma / volumeDrivingForce[indices]
#         Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
#         Gcrit[indices] = (4*np.pi/3) * precipitate.gamma * Rcrit[indices]**2

#     else:
#         RcritProposal = precipitate.nucleation.Rcrit(volumeDrivingForce[indices])
#         Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
#         Gcrit[indices] = precipitate.nucleation.Gcrit(volumeDrivingForce[indices], Rcrit[indices])

#     return np.squeeze(Rcrit), np.squeeze(Gcrit)

# def zeldovich(T, Rcrit, precipitate : PrecipitateParameters):
#     '''
#     Zeldovich factor
#     Z = sqrt(3*fv/4*pi) * Vm * sqrt(gamma/kB*T) / (2*pi*Nv*Rcrit^2)
#     '''
#     T = np.atleast_1d(T)
#     Rcrit = np.atleast_1d(Rcrit)
#     indices = Rcrit != 0

#     Z = np.zeros(Rcrit.shape)
#     Z[indices] = np.sqrt(3 * precipitate.nucleation.volumeFactor / (4 * np.pi)) * precipitate.volume.Vm * np.sqrt(precipitate.gamma / (BOLTZMANN_CONSTANT * T[indices]))
#     Z[indices] /= (2 * np.pi * AVOGADROS_NUMBER * Rcrit[indices]**2)
#     return np.squeeze(Z)

# def betaBinary1(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False):
#     '''
#     Impingement rate for binary systems

#     beta = fa * Rcrit^2 * x * D / a**4
#     '''
#     x = np.atleast_1d(x)
#     T = np.atleast_1d(T)
#     Rcrit = np.atleast_1d(Rcrit)
#     indices = Rcrit != 0

#     beta = np.zeros(Rcrit.shape)
#     D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
#     beta[indices] = precipitate.nucleation.areaFactor * Rcrit[indices]**2 * x[indices] * D[:,1] / matrix.volume.a**4
#     return np.squeeze(beta)

# def betaBinary2(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, xEqAlpha = None, xEqBeta = None, removeCache = False):
#     '''
#     Impingement rate for binary systems similar to how multicomponent systems are computed

#     beta = fa * Rcrit^2 * Dterm / a**4
#     D = [(xB - xA)^2 / (xA*D) + (xB - xA)^2 / ((1-xA)*D)]^-1
#     '''
#     x = np.atleast_1d(x)
#     T = np.atleast_1d(T)
#     Rcrit = np.atleast_1d(Rcrit)
#     indices = Rcrit != 0

#     if xEqAlpha is None:
#         xEqAlpha, xEqBeta = therm.getInterfacialComposition(T[indices], np.zeros(T[indices].shape), precipitate.phase)

#     beta = np.zeros(Rcrit.shape)
#     D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
#     Dfactor = (xEqBeta - xEqAlpha)**2 / (xEqAlpha*D[:,1]) + (xEqBeta - xEqAlpha)**2 / ((1 - xEqAlpha)*D[:,0])
#     beta[indices] = precipitate.nucleation.areaFactor * Rcrit[indices]**2 * (1/Dfactor) / matrix.volume.a**4
#     return np.squeeze(beta)

# def betaMulti(therm : MulticomponentThermodynamics, x, T, Rcrit,  matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False, searchDir = None):
#     '''
#     Impingement rate for multicomponent systems

#     beta = fa * Rcrit^2 * Dterm / a**4
#     Dterm = [sum_i((xB_i - xA_i)^2 / (xA_i*D_i))]^-1
#     '''
#     x = np.atleast_2d(x)
#     T = np.atleast_1d(T)
#     Rcrit = np.atleast_1d(Rcrit)
#     indices = Rcrit != 0

#     beta = np.zeros(Rcrit.shape)
#     beta[indices] = np.array([therm.impingementFactor(xi, Ti, precPhase=precipitate.phase, removeCache=removeCache, searchDir=searchDir) for xi, Ti in zip(x[indices], T[indices])])
#     beta[indices] *= (precipitate.nucleation.areaFactor * Rcrit[indices]**2 / matrix.volume.a**4)
#     return np.squeeze(beta)

# def incubationTime(beta, Z, matrix : MatrixParameters):
#     '''
#     Returns incubation time (tau) and a time offset (this is to be compatible with the nonisothermal calculation)

#     tau = 1 / (theta * beta * Z^2)
#     '''
#     beta = np.atleast_1d(beta)
#     Z = np.atleast_1d(Z)
#     indices = Z != 0

#     tau = np.zeros(Z.shape)
#     tau[indices] = 1 / (matrix.theta * beta[indices] * Z[indices]**2)
#     return np.squeeze(tau)

# def incubationTimeNonIsothermal(Z, currBeta, currTime, currTemp, betas, times, temperatures, matrix : MatrixParameters):
#     '''
#     Note: beta, times and temperature is a subslide from startIndex:n+1
#         Start index is the incubationOffset term

#     Solve tau for int_0^tau (beta(t)dt) = 1 / (theta*Z(tau)^2)
#     '''
#     #Assume that Z is constant except for temperature (this will assume that Rcrit is constant)
#     #We can actually account for Rcrit since we record it in the KWN model
#     LHS = 1 / (matrix.theta * Z**2 * (currTemp / temperatures))

#     RHS = np.cumsum(betas[1:] * (times[1:] - times[:-1]))
#     if len(RHS) == 0:
#         RHS = currBeta * (times - times[0])
#     else:
#         RHS = np.concatenate((RHS, [RHS[-1] + currBeta * (currTime - times[0])]))

#     #Test for intersection
#     diff = RHS - LHS
#     signChange = np.sign(diff[:-1]) != np.sign(diff[1:])

#     #If no intersection
#     if not any(signChange):
#         #If RHS > LHS, then intersection is at t = 0
#         if diff[0] > 0:
#             tau = 0
#         #Else, RHS intersects LHS beyond simulation time
#         #Extrapolate integral of RHS from last point to intersect LHS
#         #integral(beta(t-t0)) from t0 to ti + beta_i * (tau - (ti - t0)) = 1 / theta * Z(tau+t0)^2
#         else:
#             tau = LHS[-1] / currBeta - RHS[-1] / currBeta + (currTime - times[0])
#     else:
#         tau = times[:-1][signChange][0] - times[0]

#     return tau

# def nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf):
#     '''
#     Nucleation rate

#     d#/dt = Z * beta * exp(-Gcrit / kB*T) * exp(-tau / t)

#     Units are 1/t
#     '''
#     Z = np.atleast_1d(Z)
#     beta = np.atleast_1d(beta)
#     Gcrit = np.atleast_1d(Gcrit)
#     T = np.atleast_1d(T)
#     tau = np.atleast_1d(tau)

#     nucRate = np.zeros(Gcrit.shape)
#     indices = Gcrit != 0
#     incubationTime = np.amin([np.exp(-tau[indices] / time), np.ones(tau[indices].shape)], axis=0)
#     nucRate[indices] = np.squeeze(Z[indices] * beta[indices] * np.exp(-Gcrit[indices] / (BOLTZMANN_CONSTANT * T[indices])) * incubationTime)
#     return np.squeeze(nucRate)

# def nucleationRadius(T, Rcrit, precipitateParameters: PrecipitateParameters):
#     '''
#     Adds 1/2 * sqrt(kb T / pi gamma) to critical radius to ensure they grow when growth rates are calculated
#     '''
#     T = np.squeeze(T)
#     Rcrit = np.squeeze(Rcrit)
#     Rad = Rcrit + 0.5*np.sqrt(BOLTZMANN_CONSTANT * T / (np.pi * precipitateParameters.gamma))
#     return np.squeeze(Rad)

# def computeSteadyStateNucleation(therm : GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, matrix : MatrixParameters, betaFunc = None, aspectRatio = 1, removeCache = False):
#     x, T = _process_xT_arrays(x, T, therm.numElements == 2)
#     chemDGs, volDGs, betaComp = volumetricDrivingForce(therm, x, T, precipitate, aspectRatio = aspectRatio, removeCache = removeCache)
#     Rcrit, Gcrit = nucleationBarrier(volDGs, precipitate, aspectRatio = aspectRatio)
#     Z = zeldovich(T, Rcrit, precipitate)
#     if betaFunc is None:
#         betaFunc = betaBinary2 if therm.numElements == 2 else betaMulti
#     beta = betaFunc(therm, x, T, Rcrit, matrix, precipitate, removeCache = removeCache)
#     tau = incubationTime(beta, Z, matrix)
#     nucRate = nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf)
#     nucRadius = nucleationRadius(T, Rcrit, precipitate)

#     return NucleationData(
#         x = np.squeeze(x),
#         T = np.squeeze(T),
#         nucleation_rate = np.squeeze(nucRate),
#         chemical_driving_force = np.squeeze(chemDGs),
#         volumetric_driving_force = np.squeeze(volDGs),
#         Rcrit = np.squeeze(Rcrit),
#         Gcrit = np.squeeze(Gcrit),
#         precipitate_composition = np.squeeze(betaComp),
#         Z = np.squeeze(Z),
#         beta = np.squeeze(beta),
#         tau = np.squeeze(tau),
#         nucleation_radius = np.squeeze(nucRadius)
#     )