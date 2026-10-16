from collections import namedtuple

VolumeDrivingForce = namedtuple('VolumeDrivingForce', [
    'dg_vol',
    'dg_chem',
    'x_beta'
])

NucleationBarrier = namedtuple('NucleationBarrer', [
    'r_crit', 'g_crit'
])

NucleationData = namedtuple('NucleationData', [
    'x',
    'T',
    'chemical_driving_force',
    'volumetric_driving_force',
    'Gcrit',
    'Rcrit',
    'nucleation_radius',
    'nucleation_rate',
    'precipitate_composition',
    'Z',
    'beta',
    'tau'
    ])

def volumetric_driving_force(
        therm: Thermodynamics,
        driving_force_function: DrivingForceFunction,
        conditions: dict[v.StateVariable, float | list[float]],
        mat: MatrixParameters,
        prec: PrecipitateParameters,
        aspect_ratio = 1,
        cache = None
        ):
    if cache is None:
        cache = {}
    output = driving_force_function(therm, conditions, mat.phase, prec.phase, cache=cache)
    dg_vol = output.driving_force / prec.volume.vm
    dg_vol -= prec.strain_energy.compute(mat.elastic_parameters, prec.shape.normal_radii(aspect_ratio))
    return VolumeDrivingForce(
        'dg_vol' = dg_vol,
        'dg_chem' = output.driving_force,
        'x_beta' = output.x_beta
    )

def nucleation_barrier(volume_driving_force: VolumeDrivingForce, mat: MatrixParameters, prec: PrecipitateParameters, aspect_ratio = 1):
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in NucleationBarrierParameters to account for grain boundary energy
    '''
    dg_vol = np.atleast_1d(volume_driving_force.dg_vol)
    if dg_vol < 0:
        rcrit = np.nan
        gcrit = np.nan
    else:
        if not prec.nucleation_site.is_grain_boundary:
            r_crit_proposal = 2*prec.shape.thermo_factor(aspect_ratio)*prec.gamma / dg_vol
            r_crit = np.amax([r_crit_proposal, prec.r_min], axis=0)
            g_crit = (4*np.pi/3) * prec.gamma * r_crit**2
        else:
            r_crit_proposal = nucleation_rcrit(prec.nucleation_site, dg_vol, prec.gamma, mat.gb_energy)
            r_crit = np.amax([r_crit_proposal, prec.r_min], axis=0)
            g_crit = nucleation_gcrit_from_rcrit(dg_vol, r_crit)
    return NucleationBarrier(
            rcrit=np.nan,
            gcrit=np.nan
        )

def zeldovich(conditions, r_crit, mat: MatrixParameters, prec: PrecipitateParameters):
    if r_crit == np.nan:
        return np.nan
    else:
        T = conditions[v.T]
        vol_factor = np.sqrt(3*prec.nucleation_sites.volume_factor(prec.gamma, mat.gb_energy)/(4*np.pi))
        z = vol_factor*prec.volume.vm*np.sqrt(prec.gamma/BOLTZMANN_CONSTANT*T)
        z /= (2*np.pi*AVOGADROS_NUMBER*r_crit**2)
        return z

def beta_binary_simple(therm: Thermodynamics, conditions, r_crit, mat: MatrixParameters, prec: PrecipitateParameters, cache=None, *args, **kwargs):
    if cache is None:
        cache = {}
    d_trace = compute_tracer_diffusivity(therm, conditions, mat.phase, cache)
    d = d_trace[therm.nonvacant_elements.index(mat.solutes[0])]
    x = conditions[v.X(mat.solutes[0])]
    beta = prec.nucleation_sites.area_factor(prec.gamma, mat.gb_energy) * r_crit**2 * x * d / mat.volume.a**4
    return beta

def beta_binary_general(therm: Thermodynamics, conditions, r_crit, mat: MatrixParameters, prec: PrecipitateParameters, cache=None, *args, **kwargs):
    if cache is None:
        cache = {}
    if kwargs.get('x_eq_alpha', None) is None:
        x_eq_alpha, x_eq_beta = compute_interfacial_composition(therm, conditions)

def beta_multi(therm: Thermodynamics, conditions, r_crit, mat: MatrixParameters, prec: PrecipitateParameters, cache=None):
    pass

def incubation_time(beta, z, theta=2):
    if z == np.nan:
        return np.nan
    else:
        return 1/(theta*beta*z**2)

def nucleation_rate(z, beta, g_crit, conditions, tau, time=np.inf):
    incubation_time = np.amin([np.exp(-tau/time), 1])
    nuc_rate = z*beta*np.exp(-g_crit/(BOLTZMANN_CONSTANT*conditions[v.T]))*incubation_time
    return nuc_rate

def nucleation_radius(conditions, r_crit, prec: PrecipitateParameters):
    r_nuc = r_crit + 0.5*np.sqrt(BOLTZMANN_CONSTANT*conditions[v.T]/np.pi*prec.gamma)
    return r_nuc

from collections import namedtuple

import numpy as np

from kawin.constants import AVOGADROS_NUMBER, BOLTZMANN_CONSTANT
from kawin.thermo import GeneralThermodynamics, BinaryThermodynamics, MulticomponentThermodynamics
from kawin.thermo.utils import _process_xT_arrays
from kawin.precipitation.PrecipitationParameters import MatrixParameters, PrecipitateParameters

NucleationData = namedtuple('NucleationData', [
    'x', 'T',
    'nucleation_rate', 'chemical_driving_force', 'volumetric_driving_force', 'Rcrit', 'Gcrit',
    'precipitate_composition', 'Z', 'beta', 'tau', 'nucleation_radius'])

def volumetricDrivingForce(therm: GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, aspectRatio = 1, removeCache = False):
    '''
    Computes volumetric driving force (chemical DG / VM - strain energy)
        Strain energy will always reduce the driving force since the precipitate will add strain to the matrix
        In the case where the matrix is prestrained and the precipitate will relax the matrix, then the strain
        energy is negative
    '''
    x, T = _process_xT_arrays(x, T, therm.numElements == 2)
    chemDGs, betaComp = therm.getDrivingForce(x, T, precPhase=precipitate.phase, removeCache=removeCache)
    volDGs = chemDGs / precipitate.volume.Vm
    volDGs -= precipitate.strainEnergy.compute(precipitate.shapeFactor.description.normalRadii(aspectRatio))

    return np.squeeze(chemDGs), np.squeeze(volDGs), np.squeeze(betaComp)

def nucleationBarrier(volumeDrivingForce, precipitate : PrecipitateParameters, aspectRatio = 1):
    '''
    Critical Gibbs free energy and radius at the nucleation barrier
    For bulk and dislocation nucleation
        Rcrit = 2*f*gamma / dG  (where f is the thermodynamic correction factor from the precipitate shape)
        Gcrit = (4*pi/3)*gamma*Rcrit^2
    For grain boundary, edge or corner nucleation, critical G and R is computed to in NucleationBarrierParameters to account for grain boundary energy
    '''
    volumeDrivingForce = np.atleast_1d(volumeDrivingForce)
    indices = volumeDrivingForce > 0

    Rmin = precipitate.Rmin*np.ones(np.array(volumeDrivingForce).shape)
    Rcrit = np.zeros(volumeDrivingForce.shape)
    Gcrit = np.zeros(volumeDrivingForce.shape)

    if not precipitate.nucleation.isGrainBoundaryNucleation:
        RcritProposal = 2*precipitate.shapeFactor.description.thermoFactor(aspectRatio) * precipitate.gamma / volumeDrivingForce[indices]
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = (4*np.pi/3) * precipitate.gamma * Rcrit[indices]**2

    else:
        RcritProposal = precipitate.nucleation.Rcrit(volumeDrivingForce[indices])
        Rcrit[indices] = np.amax([RcritProposal, Rmin[indices]], axis=0)
        Gcrit[indices] = precipitate.nucleation.Gcrit(volumeDrivingForce[indices], Rcrit[indices])

    return np.squeeze(Rcrit), np.squeeze(Gcrit)

def zeldovich(T, Rcrit, precipitate : PrecipitateParameters):
    '''
    Zeldovich factor
    Z = sqrt(3*fv/4*pi) * Vm * sqrt(gamma/kB*T) / (2*pi*Nv*Rcrit^2)
    '''
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    Z = np.zeros(Rcrit.shape)
    Z[indices] = np.sqrt(3 * precipitate.nucleation.volumeFactor / (4 * np.pi)) * precipitate.volume.Vm * np.sqrt(precipitate.gamma / (BOLTZMANN_CONSTANT * T[indices]))
    Z[indices] /= (2 * np.pi * AVOGADROS_NUMBER * Rcrit[indices]**2)
    return np.squeeze(Z)

def betaBinary1(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False):
    '''
    Impingement rate for binary systems

    beta = fa * Rcrit^2 * x * D / a**4
    '''
    x = np.atleast_1d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
    beta[indices] = precipitate.nucleation.areaFactor * Rcrit[indices]**2 * x[indices] * D[:,1] / matrix.volume.a**4
    return np.squeeze(beta)

def betaBinary2(therm : BinaryThermodynamics, x, T, Rcrit, matrix : MatrixParameters, precipitate : PrecipitateParameters, xEqAlpha = None, xEqBeta = None, removeCache = False):
    '''
    Impingement rate for binary systems similar to how multicomponent systems are computed

    beta = fa * Rcrit^2 * Dterm / a**4
    D = [(xB - xA)^2 / (xA*D) + (xB - xA)^2 / ((1-xA)*D)]^-1
    '''
    x = np.atleast_1d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    if xEqAlpha is None:
        xEqAlpha, xEqBeta = therm.getInterfacialComposition(T[indices], np.zeros(T[indices].shape), precipitate.phase)

    beta = np.zeros(Rcrit.shape)
    D = np.atleast_2d(therm.getTracerDiffusivity(x[indices], T[indices], removeCache=removeCache))
    Dfactor = (xEqBeta - xEqAlpha)**2 / (xEqAlpha*D[:,1]) + (xEqBeta - xEqAlpha)**2 / ((1 - xEqAlpha)*D[:,0])
    beta[indices] = precipitate.nucleation.areaFactor * Rcrit[indices]**2 * (1/Dfactor) / matrix.volume.a**4
    return np.squeeze(beta)

def betaMulti(therm : MulticomponentThermodynamics, x, T, Rcrit,  matrix : MatrixParameters, precipitate : PrecipitateParameters, removeCache = False, searchDir = None):
    '''
    Impingement rate for multicomponent systems

    beta = fa * Rcrit^2 * Dterm / a**4
    Dterm = [sum_i((xB_i - xA_i)^2 / (xA_i*D_i))]^-1
    '''
    x = np.atleast_2d(x)
    T = np.atleast_1d(T)
    Rcrit = np.atleast_1d(Rcrit)
    indices = Rcrit != 0

    beta = np.zeros(Rcrit.shape)
    beta[indices] = np.array([therm.impingementFactor(xi, Ti, precPhase=precipitate.phase, removeCache=removeCache, searchDir=searchDir) for xi, Ti in zip(x[indices], T[indices])])
    beta[indices] *= (precipitate.nucleation.areaFactor * Rcrit[indices]**2 / matrix.volume.a**4)
    return np.squeeze(beta)

def incubationTime(beta, Z, matrix : MatrixParameters):
    '''
    Returns incubation time (tau) and a time offset (this is to be compatible with the nonisothermal calculation)

    tau = 1 / (theta * beta * Z^2)
    '''
    beta = np.atleast_1d(beta)
    Z = np.atleast_1d(Z)
    indices = Z != 0

    tau = np.zeros(Z.shape)
    tau[indices] = 1 / (matrix.theta * beta[indices] * Z[indices]**2)
    return np.squeeze(tau)

def incubationTimeNonIsothermal(Z, currBeta, currTime, currTemp, betas, times, temperatures, matrix : MatrixParameters):
    '''
    Note: beta, times and temperature is a subslide from startIndex:n+1
        Start index is the incubationOffset term

    Solve tau for int_0^tau (beta(t)dt) = 1 / (theta*Z(tau)^2)
    '''
    #Assume that Z is constant except for temperature (this will assume that Rcrit is constant)
    #We can actually account for Rcrit since we record it in the KWN model
    LHS = 1 / (matrix.theta * Z**2 * (currTemp / temperatures))

    RHS = np.cumsum(betas[1:] * (times[1:] - times[:-1]))
    if len(RHS) == 0:
        RHS = currBeta * (times - times[0])
    else:
        RHS = np.concatenate((RHS, [RHS[-1] + currBeta * (currTime - times[0])]))

    #Test for intersection
    diff = RHS - LHS
    signChange = np.sign(diff[:-1]) != np.sign(diff[1:])

    #If no intersection
    if not any(signChange):
        #If RHS > LHS, then intersection is at t = 0
        if diff[0] > 0:
            tau = 0
        #Else, RHS intersects LHS beyond simulation time
        #Extrapolate integral of RHS from last point to intersect LHS
        #integral(beta(t-t0)) from t0 to ti + beta_i * (tau - (ti - t0)) = 1 / theta * Z(tau+t0)^2
        else:
            tau = LHS[-1] / currBeta - RHS[-1] / currBeta + (currTime - times[0])
    else:
        tau = times[:-1][signChange][0] - times[0]

    return tau

def nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf):
    '''
    Nucleation rate

    d#/dt = Z * beta * exp(-Gcrit / kB*T) * exp(-tau / t)

    Units are 1/t
    '''
    Z = np.atleast_1d(Z)
    beta = np.atleast_1d(beta)
    Gcrit = np.atleast_1d(Gcrit)
    T = np.atleast_1d(T)
    tau = np.atleast_1d(tau)

    nucRate = np.zeros(Gcrit.shape)
    indices = Gcrit != 0
    incubationTime = np.amin([np.exp(-tau[indices] / time), np.ones(tau[indices].shape)], axis=0)
    nucRate[indices] = np.squeeze(Z[indices] * beta[indices] * np.exp(-Gcrit[indices] / (BOLTZMANN_CONSTANT * T[indices])) * incubationTime)
    return np.squeeze(nucRate)

def nucleationRadius(T, Rcrit, precipitateParameters: PrecipitateParameters):
    '''
    Adds 1/2 * sqrt(kb T / pi gamma) to critical radius to ensure they grow when growth rates are calculated
    '''
    T = np.squeeze(T)
    Rcrit = np.squeeze(Rcrit)
    Rad = Rcrit + 0.5*np.sqrt(BOLTZMANN_CONSTANT * T / (np.pi * precipitateParameters.gamma))
    return np.squeeze(Rad)

def computeSteadyStateNucleation(therm : GeneralThermodynamics, x, T, precipitate: PrecipitateParameters, matrix : MatrixParameters, betaFunc = None, aspectRatio = 1, removeCache = False):
    x, T = _process_xT_arrays(x, T, therm.numElements == 2)
    chemDGs, volDGs, betaComp = volumetricDrivingForce(therm, x, T, precipitate, aspectRatio = aspectRatio, removeCache = removeCache)
    Rcrit, Gcrit = nucleationBarrier(volDGs, precipitate, aspectRatio = aspectRatio)
    Z = zeldovich(T, Rcrit, precipitate)
    if betaFunc is None:
        betaFunc = betaBinary2 if therm.numElements == 2 else betaMulti
    beta = betaFunc(therm, x, T, Rcrit, matrix, precipitate, removeCache = removeCache)
    tau = incubationTime(beta, Z, matrix)
    nucRate = nucleationRate(Z, beta, Gcrit, T, tau, time = np.inf)
    nucRadius = nucleationRadius(T, Rcrit, precipitate)

    return NucleationData(
        x = np.squeeze(x),
        T = np.squeeze(T),
        nucleation_rate = np.squeeze(nucRate),
        chemical_driving_force = np.squeeze(chemDGs),
        volumetric_driving_force = np.squeeze(volDGs),
        Rcrit = np.squeeze(Rcrit),
        Gcrit = np.squeeze(Gcrit),
        precipitate_composition = np.squeeze(betaComp),
        Z = np.squeeze(Z),
        beta = np.squeeze(beta),
        tau = np.squeeze(tau),
        nucleation_radius = np.squeeze(nucRadius)
    )