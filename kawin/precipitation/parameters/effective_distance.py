import numpy as np
import scipy.special as ssp

# TODO: would this be better put in thermo.ostwald_ripening or precipitation.precipitation_parameters?
def ohm(lam):
    """
    Supersaturation as a function of lambda
    """
    # note: erfcx(x) = exp(x^2)*erfc(x)
    return 2*lam**2 - 2*np.sqrt(np.pi)*lam**3*ssp.erfcx(lam)

def lamda_low(ohm):
    """
    Lambda when supersaturation approaches 0
    """
    return np.sqrt(ohm/2)

def lambda_high(ohm):
    """
    Lambda when supersaturation approaches 1
    """
    return np.sqrt(3/2*(1-ohm))

class EffectiveDiffusionDistanceLookupTable:
    def __init__(self, n, lam_min, lam_max):
        self.lam = np.logspace(lam_min, lam_max, n)
        self.ohm = ohm(self.lam)
        self.eff_dist = self.ohm/(2*self.lam**2)
        self.ohm = np.concatenate(([0], self.ohm, [1]))
        self.eff_dist = np.concatenate(([1], self.eff_dist, [0]))

def effective_diffusion_distance(ohm):
    """
    Effective diffusion distance given supersaturation
    This gives a constant to be multiplied by the particle radius during growth rate

    The effective diffusion distance is given by
    eps = ohm/2*lambda^2

    Where Q is the super saturation and lambda is given by
    ohm = 2*lambda^2 - 2*lambda^3 * sqrt(pi) * exp(lambda^2) * erfc(lambda)

    We use a lookup table here for efficiency

    Parameters
    ----------
    ohm: supersaturation
        (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition
    """
    return np.interp(ohm, effective_diffusion_distance.table.ohm, effective_diffusion_distance.table.eff_dist)

# function attributes just seem weird, but it effectively allows for "static" variables in functions
# this lets us make this function as a standalone without a class implementation
# values are selected here that give a max error of ~1e-5
# # this is based off sweeping the supersaturation from 0 to 1 at an interval of 1e-5
effective_diffusion_distance.table = EffectiveDiffusionDistanceLookupTable(n=1000, lam_min=-6, lam_max=1)


# class EffectiveDiffusionFunctions:
#     '''
#     Stores variables and functions to calculate effective diffusion distance
#     '''
#     def __init__(self, isEnabled=True):
#         self.setupInterpolation()
#         self.isEnabled = isEnabled

#     def setupInterpolation(self, n = 250, lmin = -5, lmax = 1):
#         '''
#         Sets up interpolation points for effective diffusion distance

#         Parameters
#         ----------
#         n : int
#             Number of points to interpolate between, two endpoints will be added
#             at supersaturation of 0 and 1
#         lmin : float
#             Minimum lambda value in log scale
#         lmax : float
#             Maximum lambda value in log scale
#         '''
#         lamInterp = np.logspace(lmin, lmax, n)
#         self.ohmInterp = 2*lamInterp**2 - 2*lamInterp**3 * np.sqrt(np.pi) * np.exp(lamInterp**2) * ssp.erfc(lamInterp)
#         self.effDiffInterp = self.ohmInterp / (2 * lamInterp**2)
#         self.ohmInterp = np.concatenate(([0], self.ohmInterp, [1]))
#         self.effDiffInterp = np.concatenate(([1], self.effDiffInterp, [0]))

#     def effectiveDiffusionDistance(self, supersaturation):
#         '''
#         Effective diffusion distance given supersaturation
#         This gives a constant to be multiplied by the particle radius

#         The effective diffusion distance is given by
#         eps = Q/2*lambda^2

#         Where Q is the super saturation and lambda is given by
#         Q = 2*lambda^2 - 2*lambda^3 * sqrt(pi) * exp(lambda^2) * (1 - erf(lambda))

#         Parameters
#         ----------
#         supersaturation : float or array of floats
#             (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition

#         Returns
#         -------
#         Effective diffusion distance (eps) to be used as (eps*R) in the growth rate equation
#         '''
#         return np.interp(supersaturation, self.ohmInterp, self.effDiffInterp)

#     def lambdaLow(self, supersaturation):
#         '''
#         Lambda when Q approaches 0
#         This is done to prevent precision errors when multiplying exp*(1-erf)
#         '''
#         return np.sqrt(supersaturation / 2)

#     def lambdaHigh(self, supersaturation):
#         '''
#         Lambda when Q approaches 1
#         This is done to prevent precision errors when multiplying exp*(1-erf)
#         '''
#         return np.sqrt(3 / (2 * (1 - supersaturation)))

#     def effectiveDiffusionDistanceApprox(self, supersaturation):
#         '''
#         Effective diffusion distance given supersaturation
#         This gives a constant to be multiplied by the particle radius

#         When Q approaches 1, this equation oscillates due to floating point errors
#         Therefore, interpolation is performed between the two limits (Q -> 0 and Q -> 1)

#         This may be removed in later versions

#         Parameters
#         ----------
#         supersaturation : float or array of floats
#             (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition

#         Returns
#         -------
#         Effective diffusion distance (eps) to be used as (eps*R) in the growth rate equation
#         '''
#         #Interpolation constant
#         a = 1.2

#         supersaturation = np.atleast_1d(supersaturation)
#         diff = np.zeros(len(supersaturation))
#         indices = (supersaturation >= 0) & (supersaturation < 1)
#         diff[supersaturation >= 1] = 0
#         diff[supersaturation < 0] = 1

#         lam = (1 - supersaturation[indices]**a) * self.lambdaLow(supersaturation[indices]) + (supersaturation[indices]**a) * self.lambdaHigh(supersaturation[indices])
#         diff[indices] = supersaturation[indices] / (2 * lam**2)

#         return np.squeeze(diff)

#     def noDiffusionDistance(self, supersaturation):
#         '''
#         If neglecting the effective diffusion distance, return 1

#         Parameters
#         ----------
#         supersaturation : float or array of floats
#             (x - x_P) / (x_M - x_P) where x is matrix composition, x_M is parent composition and x_P is precipitate composition

#         Returns
#         -------
#         1 or array of 1s
#         '''
#         supersaturation = np.atleast_1d(supersaturation)
#         return np.squeeze(np.ones(supersaturation.shape, dtype=np.float64))

#     def __call__(self, supersaturation):
#         if self.isEnabled:
#             return self.effectiveDiffusionDistance(supersaturation)
#         else:
#             return self.noDiffusionDistance(supersaturation)