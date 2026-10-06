from collections import namedtuple

import numpy as np

from pycalphad import variables as v

from kawin.thermo.thermodynamics import Thermodynamics
from kawin.thermo.mobility import expand_u_frac, u_to_x_frac, interstitials
from kawin.thermo.diffusivity import compute_interdiffusivity

from kawin.diffusion.diffusion import DiffusionModel
from kawin.diffusion.mesh.mesh_base import DiffusionPair, arithmetic_mean
from kawin.diffusion.diffusion_parameters import LookupTable, query_lookup_table

class SinglePhaseModel(DiffusionModel):
    def __init__(
            self,
            mesh,
            elements,
            phase,
            therm,
            temperature,
            *args, **kwargs):
        super().__init__(mesh, elements)
        self.phase = phase
        self.therm = therm
        self.temperature = temperature
        self.conditions = kwargs.get("conditions", dict())
        self.lookup_table = LookupTable(compute_interdiffusivity)
        self.cache = dict()
        self.sort_indices = np.argsort(np.argsort(self.elements))

    def compute_dxdt(self, state):
        u = expand_u_frac(state.u, self.all_elements, self.ref_element, interstitials)
        x = u_to_x_frac(u, self.all_elements, interstitials)

        y_res, z_res = self.mesh.get_response_coordinates(x)
        y_diff, z_diff = self.mesh.get_diffusivity_coordinates(x)

        #T = self.temperature(z_diff, state.time)
        T = self.temperature
        conditions = {v.X(e): y_diff[:,self.all_elements.index(e)] for e in self.elements}
        conditions[v.T] = T
        conditions[v.P] = self.conditions.get(v.P, 101325)
        conditions[v.GE] = self.conditions.get(v.GE, 0)
        d = query_lookup_table(self.therm, conditions, self.lookup_table, ref_element=self.ref_element, phase=self.phase, cache=self.cache)

        pairs = []
        # For binary systems, we only have 1 independent component
        if len(self.elements) == 1:
            pairs.append(DiffusionPair(
                diffusivity=d[:,np.newaxis],
                response=y_res,
                averaging_function=arithmetic_mean))
        else:
            # For 2+ independent component, we want to tile x_j from (1,N) -> (e,N) -> (N,e)
            #d = d[:,self.sort_indices,:]
            #d = d[:,:,self.sort_indices]
            for i in range(len(self.elements)):
                pairs.append(DiffusionPair(
                    diffusivity=d[:,self.sort_indices,self.sort_indices[i]],
                    response=np.tile([y_res[:,i+1]], (len(self.elements), 1)).T,
                    averaging_function=arithmetic_mean
                    ))
                #pairs.append((d[:,:,i], np.tile([yR[:,i]], (numElements, 1)).T, arithmeticMean))

        self._curr_dt = 0.4 * self.mesh.dz**2 / np.amax(d) / self.mesh.dims
        return self.mesh.compute_dxdt(pairs)

    def compute_max_dt(self, state, dxdt):
        return self._curr_dt


# import numpy as np
# from kawin.thermo.Mobility import u_to_x_frac, expand_u_frac, interstitials
# from kawin.diffusion.Diffusion import DiffusionModel
# from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean

# class SinglePhaseModel(DiffusionModel):
#     def _getPairs(self, t, xCurr):
#         '''
#         Compute diffusivity-response pairs

#         J^n_k = -sum(D^n_jk dx_j/dz)
#         dx_k/dt = -dJ^n_k/dz

#         For x_k, a pair would comprise of (D^n_jk, x_j)
#         '''
#         # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
#         u = expand_u_frac(xCurr[0], self.allElements, interstitials)
#         x = u_to_x_frac(u, self.allElements, interstitials)[:,1:]
#         x = self.mesh.unflattenResponse(x)
#         yD, zD = self.mesh.getDiffusivityCoordinates(x)
#         yR, zR = self.mesh.getResponseCoordinates(x)

#         T = self.temperatureParameters(zD, t)
#         numElements = self.mesh.numResponses
#         N = len(yD)
#         d = np.zeros(N) if numElements == 1 else np.zeros((N, numElements, numElements))
#         for i in range(N):
#             inter_diff = self.hashTable.retrieveFromHashTable(yD[i], T[i])
#             if inter_diff is None:
#                 inter_diff = self.therm.getInterdiffusivity(yD[i], T[i], phase=self.phases[0])
#                 self.hashTable.addToHashTable(yD[i], T[i], inter_diff)
#             d[i] = inter_diff

#         pairs = []
#         # For binary systems, we only have 1 independent component
#         if numElements == 1:
#             pairs.append(DiffusionPair(
#                 diffusivity=d[:,np.newaxis],
#                 response=yR,
#                 averageFunction=arithmeticMean))
#         else:
#             # For 2+ independent component, we want to tile x_j from (1,N) -> (e,N) -> (N,e)
#             for i in range(len(self.elements)):
#                 pairs.append(DiffusionPair(
#                     diffusivity=d[:,:,i],
#                     response=np.tile([yR[:,i]], (numElements, 1)).T,
#                     averageFunction=arithmeticMean
#                     ))
#                 #pairs.append((d[:,:,i], np.tile([yR[:,i]], (numElements, 1)).T, arithmeticMean))
#         self._currdt = 0.4 * self.mesh.dz**2 / np.amax(d) / self.mesh.dims
#         return pairs

#     def getDt(self, dXdt):
#         '''
#         Returns dt that was calculated from _getPairs using von-Neumann stability
#         This prevents double calculation of the diffusivity just to get a time step
#         '''
#         return self._currdt