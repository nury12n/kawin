import numpy as np

from pycalphad import variables as v

from kawin.constants import GAS_CONSTANT
from kawin.thermo.thermodynamics import Thermodynamics
from kawin.thermo.mobility import expand_u_frac, u_to_x_frac, x_to_u_frac, interstitials
from kawin.diffusion.diffusion import DiffusionModel, DiffusionState
from kawin.diffusion.diffusion_parameters import LookupTable, DiffusionModelTemperature, IsothermalTemperature
from kawin.diffusion.homogenization_parameters import AveragingFunction, HomogenizationFunction, PostProcessFunction, MobilityFunction, hashin_shtrikman_lower, post_process_do_nothing, compute_mobility, compute_homogenization_function
from kawin.diffusion.mesh.mesh_base import MeshBase, DiffusionPair, arithmetic_mean, log_mean, harmonic_mean

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

    def compute_dxdt(self, state: DiffusionState) -> np.ndarray:
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
        return self.mesh.compute_dxdt(pairs)

    def compute_max_dt(self, state: DiffusionState, dxdt: np.ndarray) -> float:
        return self.max_composition_change / np.amax(np.abs(dxdt[dxdt!=0]))

# def homogenization_diffusion_pairs(therm: Thermodynamics, u, time, diff_coords, response_coords, ref_element, T_func, homogenization_function, homogenization_parameters, **kwargs) -> list[DiffusionPair]:
#     '''
#     Compute diffusivity-response pairs

#     J_k = -\Gamma_k d\mu_k/dz - \eps*RT*\Gamma_k/u_k du_k/dz
#     J^n_k = -sum(\delta_jk - u_k) J_j
#     dx_k/dt = -dJ^n_k/dz

#     For x_k, a pair would comprise of:
#         (\delta_jk - u_k) \Gamma_j, \mu_k
#         (\delta_jk - u_k) \eps*RT*\Gamma_k/u_k, u_k
#     '''
#     # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
#     # TODO: this feels inefficient to convert u->x for mobility, then back to u for flux calculation
#     u = expand_u_frac(u, therm.nonvacant_elements, ref_element, interstitials)
#     x = u_to_x_frac(u, therm.nonvacant_elements, interstitials)

#     # temp is (N,e)
#     T_d = T_func(diff_coords.z, time)
#     T_r = T_func(response_coords.z, time)
#     # mob and mu are (N,e)
#     mob_d, mu_d = homogenization_function(therm, diff_coords.y, T_d, homogenization_parameters)
#     mob_r, mu_r = homogenization_function(therm, response_coords.y, T_r, homogenization_parameters)

#     # Full composition
#     # x_full = (N,e+1), u_full = (N,e+1), u_term = (N,e+1,e+1)
#     x_fullD = np.concatenate((1-np.sum(diff_coords.y, axis=1)[:,np.newaxis], diff_coords.y), axis=1)
#     u_fullD = x_to_u_frac(x_fullD, therm.nonvacant_elements, interstitials)
#     # u_term is the (\delta_jk - u_k), which converts from a lattice fixed frame to a volume fixed frame
#     u_termD = (np.eye(len(therm.nonvacant_elements))[np.newaxis,:,:] - u_fullD[:,:,np.newaxis])
#     # When converting to a volume fixed frame, only the substitutional (or volume contributing) elements
#     # contribute to the corrected flux. To account for interstitials, we replace the column
#     # corresponding to the interstitial (which at this point is [u_A, u_B, 1-u_I, u_D] where I is interstital)
#     # to be [0, 0, 1, 0]. So the column is 0 except for the interstitial row
#     for i,e in enumerate(therm.nonvacant_elements):
#         if e in interstitials:
#             u_termD[:,:,i] = 0
#             u_termD[:,i,i] = 1

#     x_r = np.concatenate((1-np.sum(response_coords.y, axis=1)[:,np.newaxis], response_coords.y), axis=1)
#     u_r = x_to_u_frac(x_r, therm.nonvacant_elements, interstitials)

#     # mobility matrix is repeated among rows
#     mob_d_matrix = np.repeat(mob_d[:,np.newaxis,:], therm.num_elements, axis=1)
#     # We'll group eps*R with T here
#     T_matrix = np.tile(T_d[:,np.newaxis,np.newaxis], (1, therm.num_elements, therm.num_elements)) * homogenizationParameters.eps * GAS_CONSTANT
#     # inverse composition and response is repeated among rows
#     inv_u_matrix = np.repeat(1/u_fullD[:,np.newaxis,:], therm.num_elements, axis=1)
#     mu_matrix = np.repeat(mu_r[:,np.newaxis,:], therm.num_elements, axis=1)
#     u_r_matrix = np.repeat(u_r[:,np.newaxis,:], therm.num_elements, axis=1)

#     pairs = []
#     # Since volume fixed frame leads to 1 dependent component (which we take as the first)
#     # we don't need to take the 1st row of mob_term and ideal_term
#     for i in range(len(therm.nonvacant_elements)):
#         # homogenization contribution - (vol transform * \Gamma) * dmu/dz
#         pairs.append(DiffusionPair(
#             diffusivity=np.transpose(np.array([u_termD[:,1:,i], mob_d_matrix[:,1:,i]]), axes=(1,2,0)),
#             response=mu_matrix[:,1:,i],
#             averageFunction=_homogenizationMean,
#             atNodeFunction=_atNodeProduct
#         ))
#         # ideal contribution - (vol transform * eps*R*T * \Gamma / u) * du/dz
#         pairs.append(DiffusionPair(
#             diffusivity=np.transpose(np.array([u_termD[:,1:,i], T_matrix[:,1:,i], mob_d_matrix[:,1:,i], inv_u_matrix[:,1:,i]]), axes=(1,2,0)),
#             response=u_r_matrix[:,1:,i],
#             averageFunction=_idealMean,
#             atNodeFunction=_atNodeProduct
#         ))
#     return pairs

# import numpy as np

# from kawin.Constants import GAS_CONSTANT
# from kawin.diffusion.Diffusion import DiffusionModel
# from kawin.thermo.Mobility import interstitials, x_to_u_frac, u_to_x_frac, expand_u_frac
# from kawin.diffusion.HomogenizationParameters import HomogenizationParameters, computeHomogenizationFunction
# from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean, harmonicMean, logMean

# def _homogenizationMean(Ds):
#     '''
#     Average homogenization flux
#     Ds should be in the shape of (m x N x y x 2)
#         m - number of items to average over
#         N - number of nodes
#         y - number of responses
#         2 - (transformation to volume fixed frame, homogenization term (\Gamma))
#             transformation is in composition, so use arithmetic average
#             homogenization term is in terms of mobility, so use log average (mobility is always positive)
#     '''
#     v = arithmeticMean([D[...,0] for D in Ds])
#     m = logMean([D[...,1] for D in Ds])
#     return v * m

# def _idealMean(Ds):
#     '''
#     Average homogenization flux
#     Ds should be in the shape of (m x N x y x 4)
#         m - number of items to average over
#         N - number of nodes
#         y - number of responses
#         4 - (transformation to volume fixed frame, temperature, homogenization term (\Gamma), inverse u term)
#             transformation is in composition, so use arithmetic average
#             temperature (+eps*R) is linear, so use arithmetic average
#             homogenization term is in terms of mobility, so use log average
#             inverse u term is inverse composition, so use harmonic mean (composition should always be > 0 based off constraints in diffusion model)
#     '''
#     v = arithmeticMean([D[...,0] for D in Ds])
#     t = arithmeticMean([D[...,1] for D in Ds])
#     m = logMean([D[...,2] for D in Ds])
#     invU = harmonicMean([D[...,3] for D in Ds])
#     #print(v.shape, t.shape, m.shape, invU.shape)
#     return v * t * m * invU

# def _atNodeProduct(Ds):
#     return np.prod(Ds, axis=-1)

# class HomogenizationModel(DiffusionModel):
#     def __init__(self, mesh, elements, phases,
#                  thermodynamics = None,
#                  temperature = None,
#                  homogenizationParameters = None,
#                  constraints = None,
#                  record = False):
#         super().__init__(mesh=mesh, elements=elements, phases=phases,
#                          thermodynamics=thermodynamics,
#                          temperature=temperature,
#                          constraints=constraints,
#                          record=record)
#         self.homogenizationParameters = homogenizationParameters if homogenizationParameters is not None else HomogenizationParameters()

#     def _getPairs(self, t, xCurr):
#         '''
#         Compute diffusivity-response pairs

#         J_k = -\Gamma_k d\mu_k/dz - \eps*RT*\Gamma_k/u_k du_k/dz
#         J^n_k = -sum(\delta_jk - u_k) J_j
#         dx_k/dt = -dJ^n_k/dz

#         For x_k, a pair would comprise of:
#             (\delta_jk - u_k) \Gamma_j, \mu_k
#             (\delta_jk - u_k) \eps*RT*\Gamma_k/u_k, u_k
#         '''
#         # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
#         # TODO: this feels inefficient to convert u->x for mobility, then back to u for flux calculation
#         u = expand_u_frac(xCurr[0], self.allElements, interstitials)
#         x = u_to_x_frac(u, self.allElements, interstitials)[:,1:]
#         x = self.mesh.unflattenResponse(x)
#         yD, zD = self.mesh.getDiffusivityCoordinates(x)
#         yR, zR = self.mesh.getResponseCoordinates(x)

#         # temp is (N,e)
#         tempD = self.temperatureParameters(zD, t)
#         tempR = self.temperatureParameters(zR, t)
#         # mob and mu are (N,e)
#         mobD, muD = computeHomogenizationFunction(self.therm, yD, tempD, self.homogenizationParameters, self.hashTable)
#         mobR, muR = computeHomogenizationFunction(self.therm, yR, tempR, self.homogenizationParameters, self.hashTable)

#         # Full composition
#         # x_full = (N,e+1), u_full = (N,e+1), u_term = (N,e+1,e+1)
#         x_fullD = np.concatenate((1-np.sum(yD, axis=1)[:,np.newaxis], yD), axis=1)
#         u_fullD = x_to_u_frac(x_fullD, self.allElements, interstitials)
#         # u_term is the (\delta_jk - u_k), which converts from a lattice fixed frame to a volume fixed frame
#         u_termD = (np.eye(len(self.allElements))[np.newaxis,:,:] - u_fullD[:,:,np.newaxis])
#         # When converting to a volume fixed frame, only the substitutional (or volume contributing) elements
#         # contribute to the corrected flux. To account for interstitials, we replace the column
#         # corresponding to the interstitial (which at this point is [u_A, u_B, 1-u_I, u_D] where I is interstital)
#         # to be [0, 0, 1, 0]. So the column is 0 except for the interstitial row
#         for i,e in enumerate(self.allElements):
#             if e in interstitials:
#                 u_termD[:,:,i] = 0
#                 u_termD[:,i,i] = 1

#         x_fullR = np.concatenate((1-np.sum(yR, axis=1)[:,np.newaxis], yR), axis=1)
#         u_fullR = x_to_u_frac(x_fullR, self.allElements, interstitials)

#         nEle = len(self.allElements)
#         # mobility matrix is repeated among rows
#         mobD_matrix = np.repeat(mobD[:,np.newaxis,:], nEle, axis=1)
#         # We'll group eps*R with T here
#         T_matrix = np.tile(tempD[:,np.newaxis,np.newaxis], (1, nEle, nEle)) * self.homogenizationParameters.eps * GAS_CONSTANT
#         # inverse composition and response is repeated among rows
#         invU_matrix = np.repeat(1/u_fullD[:,np.newaxis,:], nEle, axis=1)
#         mu_matrix = np.repeat(muR[:,np.newaxis,:], nEle, axis=1)
#         uR_matrix = np.repeat(u_fullR[:,np.newaxis,:], nEle, axis=1)

#         pairs = []
#         # Since volume fixed frame leads to 1 dependent component (which we take as the first)
#         # we don't need to take the 1st row of mob_term and ideal_term
#         for i in range(len(self.allElements)):
#             # homogenization contribution - (vol transform * \Gamma) * dmu/dz
#             pairs.append(DiffusionPair(
#                 diffusivity=np.transpose(np.array([u_termD[:,1:,i], mobD_matrix[:,1:,i]]), axes=(1,2,0)),
#                 response=mu_matrix[:,1:,i],
#                 averageFunction=_homogenizationMean,
#                 atNodeFunction=_atNodeProduct
#             ))
#             # ideal contribution - (vol transform * eps*R*T * \Gamma / u) * du/dz
#             pairs.append(DiffusionPair(
#                 diffusivity=np.transpose(np.array([u_termD[:,1:,i], T_matrix[:,1:,i], mobD_matrix[:,1:,i], invU_matrix[:,1:,i]]), axes=(1,2,0)),
#                 response=uR_matrix[:,1:,i],
#                 averageFunction=_idealMean,
#                 atNodeFunction=_atNodeProduct
#             ))
#         return pairs

#     def getDt(self, dXdt):
#         '''
#         Time increment
#         This is done by finding the time interval such that the composition
#             change caused by the fluxes will be lower than self.maxCompositionChange
#         '''
#         return self.constraints.maxCompositionChange / np.amax(np.abs(dXdt[0][dXdt[0]!=0]))
