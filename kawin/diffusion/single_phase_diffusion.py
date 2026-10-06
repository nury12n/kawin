import numpy as np

from pycalphad import variables as v

from kawin.thermo.thermodynamics import Thermodynamics
from kawin.thermo.mobility import expand_u_frac, u_to_x_frac, interstitials
from kawin.thermo.diffusivity import InterdiffusivityFunction, compute_interdiffusivity

from kawin.diffusion.diffusion import DiffusionModel, DiffusionState, LookupTable, DiffusionModelTemperature, IsothermalTemperature
from kawin.diffusion.mesh.mesh_base import MeshBase, DiffusionPair, arithmetic_mean

class SinglePhaseModel(DiffusionModel):
    def __init__(
            self,
            mesh: MeshBase,
            elements: list[str],
            phase: str,
            therm: Thermodynamics,
            temperature: DiffusionModelTemperature | float,
            diffusivity_function: InterdiffusivityFunction = None,
            *args, **kwargs):
        super().__init__(mesh, elements)

        # model parameters
        self.phase = phase
        self.therm = therm
        self.temperature = temperature
        if np.isscalar(self.temperature):
            self.temperature = IsothermalTemperature(temperature)
        self.conditions = kwargs.get("conditions", dict())
        self.diffusivity_function = diffusivity_function
        if self.diffusivity_function is None:
            self.diffusivity_function = LookupTable(compute_interdiffusivity, squeeze_function=InterdiffusivityFunction.squeeze)

        # utility
        self.cache = dict()
        self.sort_indices = np.argsort(np.argsort(self.elements))

        # constraints
        self.von_neumann_threshold = 0.4

    def compute_dxdt(self, state: DiffusionState) -> np.ndarray:
        u = expand_u_frac(state.u, self.all_elements, self.ref_element, interstitials)
        x = u_to_x_frac(u, self.all_elements, interstitials)[:,1:]

        y_res, z_res = self.mesh.get_response_coordinates(x)
        y_diff, z_diff = self.mesh.get_diffusivity_coordinates(x)

        T = self.temperature(state.time, z_diff)
        conditions = {v.X(e): y_diff[:,self.elements.index(e)] for e in self.elements}
        conditions[v.T] = T
        conditions[v.P] = self.conditions.get(v.P, 101325)
        conditions[v.GE] = self.conditions.get(v.GE, 0)
        d = self.diffusivity_function(self.therm, conditions, ref_element=self.ref_element, phase=self.phase, cache=self.cache)

        pairs = []
        # For binary systems, we only have 1 independent component
        if len(self.elements) == 1:
            pairs.append(DiffusionPair(
                diffusivity=d[:,np.newaxis],
                response=y_res,
                averaging_function=arithmetic_mean))
        else:
            # For 2+ independent component, we want to tile x_j from (1,N) -> (e,N) -> (N,e)
            for i in range(len(self.elements)):
                pairs.append(DiffusionPair(
                    diffusivity=d[:,self.sort_indices,self.sort_indices[i]],
                    response=np.tile([y_res[:,i]], (len(self.elements), 1)).T,
                    averaging_function=arithmetic_mean
                    ))

        self._curr_dt = self.von_neumann_threshold * self.mesh.dz**2 / np.amax(d) / self.mesh.dims
        return self.mesh.compute_dxdt(pairs)

    def compute_max_dt(self, state: DiffusionState, dxdt: np.ndarray) -> float:
        return self._curr_dt