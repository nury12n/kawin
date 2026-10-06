from typing import Union
from abc import ABC, abstractmethod
import numpy as np
from kawin.diffusion.mesh.mesh_base import MeshBase, BoundaryCondition, DiffusionPair, ProfileFunction
from kawin.diffusion.mesh.mesh_base import no_change_at_node, arithmetic_mean, _get_response_variables, _get_response_index, _format_spatial, diffusive_flux

class PeriodicBoundary1D(BoundaryCondition):
    '''
    This doesn't do anything, but rather is used in the Mesh to compute the fluxes accordingly
    Unfortunately, we can't just wrap the fluxes around and instead have to wrap the response variable
    around. Which means that we would have to have knowledge on how to compute the flux (which is done in the mesh)
    '''
    def set_initial_response(self, mesh, y):
        pass

    def adjust_fluxes(self, mesh, fluxes):
        pass

    def adjust_dxdt(self, mesh, dxdt):
        pass

class MixedBoundary1D(BoundaryCondition):
    '''
    Currently, left and right boundary conditions both are defined here
    We'll need a way to define boundary conditions at any edge, which will
    especially be the case if we add a 2D mesh

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length
    '''
    NEUMANN = 0
    DIRICHLET = 1

    def __init__(self, responses: Union[int, list[str]]):
        self.num_responses, self.responses = _get_response_variables(responses)
        self.lbc_type = MixedBoundary1D.NEUMANN * np.ones(self.num_responses)
        self.lbc_value = np.zeros(self.num_responses)
        self.rbc_type = MixedBoundary1D.NEUMANN * np.ones(self.num_responses)
        self.rbc_value = np.zeros(self.num_responses)

    def _set_bc(self, response_var, bc_type, value, bc_type_array, bc_value_array):
        '''Sets boundary condition to given side (left or right)'''
        index = _get_response_index(response_var, self.responses)

        is_set = True
        if bc_type == MixedBoundary1D.NEUMANN:
            bc_type_array[index] = MixedBoundary1D.NEUMANN
        elif bc_type == MixedBoundary1D.DIRICHLET:
            bc_type_array[index] = MixedBoundary1D.DIRICHLET
        else:
            is_set = False
        if not is_set:
            raise ValueError(f"bcType must be one of the following [MixedBoundary1D.NEUMANN, MixedBoundary1D.DIRICHLET]")
        bc_value_array[index] = value

    def set_lbc(self, response_var, bc_type, value):
        '''
        Left boundary condition

        Parameters
        ----------
        responseVar: int | str
            Response variable name or index
        bcType: str
            For flux/Neumann condition, it could be [flux, neumman, MixedBoundary1D.NEUMANN]
            For composition/dirichlet condition, it could be [composition, dirichlet, MixedBoundary1D.DIRICHLET]
        value: float
            Value of boundary condition
        '''
        self._set_bc(response_var, bc_type, value, self.lbc_type, self.lbc_value)

    def set_rbc(self, response_var, bc_type, value):
        '''
        Left boundary condition

        Parameters
        ----------
        responseVar: int | str
            Response variable name or index
        bcType: str
            For flux/Neumann condition, it could be [flux, neumman, MixedBoundary1D.NEUMANN]
            For composition/dirichlet condition, it could be [composition, dirichlet, MixedBoundary1D.DIRICHLET]
        value: float
            Value of boundary condition
        '''
        self._set_bc(response_var, bc_type, value, self.rbc_type, self.rbc_value)

    def set_initial_response(self, mesh, y):
        '''
        Sets dirichlet boundary conditions onto y
        '''
        for d in range(self.num_responses):
            if self.lbc_type[d] == MixedBoundary1D.DIRICHLET:
                y[0,d] = self.lbc_value[d]
            if self.rbc_type[d] == MixedBoundary1D.DIRICHLET:
                y[-1,d] = self.rbc_value[d]

    def adjust_fluxes(self, mesh, fluxes):
        '''
        Adjust flux values using boundary conditions

        For Neumann boundary conditions, this overrides the flux with the BC value
        '''
        for d in range(self.num_responses):
            if self.lbc_type[d] == MixedBoundary1D.NEUMANN:
                fluxes[0,d] = self.lbc_value[d]
            if self.rbc_type[d] == MixedBoundary1D.NEUMANN:
                fluxes[-1,d] = self.rbc_value[d]

    def adjust_dxdt(self, mesh, dxdt):
        '''
        Adjust dx/dt using boundary conditions
        For Dirichlet boundary conditions, this sets dx/dt to 0
        '''
        for d in range(self.num_responses):
            if self.lbc_type[d] == MixedBoundary1D.DIRICHLET:
                dxdt[0,d] = 0
            if self.rbc_type[d] == MixedBoundary1D.DIRICHLET:
                dxdt[-1,d] = 0

class StepProfile1D(ProfileFunction):
    '''
    Step function at z
    '''
    def __init__(self, z, left_value, right_value):
        self.lv = np.atleast_1d(left_value)
        self.rv = np.atleast_1d(right_value)
        self.z = z

    def __call__(self, z):
        z = _format_spatial(z)[:,0]
        y = self.lv[np.newaxis,:]*np.ones((z.shape[0], self.lv.shape[0]))
        y[z >= self.z,:] = self.rv
        return np.squeeze(y)

class LinearProfile1D(ProfileFunction):
    '''
    Linear function from (leftZ, leftValue) to (rightZ, rightValue)
    lowerLeftValue and upperRightValue corresponds to the value outside the bounds of
    leftZ and rightZ. By default they will use leftValue and rightValue
    '''
    def __init__(self, left_z, left_value, right_z, right_value, lower_left_value = None, upper_right_value = None):
        self.lv = np.atleast_1d(left_value)
        self.lz = left_z
        self.rv = np.atleast_1d(right_value)
        self.rz = right_z
        self.llv = self.lv if lower_left_value is None else np.atleast_1d(lower_left_value)
        self.urv = self.rv if upper_right_value is None else np.atleast_1d(upper_right_value)

    def __call__(self, z):
        z = _format_spatial(z)[:,0]
        y = np.zeros((z.shape[0], self.lv.shape[0]))
        y[z <= self.lz,:] = self.llv
        midIndices = (self.lz < z) & (z <= self.rz)
        for i in range(self.lv.shape[0]):
            y[midIndices,i] = np.interp(z[midIndices], [self.lz, self.rz], [self.lv[i], self.rv[i]])
        y[self.rz < z,:] = self.urv
        return np.squeeze(y)

class ExperimentalProfile1D(ProfileFunction):
    '''
    Given experimental values at z, this will interpolate in between
    left and right are the values used if the mesh goes beyond z. By default, they
    will use the left-most and right-most values
    '''
    def __init__(self, z, values, left=None, right=None):
        values = np.atleast_2d(values)
        if values.shape[0] == 1:
            values = values.T
        z = np.array(z)
        sortIndices = np.argsort(z)
        self.values = values[sortIndices]
        self.z = z[sortIndices]
        self.left = left if left is None else np.atleast_1d(left)
        self.right = right if right is None else np.atleast_1d(right)

    def __call__(self, z):
        z = _format_spatial(z)[:,0]
        y = np.zeros((z.shape[0], self.values.shape[1]))
        for i in range(self.values.shape[1]):
            y[:,i] = np.interp(z, self.z, self.values[:,i], left=self.left, right=self.right)
        return np.squeeze(y)

class MidPointCalculator(ABC):
    '''
    In the finite volume method, we need to compute diffusivity at the
    edges of the nodes, but the response variables are always at the cell centers

    This class is intended to provide functionality of how to we approach computing
    diffusivities at the edges
    '''
    @staticmethod
    @abstractmethod
    def get_diffusivity_coordinates(y, z, z_edge, is_periodic = False, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def get_D_mid(D, is_periodic=False, at_node_func=no_change_at_node, avg_func=arithmetic_mean, *args, **kwargs):
        pass

class FVM1DMidpoint(MidPointCalculator):
    @staticmethod
    def get_diffusivity_coordinates(y, z, z_edge, is_periodic = False, *args, **kwargs):
        '''
        This returns the mid point of y and z (which is zEdge)
        Note: first element in ymid is second element of zEdge
        For periodic conditions, we also include y value between the first and last node
        '''
        ymid = (y[:-1] + y[1:]) / 2
        if is_periodic:
            yend = (y[0] + y[-1]) / 2
            return np.concatenate((ymid, [yend])), z_edge[1:]
        else:
            return ymid, z_edge[1:-1]

    @staticmethod
    def get_D_mid(D, is_periodic=False, at_node_func=no_change_at_node, avg_func=arithmetic_mean, *args, **kwargs):
        '''For periodic conditions, the last D will correspond to between the first and last node'''
        D = at_node_func(D)
        if is_periodic:
            return D[:-1], D[-1]
        else:
            return D, None

class FVM1DEdge(MidPointCalculator):
    @staticmethod
    def get_diffusivity_coordinates(y, z, z_edge, is_periodic = False, *args, **kwargs):
        '''Return y and z since we compute diffusivity at these coordinates'''
        return y, z

    @staticmethod
    def get_D_mid(D, is_periodic=False, at_node_func=no_change_at_node, avg_func=arithmetic_mean, *args, **kwargs):
        '''For periodic conditions, we average the first and last D'''
        Davg = avg_func([D[:-1], D[1:]])
        Dend = avg_func([D[0], D[-1]]) if is_periodic else None
        return Davg, Dend

class FiniteVolume1D(MeshBase):
    '''
    1D finite volume mesh

    Cell will be a volume with thickness dz
        y - value at node center
        z - spatial coordinate at node center
        zEdge - spatial coordinate at node edge

    Parameters
    ----------
    zlim: list[float]
        Left and right boundary position of mesh
    N: int
        Number of cells
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length
    boundaryConditions: BoundaryCondition1D
        Boundary conditions on mesh
    computeMidpoint: bool
        Whether to compute diffusivity at average midpoint composition (True) or
        average diffusivities computed on nearby cell centers (False)
    '''
    def __init__(self, responses, zlim, N, midpoint_calculator = FVM1DEdge()):
        super().__init__(responses, N, 1)
        self.zlim = zlim
        self.edges = np.linspace(self.zlim[0], self.zlim[1], self.N+1)
        self.z = 0.5*(self.edges[1:] + self.edges[:-1])
        self.dz = self.z[1] - self.z[0]
        self.midpoint_calculator = midpoint_calculator

    @abstractmethod
    def flux_to_dxdt(self, fluxes):
        '''Given fluxes, compute dx/dt. This depends on coordinate system'''
        pass

    def build_response_profile(self, profile_builder, bcs = None):
        if bcs is None:
            bcs = MixedBoundary1D(self.num_responses)
        self.bcs = bcs
        return super().build_response_profile(profile_builder, self.bcs)

    def compute_fluxes(self, pairs: list[DiffusionPair]):
        '''Compute fluxes from (diffusivity, response) pairs on a 1D FVM mesh'''
        fluxes = np.zeros((self.N+1, self.num_responses))
        is_periodic = isinstance(self.bcs, PeriodicBoundary1D)
        for p in pairs:
            # compute fluxes for all but first and last edge (these are handled by boundary condition)
            D, r = p.diffusivity, p.response
            avg_func = arithmetic_mean if p.averaging_function is None else p.averaging_function
            at_node_func = no_change_at_node if p.at_node_function is None else p.at_node_function
            D_mid, D_end = self.midpoint_calculator.get_D_mid(D, is_periodic=is_periodic, avg_func=avg_func, at_node_func=at_node_func)
            fluxes[1:-1] += diffusive_flux(D_mid, r[1:], r[:-1], self.dz)

            # if periodic, then wrap first cell to the end, and first/last flux will be the same
            if is_periodic:
                end_flux = diffusive_flux(D_end, r[0], r[-1], self.dz)
                fluxes[0] += end_flux
                fluxes[-1] += end_flux

        return fluxes

    def compute_dxdt(self, pairs: list[DiffusionPair]):
        '''Computes fluxes, correct fluxes for boundary conditions, compute dx/dt and correct dx/dt for boundary conditions'''
        fluxes = self.compute_fluxes(pairs)
        self.bcs.adjust_fluxes(self, fluxes)
        dxdt = self.flux_to_dxdt(fluxes)
        self.bcs.adjust_dxdt(self, dxdt)
        return dxdt

    def get_diffusivity_coordinates(self, y):
        '''Return y and z to compute diffusivities at'''
        return self.midpoint_calculator.get_diffusivity_coordinates(y, self.z, self.edges, is_periodic=isinstance(self.bcs, PeriodicBoundary1D))

class Cartesian1D(FiniteVolume1D):
    def flux_to_dxdt(self, fluxes):
        '''For cartesian: dx/dt = -dJ/dz'''
        return -(fluxes[1:] - fluxes[:-1]) / self.dz

class Cylindrical1D(FiniteVolume1D):
    def __init__(self, responses, rlim, N, midpoint_calculator = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')
        super().__init__(responses, rlim, N, midpoint_calculator)

    def build_response_profile(self, profile_builder, bcs=None):
        if isinstance(bcs, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on cylindrical coordinates')
        return super().build_response_profile(profile_builder, bcs)

    def flux_to_dxdt(self, fluxes):
        '''For cylindrical: dx/dt = -1/r dJ/dz'''
        fr = fluxes*self.zEdge
        return -(fr[1:] - fr[:-1]) / self.z / self.dz

class Spherical1D(FiniteVolume1D):
    def __init__(self, responses, rlim, N, midpoint_calculator = False):
        if rlim[0] < 0 or rlim[1] < 0:
            raise ValueError('Radial limits must be positive')

        super().__init__(responses, rlim, N, midpoint_calculator)

    def build_response_profile(self, profile_builder, bcs=None):
        if isinstance(bcs, PeriodicBoundary1D):
            raise ValueError('Periodic boundary conditions are not-defined on spherical coordinates')
        return super().build_response_profile(profile_builder, bcs)

    def flux_to_dxdt(self, fluxes):
        '''For spherical: dx/dt = -1/r^2 dJ/dz'''
        fr = fluxes*self.zEdge**2
        return -(fr[1:] - fr[:-1]) / self.z**2 / self.dz

class Cartesian2D(MeshBase):
    '''
    2D finite volume mesh

    y - value at node center (Nx, Ny, dims)
    z - spatial coordinate at node center (Nx, Ny, 2)
    zCorner - spatial coordinate at node corners (Nx+1, Ny+1, 2)
    dz - thickness of node in both dimensions

    TODO: boundary conditions not supported yet, so only no flux conditions

    Parameters
    ----------
    responses: int | list[str]
        If int, then this is the number of responses and the names will be R{i}
        If list[str], then theses are the response names and the number of responses will be the list length
    zx: list[float]
        Left and right boundary position of mesh
    Nx: int
        Number of cells along x
    zy: list[float]
        Top and bottom boundary position of mesh
    Ny: int
        Number of cells along y
    '''
    def __init__(self, responses, zx, Nx, zy, Ny):
        super().__init__(responses, Nx*Ny, 2)
        self.Nx, self.Ny = Nx, Ny
        self.zx, self.zy = zx, zy
        self.edges_x = np.linspace(self.zx[0], self.zx[1], self.Nx+1)
        self.edges_y = np.linspace(self.zy[0], self.zy[1], self.Ny+1)
        self.corners = np.transpose(np.meshgrid(self.edges_x, self.edges_y), axes=(2,1,0))
        self.z_mesh = 0.25*(self.corners[:-1,:-1] + self.corners[1:,:-1] + self.corners[:-1,1:] + self.corners[1:,1:])
        self.z = np.reshape(self.z_mesh, (self.Nx*self.Ny, self.dims))
        self.dzx = self.edges_x[1]-self.edges_x[0]
        self.dzy = self.edges_y[1]-self.edges_y[0]
        self.dz = np.amin((self.dzx, self.dzy))

    def compute_fluxes(self, pairs: list[DiffusionPair]):
        '''
        Compute fluxes from (diffusivity, response) pairs on a 2D FVM mesh
        '''
        flux_x = np.zeros((self.Nx+1, self.Ny, self.num_responses))
        flux_y = np.zeros((self.Nx, self.Ny+1, self.num_responses))
        for p in pairs:
            D = np.reshape(p.diffusivity, (self.Nx, self.Ny, self.num_responses, *p.diffusivity.shape[2:]))
            r = np.reshape(p.response, (self.Nx, self.Ny, self.num_responses, *p.response.shape[2:]))
            avg_func = arithmetic_mean if p.averaging_function is None else p.averaging_function

            # flux along x (neighboring cells to compute D and flux are along x direction, 1st index)
            Dx_mid = avg_func([D[:-1,:], D[1:,:]])
            #flux_x[1:-1,:] += self._diffusiveFlux(Dx_mid, r[1:,:], r[:-1,:], self.dzs[0])
            flux_x[1:-1,:] += diffusive_flux(Dx_mid, r[1:,:], r[:-1,:], self.dzx)

            # flux along y (neighboring cells to compute D and flux are along y direction, 2nd index)
            Dy_mid = avg_func([D[:,:-1], D[:,1:]])
            #flux_y[:,1:-1] += self._diffusiveFlux(Dy_mid, r[:,1:], r[:,:-1], self.dzs[1])
            flux_y[:,1:-1] += diffusive_flux(Dy_mid, r[:,1:], r[:,:-1], self.dzy)

        return flux_x, flux_y

    def compute_dxdt(self, pairs: list[DiffusionPair]):
        '''
        dx/dt = -dJx/dz + -dJy/dz
        '''
        flux_x, flux_y = self.compute_fluxes(pairs)
        dxdt = -(flux_x[1:,:] - flux_x[:-1,:]) / self.dzx + -(flux_y[:,1:] - flux_y[:,:-1]) / self.dzy
        return np.reshape(dxdt, (self.N, self.num_responses))