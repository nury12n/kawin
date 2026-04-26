import numpy as np
from kawin.thermo.Mobility import u_to_x_frac, x_to_u_frac, expand_u_frac, expand_x_frac, interstitials
from kawin.diffusion.Diffusion import DiffusionModel
from kawin.diffusion.mesh.MeshBase import DiffusionPair, arithmeticMean
from kawin.diffusion.mesh import Cartesian1D, MixedBoundary1D, PeriodicBoundary1D

def build_matrix(mesh: Cartesian1D, x, D):
    N = mesh.N
    num_elements = mesh.numResponses
    dz = mesh.dz
    A = np.zeros((N, N, num_elements, num_elements))
    Dmid = (D[1:] + D[:-1])/2
    Dmid = np.transpose(Dmid, (0, 2, 1))
    for i in range(N):
        if isinstance(mesh.boundaryConditions, PeriodicBoundary1D):
            if i == 0:
                A[i,-1] += Dmid[-1] / dz**2
                A[i,i] -= Dmid[-1] / dz**2
            if i == N-1:
                A[i,0] += Dmid[0] / dz**2
                A[i,i] -= Dmid[0] / dz**2
        if i > 0:
            A[i,i-1] += Dmid[i-1] / dz**2
            A[i,i] -= Dmid[i-1] / dz**2
        if i < N-1:
            A[i,i] -= Dmid[i] / dz**2
            A[i,i+1] += Dmid[i] / dz**2

    # concatenate into large matrix
    Alarge = np.zeros((N*num_elements, N*num_elements))
    for i in range(num_elements):
        for j in range(num_elements):
            Alarge[N*i:N*(i+1),N*j:N*(j+1)] = A[:,:,i,j]
    return Alarge

def flatten_x(x, N, num_elements):
    return np.reshape(x.T, (N*num_elements))

def unflatten_x(x, N, num_elements):
    return np.reshape(x, (num_elements, N)).T

class SemianalyticalModel(DiffusionModel):
    def __init__(self, mesh, elements, phases,
                 thermodynamics = None,
                 temperature = None,
                 constraints = None,
                 dt_eps = 1e-9,
                 dx_max = 1e-3,
                 dt_max = 1e8,
                 record = False):
        super().__init__(mesh=mesh, elements=elements, phases=phases,
                         thermodynamics=thermodynamics,
                         temperature=temperature,
                         constraints=constraints,
                         record=record)
        self.dt_eps = dt_eps
        self.dx_max = dx_max
        self.dt_max = dt_max
        # very large value that's below overflow
        self.max_exp_val = 1e300

        assert isinstance(mesh, Cartesian1D), "Only Cartesian1D is supported"
        if isinstance(mesh.boundaryConditions, MixedBoundary1D):
            assert np.all(np.array(mesh.boundaryConditions.LBCtype) == MixedBoundary1D.NEUMANN) and np.all(np.array(mesh.boundaryConditions.LBCvalue)) == 0, "Only no flux or periodic BC are supported"
            assert np.all(np.array(mesh.boundaryConditions.RBCtype) == MixedBoundary1D.NEUMANN) and np.all(np.array(mesh.boundaryConditions.RBCvalue)) == 0, "Only no flux or periodic BC are supported"

    def getdXdt(self, t, xCurr):
        '''
        Computes dXdt from mesh and diffusivity-respones pairs
        '''
        # x is shape (N,e), so convert to mesh shape to obtain diffusion/response coordinates
        u = expand_u_frac(xCurr[0], self.allElements, interstitials)
        x = u_to_x_frac(u, self.allElements, interstitials)[:,1:]
        x = self.mesh.unflattenResponse(x)
        yD, zD = self.mesh.getDiffusivityCoordinates(x)

        # get diffusivities
        T = self.temperatureParameters(zD, t)
        N = self.mesh.N
        numElements = self.mesh.numResponses
        yD, zD = self.mesh.getDiffusivityCoordinates(x)
        D = self.therm.getInterdiffusivity(yD, T, phase=self.phases[0])
        if numElements == 1:
            D = np.reshape(D, (N,numElements,numElements))

        uflat = flatten_x(x, N, numElements)
        # store analytical solution in case we need to adjust time step
        self.A = build_matrix(self.mesh, x, D)
        self.lam, self.ev = np.linalg.eig(self.A)
        self.c = np.matmul(np.linalg.inv(self.ev), uflat)
        #print(np.amax(self.lam), np.amin(self.lam), np.amax(self.ev), np.amin(self.ev))
        if np.any(self.lam>0):
            max_possible_time = np.real(np.amin(np.log(self.max_exp_val)/self.lam[self.lam>0]))
        else:
            max_possible_time = self.dt_max

        # bisection search for dt to where dx < max
        low = 0
        high = np.amin([max_possible_time, self.dt_max])
        dt_mid = (low+high)/2
        while high-low > self.dt_eps:
            upred = np.matmul(self.ev, self.c*np.exp(self.lam*dt_mid))
            #if np.amax(np.abs(upred-uflat)/upred) > self.dx_max:
            if np.amax(np.abs(upred-uflat)) > self.dx_max:
                high = dt_mid
            else:
                low = dt_mid
            dt_mid = (low+high)/2

        # compute new u-fraction (since du/dt = D*d2x/dz2)
        # for all substitutionals, this doesn't change anything, but I will
        # need to check that this is valid when interstitials are involved
        unew = np.real(unflatten_x(np.matmul(self.ev, self.c*np.exp(self.lam*dt_mid)), N, numElements))
        self._currdt = dt_mid
        # kawin assumes models return dx/dt, but the analytical solution
        # gives x directly, so convert back to a dx/dt value
        return [(unew-xCurr[0]).squeeze()/dt_mid]

    def getDt(self, dXdt):
        return self._currdt

    def correctdXdt(self, dt, x, dXdt):
        if dt == self._currdt:
            return dXdt
        N = self.mesh.N
        numElements = self.mesh.numResponses
        # if dt is modified, compute the new u-fraction and corresponding dx/dt
        unew = np.real(unflatten_x(np.matmul(self.ev, self.c*np.exp(self.lam*dt)), N, numElements))
        return [(unew-x[0])/dt]