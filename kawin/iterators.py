from typing import Protocol
import copy

class ComputeDerivativeFunction(Protocol):
    def __call__(self, state) -> any:
        """
        Structure of dxdt can be anything since ApplyDerivativeFunction
        handles the relationship between state and dxdt
        """
        ...

class ApplyDerivativeFunction(Protocol):
    def __call__(self, state, dxdt: any, dt: float):
        """Directly modifies state"""
        ...

class Iterator(Protocol):
    """
    Protocol for iterating a differential equation
    dxdt_0 is the derivative computed from the Solver at time t
    It is the same output as compute_dxdt but is added to avoid redundant calculation
    """
    def __call__(
            self, state,
            compute_dxdt: ComputeDerivativeFunction, apply_dxdt: ApplyDerivativeFunction,
            dxdt_0: any, dt: float):
        ...

    @staticmethod
    def explicit_euler(
            state,
            compute_dxdt: ComputeDerivativeFunction, apply_dxdt: ApplyDerivativeFunction,
            dxdt_0: any, dt: float):
        apply_dxdt(state, dxdt_0, dt)

    @staticmethod
    def rk4(
            state,
            compute_dxdt: ComputeDerivativeFunction, apply_dxdt: ApplyDerivativeFunction,
            dxdt_0: any, dt: float):
        # TODO: I don't like invoking deepcopy here, should we have a method in
        # state to handle copies?
        k1 = dxdt_0
        x1 = copy.deepcopy(state)
        apply_dxdt(x1, k1, dt/2)
        x1.time += dt/2

        k2 = compute_dxdt(x1)
        x2 = copy.deepcopy(state)
        apply_dxdt(x2, k2, dt/2)
        x2.time += dt/2

        k3 = compute_dxdt(x2)
        x3 = copy.deepcopy(state)
        apply_dxdt(x3, k3, dt)
        x3.time += dt

        k4 = compute_dxdt(x3)

        # RK4 is generally shown as f_t+1 = f_t + (k1+2*k2+2*k3+k4)*dt
        # We split it up here instead to avoid assumptions on whether dxdt can be added together
        #apply_dxdt(state, (k1+2*k2+2*k3+k4)/6, dt)
        apply_dxdt(state, k1, dt/6)
        apply_dxdt(state, k2, dt/3)
        apply_dxdt(state, k3, dt/3)
        apply_dxdt(state, k4, dt/6)