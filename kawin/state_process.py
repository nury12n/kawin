from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol
import copy

import numpy as np

class ModelState:
    def __init__(self):
        self.time: float = 0

class ModelProcess(ABC):
    @abstractmethod
    def get_next_time(self, state: ModelState) -> float:
        ...

    @abstractmethod
    def progress_state(self, state: ModelState, time: float):
        ...

    @abstractmethod
    def finalize(self, state: ModelState):
        ...

class Iterator(Protocol):
    def __call__(self, state: ModelState, compute_dxdt, apply_dxdt, dxdt_0, dt):
        ...

    @staticmethod
    def explicit_euler(state: ModelState, compute_dxdt, apply_dxdt, dxdt_0, dt):
        apply_dxdt(state, dxdt_0, dt)

    @staticmethod
    def rk4(state: ModelState, compute_dxdt, apply_dxdt, dxdt_0, dt):
        k1 = dxdt_0
        dxdtsum = k1
        x1 = copy.deepcopy(state)
        apply_dxdt(x1, k1, dt/2)

        k2 = compute_dxdt(x1)
        dxdtsum += 2*k2
        x2 = copy.deepcopy(state)
        apply_dxdt(x2, k2, dt/2)

        k3 = compute_dxdt(x2)
        dxdtsum += 2*k3
        x3 = copy.deepcopy(state)
        apply_dxdt(x3, k3, dt)

        k4 = compute_dxdt(x3)
        dxdtsum += k4

        apply_dxdt(state, dxdtsum/6, dt)

class DifferentialEquationProcess(ModelProcess):
    def __init__(self, iterator: Iterator = Iterator.explicit_euler):
        self.dxdt = None
        self.dt = None
        self.iterator = iterator

    @abstractmethod
    def compute_dxdt(self, state: ModelState) -> any:
        ...

    @abstractmethod
    def compute_max_dt(self, state: ModelState, dxdt) -> float:
        ...

    @abstractmethod
    def apply_dxdt(self, state: ModelState, dxdt, dt: float):
        ...

    def get_next_time(self, state: ModelState):
        self.dxdt = self.compute_dxdt(state)
        self.dt = self.compute_max_dt(state, self.dxdt)
        return state.time + self.dt

    def progress_state(self, state: ModelState, time: float):
        self.iterator(state, self.compute_dxdt, self.apply_dxdt, self.dxdt, time-state.time)
        #state.time = time

class EventGenerator(ABC):
    @abstractmethod
    def query(self, state: ModelState) -> float:
        ...

    @abstractmethod
    def execute(self, state: ModelState):
        ...

    @abstractmethod
    def reset(self, state: ModelState):
        ...

class DiscreteEventProcess(ModelProcess):
    def __init__(self, event_generators: list[EventGenerator], update_all_events: bool=False):
        self.event_generators = event_generators
        self.update_all_events = update_all_events
        self.next_event_id = None
        self.next_time = None

    def query_next_time(self, state: ModelState):
        times = [e.query(state) for e in self.event_generators]
        return np.amin(times), np.argmin(times)

    def execute_event(self, state: ModelState, event_id: int):
        self.event_generators[event_id].execute(state)

    def get_next_time(self, state: ModelState) -> float:
        self.next_time, self.next_event_id = self.query_next_time(state)
        return self.next_time

    def progress_state(self, state: ModelState, time: float):
        self.execute_event(state, self.next_event_id)
        #state.time = time

    def finalize(self, state: ModelState):
        if self.update_all_events:
            for e in self.event_generators:
                e.reset(state)
        else:
            self.event_generators[self.next_event_id].reset(state)

class MixedProcess(ModelProcess):
    def __init__(self, processes: list[ModelProcess]):
        self.processes: list[float] = processes
        self.p_next: int = -1

    def get_next_time(self, state: ModelState) -> float:
        times = [p.get_next_time(state) for p in self.processes]
        t_next = np.amin(times)
        self.p_next = self.processes[np.argmin(times)]
        return t_next

    def progress_state(self, state: ModelState, time: float):
        self.p_next.progress_state(state, time)

    def finalize(self, state: ModelState):
        for p in self.processes:
            p.finalize(state)

class Solver:
    @staticmethod
    def iterate(state: ModelState, process: ModelProcess, t_final: float=None):
        t_next = process.get_next_time(state)
        if t_final is not None:
            t_next = np.amin([t_next, t_final])
        process.progress_state(state, t_next)
        state.time = t_next
        process.finalize(state)

    @staticmethod
    def solve(state: ModelState, process: ModelProcess, delta_time: float):
        t_final = state.time + delta_time
        i = 0
        while state.time < t_final:
            if i % 100 == 0:
                print(i, state.time)
            Solver.iterate(state, process, t_final)
            i += 1