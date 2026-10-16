from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections.abc import Sequence
from typing import Protocol
from functools import wraps
import copy

import numpy as np

from kawin.iterators import Iterator

@dataclass
class ModelState:
    """
    Holds variables pertaining to a state. This should contain all the information needed
    to tell a model how to progress the state (i.e. the model itself should be stateless)
        Technically, a model is allowed to hold a state to help with calculations, but
        should be logically stateless in that the model acts as a statelss object

    So model states do not actually need to subclass this as long as it defines
    the "time" attribute.

    to_data, from_data and interpolate is for the Recorder
      to_data must output to a dictionary of numpy arrays
      from_data must create a valid state from a dictionary of numpy arrays
      interpolate must create a valid state by interpolating between two states

      With this, the state itself does not need to abide to any type of structure
      as long as it is able to convert itself to the necessary data types for
      the Recorder.

      These functions do not need to be overridden in the following conditions
      - the Recorder is not used
      - all attributes defined in the state can be trivially converted to a numpy array
    """
    time: float

    @staticmethod
    def to_data(state):
        return {key: np.array(val) for key, val in state.__dict__.items()}

    @staticmethod
    def from_data(data):
        state = ModelState(time=0)
        for key, val in data.items():
            setattr(state, key, np.array(val))
        return state

    @staticmethod
    def interpolate(a, b, time):
        alpha = (time-a.time)/(b.time-a.time)
        state = ModelState(time=time)
        for key in a.__dict__.keys():
            setattr(state, key, (1-alpha)*getattr(a, key) + alpha*getattr(b, key))
        return state

class ModelProcess(ABC):
    def initialize_state(self, state: ModelState = None):
        if state is None:
            state = ModelState(time=0)
            self.setup_initial_state_data(state)
        return state

    @abstractmethod
    def setup_initial_state_data(self, state: ModelState) -> ModelState:
        pass

    @abstractmethod
    def get_next_time(self, state: ModelState) -> float:
        pass

    @abstractmethod
    def progress_state(self, state: ModelState, time: float):
        pass

    @abstractmethod
    def finalize(self, state: ModelState):
        pass

class DifferentialEquationProcess(ModelProcess):
    """
    TODO: does this implementation only make sense for explicit schemes?
    """
    def __init__(self, iterator: Iterator = Iterator.explicit_euler):
        self.dxdt = None
        self.dt = None
        self.iterator = iterator

    @abstractmethod
    def compute_dxdt(self, state: ModelState) -> any:
        """Follows ComputeDerivativeFunction protocol"""
        pass

    @abstractmethod
    def compute_max_dt(self, state: ModelState, dxdt: any) -> float:
        pass

    @abstractmethod
    def apply_dxdt(self, state: ModelState, dxdt: any, dt: float):
        """Follows ApplyDerivativeFunction protocol"""
        pass

    def get_next_time(self, state: ModelState):
        self.dxdt = self.compute_dxdt(state)
        self.dt = self.compute_max_dt(state, self.dxdt)
        return state.time + self.dt

    def progress_state(self, state: ModelState, time: float):
        self.iterator(state, self.compute_dxdt, self.apply_dxdt, self.dxdt, time-state.time)

class EventGenerator(ABC):
    @abstractmethod
    def query(self, state: ModelState) -> float:
        pass

    @abstractmethod
    def execute(self, state: ModelState):
        pass

    @abstractmethod
    def reset(self, state: ModelState):
        pass

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

    def finalize(self, state: ModelState):
        if self.update_all_events:
            for e in self.event_generators:
                e.reset(state)
        else:
            self.event_generators[self.next_event_id].reset(state)

# class MixedProcess(ModelProcess):
#     def __init__(self, processes: list[ModelProcess]):
#         self.processes: list[float] = processes
#         self.p_next: int = -1

#     def get_next_time(self, state: ModelState) -> float:
#         times = [p.get_next_time(state) for p in self.processes]
#         t_next = np.amin(times)
#         self.p_next = self.processes[np.argmin(times)]
#         return t_next

#     def progress_state(self, state: ModelState):
#         self.p_next.progress_state(state, time)

#     def finalize(self, state: ModelState):
#         for p in self.processes:
#             p.finalize(state)

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

class Recorder:
    def __init__(self, state: ModelState):
        self.n = 0
        self.state_obj = state
        self.data = {}

    def record(self, state: ModelState):
        self.n += 1
        data = self.state_obj.to_data(state)
        if len(self.data) == 0:
            for key, val in data.items():
                self.data[key] = np.array(val)[np.newaxis,...]
        else:
            for key, val in data.items():
                self.data[key] = np.concatenate([self.data[key], np.array(val)[np.newaxis,...]])

    def index(self, idx: int) -> ModelState:
        if idx >= self.n:
            print('idx > n, setting idx to n')
            idx = self.n
        if idx < 0:
            print('idx < 0, setting idx to 0')
            idx = 0
        sliced_data = {}
        for key, val in self.data.items():
            sliced_data[key] = val[idx,...]
        return self.state_obj.from_data(sliced_data)

    def get_state(self, time: float) -> ModelState:
        if time > self.data["time"][self.n-1]:
            tf = self.data["time"][self.n-1]
            print(f'Time > last recorded time, setting time to {tf}')
            return self.index(self.n-1)
        if time < self.data["time"][0]:
            t0 = self.data["time"][0]
            print(f'Time < initial time, setting time to {t0}')
            return self.index(0)

        idx = np.where(time>=self.data["time"])[0][-1]
        s1 = self.index(idx)
        s2 = self.index(idx+1)
        return self.state_obj.interpolate(s1, s2, time)