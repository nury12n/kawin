from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections.abc import Sequence
from typing import Protocol, Self
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

    to_data, from_data, create_dataset, append_to_dataset,
    get_dataset_slice and interpolate is for the Recorder
      For use in the Recorder, the state and data representation does nto need
      to follow any structure as long as these functions are properly defined

      These functions do not need to be overridden in the following conditions
      - the Recorder is not used
      - or all attributes defined in the state can be
        - trivially converted to a numpy array
        - attributes that are Sequences are of fixed length
        - attributes can be interpolated by (1-alpha)*a + alpha*b
    """
    time: float

    @staticmethod
    def to_data(state: Self) -> any:
        """
        Defines how a state should be represented as raw data
        Any format is okay for the data representation given that:
          - We can convert the data back to the state
          - We can define how a list of data will be represented
        """
        return {key: np.array(val) for key, val in state.__dict__.items()}

    @staticmethod
    def from_data(data: any) -> Self:
        """Converts data representation to a state"""
        state = ModelState(time=0)
        for key, val in data.items():
            setattr(state, key, np.array(val))
        return state

    @staticmethod
    def create_dataset() -> any:
        """Defines how a list of data should be defined"""
        return dict()

    @staticmethod
    def append_to_dataset(dataset: any, data: any):
        """Defines how a state's data representation should be appended to the dataset"""
        if len(dataset) == 0:
            for key, val in data.items():
                dataset[key] = np.array(val)[np.newaxis,...]
        else:
            for key, val in data.items():
                dataset[key] = np.concatenate([dataset[key], np.array(val)[np.newaxis,...]])

    @staticmethod
    def get_dataset_slice(dataset: any, idx: int) -> any:
        """
        Defines how a dataset is indexed to return a data representation
        This is not responsible for converting the data to a state, from_data should be called to do that instead
        """
        return {key: val[idx,...] for key, val in dataset.items()}

    @staticmethod
    def interpolate(a: Self, b: Self, time: float) -> Self:
        """Defines how to interpolation between two states"""
        alpha = (time-a.time)/(b.time-a.time)
        state = ModelState(time=time)
        for key in a.__dict__.keys():
            setattr(state, key, (1-alpha)*getattr(a, key) + alpha*getattr(b, key))
        return state

class ModelProcess(ABC):
    state_definition = ModelState

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

class Recorder:
    """
    Handles recording of states
    """
    def __init__(self, model: ModelProcess):
        self.state_def = model.state_definition
        # we store our copy of times since we don't know how dataset is implemented
        self.times = np.array([])
        self.dataset = self.state_def.create_dataset()

    @property
    def n(self):
        return len(self.times)

    def record(self, state: ModelState):
        self.state_def.append_to_dataset(self.dataset, self.state_def.to_data(state))
        self.times = np.concatenate([self.times, [state.time]])

    def __getitem__(self, idx: int) -> ModelState:
        # TODO: should we support slicing? Would that return an instance of Recorder
        #       or a slice of dataset. The single idx returns a State however, so I
        #       don't want to be returning different objects
        # TODO: should we silently clamp idx? Or should this raise an error, where we
        #       would have a second indexing function that does the clamping?
        if idx >= self.n:
            print('idx > n, setting idx to n')
            idx = self.n
        if idx < 0:
            print('idx < 0, setting idx to 0')
            idx = 0
        return self.state_def.from_data(self.state_def.get_dataset_slice(self.dataset, idx))

    def __setitem__(self, idx: int, value: any):
        raise ValueError(f"'{self.__class__.__name__}' is read-only")

    def __delitem__(self, idx: int):
        raise ValueError(f"'{self.__class__.__name__}' is read-only")

    def get_state(self, time: float) -> ModelState:
        if time < self.times[0]:
            t0 = self.times[0]
            print(f'Time < initial time, setting time to {t0}')
            return self.index(0)
        elif time > self.times[self.n-1]:
            tf = self.times[self.n-1]
            print(f'Time > last recorded time, setting time to {tf}')
            return self.index(self.n-1)

        idx = np.where(time >= self.times)[0][-1]
        return self.state_def.interpolate(self[idx], self[idx+1], time)

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