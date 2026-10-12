import numpy as np

from kawin.Constants import AVOGADROS_NUMBER

class Volume:
    def __init__(self, a: float = 0, va: float = 0, vm: float = 0, atoms_per_cell: int = 0):
        self.a = a
        self.va = va
        self.vm = vm
        self.atoms_per_cell = atoms_per_cell

    @staticmethod
    def vm_to_va(vm: float, atoms_per_cell: int):
        return vm*atoms_per_cell/AVOGADROS_NUMBER

    @staticmethod
    def vm_to_a(vm: float, atoms_per_cell: int):
        return np.cbrt(Volume.vm_to_va(vm, atoms_per_cell))

    @staticmethod
    def va_to_vm(va: float, atoms_per_cell: int):
        return va*AVOGADROS_NUMBER/atoms_per_cell

    @staticmethod
    def va_to_a(va: float):
        return np.cbrt(va)

    @staticmethod
    def a_to_va(a: float):
        return a**3

    @staticmethod
    def a_to_vm(self, a: float, atoms_per_cell: int):
        return Volume.a_to_va(a) * AVOGADROS_NUMBER / atoms_per_cell

    def set_vm(self, vm: float, atoms_per_cell: int):
        self.vm = vm
        self.atoms_per_cell = atoms_per_cell
        self.va = Volume.vm_to_va(self.vm, self.atoms_per_cell)
        self.a = Volume.va_to_a(self.va)

    def set_va(self, va: float, atoms_per_cell: int):
        self.va = va
        self.atoms_per_cell = atoms_per_cell
        self.vm = Volume.va_to_vm(self.va, self.atoms_per_cell)
        self.a = Volume.va_to_a(self.va)

    def set_lattice_parameter(self, a: float, atoms_per_cell: int):
        self.a = a
        self.atoms_per_cell = atoms_per_cell
        self.va = Volume.a_to_va(self.a)
        self.vm = Volume.a_to_vm(self.a, self.atoms_per_cell)

    @classmethod
    def from_vm(cls, vm: float, atoms_per_cell: int):
        return cls(Volume.vm_to_a(vm, atoms_per_cell), Volume.vm_to_va(vm, atoms_per_cell), vm, atoms_per_cell)

    @classmethod
    def from_va(cls, va: float, atoms_per_cell: int):
        return cls(Volume.va_to_a(va), va, Volume.va_to_vm(va, atoms_per_cell), atoms_per_cell)

    @classmethod
    def from_lattice_parameter(cls, a: float, atoms_per_cell: int):
        return cls(a, Volume.a_to_va(a), Volume.a_to_vm(a, atoms_per_cell), atoms_per_cell)

class VolumeParameter:
    MOLAR_VOLUME = 0
    ATOMIC_VOLUME = 1
    LATTICE_PARAMETER = 2

    def __init__(self, value=None, volumeType=None, atomsPerCell=None):
        if value is None or volumeType is None or atomsPerCell is None:
            self.a = None
            self.Va = None
            self.Vm = None
            self.atomsPerCell = None
        else:
            self.setVolume(value, volumeType, atomsPerCell)
        self._updateCallbacks = []

    def setVolume(self, value, volumeType, atomsPerCell):
        '''
        Function to set lattice parameter, atomic volume and molar volume

        Parameters
        ----------
        value : float
            Value for volume parameters (lattice parameter, atomic (unit cell) volume or molar volume)
        valueType : VolumeParameter or str
            States what volume term that value is
        atomsPerCell : int
            Number of atoms in the unit cell
        '''
        self.atomsPerCell = atomsPerCell
        if volumeType == self.MOLAR_VOLUME or volumeType == 'VM':
            self.Vm = value
            self.Va = atomsPerCell * self.Vm / AVOGADROS_NUMBER
            self.a = np.cbrt(self.Va)
        elif volumeType == self.ATOMIC_VOLUME or volumeType == 'VA':
            self.Va = value
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell
            self.a = np.cbrt(self.Va)
        elif volumeType == self.LATTICE_PARAMETER or volumeType == 'a':
            self.a = value
            self.Va = self.a**3
            self.Vm = self.Va * AVOGADROS_NUMBER / atomsPerCell
        else:
            valid_values = "['VM', 'VA', 'a', VolumeParameter.MOLAR_VOLUME, VolumeParameter.ATOMIC_VOLUME, VolumeParameter.LATTICE_PARAMETER]"
            raise ValueError(f'Unknown volume type {volumeType}. Values must be: {valid_values}')

        for callback in self._updateCallbacks:
            callback()