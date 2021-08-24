from collections import namedtuple
from collections.abc import Iterable
from typing import List, Tuple, Optional
import numpy as np


SharedLineConnection = namedtuple("SharedLineConnection",
                                  ["start", "stop", "row"])


class Compartment:
    '''
    A single iso-potential compartment of a multi-compartmental neuron model.
    '''
    def __init__(self, *,
                 positions: List[int],
                 label: Optional[str] = None,
                 connect_shared_line: Optional[List[int]] = None,
                 connect_conductance: Optional[List[Tuple[int, int]]] = None,
                 **parameters):
        '''
        Create a single compartment.

        Additional arguments are saved as parameters of the compartment.

        :param positions: Enums of AtomicNeuronOnLogicalNeuron which belong to
            this compartment.
        :param label: Label of the given compartment.
        :param connect_shared_line: Enums of AtomicNeuronOnLogicalNeuron for
            neuron circuits which are directly connected to the shared line.
        :param connect_conductance: Enums of AtomicNeuronOnLogicalNeuron for
            neuron circuits which are connected via a resistor to the shared
            line and their resistance.
        '''
        self._positions: Iterable = []
        self._label: str = ''
        self._connect_shared_line: Iterable = []
        self._connect_conductance: Iterable = []

        self.positions = positions
        self.label = label
        if connect_shared_line is not None:
            self.connect_shared_line = connect_shared_line
        if connect_conductance is not None:
            self.connect_conductance = connect_conductance
        self.parameters = parameters

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions: Iterable):
        if not (isinstance(positions, Iterable) and np.all(
                [isinstance(pos, int) for pos in positions])):
            raise TypeError('`positions` needs to be a sequence of integers '
                            '(List, Tuple, ...)')

        if len(set(positions)) != len(positions):
            raise TypeError('`positions` contains non-unique entries.')

        # check that parameters and connections are still valid
        try:
            self._check_connect_shared_line(self.connect_shared_line,
                                            positions)
        except TypeError as err:
            raise TypeError('The supplied `positions` do not agree with the '
                            'given value of `connect_shared_line`.') from err
        try:
            self._check_connect_conductance(self.connect_conductance,
                                            positions)
        except TypeError as err:
            raise TypeError('The supplied `positions` do not agree with the '
                            'given value of `connect_conductance`.') from err

        self._positions = positions

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label: str):
        '''
        Check label is a string.

        :param label: Label to check.
        '''
        if not isinstance(label, str):
            raise TypeError('Provide a string as a `label`.')
        self._label = label

    @property
    def connect_shared_line(self):
        return self._connect_shared_line

    @connect_shared_line.setter
    def connect_shared_line(self, connect_shared_line: Iterable):
        self._check_connect_shared_line(connect_shared_line, self.positions)
        self._connect_shared_line = connect_shared_line

    @staticmethod
    def _check_connect_shared_line(connect_shared_line: Iterable,
                                   positions: Iterable) -> None:
        '''
        Check type and validness of `connect_shared_line`

        :param connect_shared_line: Sequence of connect_shared_line to check.
        :param positions: Sequence of positions which belong to the
            compartment.
        '''
        if not (isinstance(connect_shared_line, Iterable) and np.all(
                [isinstance(pos, int) for pos in connect_shared_line])):
            raise TypeError('`connect_shared_line` needs to be a sequence of '
                            'integers (List, Tuple, ...)')

        if len(set(connect_shared_line)) != len(connect_shared_line):
            raise TypeError('`connect_shared_line` contains non-unique '
                            'entries.')

        if not np.isin(connect_shared_line, positions).all():
            raise TypeError('`connect_shared_line` contains entries which '
                            'were not defined in `positions`.')

    @property
    def connect_conductance(self):
        return self._connect_conductance

    @connect_conductance.setter
    def connect_conductance(self, connect_conductance: Iterable):
        self._check_connect_conductance(connect_conductance, self.positions)
        self._connect_conductance = connect_conductance

    @staticmethod
    def _check_connect_conductance(connect_conductance: Iterable,
                                   positions: Iterable) -> None:
        '''
        Check type and validness of `connect_conductance`

        :param connect_conductance: Sequence of connect_conductance to check.
        :param positions: Sequence of positions which belong to the
            compartment.
        '''
        if not isinstance(connect_conductance, Iterable) or \
                isinstance(connect_conductance, str):
            raise TypeError('`connect_conductance` needs to be a sequence of '
                            'values (List, Tuple, ...)')

        if len(connect_conductance) == 0:
            return

        is_tuple = [isinstance(con, tuple) for con in connect_conductance]
        if not np.all(is_tuple):
            raise TypeError('`connect_conductance` has to be a sequence in '
                            'which each entry is a Tuple.')

        length_of_entries = np.array([len(con) for con in connect_conductance])
        if not np.all(length_of_entries == 2):
            raise TypeError('`connect_conductance` has to be a sequence in '
                            'which each entry is of the form '
                            '`(<position>, <conductance>)`.')

        con_array = np.array(connect_conductance)
        if len(set(con_array[:, 0])) != len(connect_conductance):
            raise TypeError('`connect_conductance` contains non-unique '
                            'entries.')

        if not np.isin(con_array[:, 0], positions).all():
            raise TypeError('`connect_conductance` contains entries which '
                            'were not defined in `positions`.')

    @property
    def size(self):
        return len(self.positions)
