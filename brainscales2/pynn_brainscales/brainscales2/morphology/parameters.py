from typing import Sequence
import numpy as np
from pyNN.parameters import ArrayParameter


class McCircuitParameters(ArrayParameter):
    '''
    Represents the parameters of a single multi-compartmental neuron.

    It saves a parameter value for each compartment and for each neuron circuit
    in these compartments.
    '''
    # `self.value` is a ragged array -> needs `object` as dtype
    def __init__(self, value: Sequence) -> None:
        '''
        :param value: Nested list with parameter for each neuron circuit.
            The outer list is over the different compartments, the inner list
            over the neuron circuits in each compartment.
        '''
        # pylint: disable=super-init-not-called

        # convert lists to numpy arrays
        self.value = np.array([np.asarray(circuit_params) for
                               circuit_params in value],
                              dtype=object)

    # Need to handle differently since `self.value` is a ragged array
    def max(self):
        return np.max([value.max() for value in self.value])

    # Need to handle differently since `self.value` is a ragged array
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.value.size != other.value.size:
                return False
            return np.all([np.all(my_val == o_val) for my_val, o_val in
                           zip(self.value, other.value)])

        if isinstance(other, np.ndarray) and other.size > 0 and \
                isinstance(other[0], self.__class__):
            return np.array([(self == seq).all() for seq in other])

        return False
