from collections.abc import Iterable
from typing import List, Optional
import numpy as np

from dlens_vx_v3 import lola, halco

from pynn_brainscales.brainscales2.morphology.parts import Compartment, \
    SharedLineConnection
from pynn_brainscales.brainscales2.morphology.mc_neuron_base import \
    McNeuronBase
from pynn_brainscales.brainscales2.morphology.parameters import \
    McCircuitParameters


def create_mc_neuron(name: str,
                     compartments: List[Compartment],
                     connections: Optional[List[SharedLineConnection]] = None,
                     single_active_circuit: bool = False
                     ) -> McNeuronBase:
    '''
    Create a multicompartment neuron class.

    :param name: Name of the newly created class.
    :param compartments: Compartments of the multicompartment neuron.
    :param connections: Specifies where the shared line is connected.
    :param single_active_circuit: Disable leak, capacitance and threshold for
        all but the first circuit in each comaprtment.

    :return: Class for a multi-compartmental neuron model with the given
        compartments and connections.
    '''
    morphology = lola.Morphology()

    ids = _add_compartments(morphology, compartments)
    comp_dict = dict(zip(ids, compartments))

    if connections is not None:
        _add_connections(morphology, connections)

    logical_compartments, logical_neuron = morphology.done()
    neuron_class = type(name,
                        (McNeuronBase,),
                        {"logical_neuron": logical_neuron,
                         "logical_compartments": logical_compartments,
                         "compartments": comp_dict,
                         "single_active_circuit": single_active_circuit})
    return neuron_class


def _add_compartments(morphology: lola.Morphology,
                      compartments: List[Compartment]
                      ) -> List[halco.CompartmentOnLogicalNeuron]:
    '''
    Add compartments to the given morphology.

    :param morphology: Morphology to which the compartments are added.
    :param compartments: Compartments to add to the morphology.
    :return: Indices of the added compartments.
    '''
    if not isinstance(compartments, Iterable) or \
            isinstance(compartments, str):
        raise TypeError('The `compartments` argument needs to be a sequence '
                        'of Compartments (List, Tuple, ...).')
    ids = []
    for compartment in compartments:
        neuron_circuits = []
        for position in compartment.positions:
            neuron_circuits.append(halco.AtomicNeuronOnLogicalNeuron(
                halco.common.Enum(position)))
        ids.append(morphology.create_compartment(neuron_circuits))

        for neuron in compartment.connect_shared_line:
            assert neuron in compartment.positions
            morphology.connect_to_soma(halco.AtomicNeuronOnLogicalNeuron(
                halco.common.Enum(neuron)))

        for neuron, conductance in compartment.connect_conductance:
            assert neuron in compartment.positions

            # check that neuron was not already used for direct connection
            if np.isin(neuron, compartment.connect_shared_line):
                raise RuntimeError(
                    f'The circuit at position {neuron} was already directly '
                    'connected to the shared line. It can not be connected '
                    'via a conductance at the same time.')
            morphology.connect_resistor_to_soma(
                halco.AtomicNeuronOnLogicalNeuron(
                    halco.common.Enum(neuron)))

            # add conductance to compartment parameters
            prev_cond = compartment.parameters.get(
                'multicompartment_i_bias_nmda', [0] * compartment.size)
            prev_cond[compartment.positions.index(neuron)] = conductance
            compartment.parameters['multicompartment_i_bias_nmda'] = prev_cond

    return ids


def _add_connections(morphology: lola.Morphology,
                     connections: List[SharedLineConnection]) -> None:
    '''
    Add connections which involve the shared line (somatic line) to the
    morphology.

    :param morphology: Morphology to which the connections are added.
    :param compartments: Connections to add to the morphology.
    '''
    if not (isinstance(connections, Iterable) and np.all(
            [isinstance(con, SharedLineConnection) for con in connections])):
        raise TypeError('The `connections` argument needs to be a sequence '
                        ' of SharedLineConnection (List, Tuple, ...)')

    for start, stop, row in connections:
        morphology.connect_soma_line(halco.NeuronColumnOnLogicalNeuron(start),
                                     halco.NeuronColumnOnLogicalNeuron(stop),
                                     halco.NeuronRowOnLogicalNeuron(row))
