"""
Add `location_selector` and `source_location_selector` to build-in pyNN
connectors. This allows connections between neurons with several locations.
"""
import pyNN


# FIXME: Get rid of unused COnnectors (in later CS)? I.e. CSAConnector,
# DistanceDependentProbabilityConnector,
# DisplacementDependentProbabilityConnector, SmallWorldConnector, ...
class MapConnector(pyNN.connectors.MapConnector):
    def __init__(self, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector

    def connect(self, projection):
        raise NotImplementedError()


class AllToAllConnector(pyNN.connectors.AllToAllConnector):
    def __init__(self, allow_self_connections=True, safe=True,
                 callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(allow_self_connections=allow_self_connections,
                         safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class FixedProbabilityConnector(pyNN.connectors.FixedProbabilityConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, p_connect, allow_self_connections=True,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(p_connect=p_connect,
                         allow_self_connections=allow_self_connections,
                         rng=rng, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class DistanceDependentProbabilityConnector(
        pyNN.connectors.DistanceDependentProbabilityConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, d_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(d_expression=d_expression,
                         allow_self_connections=allow_self_connections,
                         rng=rng, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class IndexBasedProbabilityConnector(
        pyNN.connectors.IndexBasedProbabilityConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, index_expression, allow_self_connections=True,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(index_expression=index_expression,
                         allow_self_connections=allow_self_connections,
                         rng=rng, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class DisplacementDependentProbabilityConnector(
        pyNN.connectors.DisplacementDependentProbabilityConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, disp_function, allow_self_connections=True,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(disp_function=disp_function,
                         allow_self_connections=allow_self_connections,
                         rng=rng, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class FromListConnector(pyNN.connectors.FromListConnector):
    def __init__(self, conn_list, column_names=None, safe=True, callback=None,
                 *, location_selector=None, source_location_selector=None):
        super().__init__(conn_list=conn_list, column_names=column_names,
                         safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class FromFileConnector(pyNN.connectors.FromFileConnector):
    def __init__(self, file, distributed=False, safe=True, callback=None,
                 *, location_selector=None, source_location_selector=None):
        super().__init__(file=file, distributed=distributed, safe=safe,
                         callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class FixedNumberConnector(pyNN.connectors.FixedNumberConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, n, allow_self_connections=True, with_replacement=False,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(n=n, allow_self_connections=allow_self_connections,
                         with_replacement=with_replacement, rng=rng, safe=safe,
                         callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector

    def connect(self, projection):
        raise NotImplementedError()


class FixedNumberPostConnector(pyNN.connectors.FixedNumberPostConnector,
                               FixedNumberConnector):
    pass


class FixedNumberPreConnector(pyNN.connectors.FixedNumberPreConnector,
                              FixedNumberConnector):
    pass


class OneToOneConnector(pyNN.connectors.OneToOneConnector, MapConnector):
    pass


class SmallWorldConnector(pyNN.connectors.SmallWorldConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, degree, rewiring, allow_self_connections=True,
                 n_connections=None, rng=None, safe=True, callback=None,
                 *, location_selector=None, source_location_selector=None):
        super().__init__(degree=degree, rewiring=rewiring,
                         allow_self_connections=allow_self_connections,
                         n_connections=n_connections, rng=rng, safe=safe,
                         callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector

    def connect(self, projection):
        raise NotImplementedError


class CSAConnector(pyNN.connectors.CSAConnector):
    def __init__(self, cset, safe=True, callback=None, *,
                 location_selector=None, source_location_selector=None):
        super().__init__(cset=cset, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class CloneConnector(pyNN.connectors.CloneConnector):
    def __init__(self, reference_projection, safe=True, callback=None,
                 *, location_selector=None, source_location_selector=None):
        super().__init__(reference_projection=reference_projection, safe=safe,
                         callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class ArrayConnector(pyNN.connectors.ArrayConnector):
    def __init__(self, array, safe=True, callback=None, *,
                 location_selector=None, source_location_selector=None):
        super().__init__(array=array, safe=safe, callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector


class FixedTotalNumberConnector(pyNN.connectors.FixedTotalNumberConnector):
    # pylint: disable=too-many-arguments
    def __init__(self, n, allow_self_connections=True, with_replacement=True,
                 rng=None, safe=True, callback=None, *, location_selector=None,
                 source_location_selector=None):
        super().__init__(n=n, allow_self_connections=allow_self_connections,
                         with_replacement=with_replacement, rng=rng, safe=safe,
                         callback=callback)
        self.location_selector = location_selector
        self.source_location_selector = source_location_selector
