from abc import ABC, abstractmethod
from types import SimpleNamespace

from aptrade.metabase import MetaParams


class AbstractSizer(ABC):
    """
    The Abstract Factory interface declares a set of methods that return
    different abstract products. These products are called a family and are
    related by a high-level theme or concept. Products of one family are usually
    able to collaborate among themselves. A family of products may have several
    variants, but the products of one variant are incompatible with products of
    another.
    """

    strategy = None
    broker = None

    def getsizing(self, data, isbuy: bool):
        comminfo = self.broker.getcommissioninfo(data)
        return self._getsizing(comminfo, self.broker.getcash(), data, isbuy)

    def __init__(self, *args, **kwargs):
        """Initialize params for sizers that declare a `params` tuple.

        Supports positional and keyword parameters to remain compatible
        with the previous `MetaParams`-based behavior used by sizers.
        """
        params_def = getattr(self.__class__, "params", ()) or ()
        # Start with defaults
        params_obj = SimpleNamespace()
        param_names = [name for name, _ in params_def]
        for name, default in params_def:
            setattr(params_obj, name, default)

        # Apply positional args in order
        for name, val in zip(param_names, args):
            setattr(params_obj, name, val)

        # Apply keyword overrides
        for name in param_names:
            if name in kwargs:
                setattr(params_obj, name, kwargs.pop(name))

        # expose as both `params` and `p` for compatibility
        self.params = params_obj
        self.p = params_obj

    @abstractmethod
    def _getsizing(self, comminfo, cash, data, isbuy: bool) -> int:
        """This method has to be overriden by subclasses of Sizer to provide
        the sizing functionality

        Params:
          - ``comminfo``: The CommissionInfo instance that contains
            information about the commission for the data and allows
            calculation of position value, operation cost, commision for the
            operation

          - ``cash``: current available cash in the *broker*

          - ``data``: target of the operation

          - ``isbuy``: will be ``True`` for *buy* operations and ``False``
            for *sell* operations

        The method has to return the actual size (an int) to be executed. If
        ``0`` is returned nothing will be executed.

        The absolute value of the returned value will be used

        """
        raise NotImplementedError

    def set(self, strategy, broker):
        self.strategy = strategy
        self.broker = broker


# class Sizer(metaclass=MetaParams):
#     """This is the base class for *Sizers*. Any *sizer* should subclass this
#     and override the ``_getsizing`` method

#     Member Attribs:

#       - ``strategy``: will be set by the strategy in which the sizer is working

#         Gives access to the entire api of the strategy, for example if the
#         actual data position would be needed in ``_getsizing``::

#            position = self.strategy.getposition(data)

#       - ``broker``: will be set by the strategy in which the sizer is working

#         Gives access to information some complex sizers may need like portfolio
#         value, ..
#     """

#     strategy = None
#     broker = None

#     def getsizing(self, data, isbuy: bool):
#         comminfo = self.broker.getcommissioninfo(data)
#         return self._getsizing(comminfo, self.broker.getcash(), data, isbuy)

#     def _getsizing(self, comminfo, cash, data, isbuy: bool) -> int:
#         """This method has to be overriden by subclasses of Sizer to provide
#         the sizing functionality

#         Params:
#           - ``comminfo``: The CommissionInfo instance that contains
#             information about the commission for the data and allows
#             calculation of position value, operation cost, commision for the
#             operation

#           - ``cash``: current available cash in the *broker*

#           - ``data``: target of the operation

#           - ``isbuy``: will be ``True`` for *buy* operations and ``False``
#             for *sell* operations

#         The method has to return the actual size (an int) to be executed. If
#         ``0`` is returned nothing will be executed.

#         The absolute value of the returned value will be used

#         """
#         raise NotImplementedError

#     def set(self, strategy, broker):
#         self.strategy = strategy
#         self.broker = broker


# SizerBase = Sizer  # alias for old naming


from .fixedsize import *
from .percents_sizer import *
from .simple import *
