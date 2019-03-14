"""Convert scikit-learn estimators into an OpenMLFlows and vice versa."""

from collections import OrderedDict
import importlib
import json
import json.decoder
import re
import six
import warnings
import sys
import inspect
import copy
import functools

import numpy as np
import scipy.stats.distributions
import shogun as sg
import shogun

import openml
from openml.flows import OpenMLFlow
from openml.exceptions import PyOpenMLError

from .abstract_converter import AbstractConverter, DEPENDENCIES_PATTERN
from distutils.version import LooseVersion

if sys.version_info >= (3, 5):
    from json.decoder import JSONDecodeError
else:
    JSONDecodeError = ValueError


class ShogunConverter(AbstractConverter):
    def __init__(self, model):
        super().__init__(model)

    @staticmethod
    def from_flow(flow, components=None, initialize_with_defaults=False):
        """Initializes a model based on a flow.

        Parameters
        ----------
        o : mixed
            the object to deserialize (can be flow object, or any serialzied
            parameter value that is accepted by)

        components : dict


        initialize_with_defaults : bool, optional (default=False)
            If this flag is set, the hyperparameter values of flows will be
            ignored and a flow with its defaults is returned.

        Returns
        -------
        mixed
        """
        # raise NotImplementedError("")
        return ShogunConverter._flow_to_shogun(flow, components=components,
                                               initialize_with_defaults=False)

    @staticmethod
    def _flow_to_shogun(o, components=None, initialize_with_defaults=False):

        if isinstance(o, six.string_types):
            try:
                o = json.loads(o)
            except JSONDecodeError:
                pass
        if isinstance(o, dict):
            # Check if the dict encodes a 'special' object, which could not
            # easily converted into a string, but rather the information to
            # re-create the object were stored in a dictionary.
            if 'oml-python:serialized_object' in o:
                serialized_type = o['oml-python:serialized_object']
                value = o['value']
                key = o["key"]
                if serialized_type == 'function':
                    rval = ShogunConverter._deserialize_model(
                        components[key], initialize_with_defaults)
                    return rval
                else:
                    raise ValueError("oml-python:serialized_object", o)
        elif isinstance(o, (bool, int, float, six.string_types)) or o is None:
            rval = o
        elif isinstance(o, OpenMLFlow):
            rval = ShogunConverter._deserialize_model(o, initialize_with_defaults)
        elif isinstance(o, list):
            rval = None
        elif isinstance(o, (np.ndarray, np.generic)):
            rval = o
        else:
            raise ValueError("Error with type {}".format(type(o)))
        return rval

    @staticmethod
    def _shogun_deserialize_function(component, initialize_with_defaults=False):
        return ShogunConverter._flow_to_shogun(component,

                                               initialize_with_defaults=initialize_with_defaults)
    @staticmethod
    def get_model_class(flow):
        module_name = flow.class_name.rsplit('.', 1)
        return functools.partial(eval(module_name[0]), module_name[1])

    def to_flow(self):
        """Creates an OpenML flow of the models.

        Returns
        -------
        OpenMLFlow
        """

        class_name = "{}.{}.{}".format("shogun", type(self._model).__name__.lower(),
                                       self._model.get_name())
        name = class_name

        dependencies = [self.format_external_version('shogun', sg.__version__),
                        'numpy>=1.7.0']
        dependencies = '\n'.join(dependencies)

        return OpenMLFlow(name=name,
                          class_name=class_name,
                          description='Automatically created shogun flow.',
                          model=self._model,
                          components=self._sub_components,
                          parameters=self._parameters,
                          parameters_meta_info=self._parameters_meta_info,
                          external_version=self.external_version,
                          tags=['openml-python', 'shogun', 'shogun-toolbox',
                                'python',
                                self.format_external_version(
                                    'shogun', sg.__version__).replace('==', '_'),
                                self._model.get_name()
                                ],
                          language='English',
                          # TODO fill in dependencies!
                          dependencies=dependencies)

    def extract_information_from_model(self):
        """
        Extract information from shogun model.
        Populates self._parameters, a OrderedDict with
        the parameter names and corresponding values of
        the model, and self._parameters_meta_info which
        has a description and the data type of each
        parameter
        """

        model_parameters = self._model.get_params()

        for param_name, param_value in sorted(model_parameters.items(), key=lambda t: t[0]):
            rval = self._shogun_to_flow(param_value, self._model)
            if isinstance(rval, OpenMLFlow):
                self._extract_openml_flow_information(rval, param_name)

            else:
                if not (hasattr(rval, '__len__') and len(rval) == 0):
                    rval = json.dumps(rval)
                    self._parameters[param_name] = rval
                else:
                    self._parameters[param_name] = None

            # self._parameters_meta_info[param_name] = \
            #     OrderedDict((('description', self._model.parameter_description(param_name)),
            #                  ('data_type', self._model.parameter_type(param_name))))
            # TODO: descriptions can cause issues with upload so ignore for now
            self._parameters_meta_info[param_name] = OrderedDict((('description', None),
                                                         ('data_type', self._model.parameter_type(param_name))))

    @staticmethod
    def _shogun_to_flow(o, parent_model=None):
        """
        Returns the rvalue of name from self._model
        if there is a getter that can handle it
        """

        # is it a model? If so recursive call
        if ShogunConverter._is_shogun_trainable_model(o):
            rval = ShogunConverter(o).to_flow()
        if ShogunConverter._is_sg_base(o):
            rval = ShogunConverter(o).to_flow()
        elif isinstance(o, (float, str, int)) or o is None:
            rval = o
            # rval = [ShogunConverter._shogun_to_flow(value, name)
            # for name in value.parameter_names()]
        # handle primitive types
        elif isinstance(o, np.ndarray):
            # rval = [ShogunConverter._flow_to_shogun(o_i) for o_i in o]
            rval = None
        else:
            # warnings.warn("Not implemented for type {}".format(type(o)))
            rval = None
            # raise ValueError("Not implemented for type {}".format(type(o)))

        return rval

    def _extract_openml_flow_information(self, rval, parameter_name):
        """

        """
        self._sub_components[parameter_name] = rval
        self._sub_components_explicit.add(parameter_name)
        component_reference = OrderedDict()
        # in shogun for now every serialized object is a function
        component_reference = OrderedDict()
        component_reference[
            'oml-python:serialized_object'] = 'function'
        component_reference['key'] = parameter_name
        component_reference['value'] = "shogun.{}".format(parameter_name)
        self._parameters[parameter_name] = json.dumps(component_reference)

    @staticmethod
    def _is_shogun_trainable_model(obj):
        return hasattr(obj, "train")

    @property
    def external_version(self):
        if self._external_version:
            return self._external_version
        # Create external version string for a flow, given the model and the
        # already parsed dictionary of sub_components. Retrieves the external
        # version of all subcomponents, which themselves already contain all
        # requirements for their subcomponents. The external version string is a
        # sorted concatenation of all modules which are present in this run.
        model_package_version_number = sg.__version__
        external_version = self.format_external_version("shogun",
                                                        model_package_version_number)
        openml_version = self.format_external_version('openml', openml.__version__)
        external_versions = set()
        external_versions.add(external_version)
        external_versions.add(openml_version)
        for visitee in self._sub_components.values():
            for external_version in visitee.external_version.split(','):
                external_versions.add(external_version)
        external_versions = list(sorted(external_versions))
        self._external_version = ','.join(external_versions)
        return self._external_version

    @staticmethod
    def _is_sg_base(obj):
        return isinstance(obj, (sg.Kernel, sg.Distance))
