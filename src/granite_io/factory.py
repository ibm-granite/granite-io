# SPDX-License-Identifier: Apache-2.0

"""
This factory utility provides common factory construction semantics and a common
base class for classes that can be constructed via config

Derived from: https://github.com/caikit/caikit/blob/main/caikit/core/toolkit/factory.py
"""

# Standard
from functools import partial, wraps
from typing import Protocol
import abc
import copy
import importlib
import inspect

# Third Party
import aconfig
import jsonschema
import jsonschema.exceptions


class FactoryConstructible(abc.ABC):
    """A class can be constructed by a factory if its constructor takes exactly
    one argument that is an aconfig.Config object and it has a name to identify
    itself with the factory.
    """

    # This is the set of default values for the config that me be set by child
    # implementations
    config_defaults = {}

    # This is the jsonschema dict for this instance's config
    config_schema = {}

    @abc.abstractmethod
    def __init__(self, config: aconfig.Config, **kwargs):
        """A FactoryConstructible object must be constructed with a config
        object that it uses to pull in all configuration
        """


class FactoryCallable(Protocol):
    """A callable that can be stored and constructed by a factory"""

    def __call__(self, config: aconfig.Config, **kwargs):
        """Factory callables must take config as the first argument"""


class Factory:
    """The base factory class implements all common factory functionality to
    read a designated portion of config and instantiate an instance of the
    registered classes.
    """

    # The keys in the instance config
    TYPE_KEY = "type"
    CONFIG_KEY = "config"

    def __init__(self, name: str):
        """Construct with the path in the global config where this factory's
        configuration lives.
        """
        self._name = name
        self._registered_types = {}

    @property
    def name(self) -> str:
        return self._name

    def register(
        self, constructible: type[FactoryConstructible] | FactoryCallable, *aliases
    ):
        """Register the given constructible/callable"""
        base_name = getattr(constructible, "name", None) or constructible.__name__
        names = [base_name] + list(aliases)
        assert not any(name in self._registered_types for name in names), (
            f"Conflicting registration of {base_name} (aliases: {aliases})"
        )
        for name in names:
            self._registered_types[name] = constructible

    def registered_types(self) -> list[str]:
        """Get the list of registered types"""
        return list(sorted(self._registered_types.keys()))

    @classmethod
    def _get_construct_instance_config(
        cls,
        arg_one: str | dict,
        instance_config: dict | None = None,
    ) -> dict:
        """Shared helper for parsing args in construct"""
        if isinstance(arg_one, str):
            inst_type = arg_one
            cfg = instance_config or {}
            instance_config = {
                cls.TYPE_KEY: inst_type,
                cls.CONFIG_KEY: cfg,
            }
        elif isinstance(arg_one, dict):
            assert instance_config is None, (
                "Cannot pass config as both first and second argument"
            )
            instance_config = arg_one
        else:
            raise TypeError(f"Invalid argument type {type(arg_one)} for construct")
        return instance_config

    def construct(
        self,
        arg_one: str | dict,
        instance_config: dict | None = None,
        *,
        validate: bool = True,
        **kwargs,
    ) -> FactoryConstructible:
        """Construct an instance of the given type"""
        instance_config = self._get_construct_instance_config(arg_one, instance_config)
        inst_type = instance_config.get(self.__class__.TYPE_KEY)
        cfg = instance_config.get(self.__class__.CONFIG_KEY)
        inst_cls = self._registered_types.get(inst_type)
        if inst_cls is None:
            raise ValueError(f"No {self.name} class registered for type {inst_type}")
        inst_cfg = aconfig.Config(
            _merge_configs(
                copy.deepcopy(inst_cls.config_defaults),
                cfg,
            ),
            override_env_vars=True,
        )
        if validate:
            # NOTE: This explicitly allows additional properties
            jsonschema.validate(instance=inst_cfg, schema=inst_cls.config_schema)
        return inst_cls(inst_cfg, **kwargs)

    def decorator(
        self,
        name: str,
        *aliases: list[str],
        config_schema: dict | None = None,
        config_defaults: dict | None = None,
    ) -> callable:
        """Decorator to register a class or function with the factory"""

        def _decorator(cls_or_fn, config_schema, config_defaults):
            if inspect.isfunction(cls_or_fn):
                sig = inspect.signature(cls_or_fn)

                # If no schema given, deduce it from the type signature
                if config_schema is None:
                    config_schema = _get_jsonschema(
                        {k: v.annotation for k, v in sig.parameters.items()}
                    )
                if config_defaults is None:
                    config_defaults = {
                        k: p.default
                        for k, p in sig.parameters.items()
                        if p.default is not inspect._empty
                    }

                # Make sure the schema matches the kwargs in the function
                schema_properties = (config_schema or {}).get("properties", {})
                if not all(prop in sig.parameters for prop in schema_properties):
                    raise ValueError(
                        f"Schema properties {schema_properties} do not match "
                        f"function kwargs {sig.parameters}"
                    )
                sig_kwargs = list(sig.parameters.keys())

                @wraps(cls_or_fn)
                def _wrapper_callable(config: aconfig.Config, **kwargs):
                    config_kwargs = {
                        k: v
                        for k, v in (config or aconfig.Config({})).items()
                        if k in sig_kwargs
                    }
                    return partial(cls_or_fn, **config_kwargs, **kwargs)

                cls = _wrapper_callable

            elif inspect.isclass(cls_or_fn):
                cls = cls_or_fn

            else:
                raise TypeError("Decorator must wrap a class or function")

            # Add the class attributes
            cls.name = name
            if config_schema is not None:
                cls.config_schema = config_schema
            if config_defaults is not None:
                cls.config_defaults = config_defaults

            # Register with the factory
            self.register(cls, *aliases)

            # Return the wrapped object (class or function)
            return cls_or_fn

        return partial(
            _decorator, config_schema=config_schema, config_defaults=config_defaults
        )


class ImportableFactory(Factory):
    """An ImportableFactory extends the base Factory to allow the construction
    to specify an "import_class" field that will be used to import and register
    the implementation class before attempting to initialize it.
    """
    
    def construct(
        self,
        arg_one: str | dict,
        instance_config: dict | None = None,
        **kwargs,
    ):
        # Look for an import_class and import and register it if found
        instance_config = self._get_construct_instance_config(arg_one, instance_config)
        return super().construct(instance_config, **kwargs)


## Implementation Details ######################################################


_CONFIG_TYPE = dict | aconfig.Config


def _merge_configs(
    base: _CONFIG_TYPE | None,
    overrides: _CONFIG_TYPE | None,
) -> aconfig.Config:
    """Helper to perform a deep merge of the overrides into the base. The merge
    is done in place, but the resulting dict is also returned for convenience.
    The merge logic is quite simple: If both the base and overrides have a key
    and the type of the key for both is a dict, recursively merge, otherwise
    set the base value to the override value.
    Args:
        base: The base config that will be updated with the overrides
        overrides: The override config
    Returns:
        merged:
            The merged results of overrides merged onto base
    """
    # Handle none args
    if base is None:
        return overrides or {}
    if overrides is None:
        return base or {}

    # Do the deep merge
    for key, value in overrides.items():
        if (
            key not in base
            or not isinstance(base[key], dict)
            or not isinstance(value, dict)
        ):
            base[key] = value
        else:
            base[key] = _merge_configs(base[key], value)

    return base


def _get_jsonschema(sig_types: dict[str:type]) -> dict:
    """Given a mapping from keywords to type hints, create a jsonschema"""
    if None in sig_types.values() or inspect._empty in sig_types.values():
        raise ValueError("Cannot deduce config schema from function without type hints")
    all_sig_types = set(sig_types.values())
    validator = jsonschema.validators.validator_for({})
    type_mapping = {
        t: x[0] if x else None
        for t, x in zip(
            all_sig_types,
            [
                [
                    y
                    for y in validator.TYPE_CHECKER._type_checkers
                    if validator.TYPE_CHECKER.is_type(t(), y)
                ]
                for t in all_sig_types
            ],
            strict=True,
        )
    }
    return {"properties": {k: {"type": type_mapping[v]} for k, v in sig_types.items()}}
