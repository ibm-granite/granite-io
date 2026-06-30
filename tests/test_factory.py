# SPDX-License-Identifier: Apache-2.0

"""
Tests for the shared factory implementation
"""
# NOTE: We use lots of "bad" names in tests to indicate that they're not real!
# pylint: disable=disallowed-name

# Standard
from unittest import mock
import abc

# Third Party
from jsonschema.exceptions import ValidationError
import aconfig
import pytest

# Local
from granite_io import factory

## Helpers #####################################################################


class BaseDummy(factory.FactoryConstructible):
    """Base class for dummy classes"""

    def __init__(self, config: aconfig.Config):
        self._config = config

    @abc.abstractmethod
    def doit(self, it: str) -> str:
        """Do something"""


## Tests #######################################################################


def test_factory_decorate_and_construct_class():
    """Test using the decorator on classes works to register and construct"""
    dummy_factory = factory.Factory("dummy")
    assert dummy_factory.name == "dummy"
    assert not dummy_factory.registered_types()

    @dummy_factory.decorator("dummy-one", "Dummy One")
    class DummyOne(BaseDummy):
        def doit(self, it: str) -> str:
            return f"Doing ONE it: {it}"

    @dummy_factory.decorator("dummy-two", "d2")
    class DummyTwo(BaseDummy):  # pylint: disable=unused-variable
        def doit(self, it: str) -> str:
            return f"Doing TWO its: {it} {it}"

    assert len(dummy_factory.registered_types()) == 4

    d1_a = dummy_factory.construct("dummy-one")
    assert isinstance(d1_a, DummyOne)
    d1_b = dummy_factory.construct("Dummy One")
    assert isinstance(d1_b, DummyOne)


def test_construct_with_config():
    """Test that constructing an object with a config works as expected."""
    dummy_factory = factory.Factory("dummy")

    @dummy_factory.decorator("dummy-one", "Dummy One")
    class DummyOne(BaseDummy):
        def doit(self, it: str) -> str:
            return f"Doing ONE it: {it}"

    d1 = dummy_factory.construct({"type": "dummy-one"})
    assert isinstance(d1, DummyOne)


def test_decorate_function():
    """Test that a function can be decorated."""
    dummy_factory = factory.Factory("dummy")

    @dummy_factory.decorator(
        "d1",
        config_schema={
            "properties": {"foo": {"type": "string"}, "bar": {"type": "string"}}
        },
        config_defaults={"bar": "BAR"},
    )
    def d(it: str, foo: str, bar: str) -> str:
        return f"Foo: {foo}, Bar: {bar}, it: {it}"

    d1 = dummy_factory.construct("d1", {"foo": "FOO"})
    assert d1("it") == "Foo: FOO, Bar: BAR, it: it"


def test_decorate_function_config_mismatch():
    """Make sure that the config schema must match the function signature"""
    dummy_factory = factory.Factory("dummy")

    def d1(it: str, foo: str, bar: str) -> str:
        return f"Foo: {foo}, Bar: {bar}, it: {it}"

    with pytest.raises(ValueError):
        dummy_factory.decorator(
            "d1",
            config_schema={"properties": {"buz": {"type": "string"}}},
            config_defaults={"bar": "BAR"},
        )(d1)


def test_decorate_function_bad_type():
    """Make sure that decorating something that isn't a class or function raises
    an error
    """
    dummy_factory = factory.Factory("dummy")
    x = 1
    with pytest.raises(TypeError):
        dummy_factory.decorator("d1")(x)


def test_nested_config():
    """Make sure that nested config defaults and types are handled correctly"""
    dummy_factory = factory.Factory("dummy")

    @dummy_factory.decorator(
        "d",
        config_defaults={"foo": {"bar": 1}},
        config_schema={
            "type": "object",
            "properties": {
                "foo": {
                    "type": "object",
                    "properties": {
                        "bar": {
                            "type": "number",
                        },
                    },
                },
            },
        },
    )
    def d(foo: dict, bar_exp: int) -> str:
        assert "bar" in foo
        assert foo["bar"] == bar_exp

    dummy_factory.construct("d")(bar_exp=1)
    dummy_factory.construct("d", {"foo": {"bar": 2}})(bar_exp=2)


def test_config_schema_validation():
    """Make sure that the config schema must match the function signature"""
    dummy_factory = factory.Factory("dummy")

    @dummy_factory.decorator("d")
    def d(foo: int):
        return foo

    with pytest.raises(ValidationError):
        dummy_factory.construct("d", {"foo": "bar"})


def test_schema_inference_with_defaults():
    """Make sure the config schema and defaults can be correctly inferred"""
    dummy_factory = factory.Factory("dummy")

    @dummy_factory.decorator("d")
    def d(foo: int = 1, bar: str = "asdf"):
        return f"{bar}[{foo}]"

    assert dummy_factory.construct("d")() == "asdf[1]"
    assert dummy_factory.construct("d", {"foo": 2})() == "asdf[2]"


def test_schema_inference_no_hint():
    """Make sure schema inference raises if the function doesn't have type hints"""
    dummy_factory = factory.Factory("dummy")

    def d(foo):
        return foo

    with pytest.raises(ValueError):
        dummy_factory.decorator("d")(d)


def test_merge_configs_corners():
    """Make sure merge config corner cases are handled correctly"""
    base = {"a": 1, "b": 2}
    overrides = {"a": 11, "c": 3}
    assert factory._merge_configs(None, overrides) is overrides
    assert factory._merge_configs(base, None) is base
    assert factory._merge_configs(base, overrides) == {"a": 11, "b": 2, "c": 3}


def test_importable_factory():
    """Test that registration can happen with a dynamic import using
    ImportableFactory
    """
    dummy_ifactory = factory.ImportableFactory("dummy")

    class DummyOne(BaseDummy):
        name = "d1"

        def doit(self, it: str) -> str:
            return f"ONE[{it}]"

    mock_module = aconfig.Config({"DummyOne": DummyOne}, override_env_vars=False)

    with mock.patch("importlib.import_module", return_value=mock_module) as import_mock:
        # Without the import keyword, it should fail to construct
        with pytest.raises(ValueError):
            dummy_ifactory.construct({"type": "d1"})
        assert import_mock.call_count == 0

        # With the import keyword, it should succeed
        # dummy_one = dummy_ifactory.construct(
        #     {"type": "d1", dummy_ifactory.IMPORT_CLASS_KEY: "foo.bar.DummyOne"}
        # )
        # assert isinstance(dummy_one, DummyOne)
        # assert import_mock.call_count == 1


def test_bad_construct_arg_type():
    """Test that calling construct with a bad type raises"""
    dummy_factory = factory.Factory("dummy")
    with pytest.raises(TypeError):
        dummy_factory.construct(1)
