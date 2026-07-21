import pytest

from auto_infer.models.registry import get_model_class, register


def test_model_registry_uses_guarded_extension_seam():
    class SyntheticModel:
        pass

    register("SyntheticModelForRegistryTest", SyntheticModel)
    assert get_model_class("SyntheticModelForRegistryTest") is SyntheticModel
    with pytest.raises(ValueError, match="already registered"):
        register("SyntheticModelForRegistryTest", SyntheticModel)
