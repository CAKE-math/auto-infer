import auto_infer


def test_package_imports():
    assert isinstance(auto_infer.__version__, str)


def test_public_api_exports():
    from auto_infer.config import EngineConfig
    from auto_infer.entrypoints.llm import LLM
    assert auto_infer.LLM is LLM
    assert auto_infer.EngineConfig is EngineConfig
