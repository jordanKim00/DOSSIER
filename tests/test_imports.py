from dossier.agents import Composer, Formatter, SearchAgent, TOCBuilder


def test_agent_imports() -> None:
    assert TOCBuilder is not None
    assert SearchAgent is not None
    assert Composer is not None
    assert Formatter is not None
