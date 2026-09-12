import ast
from pathlib import Path
from types import SimpleNamespace


_SESSION_SRC = Path("src/bloombee/client/inference_session.py")


def _load_seed_helper():
    tree = ast.parse(_SESSION_SRC.read_text())
    helper = None
    update_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "seed_replacement_session_history":
            helper = node
        if isinstance(node, ast.ClassDef) and node.name == "InferenceSession":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_update_sequence":
                    update_fn = item
    assert helper is not None, "seed_replacement_session_history is missing"
    assert update_fn is not None, "InferenceSession._update_sequence is missing"
    called = [
        n.func.id
        for n in ast.walk(update_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "seed_replacement_session_history" in called
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "<helper>", "exec"), namespace)
    return namespace["seed_replacement_session_history"]


class _FakeTensor:
    def __init__(self, name: str):
        self.name = name

    def clone(self):
        return _FakeTensor(self.name + "_clone")


def test_recovery_history_only_seeds_first_replacement_session():
    seed = _load_seed_helper()
    sessions = [
        SimpleNamespace(history="stale", _history_storage="buf"),
        SimpleNamespace(history="stale", _history_storage="buf"),
    ]
    seed(sessions, _FakeTensor("input_acts"))
    assert sessions[0].history.name == "input_acts_clone"
    assert sessions[0]._history_storage is None
    assert sessions[1].history is None
    assert sessions[1]._history_storage is None


def test_recovery_history_none_clears_first_session_only():
    seed = _load_seed_helper()
    sessions = [SimpleNamespace(history="stale", _history_storage="buf")]
    seed(sessions, None)
    assert sessions[0].history is None
    assert sessions[0]._history_storage is None


def test_first_step_uses_local_history_when_seeded():
    """A replacement hop with cloned history ignores incoming activations.

    This is why later hops in a split replacement must start empty: their
    first step substitutes ``self.history`` for ``inputs``.
    """
    src = _SESSION_SRC.read_text()
    assert "if not self.stepped:" in src
    assert "inputs = self.history" in src
