import importlib.util
from pathlib import Path


def load_module():
    file_path = Path(__file__).with_name("32_tebd.py")
    spec = importlib.util.spec_from_file_location("tebd32", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runtime_comparison_original_vs_vidal():
    tebd = load_module()

    n = 12
    depth = 8
    max_dim = 4
    cutoff = 1e-10
    seed = 0

    gates_1q = tebd.generate_single_qubit_gates(depth, n, seed=seed)
    exact_state, _ = tebd.run_exact(n, depth, gates_1q)

    result_original = tebd.run_original(
        n,
        depth,
        gates_1q,
        max_dim=max_dim,
        cutoff=cutoff,
        exact_state=exact_state,
        verbose=False,
    )
    result_vidal = tebd.vidal(
        n,
        depth,
        gates_1q,
        max_dim=max_dim,
        cutoff=cutoff,
        exact_state=exact_state,
        verbose=False,
    )

    print("\n[pytest runtime comparison]")
    print(f"original main-style: {result_original['elapsed']:.6f} sec")
    print(f"vidal canonical   : {result_vidal['elapsed']:.6f} sec")
    print(
        f"speed ratio (original/vidal): {result_original['elapsed'] / result_vidal['elapsed']:.3f}"
    )

    fidelity_improvement = result_vidal["fidelity"] - result_original["fidelity_trunc"]
    fidelity_improvement_ratio = result_vidal["fidelity"] / result_original["fidelity_trunc"]
    print(f"original trunc fidelity: {result_original['fidelity_trunc']:.6f}")
    print(f"vidal fidelity         : {result_vidal['fidelity']:.6f}")
    print(f"fidelity improvement   : {fidelity_improvement:.6f}")
    print(f"fidelity ratio (vidal/original_trunc): {fidelity_improvement_ratio:.3f}")

    assert result_original["elapsed"] > 0.0
    assert result_vidal["elapsed"] > 0.0
    assert 0.0 <= result_original["fidelity_trunc"] <= 1.0000001
    assert 0.0 <= result_vidal["fidelity"] <= 1.0000001
    assert fidelity_improvement > 0.0
