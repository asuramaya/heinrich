import numpy as np
from heinrich.inspect.matrix import (
    analyze_matrix, sparsity_signals, norm_signals, entropy_signals,
    symmetry_signals, discrete_signals, connected_components, diff_matrices, _is_discrete,
)

def test_analyze_matrix_basic():
    m = np.eye(4)
    signals = analyze_matrix(m, label="test", name="identity")
    kinds = {s.kind for s in signals}
    assert "matrix_rows" in kinds
    assert "matrix_sparsity" in kinds
    assert "matrix_fro_norm" in kinds

def test_analyze_matrix_square_has_symmetry():
    m = np.eye(4)
    signals = analyze_matrix(m, label="test", name="identity")
    kinds = {s.kind for s in signals}
    assert "matrix_transpose_symmetry" in kinds

def test_analyze_matrix_nonsquare_no_symmetry():
    m = np.ones((3, 5))
    signals = analyze_matrix(m, label="test", name="rect")
    kinds = {s.kind for s in signals}
    assert "matrix_transpose_symmetry" not in kinds

def test_analyze_matrix_discrete_has_histogram():
    m = np.array([[0, 1, 1], [2, 0, 1]])
    signals = analyze_matrix(m, label="test", name="grid")
    kinds = {s.kind for s in signals}
    assert "matrix_unique_values" in kinds
    assert "matrix_value_count" in kinds

def test_sparsity_identity():
    signals = sparsity_signals(np.eye(4), label="t", name="n")
    assert signals[0].value == 0.75  # 12 of 16 are zero

def test_sparsity_dense():
    signals = sparsity_signals(np.ones((3, 3)), label="t", name="n")
    assert signals[0].value == 0.0

def test_norm_signals():
    m = np.ones((4, 4))
    signals = norm_signals(m, label="t", name="n")
    fro = [s for s in signals if s.kind == "matrix_fro_norm"][0]
    assert fro.value == 4.0  # sqrt(16)

def test_entropy_uniform():
    m = np.ones((4, 4)) / 4  # uniform rows
    signals = entropy_signals(m, label="t", name="n")
    mean_ent = [s for s in signals if s.kind == "matrix_mean_row_entropy"][0]
    assert mean_ent.value > 1.9  # close to log2(4) = 2

def test_symmetry_symmetric():
    m = np.array([[1, 2], [2, 1]], dtype=float)
    signals = symmetry_signals(m, label="t", name="n")
    trans = [s for s in signals if s.kind == "matrix_transpose_symmetry"][0]
    assert trans.value > 0.99

def test_symmetry_asymmetric():
    m = np.array([[1, 0], [5, 1]], dtype=float)
    signals = symmetry_signals(m, label="t", name="n")
    trans = [s for s in signals if s.kind == "matrix_transpose_symmetry"][0]
    assert trans.value < 0.9

def test_discrete_histogram():
    m = np.array([[0, 1, 1], [2, 0, 0]])
    signals = discrete_signals(m, label="t", name="n")
    unique = [s for s in signals if s.kind == "matrix_unique_values"][0]
    assert unique.value == 3.0  # values 0, 1, 2
    counts = [s for s in signals if s.kind == "matrix_value_count"]
    assert len(counts) == 3

def test_connected_components_simple():
    m = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 2]])
    comps = connected_components(m)
    assert len(comps) >= 3  # at least: 1-region, 0-region, 2-region
    sizes = {c["value"]: c["size"] for c in comps}
    assert sizes[1] == 3

def test_diff_matrices_identical():
    m = np.eye(4)
    signals = diff_matrices(m, m)
    norm = [s for s in signals if s.kind == "matrix_delta_norm"][0]
    assert norm.value == 0.0

def test_diff_matrices_different():
    a = np.zeros((3, 3))
    b = np.ones((3, 3))
    signals = diff_matrices(a, b)
    changed = [s for s in signals if s.kind == "matrix_cells_changed"][0]
    assert changed.value == 9.0

def test_diff_matrices_shape_mismatch():
    signals = diff_matrices(np.zeros((2, 3)), np.zeros((3, 2)))
    assert signals[0].kind == "matrix_shape_mismatch"

def test_is_discrete():
    assert _is_discrete(np.array([[0, 1], [2, 3]]))
    assert _is_discrete(np.array([[0.0, 1.0, 2.0]]))
    assert not _is_discrete(np.array([[0.5, 1.5]]))
