from pathlib import Path

from video2world.pipeline import find_sparse_model_dir


def test_find_sparse_model_dir_prefers_larger_registered_model(tmp_path: Path):
    sparse = tmp_path / "sparse"
    model_0 = sparse / "0"
    model_1 = sparse / "1"
    model_0.mkdir(parents=True)
    model_1.mkdir(parents=True)
    (model_0 / "cameras.bin").write_bytes(b"camera")
    (model_0 / "images.bin").write_bytes(b"small")
    (model_0 / "points3D.bin").write_bytes(b"p")
    (model_1 / "cameras.bin").write_bytes(b"camera")
    (model_1 / "images.bin").write_bytes(b"larger-image-registration")
    (model_1 / "points3D.bin").write_bytes(b"points")

    assert find_sparse_model_dir(sparse) == model_1
