from video2world.colmap_runner import feature_extractor_args, matcher_args


def test_feature_extractor_args_disable_gpu_for_headless_macos_colmap():
    args = feature_extractor_args(
        database_path="database.db",
        image_path="frames",
        camera_model="SIMPLE_PINHOLE",
    )

    assert "--FeatureExtraction.use_gpu" in args
    assert args[args.index("--FeatureExtraction.use_gpu") + 1] == "0"
    assert "--ImageReader.single_camera" in args
    assert args[args.index("--ImageReader.single_camera") + 1] == "1"


def test_matcher_args_disable_gpu_for_headless_macos_colmap():
    args = matcher_args("sequential", database_path="database.db")

    assert args[0] == "sequential_matcher"
    assert "--FeatureMatching.use_gpu" in args
    assert args[args.index("--FeatureMatching.use_gpu") + 1] == "0"
