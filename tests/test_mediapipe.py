import pathlib
import platform

import cv2

import astrapia


PATH = pathlib.Path(__file__).resolve().parent
(PATH / "results").mkdir(parents=True, exist_ok=True)


def test_short():
    r"""Test mediapipe.mesh."""
    if platform.system() == "Darwin":
        astrapia.utils.timer.ENABLE_TIMER = True
        process = astrapia.examples.mediapipe.detector_mesh.Process(version="short", do_mesh=True, engine="coreml")
        astrapia.utils.timer.ENABLE_TIMER = False
        for _ in range(10):  # warmup
            process(astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg"))

        astrapia.utils.timer.ENABLE_TIMER = True
        # just short
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process.detector(itensor)
        assert len(itensor.detections) == 1
        assert itensor.detections[0].points.shape[0] == 6
        image = itensor.detections[0].aligned_face(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_short_aligned_coreml.webp"), image[..., ::-1])

        # short + mesh
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1
        assert itensor.detections[0].points.shape[0] == 468
        image = itensor.detections[0].aligned_face(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_short_mesh_aligned_coreml.webp"), image[..., ::-1])
        astrapia.utils.timer.ENABLE_TIMER = False

    # onnxruntime
    astrapia.utils.timer.ENABLE_TIMER = True
    process = astrapia.examples.mediapipe.detector_mesh.Process(version="short", do_mesh=True, engine="onnxruntime")
    astrapia.utils.timer.ENABLE_TIMER = False
    for _ in range(10):  # warmup
        process(astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg"))

    astrapia.utils.timer.ENABLE_TIMER = True
    # just short
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process.detector(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 6
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_aligned.webp"), image[..., ::-1])

    # short + mesh
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_mesh_aligned.webp"), image[..., ::-1])
    itensor.save_detections(PATH / "results/sample_short_mesh.yaml")
    astrapia.utils.timer.ENABLE_TIMER = False


def test_long():
    r"""Test mediapipe.mesh."""
    if platform.system() == "Darwin":
        astrapia.utils.timer.ENABLE_TIMER = True
        process = astrapia.examples.mediapipe.detector_mesh.Process(version="long", do_mesh=False, engine="coreml")
        astrapia.utils.timer.ENABLE_TIMER = False
        for _ in range(10):  # warmup
            process(astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg"))

        astrapia.utils.timer.ENABLE_TIMER = True
        # just long
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1
        assert itensor.detections[0].points.shape[0] == 6
        image = itensor.detections[0].aligned_face(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_long_aligned_coreml.webp"), image[..., ::-1])

    # onnxruntime
    astrapia.utils.timer.ENABLE_TIMER = True
    process = astrapia.examples.mediapipe.detector_mesh.Process(version="long", do_mesh=True, engine="onnxruntime")
    astrapia.utils.timer.ENABLE_TIMER = False
    for _ in range(10):  # warmup
        process(astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg"))

    astrapia.utils.timer.ENABLE_TIMER = True
    # just long
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process.detector(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 6
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_long_aligned.webp"), image[..., ::-1])

    # long + mesh
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_long_mesh_aligned.webp"), image[..., ::-1])
    itensor.save_detections(PATH / "results/sample_long_mesh.yaml")
    astrapia.utils.timer.ENABLE_TIMER = False


def test_short_2x():
    r"""Test mediapipe.mesh."""
    # onnxruntime
    process = astrapia.examples.mediapipe.detector_mesh.Process(version="short", do_mesh=True, engine="onnxruntime")
    process.add_extra_size((1024, 1024))

    # short + mesh
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_mesh_aligned_2x.webp"), image[..., ::-1])


def test_long_2x():
    r"""Test mediapipe.mesh."""
    # onnxruntime
    process = astrapia.examples.mediapipe.detector_mesh.Process(version="short", do_mesh=True, engine="onnxruntime")
    process.add_extra_size((1024, 1024))

    # short + mesh
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468
    image = itensor.detections[0].aligned_face(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_long_mesh_aligned_2x.webp"), image[..., ::-1])
