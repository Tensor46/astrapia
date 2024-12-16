import pathlib
import platform

import cv2

import astrapia
import astrapia.geometry


PATH = pathlib.Path(__file__).resolve().parent
(PATH / "results").mkdir(parents=True, exist_ok=True)


def test_short_coreml():
    r"""Test mediapipe.detector."""
    if platform.system() == "Darwin":
        process = astrapia.examples.mediapipe.detector.Process.load_short(engine="coreml")
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1

        image = itensor.detections[0].aligned(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_short_coreml_aligned.webp"), image[..., ::-1])
        image = itensor.detections[0].annotate(itensor.tensor)
        cv2.imwrite(str(PATH / "results/sample_short_coreml_annotated.webp"), image[..., ::-1])


def test_short_coreml_2X():
    r"""Test mediapipe.detector."""
    if platform.system() == "Darwin":
        process = astrapia.examples.mediapipe.detector.Process.load_short(engine="coreml")
        process.specs.extra_sizes.append((1024, 1024))

        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1

        image = itensor.detections[0].aligned(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_short_coreml_2x_aligned.webp"), image[..., ::-1])


def test_short_ort():
    r"""Test mediapipe.detector."""
    process = astrapia.examples.mediapipe.detector.Process.load_short(engine="onnxruntime")
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1

    image = itensor.detections[0].aligned(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_ort_aligned.webp"), image[..., ::-1])

    # save detection
    itensor.save_detections(PATH / "results/sample_short_ort.yaml")
    itensor_ = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    itensor_.load_detections(PATH / "results/sample_short_ort.yaml")
    assert itensor.detections[0].iod == itensor_.detections[0].iod


def test_short_ort_2X():
    r"""Test mediapipe.detector."""
    process = astrapia.examples.mediapipe.detector.Process.load_short(engine="onnxruntime")
    process.specs.extra_sizes.append((1024, 1024))

    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    assert len(itensor.detections) == 1

    image = itensor.detections[0].aligned(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_ort_2x_aligned.webp"), image[..., ::-1])


def test_long_coreml():
    r"""Test mediapipe.detector."""
    if platform.system() == "Darwin":
        process = astrapia.examples.mediapipe.detector.Process.load_long(engine="coreml")
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1

        image = itensor.detections[0].aligned(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_long_coreml_aligned.webp"), image[..., ::-1])


def test_long_ort():
    r"""Test mediapipe.detector."""
    if platform.system() == "Darwin":
        process = astrapia.examples.mediapipe.detector.Process.load_long(engine="onnxruntime")
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        assert len(itensor.detections) == 1

        image = itensor.detections[0].aligned(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_long_ort_aligned.webp"), image[..., ::-1])


def test_short_mesh_coreml():
    r"""Test mediapipe.mesh."""
    if platform.system() == "Darwin":
        process = astrapia.examples.mediapipe.detector.Process.load_short(engine="coreml")
        process_mesh = astrapia.examples.mediapipe.mesh.Process.load_mesh(engine="coreml")
        itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
        process(itensor)
        process_mesh(itensor)

        assert len(itensor.detections) == 1
        assert itensor.detections[0].points.shape[0] == 468

        image = itensor.detections[0].aligned(itensor.tensor, 128)
        cv2.imwrite(str(PATH / "results/sample_short_mesh_coreml_aligned.webp"), image[..., ::-1])


def test_short_mesh_ort():
    r"""Test mediapipe.mesh."""
    process = astrapia.examples.mediapipe.detector.Process.load_short(engine="onnxruntime")
    process_mesh = astrapia.examples.mediapipe.mesh.Process.load_mesh(engine="onnxruntime")
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)
    process_mesh(itensor)

    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468

    image = itensor.detections[0].aligned(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_short_mesh_ort_aligned.webp"), image[..., ::-1])

    # save detection
    itensor.save_detections(PATH / "results/sample_short_mesh_ort.yaml")


def test_fd_ort():
    r"""Test mediapipe.facedetector."""
    process = astrapia.examples.mediapipe.detector_mesh.Process(
        version="short",
        do_mesh=True,
        engine="onnxruntime",
        extra_sizes=[(1280, 1280)],
    )
    itensor = astrapia.data.ImageTensor(tensor=PATH / "sample/sample.jpeg")
    process(itensor)

    assert len(itensor.detections) == 1
    assert itensor.detections[0].points.shape[0] == 468

    image = itensor.detections[0].aligned(itensor.tensor, 128)
    cv2.imwrite(str(PATH / "results/sample_fd_ort_aligned.webp"), image[..., ::-1])
