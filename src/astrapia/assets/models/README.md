### CoreML and ONNX models
MediaPipe models ([MediaPipe BlazeFace Short Range](https://drive.google.com/file/d/1d4-xJP9PVzOvMBDgIjz6NhvpnlG9_i0S/preview) and [MediaPipe BlazeFace Full Range](https://drive.google.com/file/d/1jpQt8TB1nMFQ49VSSBKdNEdQOygNRvCP/preview)) are ported to CoreML and ONNX. Certain post processing steps are moved into the CoreML and ONNX models.
#### Short2X
Short2X is short model running at two resolutions (128x128 and 1024x1024). Input must be 1024x1024.
#### Long2X
Long2X is long model running at two resolutions (256x256 and 1280x1280). Input must be 1280x1280.
### LICENSE (Apache License 2.0)
All the models are licensed under the same terms as [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).
