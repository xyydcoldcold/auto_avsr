"""Visual speech recognition pipeline shared by the inference scripts.

Same steps as tutorials/inference.ipynb (video modality), using the mediapipe
detector so it runs on CPU-only machines.
"""

import argparse

import torch
import torchvision

from datamodule.transforms import VideoTransform
from lightning import ModelModule
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

# The models were trained on 25 fps video (LRS3 / VoxCeleb2).
MODEL_FPS = 25


class VSRPipeline:
    def __init__(self, model_path):
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(argparse.Namespace(modality="video"))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()

    def transcribe(self, frames):
        """frames: uint8 RGB array of shape T x H x W x 3, sampled at 25 fps."""
        landmarks = self.landmarks_detector(frames)
        video = self.video_process(frames, landmarks)
        if video is None:
            raise RuntimeError("No face detected in the clip.")
        video = torch.tensor(video).permute((0, 3, 1, 2))
        video = self.video_transform(video)
        with torch.no_grad():
            return self.modelmodule(video)

    def transcribe_file(self, video_path):
        frames = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
        return self.transcribe(frames)
