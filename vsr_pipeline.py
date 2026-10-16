"""Visual speech recognition pipeline shared by the inference scripts.

Same steps as tutorials/inference.ipynb (video modality), using the mediapipe
detector so it runs on CPU-only machines.
"""

import argparse
import time

import torch
import torchvision

from datamodule.transforms import VideoTransform
from lightning import ModelModule, get_beam_search_decoder
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

    def preprocess(self, frames):
        """frames: uint8 RGB array of shape T x H x W x 3, sampled at 25 fps."""
        landmarks = self.landmarks_detector(frames)
        video = self.video_process(frames, landmarks)
        if video is None:
            raise RuntimeError("No face detected in the clip.")
        video = torch.tensor(video).permute((0, 3, 1, 2))
        return self.video_transform(video)

    def transcribe(self, frames):
        video = self.preprocess(frames)
        with torch.no_grad():
            return self.modelmodule(video)

    def transcribe_nbest(self, frames, beam_size=10, nbest=10):
        """Like transcribe, but returns the top-n hypotheses with their beam scores.

        Returns (hyps, timings): hyps is a list of {"text", "score"} sorted best first,
        with duplicate texts merged; timings holds crop_ms and decode_ms.
        """
        t0 = time.perf_counter()
        video = self.preprocess(frames)
        t1 = time.perf_counter()

        model = self.modelmodule.model
        beam_search = get_beam_search_decoder(model, self.modelmodule.token_list, beam_size=beam_size)
        with torch.no_grad():
            x = model.proj_encoder(model.frontend(video.unsqueeze(0)))
            enc_feat, _ = model.encoder(x, None)
            ended = beam_search(enc_feat.squeeze(0))
        t2 = time.perf_counter()

        hyps, seen = [], set()
        for h in ended:  # already sorted by score, best first
            token_ids = torch.tensor(list(map(int, h.yseq[1:])))
            text = self.modelmodule.text_transform.post_process(token_ids).replace("<eos>", "")
            if text in seen:  # different token splits can spell the same text
                continue
            seen.add(text)
            hyps.append({"text": text, "score": round(float(h.score), 4)})
            if len(hyps) == nbest:
                break
        return hyps, {"crop_ms": round((t1 - t0) * 1000), "decode_ms": round((t2 - t1) * 1000)}

    def transcribe_file(self, video_path):
        frames = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
        return self.transcribe(frames)
