"""Visual speech recognition pipeline shared by the inference scripts.

Same steps as tutorials/inference.ipynb (video modality), using the mediapipe
detector so it runs on CPU-only machines.
"""

import argparse
import os
import time

# The 3D max-pool in the visual frontend has no MPS kernel; run that one op on the CPU.
# Must be set before torch is imported (scripts that import torch first set it themselves).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import torchvision

from datamodule.transforms import VideoTransform
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from lightning import ModelModule, get_beam_search_decoder
from preparation.detectors.mediapipe.detector import LandmarksDetector
from preparation.detectors.mediapipe.video_process import VideoProcess

# The models were trained on 25 fps video (LRS3 / VoxCeleb2).
MODEL_FPS = 25


class VSRPipeline:
    def __init__(self, model_path, device="cpu"):
        """device: "cpu" or "mps" (mps is only tested for encode() / score_phrases())."""
        self.device = torch.device(device)
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")

        self.modelmodule = ModelModule(argparse.Namespace(modality="video"))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()
        self.modelmodule.model.to(self.device)

    def preprocess(self, frames):
        """frames: uint8 RGB array of shape T x H x W x 3, sampled at 25 fps."""
        landmarks = self.landmarks_detector(frames)
        video = self.video_process(frames, landmarks)
        if video is None:
            raise RuntimeError("No face detected in the clip.")
        video = torch.tensor(video).permute((0, 3, 1, 2))
        return self.video_transform(video)

    def transcribe(self, frames):
        video = self.preprocess(frames).to(self.device)
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
            x = model.proj_encoder(model.frontend(video.unsqueeze(0).to(self.device)))
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

    def encode(self, frames):
        """Crop the mouth and run the visual encoder once. Returns 1 x T x D features."""
        video = self.preprocess(frames)
        model = self.modelmodule.model
        with torch.no_grad():
            x = model.proj_encoder(model.frontend(video.unsqueeze(0).to(self.device)))
            enc_feat, _ = model.encoder(x, None)
        return enc_feat

    def tokenize(self, text):
        return self.modelmodule.text_transform.tokenize(text.upper())

    def score_phrases(self, enc_feat, token_seqs, batch_size=128):
        """Score given sentences against the video instead of decoding freely (teacher forcing).

        The same two log-probabilities the model is trained on: the attention decoder's
        log P(tokens + <eos>) and the CTC log P(tokens). Returns (att_logp, ctc_logp) lists.
        """
        model = self.modelmodule.model
        ctc_logp_frames = model.ctc.log_softmax(enc_feat).cpu()  # 1 x T x V; CTC is cheap, keep it on the CPU
        n_frames = enc_feat.size(1)
        att_out, ctc_out = [], []
        with torch.no_grad():
            for i in range(0, len(token_seqs), batch_size):
                ys = token_seqs[i : i + batch_size]
                b = len(ys)
                ys_dev = [y.to(self.device) for y in ys]
                sos = eos = model.eos
                ys_in = pad_list([torch.cat([y.new([sos]), y]) for y in ys_dev], eos)
                ys_out = pad_list([torch.cat([y, y.new([eos])]) for y in ys_dev], -1)
                logits, _ = model.decoder(ys_in, target_mask(ys_in, -1), enc_feat.expand(b, -1, -1), None)
                logp = logits.log_softmax(-1)
                valid = ys_out != -1
                tok_logp = logp.gather(-1, ys_out.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                att_out += (tok_logp * valid).sum(-1).cpu().tolist()

                ctc_loss = torch.nn.functional.ctc_loss(
                    ctc_logp_frames.expand(b, -1, -1).transpose(0, 1),
                    torch.cat(ys),
                    torch.full((b,), n_frames, dtype=torch.long),
                    torch.tensor([len(y) for y in ys]),
                    blank=0, reduction="none",
                )
                ctc_out += (-ctc_loss).tolist()  # -inf when the sentence is too long for the clip
        return att_out, ctc_out

