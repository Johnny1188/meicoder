import os
import torch
from torch import nn

from csng.comparison import load_decoder_from_ckpt
from csng.utils import crop


class Ensemble(nn.Module):
    def __init__(
        self,
        decoders=None,
        ckpt_paths=None,
        crop_win=None,
        device="cpu",
    ):
        super().__init__()
        assert ((0 if ckpt_paths is None else len(ckpt_paths)) + (0 if decoders is None else len(decoders))) > 0, \
            "Neither decoders nor ckpt_paths specified"
        self.decoders = decoders
        self.ckpt_paths = ckpt_paths
        self.crop_win = crop_win
        self.device = device

    def forward(self, resp, data_key=None, neuron_coords=None, pupil_center=None):
        stim_pred = 0
        n_preds = 0

        if self.decoders is not None:
            for decoder in self.decoders:
                n_preds += 1
                if decoder.__class__.__name__ in ("InvertedEncoder", "InvertedEncoderBrainreader"):
                    stim_pred += crop(decoder(
                        resp_target=resp,
                        stim_target=None,
                        additional_encoder_inp={
                            "data_key": data_key,
                            "pupil_center": pupil_center,
                        }
                    )[0], self.crop_win)
                else:
                    stim_pred += crop(decoder(
                        resp,
                        data_key=data_key,
                        neuron_coords=neuron_coords,
                        pupil_center=pupil_center,
                    ), self.crop_win)
        if self.ckpt_paths is not None:
            for ckpt_path in self.ckpt_paths:
                n_preds += 1
                decoder, _ = load_decoder_from_ckpt(
                    ckpt_path=ckpt_path,
                    load_best=False,
                    device=self.device,
                )
                stim_pred += crop(decoder(
                    resp,
                    data_key=data_key,
                    neuron_coords=neuron_coords,
                    pupil_center=pupil_center,
                ), self.crop_win)

        return stim_pred / n_preds
