import os
import torch
from torch import nn

from csng.models.inverted_encoder import InvertedEncoder, InvertedEncoderBrainreader
# from csng.brainreader_mouse.encoder import get_encoder
from csng.utils.comparison import load_decoder_from_ckpt
from csng.utils.data import crop


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


class EnsembleInvEnc(Ensemble):
    def __init__(
        self,
        encoder_paths, # list
        encoder_config, # dict or list[dict]
        use_brainreader_encoder=True,
        **kwargs,
    ):
        assert "decoders" not in kwargs
        kwargs.pop("encoder_paths", None)
        if type(encoder_config) == dict:
            assert "encoder" not in encoder_config
        elif type(encoder_config) == list:
            assert len(encoder_config) == len(encoder_paths)
            for enc_inv_cfg in encoder_config:
                assert "encoder" not in enc_inv_cfg
        else:
            raise ValueError("`encoder_config` must be dict or list of dicts.")

        kwargs["decoders"] = []
        if use_brainreader_encoder:
            inv_enc_cls = InvertedEncoderBrainreader
        else:
            inv_enc_cls = InvertedEncoder
        for enc_idx, encoder_path in enumerate(encoder_paths):
            if type(encoder_config) == dict:
                enc_cfg = encoder_config
            else:
                enc_cfg = encoder_config[enc_idx]
            enc_cfg["encoder"] = get_encoder(
                ckpt_path=encoder_path,
                eval_mode=True,
                device=kwargs.get("device", "cpu"),
            )
            kwargs["decoders"].append(inv_enc_cls(**enc_cfg))

        super().__init__(**kwargs)

