from torch import nn


class TransparentDataParallel(nn.DataParallel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try: # DataParallel
            return super().__getattr__(name)
        except AttributeError: # underlying module as a fallback
            return getattr(self.module, name)
