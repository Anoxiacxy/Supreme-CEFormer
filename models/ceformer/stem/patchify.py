from .basic_stem import BasicStem


class PatchifyStem(BasicStem):

    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": 16, "stride": 16, "padding": 0, "in_channels": 3, "out_channels": 128},
        ]
        super(PatchifyStem, self).__init__(in_channels, out_channels)

