from .basic_stem import BasicStem


class ConvolutionalStem(BasicStem):

    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 24},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 24, "out_channels": 48},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 48, "out_channels": 96},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 96, "out_channels": 192},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 192, "out_channels": 128},
        ]
        super(ConvolutionalStem, self).__init__(in_channels, out_channels)