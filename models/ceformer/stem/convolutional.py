from .basic_stem import BasicStem


class ConvolutionalMStem(BasicStem):

    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 24},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 24, "out_channels": 48},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 48, "out_channels": 96},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 96, "out_channels": 192},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 192, "out_channels": 128},
        ]
        super(ConvolutionalMStem, self).__init__(in_channels, out_channels)


class ConvolutionalSStem(BasicStem):

    def __init__(self, in_channels=3, out_channels=48):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 8},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 8, "out_channels": 16},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 16, "out_channels": 32},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 32, "out_channels": 64},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 64, "out_channels": 48},
        ]
        super(ConvolutionalSStem, self).__init__(in_channels, out_channels)


class ConvolutionalLStem(BasicStem):

    def __init__(self, in_channels=3, out_channels=384):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 48},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 48, "out_channels": 96},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 96, "out_channels": 192},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 192, "out_channels": 384},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 384, "out_channels": 384},
        ]
        super(ConvolutionalLStem, self).__init__(in_channels, out_channels)


ConvolutionalStem = ConvolutionalMStem

