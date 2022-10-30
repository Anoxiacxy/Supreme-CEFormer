from .basic_stem import BasicStem


class Alternative1Stem(BasicStem):
    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 21},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 21, "out_channels": 52},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 52, "out_channels": 104},
            {"kernel_size": (2, 2), "stride": 2, "padding": 0, "in_channels": 104, "out_channels": 208},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 208, "out_channels": 128},
        ]
        super(Alternative1Stem, self).__init__(in_channels, out_channels)


class Alternative2Stem(BasicStem):
    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 16},
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 16, "out_channels": 32},
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 32, "out_channels": 64},
            {"kernel_size": (4, 4), "stride": 4, "padding": 0, "in_channels": 64, "out_channels": 128},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 128, "out_channels": 128},
        ]
        super(Alternative2Stem, self).__init__(in_channels, out_channels)


class Alternative3Stem(BasicStem):
    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 2, "padding": 1, "in_channels": 3, "out_channels": 9},
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 9, "out_channels": 17},
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 17, "out_channels": 34},
            {"kernel_size": (8, 8), "stride": 8, "padding": 0, "in_channels": 34, "out_channels": 68},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 68, "out_channels": 128},
        ]
        super(Alternative3Stem, self).__init__(in_channels, out_channels)


class Alternative4Stem(BasicStem):
    def __init__(self, in_channels=3, out_channels=128):
        self.conv_params = [
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 3, "out_channels": 4},
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 4, "out_channels": 8},
            {"kernel_size": (3, 3), "stride": 1, "padding": 1, "in_channels": 8, "out_channels": 16},
            {"kernel_size": (16, 16), "stride": 16, "padding": 0, "in_channels": 16, "out_channels": 32},
            {"kernel_size": (1, 1), "stride": 1, "padding": 0, "in_channels": 32, "out_channels": 128},
        ]
        super(Alternative4Stem, self).__init__(in_channels, out_channels)
