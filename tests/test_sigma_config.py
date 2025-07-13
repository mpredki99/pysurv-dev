from pysurv.adjustment import sigma_config


class TestSigmaConfig:
    def test_singleton(self):
        sigma_config_1 = sigma_config
        sigma_config_2 = sigma_config

        assert sigma_config_1 is sigma_config_2
