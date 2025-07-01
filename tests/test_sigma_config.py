from pysurv.adjustment import Adjustment, sigma_config


class TestSigmaConfig:
    def test_singleton(self):
        sigma_config_1 = sigma_config
        sigma_config_2 = sigma_config

        adjustment_1 = Adjustment()
        adjustment_2 = Adjustment()

        assert (
            sigma_config_1
            is sigma_config_2
            is adjustment_1.sigma_config
            is adjustment_2.sigma_config
        )
