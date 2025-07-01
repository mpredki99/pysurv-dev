from pysurv import config


class TestConfig:
    def test_singleton(self):
        config1 = config
        config2 = config

        assert config1 is config2
