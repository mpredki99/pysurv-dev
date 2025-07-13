from .memory_strategy.memory_strategy import MemoryStrategy
from .speed_strategy.speed_strategy import SpeedStrategy

strategies = {"speed": SpeedStrategy, "memory_safe": MemoryStrategy}


def get_strategy(parent, name=None):
    global strategies
    name = get_strategy_name(name, parent.dataset)
    strategy_constructor = strategies[name]

    return strategy_constructor(parent)


def get_strategy_name(name, dataset):
    if name is not None:
        return _validate_strategy_name(name)

    measurements_weight = dataset.measurements.memory_usage(deep=True).sum()
    controls_weight = dataset.controls.memory_usage(deep=True).sum()
    stations_weight = dataset.stations.memory_usage(deep=True).sum()

    total_weight = measurements_weight + controls_weight + stations_weight
    return "speed" if total_weight / (1024**3) < 1 else "memory_safe"


def _validate_strategy_name(name):
    global strategies

    strategies_list = list(strategies.keys())
    if name not in strategies_list:
        raise ValueError(f"Invalid strategy. Please choose one: {strategies_list}")

    return name
