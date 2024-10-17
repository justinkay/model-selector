import model_selection.methods # !!! import that automatically registers all model selection strategies
from model_selection.model_selection_strategy import AVAILABLE_STRATEGIES


def get_model_selection_strategy(method_config):
    strategy_type = method_config["type"]

    if strategy_type not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Unknown model selection strategy: {strategy_type}")

    return AVAILABLE_STRATEGIES[strategy_type](method_config) # instantiate a strategy given the name and parameters
