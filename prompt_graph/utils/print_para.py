from .logging import get_logger

logger = get_logger(__name__)


def print_model_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param
        logger.info(f"{name} has {param} parameters")
    logger.info(f"Total Parameters: {total_params}")
