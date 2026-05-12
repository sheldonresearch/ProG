"""Prompt-type strategy implementations.

Each strategy encapsulates the train/eval logic for one ``prompt_type`` value
(``'None'``, ``'GPF'``, ``'GPPT'``, etc.). Importing this package registers
all bundled strategies into ``STRATEGY_REGISTRY``.
"""
from . import none  # noqa: F401 -- import side-effect registers NoneStrategy
from . import gpf  # noqa: F401 -- import side-effect registers GPFStrategy + GPFPlusStrategy
from . import gprompt  # noqa: F401 -- import side-effect registers GpromptStrategy
from . import all_in_one  # noqa: F401 -- import side-effect registers AllInOneStrategy
