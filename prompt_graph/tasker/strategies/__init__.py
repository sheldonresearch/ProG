"""Prompt-type strategy implementations.

Each strategy encapsulates the train/eval logic for one ``prompt_type`` value
(``'None'``, ``'GPF'``, ``'GPPT'``, etc.). Importing this package registers
all bundled strategies into ``STRATEGY_REGISTRY``.
"""

from . import (
    all_in_one,  # noqa: F401 -- import side-effect registers AllInOneStrategy
    edge_prompt,  # noqa: F401 -- import side-effect registers EdgePromptStrategy
    gpf,  # noqa: F401 -- import side-effect registers GPFStrategy + GPFPlusStrategy
    gppt,  # noqa: F401 -- import side-effect registers GPPTStrategy
    gprompt,  # noqa: F401 -- import side-effect registers GpromptStrategy
    multi_gprompt,  # noqa: F401 -- import side-effect registers MultiGpromptStrategy
    none,  # noqa: F401 -- import side-effect registers NoneStrategy
    prodigy,  # noqa: F401 -- import side-effect registers ProdigyStrategy
    self_pro,  # noqa: F401 -- import side-effect registers SelfProStrategy
    uni_prompt,  # noqa: F401 -- import side-effect registers UniPromptStrategy
)
