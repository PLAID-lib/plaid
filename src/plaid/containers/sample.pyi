"""Type stub for `plaid.containers.sample`.

This stub exists purely so that static type checkers and IDEs see the public
methods that ``Sample`` receives at runtime through
``@delegate_methods("features", SampleFeatures)``. By inheriting from
``SampleFeatures`` here (only in the ``.pyi`` file, never at runtime) the
delegated methods become visible to autocompletion without confusing
documentation tools (mkdocstrings/griffe), which read ``.py`` files only.
"""

from .features import SampleFeatures
from .sample import Sample as _Sample

class Sample(_Sample, SampleFeatures):  # type: ignore[misc]
    ...
