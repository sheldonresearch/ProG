{{ fullname.split('.')[-1].replace('.', '') }}
{{ '=' * (fullname.split('.')[-1].replace('.', '')|length) }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members: update, compute, reset
