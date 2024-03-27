.. py:function:: split_fullname(fullname)

   This is a helper function that splits the fullname into module and class names.

   :param str fullname: The fullname of the class.
   :return: A tuple containing the module name and the class name.
   :rtype: tuple

   .. code-block:: python
   
      def split_fullname(fullname):
          parts = fullname.split('.')
          module = '.'.join(parts[:-1])
          classname = parts[-1]
          return module, classname

{{ split_fullname(fullname)[1].replace('.', '_') }}
{{ '=' * split_fullname(fullname)[1].replace('.', '_')|length }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :show-inheritance:
   :members:
   :inherited-members:
   :special-members: __cat_dim__, __inc__
