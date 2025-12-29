{% set title = fullname | escape %}
{{ title }}
{{ "=" * title|length }}

.. automodule:: {{ fullname }}
   :show-inheritance:
