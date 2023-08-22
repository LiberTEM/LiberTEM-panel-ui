UDFs UI
=======

Window-based web GUI framework intended to support the
`LiberTEM <https://github.com/LiberTEM/LiberTEM/>`_
library for electron microscopy data analysis.

Runs LiberTEM User Defined Functions (UDFs) in interactive
windows, to simplify adjusting parameters and exploring
data.

Capable of running certain analyses on top of the
`LiberTEM-live <https://github.com/LiberTEM/LiberTEM-live/>`_
framework, allowing for interactive live acquisitions
in a Jupyter Notebook.

To get started, in a Notebook:

.. code-block:: python

    from udfs_ui.ui_context import UIContext

    # Create the UIContext object
    # This orchestrates:
    #  - creation of analysis windows
    #  - providing work to the LiberTEM Context
    #  - managing and saving results from analyses
    ui_context = UIContext.for_offline(ctx, ds)
    # In separate cells:
    # Display main window for running analyses
    ui_context.layout()
    # Display table / dataframe-based results manager
    ui_context.results_manager.layout()

see :code:`examples/` for some more complete
example use cases.
