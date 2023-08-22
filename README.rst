UDFs UI
=======

Intended to support the `LiberTEM <https://github.com/LiberTEM/LiberTEM/>`_
library for electron microscopy data analysis.

Repo for UDF UI prototype which is split from prior work on
:code:`aperture` which was a more complete image toolkit.


In a Notebook (see :code:`prototypes/LivePlotMulti.ipynb`):

.. code-block:: python

    from aperture.udfs_ui.ui_context import UIContext

    # Create the UIContext object
    # This orchestrates:
    #  - creation of analysis windows
    #  - providing work to the LiberTEM Context
    #  - managing and saving results from analyses
    ui_context = UIContext().for_offline(ctx, ds)
    # ui_context = UIContext().for_live(live_ctx, partial(get_aq, conn))
    
    # In separate cells:
    # Display web-terminal log window
    ui_context.log_window()
    # Display main window for running analyses
    ui_context.layout()
    # Display table / dataframe-based results manager
    ui_context.results_manager.layout()
