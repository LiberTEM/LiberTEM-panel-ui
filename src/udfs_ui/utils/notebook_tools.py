from __future__ import annotations
from typing_extensions import Literal


def python_info() -> Literal['unknown', 'python', 'ipython', 'notebook']:
    """
    Try to infer information about the current Python interpreter environment

    By default return 'python' when we are not in an ipython context
    """
    try:
        shell = get_ipython().__class__.__name__  # noqa
        if shell == 'ZMQInteractiveShell':
            return 'notebook'   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return 'ipython'  # Terminal running IPython
        else:
            return 'unknown'  # Other type (?)
    except NameError:
        return 'python'


def is_notebook() -> bool:
    """Convenience method to check if we are in a notebook"""
    return python_info() == 'notebook'


def notebook_fullwidth():
    from IPython.display import display, HTML
    """Set notebook display to 95% fullwidth for more available space"""
    if is_notebook():
        display(HTML(data="""
        <style>
            div#notebook-container    { width: 95%; }
            div#menubar-container     { width: 65%; }
            div#maintoolbar-container { width: 99%; }
        </style>
        """))


def run_next_cells(ncells: int, ev):
    """
    Run the next N cells

    Parameters
    ----------
    ncells : int
        The number of cells to run
    """
    from IPython.display import display, Javascript
    display(Javascript('IPython.notebook.execute_cells('
                       f'[IPython.notebook.index_or_selected()+{ncells}])'))


def run_remaining_cells(ev):
    """
    Run from the next cell to the end of the notebook
    """
    from IPython.display import display, Javascript
    display(Javascript('IPython.notebook.execute_cell_range('
                       'IPython.notebook.index_or_selected()+1, '
                       'IPython.notebook.ncells())'))


def run_this(ev):
    from IPython.display import display, Javascript
    display(Javascript('IPython.notebook.execute_cell_range('
                       'IPython.notebook.index_or_selected(), '
                       'IPython.notebook.index_or_selected() + 1)'))


def run_this_to_end(ev):
    """
    Run from the current cell to the end of the notebook
    """
    from IPython.display import display, Javascript
    display(Javascript('IPython.notebook.execute_cell_range('
                       'IPython.notebook.index_or_selected(), '
                       'IPython.notebook.ncells())'))


def get_ipyw_continue_button(name='Continue'):
    """
    Get a button bound to the run_remaining_cells function and display it

    Parameters
    ----------
    name : str, optional
        The name of the button, by default 'Continue'
    """
    # Delayed import as this is the only function that uses ipywidgets
    from ipywidgets import widgets
    from IPython.display import display
    update_button = widgets.Button(description=name)
    update_button.on_click(run_remaining_cells)
    display(update_button)


def get_ipyw_reload_button(name='Reload cell'):
    """
    Get a button bound to the run_this function and display it

    Parameters
    ----------
    name : str, optional
        The name of the button, by default 'Reload cell'
    """
    # Delayed import as this is the only function that uses ipywidgets
    from ipywidgets import widgets
    from IPython.display import display
    reload_button = widgets.Button(description=name)
    reload_button.on_click(run_this)
    display(reload_button)


class StopExecution(Exception):
    """
    A silent Exception, shows no traceback
    """
    def _render_traceback_(self):
        pass


def stop_nb(with_continue: bool = True, continue_name: str = 'Continue'):
    """
    Stop notebook execution on this line and display a Continue button

    Parameters
    ----------
    with_continue : bool, optional
        Whether to display the continue button, by default True
    continue_name : str, optional
        The name to apply on the continue button, by default 'Continue'

    Raises
    ------
    StopExecution
        _description_
    """
    assert is_notebook(), 'Only possible in a notebook context'
    if with_continue:
        get_ipyw_continue_button(name=continue_name)
    raise StopExecution
