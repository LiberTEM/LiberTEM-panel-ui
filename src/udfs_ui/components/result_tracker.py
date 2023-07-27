from __future__ import annotations
from typing import TYPE_CHECKING

from bidict import bidict
import panel as pn

if TYPE_CHECKING:
    from .base import UIWindow
    from .live_plot import AperturePlot
    from .results import ResultRow


class ImageResultTracker:
    def __init__(
        self,
        ui_window: UIWindow,
        plot: AperturePlot,
        tags: tuple[str, ...],
        select_text: str,
        plot_initialized: bool | str = False,
        auto_refresh: bool = True,
    ):
        self.window = ui_window
        self.plot = plot
        self.tags = tags
        # Records the fact we initialised from a zeros-array
        self._plot_displayed = plot_initialized

        divider = pn.pane.HTML(
            R"""<div></div>""",
            styles={
                'border-left': '2px solid #757575',
                'height': '35px',
            }
        )
        select_text = pn.widgets.StaticText(
            value=select_text,
            align='center',
            margin=(5, 5, 5, 5),
        )
        self._select_options: bidict[str, ResultRow] = bidict({})
        self.select_box = pn.widgets.Select(
            options=list(self._select_options.keys()),
            width=200,
            max_width=300,
            width_policy='min',
            align='center',
        )
        self.load_btn = pn.widgets.Button(
            name='Load image',
            align='center',
            button_type='primary',
        )
        self.load_btn.on_click(self.load_image)
        self.refresh_cbox = pn.widgets.Checkbox(
            name='Auto-refresh',
            value=auto_refresh,
            align='center',
        )

        ui_window._header_layout.extend((
            divider,
            select_text,
            self.select_box,
            self.load_btn,
            self.refresh_cbox,
        ))

    def _select_result_name(self, result: ResultRow):
        return (f'[{result.run_id}]: {result.result_id}, '
                f'{self.window.results_manager.get_window(result.window_id).window_name}')

    def initialize(self):
        if len(self.window.results_manager.all_results):
            self.on_results_registered(
                *self.window.results_manager.all_results
            )

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        new_results = tuple(
            self.window.results_manager.yield_with_tag(*self.tags, from_rows=results)
        )
        if not new_results:
            return
        self._select_options.update({
            self._select_result_name(r): r
            for r in new_results
        })
        self.select_box.options = list(reversed(sorted(
            self._select_options.keys(),
            key=lambda x: self._select_options[x].run_id
        )))
        if not self._plot_displayed:
            # Initialize plot from first nav-tagged result
            self.select_box.value = self.select_box.options[0]
            self.load_image()
            return
        if self.refresh_cbox.value:
            current_display_id = self._plot_displayed
            current = self.window.results_manager.get_result_row(current_display_id)
            if current is None:
                # Result must have been deleted
                return
            possible_results = tuple(
                r for r in new_results
                if (r.window_id == current.window_id) and (r.name == current.name)
            )
            if not possible_results:
                return
            # Take the first, result names are supposed to be unique!
            self.select_box.value = self._select_options.inverse[possible_results[0]]
            self.load_image()

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        for r in results:
            _ = self._select_options.inverse.pop(r, None)
        self.select_box.options = list(self._select_options.keys())

    def load_image(self, *e):
        selected: str = self.select_box.value
        if not selected:
            return
        result_row = self._select_options.get(selected, None)
        if result_row is None:
            return
        rc = self.window.results_manager.get_result_container(result_row.result_id)
        if rc is None:
            return
        self.plot.im.update(rc.data)
        title = result_row.params.get('result_title', None)
        if title is None:
            title = f'{self.plot.title}'
        self.plot.fig.title.text = f'{title} [{result_row.result_id}]'
        self._plot_displayed = result_row.result_id
        pn.io.notebook.push_notebook(self.plot.pane)
