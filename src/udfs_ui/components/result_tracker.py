from __future__ import annotations
from typing import TYPE_CHECKING

from bidict import bidict
import panel as pn

if TYPE_CHECKING:
    from .live_plot import AperturePlot
    from .results import ResultRow, ResultsManager


class ResultTracker:
    def __init__(self, manager: ResultsManager):
        self.manager = manager
        self._auto_update = True
        self._layout: pn.layout.ListPanel | None = None

    @property
    def auto_update(self):
        return self._auto_update

    @auto_update.setter
    def auto_update(self, val: bool):
        self._auto_update = val

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        pass

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        pass

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, lo: pn.layout.ListPanel):
        self._layout = lo

    def components(self) -> tuple[pn.widgets.Widget, ...]:
        return tuple()

    def set_visible(self, val: bool):
        if self._layout is None:
            return
        self._layout.visible = val


class ImageResultTracker(ResultTracker):
    def __init__(
        self,
        manager: ResultsManager,
        plot: AperturePlot,
        tags: tuple[str, ...],
    ):
        super().__init__(manager)
        self.plot = plot
        self.tags = tags

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
            value=True,
            align='center',
        )

    def _select_result_name(self, result: ResultRow):
        return (f'[{result.run_id}]: {result.result_id}, '
                f'{self.manager.get_window(result.window_id).window_name}')

    def initialize(self):
        if len(self.manager.all_results):
            self.on_results_registered(
                *self.manager.all_results
            )

    def components(self) -> tuple[pn.widgets.Widget, ...]:
        return (
            self.select_box,
            self.load_btn,
            self.refresh_cbox,
        )

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        new_results = tuple(
            self.manager.yield_with_tag(*self.tags, from_rows=results)
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
        if self.plot.displayed is None:
            # Initialize plot from first nav-tagged result
            self.select_box.value = self.select_box.options[0]
            self.load_image()
            return
        elif isinstance(self.plot.displayed, float):
            pass
        elif self.refresh_cbox.value:
            current = self.plot.displayed
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
        rc = self.manager.get_result_container(result_row.result_id)
        if rc is None:
            return
        self.plot.im.update(rc.data)
        self.plot.displayed = result_row
        self.plot.fig.title.text = (
            f'{rc.title} [{result_row.result_id} from {result_row.window_id}]'
        )
        pn.io.notebook.push_notebook(self.plot.pane)
        # raise NotImplementedError(
        #     'Maybe also have some kind of follow-window mode in the dropdown '
        #     'so that we can command the display of the latest result from a given window.'
        # )

    # def get_results_tracker(self, *args, auto_update: bool = True):
        # can also auto-label nav/sig shaped results and auto-subscribe
        # Maybe auto-subscribe only to windows, and not to results themselves
        # ...
