from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import Literal

from bidict import bidict
import panel as pn

if TYPE_CHECKING:
    from ..live_plot import AperturePlot
    from .results_manager import ResultRow, ResultsManager, WindowRow
    from ..windows.base import UIWindow


class ResultTracker:
    def __init__(self, manager: ResultsManager):
        self.manager = manager
        self._auto_update = True
        self._layout: pn.layout.ListPanel | None = None
        self._latest_key: Literal['Latest'] = 'Latest'

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

    def on_window_removed(self, window: UIWindow):
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

        self._window_options: bidict[str, WindowRow | None] = bidict({})
        self._result_options: bidict[str, ResultRow] = bidict({})
        self.window_select_box = pn.widgets.Select(
            options=list(self._window_options.keys()),
            width=200,
            max_width=300,
            width_policy='min',
            align='center',
            # Top, Right, Bottom, Left
            margin=(5, 2, 5, 2),
        )
        self.result_select_box = pn.widgets.Select(
            options=list(self._result_options.keys()),
            width=150,
            max_width=300,
            width_policy='min',
            align='center',
            margin=(5, 2, 5, 2),
        )
        self.window_select_box.param.watch(self.update_result_select, 'value')
        self.load_btn = pn.widgets.Button(
            name='Load',
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
        return f'{result.run_id}: {result.name} [{result.result_id}]'

    def _sorted_result_names(self) -> list[str]:
        options = list(reversed(sorted(
            self._result_options.keys(),
            key=lambda x: self._result_options[x].run_id
        )))
        if len(options):
            options.insert(0, self._latest_key)
        return options

    def _select_window_name(self, window: WindowRow):
        return f'{window.window_name} [{window.window_id}]'

    def _sorted_window_names(self) -> list[str]:
        return list(reversed(self._window_options.keys()))

    def initialize(self):
        if len(self.manager.all_results):
            self.on_results_registered(
                *self.manager.all_results
            )

    def components(self) -> tuple[pn.widgets.Widget, ...]:
        return (
            self.window_select_box,
            self.result_select_box,
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

        new_windows = tuple(
            window for r in new_results
            if (window := self.manager.get_window(r.window_id)) is not None
            and window not in self._window_options.values()
        )
        for w in new_windows:
            if w in self._window_options.inverse.keys():
                # Mitigation for bidict assert error if replacing
                # string key with same WindowRow (due to duplication)
                continue
            self._window_options[self._select_window_name(w)] = w
        if new_windows:
            self.window_select_box.options = self._sorted_window_names()
        if not self.window_select_box.value:
            self.window_select_box.value = self.window_select_box.options[0]

        self.update_result_select(new_results=new_results)

    def update_result_select(self, *e, new_results: tuple[ResultRow] | None = None):
        current_window = self._window_options[self.window_select_box.value]
        if current_window is not None:
            current_window = (current_window,)
        results = tuple(
            self.manager.yield_with_tag(*self.tags, from_windows=current_window)
        )
        self._result_options = bidict({
            self._select_result_name(r): r
            for r in results
        })
        self.result_select_box.options = self._sorted_result_names()

        if self.plot.displayed is None:
            # Initialize plot from first correctly-tagged result
            self.result_select_box.value = self.result_select_box.options[0]
            self.load_image()
        elif isinstance(self.plot.displayed, float):
            # This is a timestamp, could come from
            # a self-generated result or liveplot and
            # shouldn't normally be overwritten
            pass
        elif new_results is None:
            # This is a manually triggered window change
            # If auto refresh is on should update the window
            self.result_select_box.value = self.result_select_box.options[0]
            if self.refresh_cbox.value:
                self.load_image()
        elif new_results is not None and self.refresh_cbox.value:
            # we have new results, check if any match the current selection
            show_latest = self.result_select_box.value == self._latest_key
            current = self.plot.displayed
            try:
                possible_results = tuple(
                    r for r in new_results
                    if (r.window_id == current.window_id)
                    and (show_latest or (r.name == current.name))
                )
            except AttributeError:
                # current display is not a ResultRow, don't overwrite it
                possible_results = None
            if possible_results:
                if show_latest:
                    self.result_select_box.value = self._latest_key
                else:
                    # Take the first, result names are supposed to be unique!
                    self.result_select_box.value = self._result_options.inverse[possible_results[0]]
                self.load_image()

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        popped = False
        for r in results:
            result = self._result_options.inverse.pop(r, None)
            popped = popped or result is not None
        if popped:
            self.result_select_box.options = self._sorted_result_names()

    def load_image(self, *e):
        selected: str = self.result_select_box.value
        if not selected:
            return
        elif selected == self._latest_key:
            # This works because the results are sorted
            # in most-recent first order and we insert 'Latest'
            # at position 0
            try:
                result_row = self._result_options.get(
                    self.result_select_box.options[1],
                    None
                )
            except IndexError:
                result_row = None
        else:
            result_row = self._result_options.get(selected, None)
        if result_row is None:
            return
        rc = self.manager.get_result_container(result_row.result_id)
        if rc is None:
            return
        self.plot.im.update(rc.data)
        self.plot.displayed = result_row
        title_suffix = ''
        if (window := self.manager.get_window(result_row.window_id)) is not None:
            title_suffix = f' from {window.window_name}'
        self.plot.fig.title.text = (
            f'{rc.title} [{result_row.result_id}]{title_suffix}'
        )
        pn.io.notebook.push_notebook(self.plot.pane)
