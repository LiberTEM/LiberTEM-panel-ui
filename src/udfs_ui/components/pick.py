from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

import panel as pn
from bidict import bidict
import libertem.api as lt
from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.base import UDF, UDFResultDict

from .live_plot import AperturePlot
from .base import RunnableUIWindow, UIType, UIState, UDFWindowJob
from ..display.display_base import Cursor
from .imaging import get_initial_pos
from .result_containers import Numpy2DResultContainer

if TYPE_CHECKING:
    from libertem.common.buffers import BufferWrapper
    from libertem.io.dataset.base.tiling import DataTile
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow


class PickNoROIUDF(UDF):
    def get_result_buffers(self) -> dict[str, BufferWrapper]:
        return {
            'intensity': self.buffer(kind='sig'),
        }

    def process_tile(self, tile: DataTile):
        sl = self.meta.slice
        nav_start = sl.origin[0]
        nav_end = nav_start + sl.shape[0]  # check for +1 ??
        # Expects self.params.pick_idx to already be in ROI-reduced form
        if nav_start <= self.params.pick_idx < nav_end:
            self.results.intensity[:] += tile[self.params.pick_idx - nav_start, ...]

    def merge(self, dest, src):
        dest.intensity += src.intensity


class PickUDFWindow(RunnableUIWindow, ui_type=UIType.TOOL):
    name = 'pick_frame'
    title_md = 'PickUDF'
    can_self_run = False
    can_save = True
    pick_cls = PickUDF

    def initialize(self, dataset: lt.DataSet):
        self._udf_pick = self.pick_cls()
        roi = np.zeros(dataset.shape.nav, dtype=bool)
        roi[0, 0] = True
        self.pick_plot = AperturePlot.new(
            dataset,
            self._udf_pick,
            roi=roi,
            channel=('intensity', lambda buffer: buffer.squeeze()),
            title='Pick frame',
        )
        self._last_pick = (None, None)
        self._udf_plots = [self.pick_plot]

        self._udf_sumsig = SumSigUDF()
        self.nav_plot = AperturePlot.new(
            dataset,
            self._udf_sumsig,
            title='Scan grid',
        )
        # Records the fact we initialised from a zeros-array
        self._nav_plot_displayed: str | None = None

        (ny, nx), _, _ = get_initial_pos(dataset.shape.nav)
        self._nav_cursor = Cursor.new().from_pos(nx, ny)
        self._nav_cursor.on(self.nav_plot.fig)
        self._nav_cursor.make_editable()
        self._nav_cursor.cds.on_change('data', self._run_pick)
        self.nav_plot.fig.toolbar.active_drag = self.nav_plot.fig.tools[-1]

        nav_divider = pn.pane.HTML(
            R"""<div></div>""",
            styles={
                'border-left': '2px solid #757575',
                'height': '35px',
            }
        )
        nav_select_text = pn.widgets.StaticText(
            value='Nav display:',
            align='center',
            margin=(5, 5, 5, 5),
        )
        self._nav_select_options: bidict[str, ResultRow] = bidict({})
        self.nav_select_box = pn.widgets.Select(
            options=list(self._nav_select_options.keys()),
            width=200,
            max_width=300,
            width_policy='min',
            align='center',
        )
        self.nav_load_btn = pn.widgets.Button(
            name='Load image',
            align='center',
            button_type='primary',
        )
        self.nav_load_btn.on_click(self.load_nav_image)
        self.nav_refresh_cbox = pn.widgets.Checkbox(
            name='Auto-refresh',
            value=True,
            align='center',
        )

        self._header_layout.extend((
            nav_divider,
            nav_select_text,
            self.nav_select_box,
            self.nav_load_btn,
            self.nav_refresh_cbox,
        ))

        self.toolbox = pn.Column()
        self.inner_layout.extend((
            pn.Column(
                self.nav_plot.pane,
                self.toolbox,
            ),
            pn.Column(
                self.pick_plot.pane
            )
        ))

        if self.can_save:
            self.add_save()

        return self

    def add_save(self):
        self._save_btn = pn.widgets.Button(
            name='Save frame',
            button_type='primary',
            max_width=150,
            align='start',
            disabled=True,
        )

        def _save_frame(e):
            channel = 'intensity'
            res, params = self._last_pick
            if res is None:
                return
            frame = res[channel].data.squeeze(axis=0)
            run_row = self.results_manager.new_run()
            window_row = self.results_manager.new_window_run(self, run_row, params)
            rc = Numpy2DResultContainer(channel, frame)
            self.results_manager.new_result(rc, run_row, window_row)

        self._save_btn.on_click(_save_frame)
        self.toolbox.append(self._save_btn)

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):

        cx = int(self._nav_cursor.cds.data['cx'][0])
        cy = int(self._nav_cursor.cds.data['cy'][0])
        pick_idx = np.ravel_multi_index(([cy], [cx]), dataset.shape.nav).item()
        udfs = []
        plots = []
        params = {
            'cx': cx,
            'cy': cy,
        }

        if roi is not None:
            if roi[cy, cx]:
                # This is quite untested!
                pick_idx = np.cumsum(roi.ravel())[pick_idx] - 1
            else:
                pick_idx = None

        if pick_idx is not None:
            udfs.append(PickNoROIUDF(pick_idx=pick_idx))
            plots.append(self.pick_plot)
            self.pick_plot.udf = udfs[-1]
            if self.can_save:
                self._save_btn.disabled = True

        return UDFWindowJob(self, udfs, plots, params=params)

    @staticmethod
    def _should_pick(ds: lt.DataSet, data: dict):
        try:
            x = int(data['cx'][0])
            y = int(data['cy'][0])
        except (KeyError, IndexError):
            return False
        h, w = ds.shape.nav
        if not ((0 <= x < w) and (0 <= y < h)):
            return False
        return (x, y)

    def _run_pick(self, attr, old, new):
        ui_state = self._ui_context._state
        if ui_state == UIState.LIVE:
            return

        ctx = self._ui_context._resources.get_ctx(ui_state)
        ds = self._ui_context._resources.get_ds_for_run(
            ui_state,
            self._ui_context.current_ds_ident
        )
        if ds is None:
            return

        coords = self._should_pick(ds, new)
        if not coords:
            return
        x, y = coords
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[y, x] = True
        # progress causes notable lag
        self.pick_plot.udf = self._udf_pick
        res = ctx.run_udf(
            dataset=ds,
            udf=self._udf_pick,
            roi=roi,
            sync=True,
            plots=self._udf_plots,
            progress=False
        )
        self._last_pick = (res, {'cx': x, 'cy': y})
        if self.can_save:
            self._save_btn.disabled = False

    def complete_job(
        self,
        run_id: str,
        job: UDFWindowJob,
        results: tuple[UDFResultDict],
        damage: BufferWrapper | None = None
    ) -> tuple[ResultRow, ...]:
        if not job.udfs:
            return tuple()
        window_row = self.results_manager.new_window_run(self, run_id, params=job.params)
        image: np.ndarray = results[0]['intensity'].data
        rc = Numpy2DResultContainer('intensity', image)
        result = self.results_manager.new_result(rc, run_id, window_row.window_id)
        return (result,)

    def _select_result_name(self, result: ResultRow):
        return (f'[{result.run_id}]: {result.result_id}, '
                f'{self.results_manager.get_window(result.window_id).window_name}')

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        new_nav_results = tuple(
            self.results_manager.yield_with_tag('nav', from_rows=results)
        )
        if not new_nav_results:
            return
        self._nav_select_options.update({
            self._select_result_name(r): r
            for r in new_nav_results
        })
        self.nav_select_box.options = list(reversed(sorted(
            self._nav_select_options.keys(),
            key=lambda x: self._nav_select_options[x].run_id
        )))
        if self._nav_plot_displayed is None:
            # Initialize plot from first nav-tagged result
            self.nav_select_box.value = self.nav_select_box.options[0]
            self.load_image()
            return
        if self.nav_refresh_cbox.value:
            current_display_id = self._nav_plot_displayed
            current = self.results_manager.get_result_row(current_display_id)
            if current is None:
                # Result must have been deleted
                return
            possible_results = tuple(
                r for r in new_nav_results
                if (r.window_id == current.window_id) and (r.name == current.name)
            )
            if not possible_results:
                return
            # Take the first, result names are supposed to be unique!
            self.nav_select_box.value = self._nav_select_options.inverse[possible_results[0]]
            self.load_image()

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        for r in results:
            _ = self._nav_select_options.inverse.pop(r, None)
        self.nav_select_box.options = list(self._nav_select_options.keys())
        self.logger.info(self.nav_select_box.options)

    def load_image(self, *e):
        selected: str = self.nav_select_box.value
        if not selected:
            return
        result_row = self._nav_select_options.get(selected, None)
        if result_row is None:
            return
        rc = self.results_manager.get_result_container(result_row.result_id)
        if rc is None:
            return
        self.nav_plot.im.update(rc.data)
        self.nav_plot.fig.title.text = f'Scan grid - {result_row.result_id}'
        self._nav_plot_displayed = result_row.result_id
        pn.io.notebook.push_notebook(self.nav_plot.pane)
