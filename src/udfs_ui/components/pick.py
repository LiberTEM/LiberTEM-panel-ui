from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

import libertem.api as lt
import panel as pn
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
            channel=('intensity', lambda buffer: buffer.squeeze())
        )
        self.pick_plot.fig.title = 'Pick frame'
        self._last_pick = (None, None)
        self._udf_plots = [self.pick_plot]

        self._udf_sumsig = SumSigUDF()
        self.nav_plot = AperturePlot.new(
            dataset,
            self._udf_sumsig,
        )
        (ny, nx), _, _ = get_initial_pos(dataset.shape.nav)
        self._nav_cursor = Cursor.new().from_pos(nx, ny)
        self._nav_cursor.on(self.nav_plot.fig)
        self._nav_cursor.make_editable()
        self._nav_cursor.cds.on_change('data', self._run_pick)
        self.nav_plot.fig.toolbar.active_drag = self.nav_plot.fig.tools[-1]
        self.nav_plot.fig.title = 'Scan grid'

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
        udfs = [self._udf_sumsig]
        plots = [self.nav_plot]
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
        image: np.ndarray = results[1]['intensity'].data
        rc = Numpy2DResultContainer('intensity', image)
        result = self.results_manager.new_result(rc, run_id, window_row.window_id)
        return (result,)
