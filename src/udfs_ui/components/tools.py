from __future__ import annotations
import uuid
import pathlib
import datetime
from typing import TYPE_CHECKING

import panel as pn

from libertem_live.udf.record import RecordUDF
from libertem_live.udf.monitor import SignalMonitorUDF
from libertem.udf.sumsigudf import SumSigUDF

from .live_plot import AperturePlot
from .base import ActivateableUIWindow, RunnableUIWindow, UIType, UIState, UDFWindowJob
from ..display.display_base import Rectangles
from .simple import SimpleUDFUIWindow
from .result_containers import RecordResultContainer

if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import BufferWrapper, UDFResultDict


class ROIWindow(ActivateableUIWindow, ui_type=UIType.TOOL):
    name = 'roi'
    title_md = 'ROI'
    is_unique = True

    def get_roi(self, dataset) -> np.ndarray | None:
        if self.is_active:
            return self._rectangles.as_mask(dataset.shape.nav)
        return None

    def initialize(self, dataset: lt.DataSet):
        udf = SumSigUDF()
        self._plot = AperturePlot.new(dataset, udf)
        self._rectangles = Rectangles.new().empty()
        self._rectangles.on(self._plot.fig)
        self._rectangles.make_editable()
        self._plot.fig.toolbar.active_drag = self._plot.fig.tools[-1]
        self.inner_layout.append(self._plot.pane)
        return self


class RecordWindow(RunnableUIWindow, ui_type=UIType.TOOL):
    name = 'record'
    title_md = 'Record'
    is_unique = True
    can_self_run = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Track previous active for a given state
        self._last_active: dict[UIState, bool] = {}

    def get_run(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if self.is_active and roi is None and (record_udf := self.get_record_udf()):
            return UDFWindowJob(self, [record_udf], [])
        return None

    def get_record_udf(self):
        try:
            file_root = pathlib.Path(self.save_dir.value)
        except (TypeError, ValueError):
            return None

        dt = datetime.datetime.now()
        fnm = (
            f'data_{dt.year:>02d}{dt.month:>02d}{dt.day:>02d}'
            f'_{dt.hour:>02d}{dt.minute:>02d}{dt.second:>02d}.npy'
        )
        filepath = file_root / fnm
        return RecordUDF(filepath)

    def initialize(self, dataset: lt.DataSet):
        self.save_dir = pn.widgets.TextInput(
            name='Save directory',
            value='.',
            min_width=200,
        )
        self.inner_layout.append(self.save_dir)
        return self

    def set_results(
        self,
        run_id: str,
        job: UDFWindowJob,
        results: tuple[UDFResultDict],
        damage: BufferWrapper | None = None
    ):
        udfs = job.udfs
        if not udfs:
            return
        record_udf: RecordUDF = udfs[0]
        filepath = record_udf.params.filename
        params = {}
        try:
            ident = f'ds-{str(uuid.uuid4())[:5]}'
            self._ui_context._resources.recordings[ident] = filepath
        except AttributeError:
            pass

        window_row = self.results_manager.new_window_run(self, run_id)
        rc = RecordResultContainer('recording', str(filepath), params=params)
        self.results_manager.new_result(rc, run_id, window_row)

    def set_state(self, old_state: UIState, new_state: UIState):
        self._last_active[old_state] = self.is_active

        if new_state == UIState.REPLAY:
            self.set_active(self._last_active.get(UIState.REPLAY, False))
        elif new_state == UIState.LIVE:
            self.set_active(self._last_active.get(UIState.LIVE, True))


class SignalMonitorUDFWindow(SimpleUDFUIWindow, ui_type=UIType.TOOL):
    name = 'frame_monitor'
    title_md = 'Monitor'
    udf_class = SignalMonitorUDF
    can_self_run = False
