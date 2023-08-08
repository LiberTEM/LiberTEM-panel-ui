from __future__ import annotations
import uuid
import pathlib
import datetime
from typing import TYPE_CHECKING

import numpy as np
from humanize import naturalsize

from libertem_live.udf.record import RecordUDF
from libertem_live.udf.monitor import SignalMonitorUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.common.math import prod

from .live_plot import AperturePlot
from .base import UIWindow, UIType, UIState, UDFWindowJob, JobResults
from .simple import SimpleUDFUIWindow
from .result_containers import RecordResultContainer

if TYPE_CHECKING:
    import numpy as np
    from libertem.api import DataSet
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow


class ROIWindow(UIWindow, ui_type=UIType.RESERVED):
    name = 'roi'
    title_md = 'Global ROI'
    can_self_run = False
    header_remove = False

    def get_roi(self, dataset: DataSet) -> np.ndarray | None:
        if self.is_active:
            return self._plot.get_mask(dataset.shape.nav.to_tuple())
        return None

    def initialize(self, dataset: DataSet):
        udf = SumSigUDF()
        self._plot = AperturePlot.new(dataset, udf)
        self._plot.add_mask_tools(activate=True)
        self.inner_layout.append(self._plot.pane)
        clear_btn = self._plot.get_clear_mask_btn()
        self.inner_layout.append(clear_btn)
        self.nav_plot_tracker = self.link_image_plot(
            'Nav',
            self._plot,
            tags=('nav',)
        )
        return self

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        self.nav_plot_tracker.on_results_registered(*results)

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        self.nav_plot_tracker.on_results_deleted(*results)


class RecordWindow(UIWindow, ui_type=UIType.RESERVED):
    name = 'record'
    title_md = 'Record'
    can_self_run = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def layout(self):
        return None

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if state in (UIState.OFFLINE, UIState.REPLAY):
            # The UI state should ensure we never get here
            # but early-return just in case
            return None
        if self.is_active and roi is None and (record_udf := self.get_record_udf()):
            return UDFWindowJob(
                window=self,
                udfs=[record_udf],
                plots=[],
                result_handler=self.complete_job
            )
        return None

    def get_record_udf(self):
        try:
            file_root = pathlib.Path(self._ui_context.save_root)
        except (TypeError, ValueError):
            return None

        dt = datetime.datetime.now()
        fnm = (
            f'data_{dt.year:>02d}{dt.month:>02d}{dt.day:>02d}'
            f'_{dt.hour:>02d}{dt.minute:>02d}{dt.second:>02d}.npy'
        )
        filepath = file_root.resolve() / fnm
        return RecordUDF(filepath)

    def initialize(self, dataset: DataSet):
        return self

    def complete_job(
        self,
        job: UDFWindowJob,
        results: JobResults,
    ) -> tuple[ResultRow, ...]:
        udfs = job.udfs
        if not udfs:
            return
        record_udf: RecordUDF = udfs[0]
        filepath = record_udf.params.filename
        try:
            n_px = prod(record_udf.meta.dataset_shape)
            filesize = np.dtype(record_udf.meta.input_dtype).itemsize * n_px
            self.logger.info(f'Data recorded at {filepath} [{naturalsize(filesize)}]')
        except (TypeError, AttributeError):
            # In case the the UDFMeta is not correctly set
            self.logger.info(f'Data recorded at {filepath}')

        params = {}
        try:
            ident = f'ds-{str(uuid.uuid4())[:5]}'
            self._ui_context._resources.recordings[ident] = filepath
        except AttributeError:
            pass

        window_row = self.results_manager.new_window_run(self, results.run_id)
        rc = RecordResultContainer('recording', str(filepath), meta=params)
        result = self.results_manager.new_result(rc, results.run_id, window_row)
        return (result,)


class SignalMonitorUDFWindow(SimpleUDFUIWindow, ui_type=UIType.RESERVED):
    name = 'frame_monitor'
    title_md = 'Monitor'
    udf_class = SignalMonitorUDF
    can_self_run = False
    header_remove = False
