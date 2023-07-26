from __future__ import annotations
from typing import TYPE_CHECKING

from libertem.udf.sum import SumUDF
from libertem.udf.logsum import LogsumUDF
from libertem.udf.sumsigudf import SumSigUDF

from .base import UIType, UIWindow, UIState, UDFWindowJob, JobResults
from .live_plot import AperturePlot
from .result_containers import Numpy2DResultContainer

if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem.udf.base import UDF
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow


class SimpleUDFUIWindow(UIWindow):
    udf_class: type[UDF] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._udf = self.udf_class()
        self._plot: AperturePlot = None

    def initialize(self, dataset: lt.DataSet):
        self._plot = AperturePlot.new(
            dataset,
            self._udf,
            title=self.title_md,
        )
        self.inner_layout.append(self._plot.pane)
        return self

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if self._plot is None:
            raise RuntimeError('Must initialise plot with .initialize(dataset) before run')
        return UDFWindowJob(self, [self._udf], [self._plot], result_handler=self.complete_job)

    def complete_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        window_row = self.results_manager.new_window_run(
            self,
            job_results.run_id,
            params=job.params,
        )
        channel = self._plot.channel
        buffer = job_results.udf_results[0][channel]
        image: np.ndarray = buffer.data
        rc = Numpy2DResultContainer(channel, image, {'tags': (buffer.kind,)})
        result = self.results_manager.new_result(rc, job_results.run_id, window_row.window_id)
        return (result,)


class SumUDFWindow(SimpleUDFUIWindow, ui_type=UIType.ANALYSIS):
    name = 'sum_over_frames'
    title_md = 'SumUDF'
    udf_class = SumUDF


class SumSigUDFWindow(SimpleUDFUIWindow, ui_type=UIType.ANALYSIS):
    name = 'whole_frame_imaging'
    title_md = 'SumSigUDF'
    udf_class = SumSigUDF


class LogSumUDFWindow(SimpleUDFUIWindow, ui_type=UIType.ANALYSIS):
    name = 'logsum_over_frames'
    title_md = 'LogsumUDF'
    udf_class = LogsumUDF
