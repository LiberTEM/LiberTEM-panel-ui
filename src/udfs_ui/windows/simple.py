from __future__ import annotations
from typing import TYPE_CHECKING

from ..base import UIState, JobResults
from .base import UIWindow, UDFWindowJob
from ..live_plot import AperturePlot
from ..results.containers import Numpy2DResultContainer

if TYPE_CHECKING:
    import numpy as np
    from libertem.api import DataSet
    from libertem.udf.base import UDF
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from ..results.results_manager import ResultRow


class SimpleUDFUIWindow(UIWindow):
    udf_class: type[UDF] = None

    def initialize(self, dataset: DataSet):
        self._udf = self.udf_class()
        self._plot: AperturePlot = None
        self._plot = AperturePlot.new(
            dataset,
            self._udf,
            title=self.properties.title_md,
        )
        self.inner_layout.append(self._plot.layout)
        return self

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
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
            job_results.run_row.run_id,
            params=job.params,
        )
        channel = self._plot.channel
        buffer = job_results.udf_results[0][channel]
        image: np.ndarray = buffer.data
        rc = Numpy2DResultContainer(channel, image, {'tags': (buffer.kind,)})
        result = self.results_manager.new_result(
            rc, job_results.run_row.run_id, window_row.window_id
        )
        return (result,)
