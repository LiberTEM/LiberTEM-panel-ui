from __future__ import annotations
from typing import TYPE_CHECKING

import panel as pn

from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.sum import SumUDF
from libertem.udf.logsum import LogsumUDF

from .live_plot import AperturePlot
from .base import RunnableUIWindow, UIType, UIState, UDFWindowJob
from ..display.display_base import Cursor
from .result_containers import Numpy2DResultContainer

if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import BufferWrapper, UDFResultDict
    from .results import ResultRow


def get_initial_pos(shape: tuple[int, int]):
    h, w = shape
    cy, cx = h // 2, w // 2
    ri, r = h // 6, w // 4
    return tuple(map(float, (cy, cx))), tuple(map(float, (ri, r))), float(max(h, w))


class SingleImagingUDFWindow(RunnableUIWindow):
    name = 'single_imaging'
    title_md = 'Single Imaging'
    analysis_class = PointMaskAnalysis

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._imaging_udf: ApplyMasksUDF = None
        self._view_plot: AperturePlot = None
        self._imaging_plot: AperturePlot = None

        self._view_udfs = {
            'Sum': (SumUDF(), 'intensity'),
            'LogSum': (LogsumUDF(), 'logsum'),
        }
        self._view_select = pn.widgets.Select(
            name='View UDF select',
            options=list(self._view_udfs.keys()),
            value='Sum',
        )
        self._view_udf: LogsumUDF | SumUDF = self._view_udfs[self._view_select.value][0]
        self._toolbox = pn.Column()
        self.inner_layout.append(self._toolbox)
        self.toolbox.append(self._view_select)

    @property
    def toolbox(self):
        return self._toolbox

    def initialize(self, dataset: lt.DataSet):
        self._sig_plot = AperturePlot.new(dataset, self._view_udf)
        self.setup_cursor(dataset)
        mask_analysis = self.analysis_class(dataset, self._get_analysis_params())
        self._imaging_udf = mask_analysis.get_udf()
        self._imaging_plot = AperturePlot.new(dataset, self._imaging_udf)
        self.inner_layout.extend((self._sig_plot.pane, self._imaging_plot.pane))
        return self

    def _get_analysis_params(self):
        return {
            'cx': self._sig_cursor.cds.data['cx'][0],
            'cy': self._sig_cursor.cds.data['cy'][0],
        }

    def setup_cursor(self, ds):
        fig_sig = self._sig_plot.fig
        (cy, cx), (ri, r), max_dim = get_initial_pos(ds.shape.sig)
        self._sig_cursor = Cursor.new().from_pos(cx, cy)
        self._sig_cursor.on(fig_sig)
        self._sig_cursor.make_editable()
        fig_sig.toolbar.active_drag = fig_sig.tools[-1]

    def update_view_udf(self):
        self._view_udf, channel = self._view_udfs[self._view_select.value]
        self._sig_plot.udf = self._view_udf
        self._sig_plot.channel = channel

    def update_imaging_udf(self, ds):
        params = self._get_analysis_params()
        analysis = self.analysis_class(ds, params)
        self._imaging_udf = analysis.get_udf()
        self._imaging_plot.udf = self._imaging_udf
        return params

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if self._imaging_plot is None:
            raise RuntimeError('Must initialise plot with .initialize(dataset) before run')
        self.update_view_udf()
        params = self.update_imaging_udf(dataset)
        return UDFWindowJob(
            self,
            [self._view_udf, self._imaging_udf],
            [self._imaging_plot],
            params=params,
        )

    def complete_job(
        self,
        run_id: str,
        job: UDFWindowJob,
        results: tuple[UDFResultDict],
        damage: BufferWrapper | None = None
    ) -> tuple[ResultRow, ...]:
        window_row = self.results_manager.new_window_run(self, run_id, params=job.params)
        buffer = results[1]['intensity']
        image: np.ndarray = buffer.data[..., 0]
        rc = Numpy2DResultContainer('intensity', image, params={'tags': (buffer.kind,)})
        result = self.results_manager.new_result(rc, run_id, window_row.window_id)
        self._sig_plot.new_data(results[0], damage, force=True)
        return (result,)


class PointImagingWindow(SingleImagingUDFWindow, ui_type=UIType.ANALYSIS):
    name = 'point_imaging'
    title_md = 'Point Imaging'
    analysis_class = PointMaskAnalysis


class DiskImagingWindow(SingleImagingUDFWindow, ui_type=UIType.ANALYSIS):
    name = 'disk_imaging'
    title_md = 'Disk Imaging'
    analysis_class = DiskMaskAnalysis

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slider = pn.widgets.FloatSlider(
            name='Disk radius',
            value=10,
            start=0.1,
            end=100,
        )
        self.toolbox.append(self._slider)

    def _get_analysis_params(self):
        base_params = super()._get_analysis_params()
        return {**base_params, 'r': self._slider.value}

    def initialize(self, dataset: lt.DataSet):
        super().initialize(dataset)
        _, (_, r), max_dim = get_initial_pos(dataset.shape.sig)
        self._slider.end = max_dim
        self._slider.value = r
        self._sig_disk = self._sig_cursor.add_disk(r)
        self._slider.param.watch(self._update_radius, 'value')

    def _update_radius(self, e):
        self._sig_disk.update(radius=e.new)


class RingImagingWindow(SingleImagingUDFWindow, ui_type=UIType.ANALYSIS):
    name = 'ring_imaging'
    title_md = 'Ring Imaging'
    analysis_class = RingMaskAnalysis

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slider = pn.widgets.RangeSlider(
            name='Ring radii',
            value=(5, 10),
            start=0.1,
            end=100,
        )
        self.toolbox.append(self._slider)

    def _get_analysis_params(self):
        base_params = super()._get_analysis_params()
        return {
            **base_params,
            'ri': self._slider.value[0],
            'ro': self._slider.value[1]
        }

    def initialize(self, dataset: lt.DataSet):
        super().initialize(dataset)
        _, radii, max_dim = get_initial_pos(dataset.shape.sig)
        self._slider.end = max_dim
        self._slider.value = radii
        self._sig_ring = self._sig_cursor.add_ring(*radii)
        self._slider.param.watch(self._update_radius, 'value')

    def _update_radius(self, e):
        r0, r1 = e.new
        self._sig_ring.update(inner_radius=r0, outer_radius=r1)
