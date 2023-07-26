from __future__ import annotations
from typing import TYPE_CHECKING

import panel as pn

from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF

from .live_plot import AperturePlot
from .base import UIWindow, UIType, UIState, UDFWindowJob
from .pick import PickUDFBaseWindow
from .result_tracker import ImageResultTracker
from ..display.display_base import Cursor, DiskSet, RingSet, PointSet
from .result_containers import Numpy2DResultContainer
from ..utils import get_initial_pos


if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import BufferWrapper, UDFResultDict, UDF
    from .results import ResultRow


class SingleImagingUDFWindow(UIWindow):
    name = 'single_imaging'
    title_md = 'Single Imaging'
    analysis_class = PointMaskAnalysis

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._toolbox = pn.Column()
        self.inner_layout.append(self._toolbox)

    @property
    def toolbox(self):
        return self._toolbox

    def initialize(self, dataset: lt.DataSet):
        self._sig_plot = AperturePlot.new(
            dataset,
            SumUDF(),
            title='Sig',
        )
        self.setup_cursor(dataset)
        mask_analysis = self.analysis_class(dataset, self._get_analysis_params())
        self._imaging_udf = mask_analysis.get_udf()
        self._imaging_plot = AperturePlot.new(
            dataset,
            self._imaging_udf,
            title='Virtual image',
        )
        # This is necessary to ensure annotations are added *before*
        # the figure is inserted into the layout. Previous approach of calling
        # super().initialize() then adding annotations meant that the
        # annotations were not visible on the plot until the cell was
        # reloaded, which in turn caused issues with clashing Bokeh models
        self._add_annotation(dataset)
        self.inner_layout.extend((self._sig_plot.pane, self._imaging_plot.pane))

        self.sig_plot_tracker = ImageResultTracker(
            self,
            self._sig_plot,
            tags=('sig',),
            select_text='Sig image',
        )
        self.sig_plot_tracker.initialize()

        return self

    def _add_annotation(self, dataset: lt.DataSet):
        # Used by subclasses to add elements to plots before
        # they are inserted into the inner_layout
        pass

    def _get_analysis_params(self):
        return {
            'cx': self._sig_cursor.cds.data['cx'][0],
            'cy': self._sig_cursor.cds.data['cy'][0],
        }

    def setup_cursor(self, ds, as_editable: bool = True):
        fig_sig = self._sig_plot.fig
        (cy, cx), (ri, r), max_dim = get_initial_pos(ds.shape.sig)
        self._sig_cursor = Cursor.new().from_pos(cx, cy)
        self._sig_cursor.on(fig_sig)
        if as_editable:
            # To enable ring/disk glyphs to be draggable without bugs
            self._sig_cursor.make_editable(tool_name='default')
            fig_sig.toolbar.active_drag = fig_sig.tools[-1]

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
        params = self.update_imaging_udf(dataset)
        return UDFWindowJob(
            self,
            [self._imaging_udf],
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
        buffer = results[0]['intensity']
        image: np.ndarray = buffer.data[..., 0]
        rc = Numpy2DResultContainer('intensity', image, params={'tags': (buffer.kind,)})
        result = self.results_manager.new_result(rc, run_id, window_row.window_id)
        return (result,)

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        self.sig_plot_tracker.on_results_registered(*results)

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        self.sig_plot_tracker.on_results_deleted(*results)


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
            start=0.3,
            end=100,
        )
        self.toolbox.append(self._slider)

    def _get_analysis_params(self):
        base_params = super()._get_analysis_params()
        return {**base_params, 'r': self._slider.value}

    def setup_cursor(self, ds):
        # Cannot have point itself editable as it causes doubling
        # of the mouse input when dragging, report upstream?
        return super().setup_cursor(ds, as_editable=False)

    def _add_annotation(self, dataset: lt.DataSet):
        # called by the parent class .initialize()
        _, (_, r), max_dim = get_initial_pos(dataset.shape.sig)
        self._slider.end = max_dim
        self._slider.value = r
        self._sig_disk = self._sig_cursor.add_disk(r)
        self._sig_disk.make_editable()
        self._sig_plot.fig.toolbar.active_drag = self._sig_plot.fig.tools[-1]
        self._slider.param.watch(self._update_radius, 'value')
        self._sig_cursor.cursor.line_alpha = 0.

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
            start=0.3,
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

    def setup_cursor(self, ds):
        # Cannot have point itself editable as it causes doubling
        # of the mouse input when dragging, report upstream?
        return super().setup_cursor(ds, as_editable=False)

    def _add_annotation(self, dataset: lt.DataSet):
        # called by the parent class .initialize()
        _, radii, max_dim = get_initial_pos(dataset.shape.sig)
        self._slider.end = max_dim
        self._slider.value = radii
        self._sig_ring = self._sig_cursor.add_ring(*radii)
        self._sig_ring.make_editable()
        self._sig_plot.fig.toolbar.active_drag = self._sig_plot.fig.tools[-1]
        self._slider.param.watch(self._update_radius, 'value')
        self._sig_cursor.cursor.line_alpha = 0.

    def _update_radius(self, e):
        r0, r1 = e.new
        self._sig_ring.update(inner_radius=r0, outer_radius=r1)


class ImagingWindow(PickUDFBaseWindow, ui_type=UIType.ANALYSIS):
    name = 'imaging'
    title_md = 'Virtual Imaging'

    def initialize(self, dataset: lt.DataSet) -> ImagingWindow:
        self._pick_base(dataset)

        (cy, cx), (ri, ro), max_dim = get_initial_pos(dataset.shape.sig)
        self._ring_db = (
            RingSet
            .new()
            .from_vectors(
                x=[cx],
                y=[cy],
                inner_radius=ri,
                outer_radius=ro,
            )
            .on(self.sig_plot.fig)
            .make_editable(add=False)
            .set_visible(False)
        )
        self._disk_db = (
            DiskSet(
                self._ring_db.cds,
                radius=self._ring_db.rings.outer_radius,
            )
            .on(self.sig_plot.fig)
            .make_editable(add=False)
        )
        self._point_db = (
            PointSet(
                self._ring_db.cds,
            )
            .on(self.sig_plot.fig)
            .make_editable(add=False)
            .set_visible(False)
        )
        self._edit_tool = self.sig_plot.fig.tools[-1]
        self._edit_tool.renderers.clear()
        self._disk_db.make_editable()
        self.sig_plot.fig.toolbar.active_drag = self._edit_tool

        widget_width = 350
        self._radius_slider = pn.widgets.FloatSlider(
            name='Disk radius',
            value=ro,
            start=0.3,
            end=max_dim,
            width=widget_width,
        )
        self._radius_slider.param.watch(self._update_radius, 'value')
        self._radii_slider = pn.widgets.RangeSlider(
            name='Annulus radii',
            value=(ri, ro),
            start=0.3,
            end=max_dim,
            visible=False,
            width=widget_width,
        )
        self._radii_slider.param.watch(self._update_radii, 'value')

        self._mode_mapping: dict[str, tuple[PointSet, pn.widgets.FloatSlider | None]] = {
            'Point': (self._point_db, None),
            'Disk': (self._disk_db, self._radius_slider),
            'Annulus': (self._ring_db, self._radii_slider),
            'Whole Frame': (None, None),
        }
        self._mode_selector = pn.widgets.RadioButtonGroup(
            name='Imaging mode',
            value='Disk',
            options=list(self._mode_mapping.keys()),
            button_type='default',
            width=widget_width,
        )
        self._mode_selector.param.watch(self._toggle_visible, 'value')

        self.nav_plot.add_mask_tools(activate=False)
        clear_roi_button = self.nav_plot.get_clear_mask_btn()

        self.toolbox.extend((
            clear_roi_button,
            self._mode_selector,
            self._radius_slider,
            self._radii_slider,
        )),
        self._standard_layout()

        self.sig_plot_tracker = ImageResultTracker(
            self,
            self.sig_plot,
            ('sig',),
            'Sig image',
        )
        self.sig_plot_tracker.initialize()

        return self

    async def _toggle_visible(self, e):
        if e.new == e.old:
            return
        sig_fig = self.sig_plot.fig
        # This could be done with a 'remove_editable' method
        self._edit_tool.renderers.clear()
        db, widget = self._mode_mapping[e.new]
        if db is not None:
            db.set_visible(True)
            if widget is not None:
                widget.visible = True
                current_ro = self._ring_db.cds.data['r1'][0]
                try:
                    ri, _ = widget.value
                    if ri >= current_ro:
                        ri = 0.5 * current_ro
                    widget.value = (ri, current_ro)
                except TypeError:
                    widget.value = current_ro
            glyph_name = db.glyph_names[0]
            renderer = db.renderers_for_fig(glyph_name, sig_fig)[0]
            self._edit_tool.renderers.append(renderer)

        for name, (db, widget) in self._mode_mapping.items():
            if e.new == name:
                continue
            if db is not None:
                db.set_visible(False)
            if widget is not None:
                widget.visible = False
        pn.io.notebook.push_notebook(self.sig_plot.pane)

    def _update_radius(self, e):
        r = e.new
        self._disk_db.update(radius=r)

    def _update_radii(self, e):
        r0, r1 = e.new
        self._ring_db.update(inner_radius=r0, outer_radius=r1)

    def _get_udf(self, dataset: lt.DataSet) -> tuple[UDF, dict[str, float]]:
        mode = self._mode_selector.value
        if mode == 'Whole Frame':
            return SumSigUDF(), {}
        params = {
            'cx': self._ring_db.cds.data['cx'][0],
            'cy': self._ring_db.cds.data['cy'][0],
        }
        if mode == 'Disk':
            params['r'] = self._ring_db.cds.data['r1'][0]
            analysis = DiskMaskAnalysis
        elif mode == 'Annulus':
            params['ri'] = self._ring_db.cds.data['r0'][0]
            params['ro'] = self._ring_db.cds.data['r1'][0]
            analysis = RingMaskAnalysis
        else:
            analysis = PointMaskAnalysis
        udf = analysis(dataset, params).get_udf()
        return udf, params

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        udf, params = self._get_udf(dataset)
        self.nav_plot.udf = udf
        roi = self.nav_plot.get_mask(dataset.shape.nav)
        return UDFWindowJob(
            self,
            [udf],
            [self.nav_plot],
            result_handler=None,
            params=params,
            roi=roi,
        )

    def on_results_registered(
        self,
        *results: ResultRow,
    ):
        self.sig_plot_tracker.on_results_registered(*results)

    def on_results_deleted(
        self,
        *results: ResultRow,
    ):
        self.sig_plot_tracker.on_results_deleted(*results)
