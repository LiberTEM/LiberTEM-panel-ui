from __future__ import annotations
from typing import TYPE_CHECKING

import panel as pn

from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.udf.sumsigudf import SumSigUDF

from .base import UIType, UIState, UDFWindowJob
from .pick import PickUDFBaseWindow
from .result_tracker import ImageResultTracker
from ..display.display_base import DiskSet, RingSet, PointSet
from .result_containers import Numpy2DResultContainer
from ..utils import get_initial_pos


if TYPE_CHECKING:
    import numpy as np
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.udf.base import UDF
    from .results import ResultRow


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
            self._mode_selector,
            self._radius_slider,
            self._radii_slider,
        )),
        self._standard_layout(left_before=(clear_roi_button,))

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


# raise NotImplementedError(
#     'Add a frame-shaped result-generating window (pick + sum + logsum) '
#     'Could re-integrate PickNoROI for "pick mode" on "Run all / Run this", and have '
#     'normal frame-picking for offline datasets'
#     'Also add ROI tools to the pick window'
# )
