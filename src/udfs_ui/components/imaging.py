from __future__ import annotations
from typing import TYPE_CHECKING, Any

import panel as pn
import numpy as np

from libertem.udf.base import UDF, DataTile
from libertem.analysis.point import PointMaskAnalysis
from libertem.analysis.disk import DiskMaskAnalysis
from libertem.analysis.ring import RingMaskAnalysis
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.sum import SumUDF
from libertem.udf.logsum import LogsumUDF

from .base import UIType, UIState, UDFWindowJob, JobResults
from .pick import PickUDFBaseWindow
from ..display.display_base import DiskSet, RingSet, PointSet
from .result_containers import Numpy2DResultContainer
from ..utils import get_initial_pos


if TYPE_CHECKING:
    import libertem.api as lt
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from .results import ResultRow


class ImagingWindow(PickUDFBaseWindow, ui_type=UIType.ANALYSIS):
    name = 'virtual_detector'
    title_md = 'Virtual Detector'

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
        ))
        self._standard_layout(
            left_before=(clear_roi_button,),
            right_after=(
                self._radius_slider,
                self._radii_slider,
            ),
        )

        self.link_image_plot('Sig', self.sig_plot, ('sig',))
        return self

    async def _toggle_visible(self, e):
        if e.new == e.old:
            return
        sig_fig = self.sig_plot.fig
        # This could be done with a 'remove_editable' method
        self._edit_tool.renderers.clear()

        for name, (db, widget) in self._mode_mapping.items():
            if e.new == name:
                continue
            if db is not None:
                db.set_visible(False)
            if widget is not None:
                widget.visible = False

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
            params = {
                'result_title': 'Frame Sum',
                'result_name': 'frame_sum',
            }
            return SumSigUDF(), params
        params = {
            'cx': self._ring_db.cds.data['cx'][0],
            'cy': self._ring_db.cds.data['cy'][0],
        }
        if mode == 'Disk':
            params['r'] = self._ring_db.cds.data['r1'][0]
            result_title = 'Disk Sum'
            result_name = 'disk_sum'
            analysis = DiskMaskAnalysis
        elif mode == 'Annulus':
            params['ri'] = self._ring_db.cds.data['r0'][0]
            params['ro'] = self._ring_db.cds.data['r1'][0]
            result_title = 'Annular Sum'
            result_name = 'annular_sum'
            analysis = RingMaskAnalysis
        else:
            result_title = 'Point Value'
            result_name = 'point_value'
            analysis = PointMaskAnalysis
        udf = analysis(dataset, params).get_udf()
        return udf, {**params, 'result_title': result_title, 'result_name': result_name}

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
            result_handler=self.complete_job,
            params=params,
            roi=roi,
        )

    def complete_job(self, job: UDFWindowJob, job_results: JobResults) -> tuple[ResultRow, ...]:
        result_title: str = job.params.pop('result_title')
        result_name: str = job.params.pop('result_name')
        self.nav_plot.fig.title.text = result_title
        window_row = self.results_manager.new_window_run(
            self,
            job_results.run_id,
            params=job.params,
        )
        channel: str = self.nav_plot.channel
        buffer = job_results.udf_results[0][channel]
        image: np.ndarray = buffer.data.squeeze()
        rc = Numpy2DResultContainer(
            result_name,
            image,
            meta={
                'channel': channel,
            },
            title=result_title,
        )
        rc.tag_as(buffer.kind)
        result = self.results_manager.new_result(rc, job_results.run_id, window_row.window_id)
        self.nav_plot.displayed = result
        return (result,)


class PickNoROIUDF(UDF):
    def get_result_buffers(self):
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


class FrameImaging(PickUDFBaseWindow, ui_type=UIType.ANALYSIS):
    name = 'frame_imaging'
    title_md = 'Frame Imaging'

    def initialize(self, dataset: lt.DataSet) -> ImagingWindow:
        self._pick_base(dataset)

        widget_width = 350
        self._mode_selector = pn.widgets.RadioButtonGroup(
            name='Imaging mode',
            value='Sum',
            options=['Pick', 'Sum', 'Logsum'],
            button_type='default',
            width=widget_width,
        )
        self._mode_selector.param.watch(self._toggle_visible, 'value')

        self.nav_plot.add_mask_tools(activate=False)
        clear_roi_button = self.nav_plot.get_clear_mask_btn()

        self.toolbox.extend((
            self._mode_selector,
        )),
        self._standard_layout(left_before=(clear_roi_button,))
        self.link_image_plot('Nav', self.nav_plot, ('nav',))
        return self

    async def _toggle_visible(self, e):
        if e.new == e.old:
            return
        self._nav_cursor.set_visible(e.new == 'Pick')
        pn.io.notebook.push_notebook(self.nav_plot.pane)

    def _get_udf(self, dataset: lt.DataSet, roi: np.ndarray | None):
        mode = self._mode_selector.value
        params: dict[str, int | str] = {'mode': mode}
        if mode == 'Pick':
            should_pick = self._should_pick(dataset, self._nav_cursor.cds.data)
            if should_pick is False:
                return None, params
            cx, cy = should_pick
            pick_idx = np.ravel_multi_index(([cy], [cx]), dataset.shape.nav).item()
            if roi is not None:
                if roi[cy, cx]:
                    pick_idx = np.cumsum(roi.ravel())[pick_idx] - 1
                else:
                    return None, params
            udf = PickNoROIUDF(pick_idx=pick_idx)
            params.update({
                'cx': cx,
                'cy': cy,
                'pick_idx': pick_idx,
                'result_title': self._pick_title((cy, cx)),
                'result_name': 'pick_frame',
            })
            self.sig_plot.channel = 'intensity'
        elif mode == 'Sum':
            udf = SumUDF()
            self.sig_plot.channel = 'intensity'
            params['result_title'] = 'Sum frame'
            params['result_name'] = 'sum_frame'
        elif mode == 'Logsum':
            udf = LogsumUDF()
            self.sig_plot.channel = 'logsum'
            params['result_title'] = 'Logsum frame'
            params['result_name'] = 'logsum_frame'
        else:
            raise RuntimeError(f'Unrecognized mode {mode}')
        return udf, params

    def _cds_pick_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        """Must reset plot--udf each time in case it was changed"""
        self.sig_plot.udf = self._udf_pick
        self.sig_plot.channel = 'intensity'
        return super()._cds_pick_job(state, dataset, roi)

    def get_job(
        self,
        state: UIState,
        dataset: lt.DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        self_roi = self.nav_plot.get_mask(dataset.shape.nav)
        udf, params = self._get_udf(dataset, self_roi)
        if udf is None:
            return None
        self.sig_plot.udf = udf
        return UDFWindowJob(
            self,
            [udf],
            [self.sig_plot],
            result_handler=self.complete_job,
            params=params,
            roi=self_roi,
        )

    def complete_job(self, job: UDFWindowJob, job_results: JobResults) -> tuple[ResultRow, ...]:
        result_title: str = job.params.pop('result_title')
        result_name: str = job.params.pop('result_name')
        self.sig_plot.fig.title.text = result_title
        window_row = self.results_manager.new_window_run(
            self,
            job_results.run_id,
            params=job.params,
        )
        channel: str = self.sig_plot.channel
        buffer = job_results.udf_results[0][channel]
        image: np.ndarray = buffer.data.squeeze()
        rc = Numpy2DResultContainer(
            result_name,
            image,
            meta={
                'channel': channel,
            },
            title=result_title,
        )
        rc.tag_as(buffer.kind)
        result = self.results_manager.new_result(rc, job_results.run_id, window_row.window_id)
        self.sig_plot.displayed = result
        return (result,)
