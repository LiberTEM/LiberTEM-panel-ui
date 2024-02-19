from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import panel as pn

from .base import UIState, JobResults
from .windows.base import (
    WindowType, UDFWindowJob, UIWindow,
    WindowProperties, WindowPropertiesTDict,
)
from .live_plot import AperturePlot
from .windows.pick import PickUDFWindow
from .display.display_base import PointSet, Rectangles
from .display.lattice import LatticeOverlay
from .utils.panel_components import button_divider

from libertem.udf.raw import PickUDF
from libertem.udf.sumsigudf import SumSigUDF

from .phase import Phase, PhaseMap, AmorphousPhase
# from .udf import MultiPhaseAutoCorrUDF, FilterUDF, PhaseMapUDF

if TYPE_CHECKING:
    import pandas as pd
    from libertem.api import DataSet
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.common.shape import Shape
    from libertem.udf.base import UDFResultDict
    from .strain_decomposition import StrainResult
    from .utils import PointYX
    from .results.results_manager import ResultRow
    from .base import UIContextBase


# classes requiring the strain_app interface to load
# declare this as their ui_type, meaning they will not
# automaticall appear in the add dropdown menu
STRAINAPP = 'strain_app'


class PhaseDef(NamedTuple):
    df: pd.DataFrame
    ref_yx: PointYX
    label: str | None = None
    amorphous: bool = False

    @property
    def display_name(self) -> str:
        base = f'{self.ref_yx.y, self.ref_yx.x}'
        if self.label:
            return base + f' - {self.label}'
        return base

    @property
    def radius(self) -> float:
        return float(self.df['radius'][0])

    @property
    def g0(self) -> complex:
        idx = 0 if self.amorphous else 1
        return self.df['cx'][idx] + self.df['cy'][idx] * 1j

    @property
    def g1(self) -> complex | None:
        if self.amorphous:
            return None
        return (self.df['cx'][0] + self.df['cy'][0] * 1j) - self.g0

    @property
    def g2(self) -> complex | None:
        if self.amorphous:
            return None
        return (self.df['cx'][2] + self.df['cy'][2] * 1j) - self.g0

    def __eq__(self, other: PhaseDef):
        try:
            return other.ref_yx == self.ref_yx
        except AttributeError:
            return False

    def to_phase(
        self,
        sig_shape: tuple[int, int],
        nav_shape: tuple[int, int]
    ) -> Phase | AmorphousPhase:
        ny, nx = self.ref_yx
        (ref_idx,) = np.ravel_multi_index(([ny], [nx]), nav_shape)
        cls = AmorphousPhase if self.amorphous else Phase
        return cls.from_shape_ref(
            sig_shape,
            g1=self.g1,
            g2=self.g2,
            centre=self.g0,
            ref_idx=ref_idx,
            label=self.label,
        )


def get_initial_lattice(frame_shape: tuple[int, int]):
    h, w = frame_shape
    min_dim = min(h, w)
    centre = w // 2 + (h // 2) * 1j
    r = float(max(1., min_dim / 10.))
    vec_len = max(1., min_dim / 4.)
    g1 = 0 + 1j * vec_len
    g2 = vec_len + 0j
    return g1, centre, g2, r


class StrainAppCompatMixin:
    # Class is compatible with StrainApplication but does not require it
    @property
    def strain_app(self: UIWindow) -> StrainApplication | None:
        return self._window_data

    def validate_data(self: UIWindow):
        if self._window_data is None:
            return
        if not isinstance(self._window_data, StrainApplication):
            raise ValueError('window_data must be either StrainApplication '
                             f'or None for {type(self).__name__}')


class StrainAppMixin:
    # Class requires StrainApplication
    def validate_data(self: UIWindow):
        if not isinstance(self._window_data, StrainApplication):
            raise ValueError(f'Need StrainApplication as window_data for {type(self).__name__}')

    @property
    def strain_app(self: UIWindow) -> StrainApplication:
        return self._window_data


class LatticeDefineWindow(StrainAppCompatMixin, PickUDFWindow, ui_type=WindowType.STANDALONE):
    @classmethod
    def default_properties(cls):
        return super().default_properties().with_other(
            name='lattice_definer',
            title_md='Lattice Definer',
            header_run=False,
        )

    def initialize(self, dataset: DataSet):
        self._saved_phases: list[PhaseDef] = []
        self._displayed_phase: int | None = None

        super().initialize(dataset, with_layout=False)
        g1, c0, g2, r = get_initial_lattice(dataset.meta.shape.sig)
        self._lattice_set = LatticeOverlay.new().from_lattice_vectors(
            c0, g1, g2, r
        ).with_labels(
            'g0', 'g1', 'g2'
        ).on(
            self.sig_plot.fig
        ).editable()
        self.sig_plot.fig.toolbar.active_drag = self.sig_plot.fig.tools[-1]
        self._slider = self._lattice_set.diskset.get_radius_slider(
            min(dataset.shape.sig) * 0.25
        )

        self._phase_points = PointSet.new().empty().on(self.nav_plot.fig)
        self._phase_points.points.fill_alpha = 0.5
        self._phase_points.cds.data.update(
            {
                'marker': [],
                'color': [],
            }
        )
        self._phase_points.points.marker = 'marker'
        self._phase_points.points.fill_color = 'color'
        self._phase_points.points.line_color = 'color'
        self._phase_points.points.size = 20

        self._is_amorph_cbox = pn.widgets.Checkbox(
            name='Amorphous',
            value=False,
        )
        self._amorph_stored_data: dict[str, list] | None = None
        self._is_amorph_cbox.param.watch(self._toggle_amorph, 'value')
        self._save_phase_btn = pn.widgets.Button(
            name='Save',
            button_type='success',
            width=90,
            align=('center', 'end'),
        )
        self._save_phase_btn.on_click(self._save_phase)
        self._phase_name_input = pn.widgets.TextInput(
            name='Phase name',
            placeholder='Optional phase name...',
            width=200,
        )
        self._next_btn = pn.widgets.Button(
            name='Next',
            button_type='primary',
            width=75,
            align='center',
            margin=(5, 2, 5, 2),
        )
        self._next_btn.on_click(self._next_phase)
        self._prev_btn = pn.widgets.Button(
            name='Previous',
            button_type='primary',
            width=75,
            align='center',
            margin=(5, 2, 5, 2),
        )
        self._prev_btn.on_click(self._previous_phase)
        self._del_phase_btn = pn.widgets.Button(
            name='Delete',
            button_type='danger',
            align='center',
            margin=(5, 2, 5, 2),
        )
        self._del_phase_btn.on_click(self._delete_phase)

        self.toolbox.extend((
            pn.Row(
                pn.widgets.StaticText(
                    value='<b>Phase select:</b>',
                    align='center',
                ),
                self._prev_btn,
                self._next_btn,
                button_divider(),
                self._del_phase_btn,
                button_divider(),
                self._save_phase_btn,
            ),
            self._phase_name_input,
        ))

        self.link_image_plot('Sig', self.sig_plot, ('sig',))
        self._standard_layout(
            right_after=(self._slider, self._is_amorph_cbox)
        )
        return self

    def _toggle_amorph(self, e):
        if e.new == e.old:
            return
        to_amorph = e.new
        cds_amorph = (self._lattice_set.data_length == 1)
        if to_amorph:
            if cds_amorph:
                # Already amorphous data
                return
            old_data = {**self._lattice_set.cds.data}
            new_data = {
                k: [v for v, label
                    in zip(old_data[k], old_data['label'])
                    if 'g0' in label]
                for k in old_data.keys()
            }
            self._lattice_set.raw_update(**new_data)
            self._amorph_stored_data = old_data
        elif self._amorph_stored_data is None:
            # Nothing to do
            return
        elif cds_amorph:
            # Only convert to crystal if CDS is amorphous
            glyph = self._lattice_set.diskset.disks
            ykey = glyph.y
            xkey = glyph.x
            rkey = glyph.radius
            old_data = self._amorph_stored_data
            central_spot = self._lattice_set.cds.data
            new_cy = central_spot[ykey][0]
            new_cx = central_spot[xkey][0]
            new_r = central_spot[rkey][0]
            dy = new_cy - old_data[ykey][1]
            dx = new_cx - old_data[xkey][1]
            old_data[ykey] = [y + dy for y in old_data[ykey]]
            old_data[xkey] = [x + dx for x in old_data[xkey]]
            old_data[rkey] = [new_r for _ in old_data[rkey]]
            self._lattice_set.raw_update(**old_data)
        self.sig_plot.push()

    async def _cycle_phases(self, step: int):
        num_phases = len(self._saved_phases)
        if num_phases == 0:
            self._displayed_phase = None
            return
        if self._displayed_phase is None:
            self._displayed_phase = 0
        else:
            self._displayed_phase = (self._displayed_phase + step) % num_phases
        phase = self._saved_phases[self._displayed_phase]
        await self._load_phase(to_load=phase)

    async def _previous_phase(self, *e):
        await self._cycle_phases(-1)

    async def _next_phase(self, *e):
        await self._cycle_phases(1)

    async def _save_phase(self, *e):
        lattice_df = self._lattice_set.cds.to_df().copy(deep=True)
        label: str | None = self._phase_name_input.value
        if label is None or len(label) == 0:
            label = None
        ref_yx: tuple[int, int] = self._nav_cursor.current_pos(to_int=True).as_yx()
        # There is a risk of having phases outside of the nav grid here
        # Downstream code needs to check that phase defs are in bounds
        phase_def = PhaseDef(
            lattice_df,
            ref_yx,
            label=label,
            amorphous=self._is_amorph_cbox.value,
        )
        if (existing := self._phase_nearby(ref_yx)) is not None:
            idx, _, _ = existing
            self._saved_phases[idx] = phase_def
        else:
            self._saved_phases.append(phase_def)
            idx = len(self._saved_phases) - 1

        self._set_title_for_phase(idx)
        self._update_phase_points(selected_idx=idx, push=False)
        self.sync_phases()
        self.nav_plot.push(self.sig_plot)

    def sync_phases(self):
        if self.strain_app is not None:
            self.strain_app.new_phases(self._saved_phases)
            self.validate_data()

    def _update_phase_points(self, selected_idx: int | None = None, push: bool = True):
        data = dict(
            cx=[p.ref_yx.x for p in self._saved_phases],
            cy=[p.ref_yx.y for p in self._saved_phases],
            marker=['circle' if p.amorphous else 'hex_dot' for p in self._saved_phases],
            color=['#8FBC8F' if (selected_idx is not None and idx == selected_idx) else '#FF7F50'
                   for idx, _ in enumerate(self._saved_phases)],
        )
        self._phase_points.raw_update(**data)
        if push:
            self.nav_plot.push()

    async def _delete_phase(self, *e):
        pos = self._nav_cursor.current_pos().as_yx()
        if (existing := self._phase_nearby(pos)) is not None:
            idx, _, _ = existing
            self._saved_phases.pop(idx)
            self.sync_phases()

            if len(self._saved_phases) == 0:
                self._saved_phases: list[PhaseDef] = []
                self._update_phase_points(push=False)
                self.reset_title()
                self._phase_name_input.param.update(value='')
                self._is_amorph_cbox.param.update(value=False)
                self.nav_plot.push(self.sig_plot)
            else:
                await self._next_phase()

    async def _load_phase(self, *e, to_load: PhaseDef | None = None):
        if to_load is None:
            pos = self._nav_cursor.current_pos().as_yx()
            if (is_nearby := self._phase_nearby(pos)) is not None:
                _, _, phase = is_nearby
            else:
                return
        else:
            phase = to_load
        # Update the cursor position
        y, x = phase.ref_yx
        self._nav_cursor.update(
            x=x,
            y=y,
        )
        self.nav_plot.push()
        # The use of cursor.update should trigger the frame update
        # but due to the Panel callback issue we need to manually
        # run the _cds_pick_job which properly schedules it
        await self.run_this(
            run_from=self._cds_pick_job,
        )

    @staticmethod
    def _distance(pt1: tuple[int, int], pt2: tuple[int, int]) -> float:
        ay, ax = pt1
        by, bx = pt2
        return np.sqrt(abs(ay - by)**2 + abs(ax - bx)**2)

    def _phase_nearby(
        self,
        cyx: tuple[int, int],
        threshold: float = 3.
    ) -> tuple[int, float, PhaseDef] | None:
        here_phase = tuple(
            (i, d, p) for i, p in enumerate(self._saved_phases)
            if (d := self._distance(p.ref_yx, cyx)) <= threshold
        )
        if here_phase:
            return min(here_phase, key=lambda x: x[1])
        return None

    def _set_title_for_phase(self, phase_idx: int, pos: tuple[int, int] | PointYX | None = None):
        phase = self._saved_phases[phase_idx]
        if pos is None:
            pos = phase.ref_yx
        title_suffix = f' - Phase {phase_idx}'
        if phase.amorphous:
            title_suffix = f'{title_suffix} (Amorphous)'
        if phase.label is not None:
            title_suffix = f'{title_suffix}: {phase.label}'
        self.sig_plot.fig.title.text = self._pick_title(
            tuple(pos),
            suffix=title_suffix,
        )

    def _complete_cds_pick_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        results = super()._complete_cds_pick_job(job, job_results)
        pos = job.params['cy'], job.params['cx']
        if (is_nearby := self._phase_nearby(pos)) is not None:
            phase_idx, _, phase = is_nearby
            ddict = self._lattice_set.cds.from_df(phase.df)
            ddict = {
                k: v.tolist() for k, v in ddict.items()
                if k in self._lattice_set.cds.data.keys()
            }
            self._lattice_set.cds.data.update(**ddict)
            self._update_phase_points(selected_idx=phase_idx, push=False)
            if phase.label is not None:
                self._phase_name_input.param.update(value=phase.label)
            else:
                self._phase_name_input.param.update(value='')
            self._slider.param.update(value=phase.radius)
            self._set_title_for_phase(phase_idx, pos=pos)
            self._is_amorph_cbox.param.update(value=phase.amorphous)
            self.sig_plot.push(self.nav_plot)
        else:
            self._update_phase_points()
            self._phase_name_input.param.update(value='')
        return results


class FilterViewer(StrainAppCompatMixin, PickUDFWindow, ui_type=WindowType.STANDALONE):
    pick_cls = FilterUDF

    @classmethod
    def default_properties(cls):
        return super().default_properties().with_other(
            name='filter_viewer',
            title_md='Filter Viewer',
        )

    def initialize(
        self,
        dataset: DataSet,
        with_layout: bool = True,
    ):
        super().initialize(dataset, with_layout=False)

        channel_map = {
            'Autocorrelation': 'autocorr',
            'Filtered': 'filtered',
            'Pick': 'intensity',
        }
        self.sig_plot._channel_map = channel_map
        self._channel_select = self.sig_plot.get_channel_select(
            selected='Pick',
            update_title=False,
        )
        self._channel_select.param.watch(lambda _: self.reset_title(), 'value')

        # Move this into toolbar ??
        self.filter_choice = pn.widgets.Select(
            name='Filter type',
            value=5,
            options=[3, 4, 5, 6],
        )
        self.toolbox.append(self.filter_choice)
        if with_layout:
            self._standard_layout()

    def _pick_title(self, cyx: tuple[int, int] | None = None, suffix: str | None = None):
        try:
            stub = self._channel_select.value
        except AttributeError:
            stub = 'Pick'
        return super()._pick_title(
            cyx,
            suffix=suffix,
            title_stub=f'{stub} frame',
        )

    def _get_udfs(self, dataset: DataSet):
        if self.filter_choice.value is None:
            raise RuntimeError('Select box has None value for filter flag')
        self._udf_pick = self.pick_cls(
            **self.get_current_args(),
        )
        self.sig_plot.udf = self._udf_pick
        return [self._udf_pick]

    def get_current_args(self):
        return dict(
            filter_flag=self.filter_choice.value,
            opt=False,
        )

    def _cds_pick_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
        quiet: bool = False,
    ):
        udfs = self._get_udfs(dataset)
        return super()._cds_pick_job(
            state,
            dataset,
            roi,
            quiet=quiet,
            with_udfs=udfs,
        )


class LatticeFitterWindow(StrainAppMixin, FilterViewer, ui_type=STRAINAPP):
    @classmethod
    def default_properties(cls):
        return super().default_properties().with_other(
            name='lattice_fitter',
            title_md='Lattice Fitter',
        )

    def initialize(self, dataset: DataSet):
        super().initialize(dataset, with_layout=False)

        g1, c0, g2, r = get_initial_lattice(dataset.meta.shape.sig)
        self._lattice_set = LatticeOverlay.new().from_lattice_vectors(
            c0, g1, g2, r
        ).with_labels(
            'g0', 'g1', 'g2'
        ).on(
            self.sig_plot.fig
        ).editable()
        self._lattice_set.clear()
        # Could display the phase information under the toolbox
        self._annotation_visible_cbox = pn.widgets.Checkbox(
            name='Show annotation',
            value=False,
            align='center',
        )
        self._autocorr_points = PointSet.new().empty().on(self.sig_plot.fig).set_visible(False)
        self._channel_select.param.watch(self._toggle_annotation, 'value')
        self._annotation_visible_cbox.param.watch(self._toggle_annotation, 'value')
        self.sig_plot._toolbar.insert(2, self._annotation_visible_cbox)
        self._toggle_annotation(None)
        self._standard_layout()

    def _toggle_annotation(self, e):
        global_visible = self._annotation_visible_cbox.value
        selected_channel = self._channel_select.value

        self._autocorr_points.set_visible(
            (selected_channel == 'Autocorrelation') and global_visible
        )
        lattice_vis = (selected_channel != 'Autocorrelation') and global_visible
        self._lattice_set.set_visible(lattice_vis)

    def _get_udfs(self, dataset: DataSet):
        udfs = super()._get_udfs(dataset)
        phase_map = self.strain_app.get_phase_map(dataset.shape)
        if phase_map is None:
            if self.strain_app.num_phases == 0:
                self.logger.warning('Need at least one phase defined to fit lattice')
            else:
                self.logger.warning('Need to compute phase map before fitting lattice')
            self._autocorr_points.clear()
            self._lattice_set.clear()
            return udfs
        elif all(isinstance(p, AmorphousPhase) for p in phase_map.phases):
            self.logger.warning('Cannot fit amorphous-only lattice definitions')
            self._autocorr_points.clear()
            return udfs
        strain_udf = MultiPhaseAutoCorrUDF.from_phase_map(
            phase_map,
            return_pos=True,
            **self.strain_app.get_filter_kwargs()
        )
        return udfs + [strain_udf]

    def _complete_cds_pick_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        super()._complete_cds_pick_job(job, job_results, with_push=False)
        if len(job_results.udf_results) <= 1:
            # Case when no phases defined or no phase map
            self.sig_plot.push()
            return tuple()
        # FIXME Should get this via job params!
        ds_shape = job_results.run_row.ds_shape
        if ds_shape is None:
            raise RuntimeError('Need dataset shape to reconstruct phase map')
        phase_map = self.strain_app.get_phase_map(ds_shape)
        try:
            cx, cy = job.params['cx'], job.params['cy']
            phase_idx = phase_map.idx_map[cy, cx]
            phase = self.strain_app.lattice_definer._saved_phases[phase_idx]
            ddict = self._lattice_set.cds.from_df(phase.df)
            ddict = {
                k: v.tolist() for k, v in ddict.items()
                if k in self._lattice_set.cds.data.keys()
            }
            self._lattice_set.cds.data.update(**ddict)
        except (ValueError, IndexError, TypeError, KeyError):
            # Better to show nothing than have a crash
            self._lattice_set.clear()
        # Update autocorr points
        strain_results = job_results.udf_results[1]
        pos_fit = strain_results['pos'].raw_data[0]
        run_row = self.results_manager.get_run(job_results.run_row.run_id)
        if run_row is not None:
            sy, sx = run_row.params['shape']['sig']
            pos_fit += ((sx / 2.) + (sy / 2.) * 1j)
            py = pos_fit.imag.tolist()
            px = pos_fit.real.tolist()
            self._autocorr_points.update(
                x=px, y=py,
            )
        self.sig_plot.push()
        return tuple()


class PhaseMapWindow(StrainAppMixin, UIWindow, ui_type=STRAINAPP):
    @staticmethod
    def default_properties():
        return WindowProperties(
            name='phase_mapper',
            title_md='Phase Mapper',
            # In principle self_run_only implies header_activate=False
            # but leave it as an option for unforseen cases
            header_activate=False,
            self_run_only=True,
        )

    def initialize(self, dataset: DataSet):
        self._phase_idx_mapping: dict[str, int] = {}
        self._display_select = pn.widgets.Select(
            name='Display phase',
            options=['Max'],
            value='Max',
            width=200,
        )
        self._display_select.param.watch(self._update_for_phase, 'value')

        self.nav_plot = AperturePlot.new(
            dataset,
            PhaseMapUDF(ref_images=(None,)),
            channel={
                'index': ('max_val', self._get_index),
                'corr_score': ('max_val', self._get_score),
                'shift_magnitude': self._get_shift,
            },
            title='Phase Map',
        )
        self.nav_plot.add_mask_tools()
        self._channel_select = self.nav_plot.get_channel_select(
            selected='corr_score'
        )
        self.inner_layout.extend((
            pn.Column(
                self._display_select,
                min_width=300,
            ),
            self.nav_plot.layout,
        ))

    def _update_for_phase(self, e):
        self.nav_plot.change_channel(self._channel_select.value)

    def _get_index(self, buffer: np.ndarray) -> np.ndarray:
        selected = self._display_select.value
        amax = np.argmax(buffer, axis=-1)
        if selected == 'Max':
            # This will give a zero index for NaN values
            # i.e. for incomplete buffers!
            return amax.astype(np.float32)
        else:
            phase_idx = self._phase_idx_mapping.get(selected, None)
            if phase_idx is None:
                return np.full(buffer.shape[:2], np.nan, dtype=np.float32)
            return (amax == phase_idx).astype(np.float32)

    def _get_score(self, buffer: np.ndarray) -> np.ndarray:
        selected = self._display_select.value
        if selected == 'Max':
            # This gives np.nan for incomplete buffer positions
            return np.max(buffer, axis=-1).astype(np.float32)
        else:
            phase_idx = self._phase_idx_mapping.get(selected, None)
            if phase_idx is None:
                return np.full(buffer.shape[:2], np.nan, dtype=np.float32)
            return buffer[..., phase_idx].astype(np.float32)

    def _get_shift(self, udf_results: UDFResultDict, damage: np.ndarray | bool) -> np.ndarray:
        selected = self._display_select.value
        max_val = udf_results['max_val'].data
        max_shift = udf_results['max_shift'].data
        shift_mag = np.abs(max_shift)
        if selected == 'Max':
            # Same zero index for NaN as above
            amax = np.argmax(max_val, axis=-1)
            max_shifts = np.take_along_axis(shift_mag, amax[..., np.newaxis], -1).squeeze(axis=-1)
            return (
                np.abs(max_shifts).astype(np.float32),
                damage
            )
        else:
            phase_idx = self._phase_idx_mapping.get(selected, None)
            if phase_idx is None:
                return np.full(shift_mag.shape[:2], np.nan, dtype=np.float32), damage
            return shift_mag[..., phase_idx].astype(np.float32), damage

    def _get_frame_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if self.strain_app.num_phases == 0:
            self.logger.info('Need at least one phase defined before mapping')
            return None

        udf_phases = self.strain_app.get_phases_for_run(dataset.shape)
        roi = np.zeros(dataset.shape.nav, dtype=bool)
        for p in udf_phases:
            roi.flat[p.ref_idx] = True
        order = np.argsort([p.ref_idx for p in udf_phases])

        return UDFWindowJob(
            self,
            [PickUDF()],
            [],
            self._complete_get_frame_job,
            params={'order': order, 'phases': udf_phases},
            roi=roi,
        )

    def _complete_get_frame_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        pick_res = job_results.udf_results[0]['intensity'].data
        pick_res = pick_res[job.params['order'], ...]
        self._phase_refs = (np.conjugate(np.fft.fft2(pick_res)), job.params['phases'])
        return tuple()

    def _get_map_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ):
        if self._phase_refs is None:
            self.logger.error('Phase map job launched without phase references, stopping')
            return None
        self._phase_refs: tuple[np.ndarray, list[Phase]]
        phase_refs, phases = self._phase_refs
        self._phase_idx_mapping: dict[str, int] = {
            f'Phase {i}: {p.label}': i for i, p in enumerate(phases)
        }
        self._display_select.param.update(
            options=['Max', *self._phase_idx_mapping.keys()],
            value='Max',
        )
        udf = PhaseMapUDF(
            ref_images=phase_refs,
            phase_centres=np.asarray([p.centre for p in phases]),
        )
        self.nav_plot.udf = udf
        return UDFWindowJob(
            self,
            [udf],
            [self.nav_plot],
            params={'phases': phases},
            result_handler=self._complete_get_map_job,
            roi=self.nav_plot.get_mask(dataset.shape.nav),
        )

    def _complete_get_map_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        try:
            phase_map = PhaseMap(
                job_results.udf_results[0]['max_val'].data.argmax(axis=-1),
                job.params['phases'],
                max_val=job_results.udf_results[0]['max_val'].data,
                max_pos=job_results.udf_results[0]['max_pos'].data,
            )
            self.strain_app.new_phase_map(phase_map)
        except Exception as e:
            self.logger.log_from_exception(e)
        return tuple()

    async def run_from_btn(self, *e):
        self._header_ns._run_btn.disabled = True
        self._phase_refs = None
        try:
            await self.run_this(run_from=self._get_frame_job)
            if self._phase_refs is not None:
                await self.run_this(run_from=self._get_map_job)
        finally:
            self._header_ns._run_btn.disabled = False
            self._phase_refs = None


class StrainAnalysis(StrainAppMixin, UIWindow, ui_type=STRAINAPP):
    @staticmethod
    def default_properties():
        return WindowProperties(
            name='strain_analysis',
            title_md='Strain Analysis',
        )

    def initialize(self, dataset: DataSet):
        self.live_plot = AperturePlot.new(
            dataset,
            MultiPhaseAutoCorrUDF(),
            channel={
                'mod_g1': ('g1', lambda buffer: np.abs(buffer.squeeze())),
                'mod_g2': ('g2', lambda buffer: np.abs(buffer.squeeze())),
                'angle_g1': ('g1', lambda buffer: np.angle(buffer.squeeze())),
                'angle_g2': ('g2', lambda buffer: np.angle(buffer.squeeze())),
            },
            title='Lattice vectors',
        )
        self.live_plot.add_mask_tools()

        self.strain_plot = AperturePlot.new(
            dataset,
            # Just to initialize as a nav-shaped plot
            SumSigUDF(),
            title='Strain components',
        )
        self.strain_plot.add_mask_tools(polygons=False)

        self.st_component_select = pn.widgets.RadioButtonGroup(
            options=['e_xx', 'e_xy', 'e_yy', 'theta'],
            value='e_xx',
            align='center',
            margin=(3, 3),
        )
        self.st_component_select.param.watch(self._update_st_plot, 'value')

        self.rot_slider = pn.widgets.FloatSlider(
            name='Angle (deg.)',
            value=0.,
            start=-180.,
            end=180.,
            step=0.1,
            disabled=True,
        )
        self.rot_slider.param.watch(self._rotate_strain, 'value_throttled')
        self.rot_options = {
            '∥ (g1)': 'g1',
            '∥ (g2)': 'g2',
            '∥ (g1 + g2)': 'g1 + g2',
            '∥ (g1 - g2)': 'g1 - g2',
            'Manual': 'Manual',
        }
        options = [*self.rot_options.keys()]
        self.rotation_option_select = pn.widgets.RadioButtonGroup(
            options=options,
            value=options[2],
            align='center',
            margin=(3, 3),
        )
        self.rotation_option_select.param.watch(self._rotate_strain, 'value')
        self.rotation_phase_select = pn.widgets.Select(
            name='g-ref phase',
            margin=(3, 3),
            value='None',
            options=['None'],
            width=150,
        )
        self.rotation_phase_select.param.watch(self._rotate_strain, 'value')

        self.show_phase_select = pn.widgets.Select(
            value='None',
            options=['None'],
            margin=(3, 3),
            width=150,
        )
        self.show_phase_select.param.watch(self._show_phase_ref, 'value')
        self.update_references_btn = pn.widgets.Button(
            name='Update ref',
            align=('center', 'end'),
        )
        self.update_references_btn.on_click(self._update_phase_ref)
        self.phase_pointset = PointSet.new().empty().on(self.strain_plot.fig)

        self.strain_plot._toolbar.insert(0, self.st_component_select)
        self.strain_plot._toolbar.insert(
            0,
            pn.widgets.StaticText(
                value='Strain component:',
                align='center',
                margin=(3, 3),
            )
        )

        self.live_plot.get_channel_select()
        self.inner_layout.extend((
            self.live_plot.layout,
            pn.Column(
                pn.Row(
                    pn.widgets.StaticText(
                        value='Show phase + ref',
                        align='center',
                        margin=(3, 3),
                    ),
                    self.show_phase_select,
                    self.update_references_btn,
                ),
                self.strain_plot.layout,
                pn.Row(
                    pn.widgets.StaticText(
                        value='Rotation:',
                        align='center',
                        margin=(3, 3),
                    ),
                    self.rotation_option_select,
                ),
                pn.Row(
                    self.rotation_phase_select,
                    self.rot_slider,
                ),
            ),
        ))

    def _update_st_plot(self, e):
        try:
            component = e.new
        except AttributeError:
            assert isinstance(e, str)
            component = e
        array = self._get_strain_comp(component, self.rot_slider.value)
        self.strain_plot.im.update(array)
        self.strain_plot.fig.title.text = component
        self.strain_plot.push()

    def _get_strain_comp(self, component: str, rotation: float | complex) -> np.ndarray | None:
        if (strain_aa := self.strain_app.get_strain_aa()) is None:
            return
        if isinstance(rotation, float):
            strain_rot = strain_aa.rotate_deg(rotation)
        else:
            strain_rot = strain_aa.to_vector(rotation)
        array: np.ndarray | float | None = getattr(strain_rot, component, None)
        if not isinstance(array, np.ndarray):
            self.logger.error(f'Unrecognized or incompatible strain component {component}')
            return
        selected = self.show_phase_select.value
        if selected != 'None':
            phase_map = self.strain_app.get_res_phasemap()
            if len(phase_map.phases) > 1:
                phase_idx = self._show_phase_select_mapping[selected]
                phase_mask = phase_map.idx_map == phase_idx
                array = array.copy()
                array[np.logical_not(phase_mask)] = np.nan
        return array

    def _rotate_strain(self, e):
        try:
            if isinstance(e.new, (np.floating, float, int)):
                rotation = e.new
            elif isinstance(e.new, str):
                self._toggle_rotation_controls()
                rotation = self._get_current_rotation()
                if isinstance(rotation, complex):
                    rotation_deg = np.angle(rotation, deg=True)
                    self.rot_slider.param.update(value=rotation_deg)
        except AttributeError:
            assert isinstance(e, (np.floating, float, int))
            rotation = e
        array = self._get_strain_comp(self.st_component_select.value, rotation)
        self.strain_plot.im.update(array)
        self.strain_plot.push()

    def _toggle_rotation_controls(self):
        is_manual = self.rotation_option_select.value == 'Manual'
        self.rot_slider.disabled = not is_manual
        self.rotation_phase_select.disabled = is_manual

    def _get_current_rotation(self):
        option = self.rot_options[self.rotation_option_select.value]
        if option == 'Manual':
            return self.rot_slider.value
        phase_idx = self._show_phase_select_mapping.get(
            self.rotation_phase_select.value,
            None,
        )
        if phase_idx is None:
            return
        phase_map = self.strain_app.get_res_phasemap()
        phase = phase_map.phases[phase_idx]
        # Here using the hand-defined g1/g2 not the
        # g1/g2 from the results (+ref_region)
        # this is possible just even more plumbing!!!
        if option == 'g1':
            return phase.g1
        elif option == 'g2':
            return phase.g2
        elif option == 'g1 + g2':
            return phase.g1 + phase.g2
        elif option == 'g1 - g2':
            return phase.g1 - phase.g2
        else:
            raise KeyError(f'Option {option} not found.')

    def _update_phase_options(self, phase_map: PhaseMap):
        phase_entries = {
            f'Phase {i}{" - " + p.label if p.label else ""}': i
            for i, p in enumerate(phase_map.phases)
        }
        self._show_phase_select_mapping: dict[str, int | None] = {
            'None': None,
            **phase_entries,
        }
        self.show_phase_select.options = [*self._show_phase_select_mapping.keys()]
        self.rotation_phase_select.options = [*phase_entries.keys()]

    def _show_phase_ref(self, e):
        phase_map = self.strain_app.get_res_phasemap()
        if phase_map is None:
            return
        nav_shape = phase_map.idx_map.shape

        rectangles: Rectangles = self.strain_plot._mask_elements[0]
        try:
            selected = e.new
        except AttributeError:
            assert isinstance(selected, str)
            selected = e
        if selected == 'None':
            rectangles.clear()
            self.phase_pointset.clear()
            self.st_component_select.param.trigger('value')
            return

        phase_idx = self._show_phase_select_mapping[selected]
        phase = phase_map.phases[phase_idx]
        if phase.ref_idx is not None:
            cy, cx = np.unravel_index(phase.ref_idx, nav_shape)
            self.phase_pointset.update(x=[cx], y=[cy])
        if (ref_region := phase.ref_region) is None:
            rectangles.clear()
        else:
            h, w = map(lambda sl: 1 if isinstance(sl, int) else abs(sl.stop - sl.start), ref_region)
            cy, cx = map(
                lambda sl: sl if isinstance(sl, int) else (sl.stop + sl.start) / 2,
                ref_region
            )
            rectangles.update(x=[cx], y=[cy], width=[w], height=[h])
        self.st_component_select.param.trigger('value')

    def _update_phase_ref(self, e):
        selected = self.show_phase_select.value
        phase_map = self.strain_app.get_res_phasemap()
        if phase_map is None or selected == 'None':
            return
        ny, nx = phase_map.idx_map.shape

        rectangles: Rectangles = self.strain_plot._mask_elements[0]
        phase_idx = self._show_phase_select_mapping[selected]
        phase = phase_map.phases[phase_idx]
        if rectangles.data_length == 0:
            had_ref_region = phase.ref_region is not None
            phase.set_ref_region(None)
            if had_ref_region:
                self.strain_app.recompute_from_fit()
                self.st_component_select.param.trigger('value')
            return
        glyph = rectangles.rectangles
        data = rectangles.cds.data
        cx = data[glyph.x][-1]
        cy = data[glyph.y][-1]
        w2 = max(1, abs(data[glyph.width][-1])) / 2
        h2 = max(1, abs(data[glyph.height][-1])) / 2
        left = min(max(0, int(np.round(cx - w2))), nx)
        right = min(max(0, int(np.round(cx + w2))), nx)
        slx = slice(*sorted((left, right)))
        top = min(max(0, int(np.round(cy - h2))), ny)
        bottom = min(max(0, int(np.round(cy + h2))), ny)
        sly = slice(*sorted((top, bottom)))
        phase.set_ref_region((sly, slx))
        try:
            self.show_phase_select.param.trigger('value')
            self.strain_app.recompute_from_fit()
            self.st_component_select.param.trigger('value')
            # # Double trigger but will update the angle slider ?
            # self.rotation_option_select.param.trigger('value')
        except Exception as err:
            self.logger.log_from_exception(err)

    def get_job(
        self,
        state: UIState,
        dataset: DataSet | AcquisitionProtocol,
        roi: np.ndarray | None,
    ) -> UDFWindowJob | None:
        if self.strain_app.num_phases == 0:
            self.logger.info('Need at least one phase defined before running')
            return None

        phase_map = self.strain_app.get_phase_map(dataset.shape)
        if phase_map is None:
            self.logger.info('Need to run phase identification before strain analysis')
            return None

        strain_udf = MultiPhaseAutoCorrUDF.from_phase_map(
            phase_map,
            **self.strain_app.get_filter_kwargs(),
            return_pos=False,
        )
        self.live_plot.udf = strain_udf
        return UDFWindowJob(
            self,
            [strain_udf],
            [self.live_plot],
            result_handler=self.complete_job,
            params={'phase_map': phase_map},
            roi=self.live_plot.get_mask(dataset.shape.nav),
        )

    def complete_job(
        self,
        job: UDFWindowJob,
        job_results: JobResults,
    ) -> tuple[ResultRow, ...]:
        buffers = job_results.udf_results[0]
        g1f = buffers['g1'].data
        g2f = buffers['g2'].data
        phase_map = job.params['phase_map']
        phase_map: PhaseMap
        self.strain_app.new_fit(phase_map, g1f, g2f)
        self.st_component_select.param.trigger('value')
        self._update_phase_options(phase_map)
        return tuple()


class StrainApplication:
    def __init__(self, phases: list[PhaseDef] | None = None):
        self._phases: list[PhaseDef] = phases if phases is not None else []
        self._phase_map: PhaseMap | None = None
        self._fit: tuple[PhaseMap, np.ndarray, np.ndarray] | None = None
        self._strain_aa: StrainResult | None = None

    @classmethod
    def construct(self, ui_context: UIContextBase):
        application = StrainApplication()
        window_props = WindowPropertiesTDict(
            header_remove=False,
        )
        collapsed_props = WindowPropertiesTDict(
            init_collapsed=True
        )
        ui_context.add(
            'virtual_detector',
            window_props=collapsed_props,
        )
        ui_context.add(
            'frame_imaging',
            window_props=collapsed_props,
        )
        application.lattice_definer = ui_context._add(
            LatticeDefineWindow,
            window_props=window_props,
            window_data=application,
        )
        application.phase_mapper = ui_context._add(
            PhaseMapWindow,
            window_props=WindowPropertiesTDict(
                **window_props,
                **collapsed_props,
            ),
            window_data=application,
        )
        application.fitter = ui_context._add(
            LatticeFitterWindow,
            window_props=WindowPropertiesTDict(
                **window_props,
                **collapsed_props,
            ),
            window_data=application,
        )
        application.strain_mapper = ui_context._add(
            StrainAnalysis,
            window_props=window_props,
            window_data=application,
        )
        return application

    @property
    def lattice_definer(self) -> LatticeDefineWindow:
        return self._lattice_window

    @lattice_definer.setter
    def lattice_definer(self, window: LatticeDefineWindow):
        self._lattice_window = window

    @property
    def phase_mapper(self) -> PhaseMapWindow:
        return self._phase_mapper_window

    @phase_mapper.setter
    def phase_mapper(self, window: PhaseMapWindow):
        self._phase_mapper_window = window

    @property
    def fitter(self) -> LatticeFitterWindow:
        return self._fitter_window

    @fitter.setter
    def fitter(self, window: LatticeFitterWindow):
        self._fitter_window = window

    @property
    def strain_mapper(self) -> StrainAnalysis:
        return self._strain_mapper_window

    @strain_mapper.setter
    def strain_mapper(self, window: StrainAnalysis):
        self._strain_mapper_window = window

    @property
    def num_phases(self):
        return len(self._phases)

    def new_phases(self, phases: list[PhaseDef]):
        self._phases = phases
        self._phase_map = None

    def get_phases_for_run(self, shape: Shape) -> list[Phase | AmorphousPhase]:
        return [
            p.to_phase(
                shape.sig,
                shape.nav,
            ) for p in self._phases
        ]

    def new_phase_map(self, phase_map: PhaseMap):
        if len(phase_map.phases) != self.num_phases:
            self._phase_map = None
            raise ValueError('Phase map does not match number of defined phases, '
                             f'{len(phase_map.phases)} supplied, {self.num_phases} defined.')
        self._phase_map = phase_map

    def get_phase_map(self, shape: Shape) -> PhaseMap | None:
        if self.num_phases == 0:
            return None
        elif self.num_phases == 1:
            return PhaseMap.for_single_phase(
                shape.nav,
                self.get_phases_for_run(shape)[0],
            )
        else:
            return self._phase_map

    def get_filter_kwargs(self):
        return self.fitter.get_current_args()

    def new_fit(self, phase_map: PhaseMap, g1f: np.ndarray, g2f: np.ndarray):
        self._fit = (phase_map, g1f, g2f)
        return self.recompute_from_fit()

    def recompute_from_fit(self):
        if self._fit is None:
            return
        phase_map, g1f, g2f = self._fit
        self._strain_aa = phase_map.compute_strain(
            g1f,
            g2f,
        )
        return self._strain_aa

    def get_strain_aa(self) -> StrainResult | None:
        return self._strain_aa

    def get_res_phasemap(self) -> PhaseMap | None:
        if self._fit is None:
            return
        return self._fit[0]
