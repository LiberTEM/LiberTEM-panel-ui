from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple, Literal

import numpy as np
import panel as pn

from ...base import UIState, JobResults
from ...windows.base import (
    WindowType, UDFWindowJob, UIWindow,
    WindowProperties, WindowPropertiesTDict,
)
from ...live_plot import AperturePlot
from ...windows.pick import PickUDFWindow
from ...display.display_base import PointSet, Rectangles
from ...display.lattice import LatticeOverlay
from ...utils.panel_components import button_divider

from libertem.udf.sumsigudf import SumSigUDF
from libertem.utils import frame_peaks
import libertem.analysis.gridmatching as grm
from libertem_blobfinder.common.patterns import (
    MatchPattern, Circular, RadialGradient
)
from libertem_blobfinder.udf.correlation import (
    FastCorrelationUDF, SparseCorrelationUDF, FullFrameCorrelationUDF
)
from libertem_blobfinder.udf.refinement import (
    AffineMixin, FastmatchMixin
)


from .phase import Phase, PhaseMap, AmorphousPhase

if TYPE_CHECKING:
    import pandas as pd
    from libertem.api import DataSet
    from libertem_live.detectors.base.acquisition import AcquisitionProtocol
    from libertem.common.shape import Shape
    from .strain_decomposition import StrainResult
    from ...utils import PointYX
    from ...results.results_manager import ResultRow
    from ...base import UIContextBase


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
        return cls(
            g1=self.g1,
            g2=self.g2,
            centre=self.g0,
            ref_idx=ref_idx,
            label=self.label,
            radius=self.radius,
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
            'z', 'a', 'b'
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
                    if 'z' in label]
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


class LatticeFitterWindow(StrainAppMixin, PickUDFWindow, ui_type=STRAINAPP):
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
            'z', 'a', 'b'
        ).on(
            self.sig_plot.fig
        )
        # Could display the phase information under the toolbox
        self._template_dropdown = pn.widgets.Select(
            name="Template type",
            options=[
                "Circular",
                "Radial Gradient",
            ],
            value="Circular",
            width=200,
        )
        self._template_radius_factor_entry = pn.widgets.FloatInput(
            name="Radius factor",
            value=1.,
            step=0.1,
            width=100,
        )
        self._template_search_cbox = pn.widgets.Checkbox(
            name="Custom search",
            value=False,
            align="center",
        )
        self._template_search_entry = pn.widgets.FloatInput(
            value=2.0,
            step=0.1,
            disabled=True,
            width=100,
        )
        disable_code = """
target.disabled = !source.active
"""
        self._template_search_cbox.jslink(
            self._template_search_entry,
            code={'value': disable_code},
        )
        self._custom_grid_cbox = pn.widgets.Checkbox(
            name="Custom grid",
            value=False,
            align="center",
        )
        self._max_grid_input_1 = pn.widgets.IntInput(
            name="N1",
            value=2,
            start=1,
            end=10,
            step=1,
            width=100,
            disabled=True,
        )
        self._max_grid_input_2 = pn.widgets.IntInput(
            name="N2",
            value=2,
            start=1,
            end=10,
            step=1,
            width=100,
            disabled=True,
        )
        self._custom_grid_cbox.jslink(
            self._max_grid_input_1,
            code={'value': disable_code},
        )
        self._custom_grid_cbox.jslink(
            self._max_grid_input_2,
            code={'value': disable_code},
        )

        self._fitted_points = PointSet.new().empty().on(self.sig_plot.fig)
        self._standard_layout(
            left_after=(
                pn.Row(
                    self._template_dropdown,
                    self._template_radius_factor_entry,
                ),
                pn.Row(
                    self._template_search_cbox,
                    self._template_search_entry,
                ),
                pn.Row(
                    self._custom_grid_cbox,
                    self._max_grid_input_1,
                    self._max_grid_input_2,
                )
            )
        )

    @staticmethod
    def _build_blobfinder(
        dataset: DataSet,
        zero,
        a,
        b,
        match_pattern: MatchPattern,
        correlation_method: Literal['fast', 'full', 'sparse'],
        match_method: Literal['affine', 'fast'],
        grid_max: tuple[int, int] = None,
    ):
        if grid_max is None:
            n1, n2 = (10, 10)
        else:
            n1, n2 = grid_max
        indices = np.mgrid[-n1: n1 + 1, -n2: n2 + 1]

        (fy, fx) = tuple(dataset.shape.sig)

        indices, peaks = frame_peaks(
            fy=fy, fx=fx, zero=zero, a=a, b=b,
            r=match_pattern.search, indices=indices
        )
        peaks = peaks.astype('int')

        if correlation_method == 'fast':
            method = FastCorrelationUDF
        elif correlation_method == 'sparse':
            method = SparseCorrelationUDF
        elif correlation_method == 'fullframe':
            method = FullFrameCorrelationUDF
        else:
            raise ValueError(
                f"Unknown correlation method {correlation_method}. Supported are "
                "fast' and 'sparse'"
            )

        if match_method == 'affine':
            mixin = AffineMixin
        elif match_method == 'fast':
            mixin = FastmatchMixin
        else:
            raise ValueError(
                f"Unknown match method {match_method}. Supported are 'fast' and 'affine'"
            )

        # The inheritance order matters: FIRST the mixin, which calls
        # the super class methods.
        class MyUDF(mixin, method):
            pass

        return MyUDF(
            peaks=peaks,
            indices=indices,
            start_zero=zero,
            start_a=a,
            start_b=b,
            match_pattern=match_pattern,
            matcher=grm.Matcher(),
            steps=5,
            # zero_shift=zero_shift,
        )

    def _get_pattern(self, radius) -> MatchPattern:
        pattern_type = self._template_dropdown.value
        pattern_cls = {
            'Circular': Circular,
            'Radial Gradient': RadialGradient
        }[pattern_type]
        search = None
        if self._template_search_cbox.value:
            search = self._template_search_entry.value
        return pattern_cls(
            radius * self._template_radius_factor_entry.value,
            search=search,
        )

    def _get_udfs(self, dataset: DataSet):
        phase_map = self.strain_app.get_phase_map(dataset.shape)
        if phase_map is None:
            if self.strain_app.num_phases == 0:
                self.logger.warning('Need at least one phase defined to fit lattice')
            else:
                self.logger.warning('Need to compute phase map before fitting lattice')
            self._fitted_points.clear()
            return []
        elif all(isinstance(p, AmorphousPhase) for p in phase_map.phases):
            self.logger.warning('Cannot fit amorphous-only lattice definitions')
            self._fitted_points.clear()
            return []
        phase = phase_map.phases[0]
        grid_max = None
        if self._custom_grid_cbox.value:
            grid_max = (
                self._max_grid_input_1.value,
                self._max_grid_input_2.value,
            )
        kwargs = dict(
            dataset=dataset,
            zero=np.asarray([phase.centre.imag, phase.centre.real]),
            a=np.asarray([phase.g1.imag, phase.g1.real]),
            b=np.asarray([phase.g2.imag, phase.g2.real]),
            match_pattern=self._get_pattern(phase.radius),
            correlation_method='fast',
            match_method='affine',
            grid_max=grid_max,
        )
        lattice_udf = self._build_blobfinder(**kwargs)
        return [lattice_udf]

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
            with_udfs=[self._udf_pick] + udfs,
        )

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
        # Update points
        strain_results = job_results.udf_results[1]
        pos_fit = strain_results['refineds'].raw_data[0]
        self._fitted_points.update(
            x=pos_fit[:, 1], y=pos_fit[:, 0],
        )
        try:
            zero = strain_results['zero'].raw_data[0]
            apos = zero + strain_results['a'].raw_data[0]
            bpos = zero + strain_results['b'].raw_data[0]
            radius = job.udfs[1].get_pattern().radius
            ddict = {
                'cx': [apos[1], zero[1], bpos[1]],
                'cy': [apos[0], zero[0], bpos[0]],
                'radius': [radius, radius, radius],
            }
            self._lattice_set.cds.data.update(**ddict)
        except (ValueError, IndexError, TypeError, KeyError):
            # Better to do nothing than have a crash
            self.logger.error('Cannot update lattice display')
        self.sig_plot.push()
        return tuple()


class StrainAnalysisWindow(StrainAppMixin, UIWindow, ui_type=STRAINAPP):
    @staticmethod
    def default_properties():
        return WindowProperties(
            name='strain_analysis',
            title_md='Strain Analysis',
        )

    @staticmethod
    def _to_complex(array):
        return array[..., 0] * 1j + array[..., 1]

    def initialize(self, dataset: DataSet):
        class FakeRefineUDF(FastmatchMixin, SumSigUDF):
            ...

        self.live_plot = AperturePlot.new(
            dataset,
            FakeRefineUDF(peaks=[0, 1]),
            channel={
                'mod_a': ('a', lambda buffer: np.abs(self._to_complex(buffer).squeeze())),
                'mod_b': ('b', lambda buffer: np.abs(self._to_complex(buffer).squeeze())),
                'angle_a': ('a', lambda buffer: np.angle(self._to_complex(buffer).squeeze())),
                'angle_b': ('b', lambda buffer: np.angle(self._to_complex(buffer).squeeze())),
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
            '∥ a': 'a',
            '∥ b': 'b',
            '∥ a + b': 'a + b',
            '∥ a - b': 'a - b',
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
        if array is None:
            return
        self.strain_plot.im.update(array)
        self.strain_plot.fig.title.text = component
        self.strain_plot.push()

    def _get_strain_comp(self, component: str, rotation: float | complex) -> np.ndarray | None:
        if (strain_aa := self.strain_app.get_strain_aa()) is None:
            self.logger.info('No strain data to get')
            return
        if isinstance(rotation, (np.floating, float, int)):
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
        if array is None:
            return
        self.strain_plot.im.update(array)
        self.strain_plot.push()

    def _toggle_rotation_controls(self):
        is_manual = self.rotation_option_select.value == 'Manual'
        self.rot_slider.disabled = not is_manual

    def _get_current_rotation(self):
        option = self.rot_options[self.rotation_option_select.value]
        if option == 'Manual':
            return self.rot_slider.value
        phase_map = self.strain_app.get_res_phasemap()
        if phase_map is None or len(phase_map.phases) == 0:
            return
        else:
            phase_idx = 0
        phase = phase_map.phases[phase_idx]
        # Here using the hand-defined g1/g2 not the
        # g1/g2 from the results (+ref_region)
        # this is possible just even more plumbing!!!
        if option == 'a':
            return phase.g1
        elif option == 'b':
            return phase.g2
        elif option == 'a + b':
            return phase.g1 + phase.g2
        elif option == 'a - b':
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

        strain_udf_list = self.strain_app.fitter._get_udfs(dataset)
        if len(strain_udf_list) == 0:
            self.logger.info("Nothing to run")
            return
        self.live_plot.udf = strain_udf_list[0]
        return UDFWindowJob(
            self,
            strain_udf_list,
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
        g1f = buffers['a'].data
        g2f = buffers['b'].data
        g1f = g1f[..., 1] + g1f[..., 0] * 1j
        g2f = g2f[..., 1] + g2f[..., 0] * 1j
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
        application.fitter = ui_context._add(
            LatticeFitterWindow,
            window_props=WindowPropertiesTDict(
                **window_props,
                # **collapsed_props,
            ),
            window_data=application,
        )
        application.strain_mapper = ui_context._add(
            StrainAnalysisWindow,
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
    def fitter(self) -> LatticeFitterWindow:
        return self._fitter_window

    @fitter.setter
    def fitter(self, window: LatticeFitterWindow):
        self._fitter_window = window

    @property
    def strain_mapper(self) -> StrainAnalysisWindow:
        return self._strain_mapper_window

    @strain_mapper.setter
    def strain_mapper(self, window: StrainAnalysisWindow):
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
