from __future__ import annotations
import datetime
import uuid
import json
from typing import Any, NamedTuple, TYPE_CHECKING, Iterator
import os
import pathlib

import panel as pn
import humanize
import pandas as pd
from libertem.common.shape import Shape


pn.extension(
    'jsoneditor',
    'tabulator',
)


if TYPE_CHECKING:
    from ..base import UIContextBase
    from ..windows.base import UIWindow
    from .containers import ResultContainer


class ResultRow(NamedTuple):
    result_id: str  # primarykey
    name: str
    run_id: int | None
    window_id: str | None
    timestamp: datetime.datetime
    params: dict[str, Any]
    result_type: type[ResultContainer]
    result_repr: str

    def __hash__(self) -> int:
        return hash(self.result_id)

    def __eq__(self, other) -> bool:
        try:
            self.result_id == other.result_id
        except AttributeError:
            raise NotImplementedError('Cannot compare')

    @property
    def tags(self) -> list[str]:
        return self.params.get('tags', [])


class WindowRow(NamedTuple):
    window_id: str
    window_name: str

    def __hash__(self) -> int:
        return hash(self.window_id)

    def __eq__(self, other) -> bool:
        try:
            self.window_id == other.window_id
        except AttributeError:
            raise NotImplementedError('Cannot compare')


class WindowRunRow(NamedTuple):
    # could multi-index between window_id and run_id as 'primarykey'
    window_id: str
    run_id: int
    params: dict[str, Any]
    timestamp: datetime.datetime


class RunRow(NamedTuple):
    run_id: int  # primarykey
    params: dict[str, Any]
    timestamp: datetime.datetime

    @property
    def ds_shape(self) -> Shape | None:
        shape_dict = self.params.get('shape', None)
        if shape_dict is None:
            return
        nav = tuple(shape_dict['nav'])
        sig = tuple(shape_dict['sig'])
        return Shape(nav + sig, sig_dims=len(sig))

    @property
    def has_roi(self) -> bool | None:
        return self.params.get('has_roi', None)


class ResultsManager:
    def __init__(self, save_root: os.PathLike | None = '.'):
        # Where to save files by default, None implies no saving
        if save_root is not None:
            self._save_root = pathlib.Path(save_root)
        else:
            self._save_root = save_root
        # This is basically a simple relational database!
        # Could implement it in squite3 without too much effort
        self._runs: list[RunRow] = []
        self._window_runs: list[WindowRunRow] = []
        self._windows: list[WindowRow] = []
        self._results: list[ResultRow] = []
        self._result_data: dict[str, ResultContainer] = {}
        self._watchers: list[UIContextBase] = []

        self._show_area = pn.Column(min_height=400)
        self._layout = pn.Column()

    @property
    def all_results(self):
        return self._results

    @property
    def save_root(self):
        return self._save_root

    def change_save_root(self, save_root: os.PathLike):
        self._save_root = pathlib.Path(save_root)

    def add_watcher(self, watcher: UIContextBase):
        self._watchers.append(watcher)

    def _next_run_id(self) -> int:
        try:
            return max(r.run_id for r in self._runs) + 1
        except ValueError:
            return 0

    def _next_result_id(self) -> str:
        return str(uuid.uuid4())[:6]

    def _get_timestamp(self) -> datetime.datetime:
        return datetime.datetime.now()

    def new_run(self, **params) -> RunRow:
        row = RunRow(
            run_id=self._next_run_id(),
            params=params,
            timestamp=self._get_timestamp(),
        )
        self._runs.append(row)
        return row

    def new_window(self, window: UIWindow):
        if window.ident in tuple(w.window_id for w in self._windows):
            return
        self._windows.append(
            WindowRow(
                window_id=window.ident,
                window_name=window.properties.title_md,
            )
        )

    def new_window_run(
        self,
        window: UIWindow,
        run_id: RunRow | str,
        params: dict[str, Any] | None = None,
    ) -> WindowRunRow:

        self.new_window(window)

        if params is None:
            params = {}
        if isinstance(run_id, RunRow):
            run_id = run_id.run_id

        row = WindowRunRow(
            run_id=run_id,
            window_id=window.ident,
            params=params,
            timestamp=self._get_timestamp()
        )
        # uniqueness check
        if (row.window_id, row.run_id) in tuple((r.window_id, r.run_id)
                                                for r in self._window_runs):
            # Could have an add_params method which merges
            raise RuntimeError('Cannot add same window--run twice')
        self._window_runs.append(row)
        return row

    def new_result(
        self,
        result_container: ResultContainer,
        run_id: RunRow | int | None = None,
        window_id: WindowRunRow | str | None = None,
    ) -> ResultRow:

        if isinstance(run_id, RunRow):
            run_id = run_id.run_id

        if isinstance(window_id, WindowRunRow):
            if run_id is not None:
                assert run_id == window_id.run_id, 'Mismatching run_id'
            run_id = window_id.run_id
            window_id = window_id.window_id

        row = ResultRow(
            result_id=self._next_result_id(),
            run_id=run_id,
            window_id=window_id,
            timestamp=self._get_timestamp(),
            params=result_container.meta,
            name=result_container.name,
            result_type=result_container.__class__,
            result_repr=result_container.table_repr(),
        )
        self._results.append(row)
        self._result_data[row.result_id] = result_container
        return row

    def windows_df(self):
        df = pd.DataFrame(self._windows, columns=WindowRow._fields)
        return df.set_index('window_id', drop=True)

    def results_df(self):
        df = pd.DataFrame(self._results, columns=ResultRow._fields)
        df = df.astype({'run_id': pd.Int32Dtype()})
        return df.set_index('result_id', drop=True)

    def window_runs_df(self):
        df = pd.DataFrame(self._window_runs, columns=WindowRunRow._fields)
        df = df.astype({'run_id': pd.Int32Dtype()})
        return df.set_index(['window_id', 'run_id'], drop=True)

    def runs_df(self):
        df = pd.DataFrame(self._runs, columns=RunRow._fields)
        df = df.astype({'run_id': pd.Int32Dtype()})
        return df.set_index('run_id', drop=True)

    def display_df(self):
        results_df = self.results_df()
        windows_df = self.windows_df()
        window_names = list(
            self.take_if(
                windows_df['window_name'],
                results_df['window_id'],
                fill_value='-',
            )
        )
        window_names = pd.Series(window_names, index=results_df.index, dtype=str)
        now = datetime.datetime.now()
        time_since = results_df['timestamp'].apply(lambda x: humanize.naturaltime(now - x))
        time_since = time_since.astype(str)
        when_time = results_df['timestamp'].apply(lambda x: x.time().isoformat('seconds'))
        when_time = when_time.astype(str)
        to_drop = ['result_type']
        return results_df.drop(columns=to_drop).assign(
            when_time=when_time,
            window_name=window_names,
            time_since=time_since,
        ).fillna({'window_id': '-'})

    def _columns_to_show(self):
        return ['when_time', 'time_since', 'run_id', 'window_id',
                'window_name', 'name', 'result_repr']

    def get_table_df(self):
        ddf = self.display_df()
        columns_to_show = self._columns_to_show()
        hidden_columns = [c for c in ddf.columns if c not in columns_to_show]
        return ddf.reindex(columns=columns_to_show + hidden_columns), hidden_columns

    def take_if(self, series, idcs, fill_value=None):
        for idx in idcs:
            try:
                yield series[idx]
            except (IndexError, KeyError):
                yield fill_value

    def get_result_container(self, result_id: str) -> ResultContainer | None:
        return self._result_data.get(result_id, None)

    def get_result_row(self, result_id: str) -> ResultRow | None:
        for row in self._results:
            if row.result_id == result_id:
                return row
        return None

    def get_window(self, window_id: str) -> WindowRow | None:
        for row in self._windows:
            if row.window_id == window_id:
                return row
        return None

    def get_run(self, run_id: str) -> RunRow | None:
        for row in self._runs:
            if row.run_id == run_id:
                return row
        return None

    def delete_result(self, result_id: str) -> ResultContainer | None:
        rc = self._result_data.pop(result_id, None)
        if rc is None:
            return
        deleted_rows = list(r for r in self._results if r.result_id == result_id)
        self._results = list(r for r in self._results if r.result_id != result_id)
        for w in self._watchers:
            w.notify_deleted_results(*deleted_rows)
        return rc

    def get_combined_params(
        self, result_id: str,
        include_empty: bool = True
    ) -> dict[str, Any] | None:
        combined_params = {}
        rc = self.get_result_container(result_id)
        if rc is None:
            return
        result_row = tuple(r for r in self._results if r.result_id == result_id)[0]
        if include_empty or result_row.params:
            combined_params['result'] = result_row.params
        if result_row.run_id is not None:
            run_row = tuple(r for r in self._runs if r.run_id == result_row.run_id)[0]
            if include_empty or run_row.params:
                combined_params['run'] = run_row.params
            if result_row.window_id is not None:
                window_run_row = tuple(r for r
                                       in self._window_runs
                                       if r.window_id == result_row.window_id
                                       and r.run_id == result_row.run_id)[0]
                if include_empty or window_run_row.params:
                    combined_params['window'] = window_run_row.params
        return combined_params

    def _make_layout(self):
        # # needs dependency, hardcoded as string instead
        # tabulator_formatters = {
        #     'time_since': {'type': 'datetimediff', 'humanize': True},
        # }

        titles = {
            'result_id': 'id',
            'when_time': 'When',
            'time_since': '(ago)',
            'run_id': 'Run-#',
            'window_id': 'Win-#',
            'window_name': 'Window name',
            'name': 'Result name',
            'result_repr': 'Result description',
        }

        refresh_btn = pn.widgets.Button(
            name='Refresh table',
            button_type='primary',
            max_width=100,
            align='center',
        )
        clear_btn = pn.widgets.Button(
            name='Clear preview',
            button_type='warning',
            max_width=100,
            align='center',
        )
        delete_btn = pn.widgets.Button(
            name='Delete selection',
            button_type='danger',
            max_width=100,
            align='center',
        )
        # gb_text = pn.widgets.StaticText(
        #     value='Group-By:',
        #     align='center',
        # )
        # gb_select = pn.widgets.RadioButtonGroup(
        #     name='Group-by',
        #     value='None',
        #     options=['None', 'Run #', 'Window #'],
        #     button_type='success',
        #     max_width=300,
        #     align='center',
        # )
        table_df, hidden_columns = self.get_table_df()
        df_widget = pn.widgets.Tabulator(
            table_df,
            hidden_columns=hidden_columns,
            show_index=True,
            groupby=[],
            editors={k: None for k in table_df.columns},
            selectable='checkbox',
            min_width=750,
            sorters=[{'field': 'run_id', 'dir': 'dsc'}],
            titles=titles,
            buttons={
                'show_btn': '🔍',
                'del_btn': '✕',
            },
        )

        # def _change_gb(e):
        #     if not e.new or e.new == 'None':
        #         df_widget.groupby = []
        #         return
        #     matches = tuple(k for k, v in titles.items() if e.new == v)
        #     if not matches:
        #         return
        #     new_gb = matches[0]
        #     df_widget.groupby = [new_gb]

        # gb_select.param.watch(_change_gb, 'value')

        current_preview = None

        def _run_show(e):
            if e.column != 'show_btn':
                return
            row: pd.Series = df_widget.value.iloc[e.row]
            rc = self.get_result_container(row.name)
            if rc is None:
                return
            _do_clear_show(e)
            self._show_area.extend(self.row_show(row))
            self._show_area.append(rc.show(standalone=False))
            nonlocal current_preview
            current_preview = row.name

        def _run_delete(e):
            if e.column != 'del_btn':
                return
            row: pd.Series = df_widget.value.iloc[e.row]
            rc = self.delete_result(row.name)
            if rc is None:
                return
            _do_refresh(e)
            if row.name == current_preview:
                _do_clear_show(e)

        def _table_on_click(e):
            if e.column == 'show_btn':
                _run_show(e)
            elif e.column == 'del_btn':
                _run_delete(e)

        df_widget.on_click(_table_on_click)

        def _do_refresh(e):
            new_df, _ = self.get_table_df()
            df_widget.value = new_df

        refresh_btn.on_click(_do_refresh)

        def _do_clear_show(e):
            nonlocal current_preview
            self._show_area.clear()
            current_preview = None

        clear_btn.on_click(_do_clear_show)

        def _do_delete_selection(e):
            selected = df_widget.selection
            if not selected:
                return
            for idx in selected:
                row: pd.Series = df_widget.value.iloc[idx]
                self.delete_result(row.name)
                if row.name == current_preview:
                    _do_clear_show(e)
            _do_refresh(e)

        delete_btn.on_click(_do_delete_selection)

        self._layout.clear()
        self._layout.extend((
            pn.Row(
                pn.pane.Markdown(object='## Result Manager'),
                refresh_btn,
            ),
            pn.Row(
                # gb_text,
                # gb_select,
                pn.layout.HSpacer(),
                delete_btn,
            ),
            df_widget,
            pn.Row(pn.layout.HSpacer(), clear_btn),
            self._show_area,
        ))
        return self._layout

    def layout(self, with_reload: bool = True):
        if with_reload:
            from ..utils.notebook_tools import get_ipyw_reload_button
            get_ipyw_reload_button()
        if len(self._layout) == 0:
            self._make_layout()
        return self._layout

    def row_show(self, row: pd.Series):
        lo = [pn.layout.Divider()]
        timestamp_time = row["timestamp"].time().isoformat("seconds")
        time_ago = humanize.naturaltime(datetime.datetime.now() - row["timestamp"])
        title_text = row["name"]
        if row["window_name"] and row["window_name"] not in ('-',):
            title_text = f'{row["window_name"]} :: {title_text}'
        # NOTE row.name is the pd.Series.name (index value) property, not the actual 'Result Name'
        result_row: ResultRow = self.get_result_row(row.name)
        md = pn.pane.Markdown(object=f'''### {title_text}

- **id**: {row.name}
- **Generated at**: {timestamp_time} ({time_ago})
- **Window**: {row["window_name"]} - *{row["window_id"]}*
- **Run #**: {row["run_id"]}
- **Description**: {row["result_repr"]}
- **Tags**: {result_row.tags}
''', min_width=400)
        combined_params = self.get_combined_params(row.name, include_empty=False)
        if combined_params:
            try:
                _ = json.dumps(combined_params)
                getter = pn.pane.Markdown(object="**Param getter**: `manager."
                                          f"get_combined_params('{row.name}')`")
                param_view = pn.Column(
                    getter,
                    self.params_as_json(combined_params)
                )
            except (TypeError, ValueError):
                param_view = pn.pane.Markdown(
                    object="**Cannot provide JSON preview**, use: "
                    f"`manager.get_combined_params('{row.name}')`",
                    max_width=400,
                    align=('start', 'center'),
                )
        else:
            param_view = pn.pane.Markdown(
                object="**No parameters registered with result**",
                max_width=400,
                align=('start', 'center'),
            )
        lo.append(pn.Row(md, pn.layout.HSpacer(), param_view))
        return lo

    def params_as_json(self, params):
        return pn.widgets.JSONEditor(
            value=params,
            mode='view',
            min_height=250,
            width=400,
        )

    def yield_with_tag(
        self,
        *tags: str,
        match_all: bool = False,
        from_rows: tuple[ResultRow] | None = None,
        from_windows: tuple[WindowRow] | None = None,
    ) -> Iterator[ResultRow]:
        tags: set[str] = set(tags)
        if not tags:
            raise ValueError('Need at least one tag to search')
        result_iter = self._results
        if from_rows is not None:
            if len(from_rows) == 0:
                raise ValueError("Must supply rows to search")
            result_iter = from_rows
        if from_windows is None:
            window_ids = None
        else:
            window_ids = tuple(w.window_id for w in from_windows)
            if len(window_ids) == 0:
                raise ValueError("Must supply windows to search")
        for result in result_iter:
            if window_ids and result.window_id not in window_ids:
                continue
            result_tags = set(result.tags)
            if not result_tags:
                continue
            intersection = tags.intersection(result_tags)
            if not intersection:
                continue
            elif match_all and len(intersection) != len(tags):
                continue
            else:
                yield result

    def yield_of_type(
        self,
        *result_type: type[ResultContainer],
        from_rows: tuple[ResultRow] | None = None,
    ) -> Iterator[ResultRow]:
        result_iter = self._results
        if from_rows is not None:
            if len(from_rows) == 0:
                raise ValueError("Must supply rows to search")
            result_iter = from_rows
        for result in result_iter:
            if result.result_type in result_type:
                yield result
