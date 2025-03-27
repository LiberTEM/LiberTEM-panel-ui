import platform
import os
import re
import contextlib
from typing import Optional, List
import numpy as np
import datetime
import pathlib
import panel as pn
import pandas as pd


pn.extension(
    'tabulator',
)


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


def is_hidden(f: pathlib.Path, stat_result: Optional[os.stat_result] = None) -> bool:
    is_dotfile = f.name.startswith('.')
    if stat_result is not None and platform.system() == 'Windows':
        try:
            attr_hidden = stat_result.st_file_attributes.FILE_ATTRIBUTE_HIDDEN
        except AttributeError:
            attr_hidden = False
        return is_dotfile or attr_hidden
    return is_dotfile


def add_file_info(acc: dict[str, list], f: pathlib.Path):
    fstat = os.stat(f)
    acc['Type'].append('F')
    acc['Name'].append(f.name)
    acc['Size'].append(sizeof_fmt(fstat.st_size))
    acc['Last Modified'].append(datetime.datetime.fromtimestamp(fstat.st_mtime))
    acc['Extension'].append(f.suffix)
    acc['Hidden'].append(is_hidden(f, fstat))


def add_folder_info(acc: dict[str, list], f: pathlib.Path):
    acc['Type'].append('D')
    acc['Name'].append(f.name)
    acc['Size'].append('')
    acc['Last Modified'].append(None)
    acc['Extension'].append('')
    acc['Hidden'].append(is_hidden(f))


def get_folder_contents(path: pathlib.Path):
    acc = {}
    acc['Type'] = ['D', 'D']
    acc['Name'] = ['..', '.']
    acc['Size'] = ['', '']
    acc['Last Modified'] = [None, None]
    acc['Extension'] = ['', '']
    acc['Hidden'] = [False, False]

    for f in path.iterdir():
        if f.is_file():
            add_file_info(acc, f)
        else:
            add_folder_info(acc, f)
    return acc


def highlight_dir(row):
    if row['Type'] == 'D':
        base = np.full((row.size,), "font-weight: bold; background-color: LightGray;")
        base[1:] = "font-weight: bold;"
        return base
    else:
        return [''] * row.size


def get_df(path: pathlib.Path, filters: Optional[List[str]] = None):
    contents = get_folder_contents(path)
    df = pd.DataFrame.from_dict(data=contents)
    df = filter_by_choice(df, filters)
    df = df.sort_values(['Type', 'Name'], ascending=[True, True])
    return df


def folder_mask(df):
    return df['Type'].str.fullmatch('D', case=False)


def visible_mask(df):
    return np.logical_not(df['Hidden'])


def filter_by_regex(
    df: pd.DataFrame,
    pattern: str,
    filter_folders: Optional[bool] = False,
    # show_hidden: Optional[bool] = True,
):
    if pattern:
        try:
            mask = df['Name'].str.contains(pattern, case=False)
            if not filter_folders:
                mask = mask | folder_mask(df)
            # if not show_hidden:
            #     mask = mask & visible_mask(df)
            sub_df = df[mask]
            if not sub_df.empty:
                return sub_df
        except re.error:
            pass
    return df


def filter_by_choice(
    df: pd.DataFrame,
    extensions: list[str],
    filter_folders: Optional[bool] = False,
    show_hidden: Optional[bool] = True,
):
    if not extensions:
        if not show_hidden:
            sub_df = df[visible_mask(df)]
            if not sub_df.empty:
                return sub_df
        return df
    mask = df['Extension'].str.fullmatch('|'.join(extensions), case=False)
    if not filter_folders:
        mask = mask | folder_mask(df)
    if not show_hidden:
        mask = mask & visible_mask(df)
    sub_df = df[mask]
    if sub_df.empty:
        return df
    return sub_df


class FileBrowser:
    def __init__(
        self,
        root: Optional[pathlib.Path] = '.',
        fixed_filters: Optional[List[str]] = None,
        optional_filters: Optional[List[str]] = None,
    ):
        root = pathlib.Path(root).resolve()
        self.home_dir = root
        self.history = []
        self.current_dir = root
        self.fixed_filters = fixed_filters
        if optional_filters is None:
            optional_filters = []
        self.optional_filters = optional_filters

    def layout(self):
        self.df = get_df(self.current_dir, filters=self.fixed_filters)
        self.df_widget = pn.widgets.Tabulator(
            self.df.style.apply(highlight_dir, axis=1),
            editors={k: None for k in self.df.columns},
            show_index=False,
            hidden_columns=['Extension', 'Hidden'],
            text_align={k: 'right' for k in self.df.columns if k != 'Type'},
            selectable=1,
            height=500,
            widths={
                'Type': '8%',
                'Name': '50%',
                'Size': '15%',
                'Last Modified': '20%',
            },
            width=750,
        )
        self.df_widget.on_click(self.navigate_cb)

        self.current_address_input = pn.widgets.TextInput(
            name='Current folder:',
            value=str(self.current_dir),
            sizing_mode='stretch_width'
        )
        self.filter_pattern_input = pn.widgets.TextInput(
            name='Filter (press enter or click away to run):',
            value='',
            placeholder='Filename filter, case-insensitive, regex accepted',
            sizing_mode='stretch_width',
            height=60,
        )

        button_params = {
            'button_type': 'primary',
            'align': 'end',
            'width': 40,
            'height': 40,
            'margin': (5, 5),
        }
        self.home_button = pn.widgets.Button(name='🏠', **button_params)
        self.back_button = pn.widgets.Button(name='🡄', **button_params)
        self.up_button = pn.widgets.Button(name='🡅', **button_params)
        self.refresh_button = pn.widgets.Button(name='⭮', **button_params)
        self.filter_folders_cbox = pn.widgets.Checkbox(
            name='Filter folders',
            value=False,
            width_policy='min',
            align='center',
        )
        self.show_hidden_cbox = pn.widgets.Checkbox(
            name='Show hidden',
            value=True,
            width_policy='min',
            align='center',
        )

        self.df_widget.add_filter(
            pn.bind(
                filter_by_regex,
                pattern=self.filter_pattern_input,
                filter_folders=self.filter_folders_cbox,
            )
        )

        def handle_up(event):
            self.update_widgets(self.current_dir / '..')

        self.up_button.on_click(handle_up)

        def handle_home(event):
            self.update_widgets(self.home_dir)

        self.home_button.on_click(handle_home)

        def handle_refresh(event):
            self.update_widgets(self.current_address_input.value, force=True)

        self.refresh_button.on_click(handle_refresh)

        def handle_back(event):
            while len(self.history) > 0:
                previous = self.history.pop(-1)
                if previous != self.current_dir:
                    self.update_widgets(previous, add_to_history=False)
                    return

        self.back_button.on_click(handle_back)

        self.extension_filter = pn.widgets.MultiChoice(
            name='Filter extension:',
            options=self.optional_filters,
            min_height=55,
        )
        self.df_widget.add_filter(
            pn.bind(
                filter_by_choice,
                extensions=self.extension_filter,
                filter_folders=self.filter_folders_cbox,
                show_hidden=self.show_hidden_cbox,
            )
        )

        self._layout = pn.Column(
            pn.Row(
                self.home_button,
                self.back_button,
                self.up_button,
                self.current_address_input,
                self.refresh_button
            ),
            pn.Row(
                self.filter_pattern_input,
                pn.Column(
                    self.filter_folders_cbox,
                    self.show_hidden_cbox,
                    align='center',
                ),
                self.extension_filter,
            ),
            self.df_widget,
        )
        return self._layout

    def update_widgets(
        self,
        new_path: pathlib.Path,
        force: bool = False,
        add_to_history: bool = True
    ):
        try:
            new_path = pathlib.Path(new_path).resolve()
        except (TypeError, ValueError, PermissionError):
            return
        if (new_path == self.current_dir) and not force:
            return
        try:
            df = get_df(new_path, filters=self.fixed_filters)
        except (OSError, PermissionError):
            return
        self.df = df
        if add_to_history:
            self.history.append(self.current_dir)
        self.current_dir = new_path
        self.current_address_input.value = str(self.current_dir)
        self.df_widget.value = df

    def navigate_cb(self, event):
        row = self.df.iloc[int(event.row)]
        if row['Type'] != 'D':
            return
        self.update_widgets(self.current_dir / row['Name'])

    @contextlib.contextmanager
    def disable_layout(self):
        self._layout.disabled = True
        yield
        self._layout.disabled = False

    def get_selection_path(self) -> Optional[pathlib.Path]:
        # Could extend to multi-file select here
        selected = self.df_widget.selection
        if not selected:
            return None
        selected = selected[0]
        row = self.df.iloc[int(selected)]
        return (self.current_dir / row['Name'])
