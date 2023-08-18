import pathlib
from typing import Optional, Type
import panel as pn

import libertem.api as lt
from libertem.io.dataset import filetypes, get_dataset_cls, get_search_order
from libertem.io.dataset.base.backend import IOBackend
from libertem.io.dataset.base.exceptions import DataSetException

from ..utils.param_converters import _converters
from .file_browser import FileBrowser


class LoadDatasetWindow:
    def __init__(
        self,
        path: Optional[pathlib.Path] = '.',
        dataset_key: Optional[str] = None,
        ctx: Optional[lt.Context] = None,
        ds: Optional[lt.DataSet] = None,
    ):
        path = pathlib.Path(path)
        if path.is_file():
            self._path = path
            self._root = path.parent
        else:
            self._path = None
            self._root = path

        self._dataset_map: dict[str, Type[lt.DataSet]] = {k: get_dataset_cls(k)
                                                          for k
                                                          in filetypes.keys()}
        self._dataset_key: Optional[str] = dataset_key
        self._dataset: Optional[lt.DataSet] = ds

        if ctx is None:
            ctx = lt.Context.make_with('inline')
        self.ctx = ctx

    def get_ds(self):
        return self._dataset

    def layout(self, with_reload: bool = True):
        if with_reload:
            from ..utils.notebook_tools import get_ipyw_reload_button
            get_ipyw_reload_button()
        self._layout = pn.Column()
        self._refresh_layout()
        return self._layout

    def _refresh_layout(self):
        self._layout.clear()

        if self._path is None:
            self._select_file_layout()
        elif self._dataset_key is None:
            self._select_dataset_layout()
        elif self._dataset is None:
            self._ds_params_layout()
        else:
            self._ds_loaded_layout()

    def _select_file_layout(self):
        self._layout.append(pn.pane.Markdown(object='## Select file'))

        fb = FileBrowser(root=self._root)
        fb_layout = fb.layout()
        self._layout.extend(fb_layout)

        load_btn = pn.widgets.Button(
            name='Load file',
            button_type='success',
            max_width=200,
            height=50,
        )

        def try_load(*e):
            selection = fb.get_selection_path()
            if selection is None:
                return
            self._path = selection
            self._root = self._path.parent
            self._refresh_layout()

        load_btn.on_click(try_load)
        self._layout.append(load_btn)

    def get_ds_options_list(self):
        if not self._path:
            return [*self._dataset_map.keys()]
        return get_search_order(self._path)

    def _select_dataset_layout(self):
        self._layout.append(pn.pane.Markdown(object='## Select dataset type'))

        ds_type_select = pn.widgets.Select(
            name='Dataset type',
            options=self.get_ds_options_list()
        )
        self._layout.append(ds_type_select)

        select_btn = pn.widgets.Button(
            name='Select',
            button_type='success',
            max_width=200,
            height=50,
        )

        back_btn = pn.widgets.Button(
            name='Back',
            button_type='warning',
            max_width=200,
            height=50,
        )

        def _select_ds_type(*e):
            self._dataset_key = ds_type_select.value
            self._refresh_layout()

        select_btn.on_click(_select_ds_type)

        def _go_back(*e):
            self._dataset = None
            self._dataset_key = None
            self._path = None
            self._refresh_layout()

        back_btn.on_click(_go_back)

        self._layout.append(pn.Row(select_btn, back_btn))

    def _ds_params_layout(self):
        dataset_cls = self._dataset_map[self._dataset_key]
        path = self._path

        try:
            detected = dataset_cls.detect_params(str(path), self.ctx.executor)
        except NotImplementedError:
            detected = False
        if detected:
            detected = detected['parameters']
        widgets, callbacks = self.schema_to_widgets(
            dataset_cls.get_msg_converter().SCHEMA,
            detected_params=detected
        )
        md = pn.pane.Markdown(
            object=f'''## Dataset parameters

- **Class:** {str(dataset_cls.__name__)}
- **Path:** {path}
- **Auto-detected:** {isinstance(detected, dict)}
''',
            min_width=700,
        )
        load_button = pn.widgets.Button(
            name='Load',
            button_type='success',
            max_width=200,
            height=50,
        )
        back_button = pn.widgets.Button(
            name='Back',
            button_type='warning',
            max_width=200,
            height=50,
        )

        def _load_ds(*e):
            params = {'path': self._path}
            for key, cb in callbacks.items():
                valid, value = cb()
                if valid:
                    params[key] = value
            if 'io_backend' in params:
                params['io_backend'] = IOBackend.get_cls_by_id(params['io_backend'])()
            try:
                self._dataset = dataset_cls(**params)
                self._dataset.initialize(self.ctx.executor)
            except DataSetException:
                self._dataset = None
                return
            self._refresh_layout()

        load_button.on_click(_load_ds)

        def _go_back(*e):
            self._dataset = None
            self._dataset_key = None
            self._refresh_layout()

        back_button.on_click(_go_back)

        self._layout.extend(
            (
                md,
                *widgets,
                pn.Row(load_button, back_button)
            )
        )

    def _ds_loaded_layout(self):
        md = pn.pane.Markdown(
            object=f'''## Dataset loaded

- **Path:** {self._path}
- **Class:** {self._dataset.__class__.__name__}
- **Description:** {repr(self._dataset)}
''',
            min_width=700,
        )

        back_button = pn.widgets.Button(
            name='Back',
            button_type='warning',
            max_width=200,
            height=50,
        )

        def _go_back(*e):
            self._dataset = None
            self._refresh_layout()

        back_button.on_click(_go_back)

        self._layout.extend(
            (
                md,
                back_button,
            )
        )

    @classmethod
    def schema_to_widgets(cls, schema, detected_params=None):
        if not detected_params:
            detected_params = {}
        props = schema['properties']
        return cls._props_to_widgets(props, detected_params)

    @staticmethod
    def _props_to_widgets(props, detected_params):
        widgets = []
        callbacks = {}
        fields = {}

        # This is no longer valid code !
        def null_coverter(x, y):
            return True, []

        for key, definition in props.items():
            fields[key] = {}
            # Find converter
            if key == 'path':
                continue
            if key in _converters.keys():
                _converter = _converters[key]
            elif 'type' in definition.keys():
                typename = definition['type']
                _converter = _converters[typename]
            else:
                def_keys = [*definition.keys()]
                _converter = null_coverter
                for k in def_keys:
                    if k in _converters.keys():
                        _converter = _converters[k]
                        break

            _widgets, cb = _converter(key, definition,
                                      default=detected_params.get(key, None))

            for _w in _widgets:
                widgets.append(pn.Row(*[*_w.values()]))
                fields[key].update(_w)

            if _widgets:
                callbacks[key] = cb

        return widgets, callbacks


if __name__ == '__main__':
    loader = LoadDatasetWindow()
    loader.layout().servable()
