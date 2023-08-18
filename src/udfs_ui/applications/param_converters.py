from __future__ import annotations
import pathlib
import functools
import numpy as np
import panel as pn


replacements = {'io': 'I/O',
                'path': 'Path',
                'sync': 'Sync',
                'offset': 'Offset',
                'nav': 'Nav',
                'sig': 'Sig'}


def _boolean_status(validator):
    current_valid = validator() is not False
    status_indicator_green = pn.indicators.BooleanStatus(value=True,
                                                         color='success',
                                                         margin=(0, 0),
                                                         visible=current_valid)
    status_indicator_red = pn.indicators.BooleanStatus(value=True,
                                                       color='danger',
                                                       margin=(0, 0),
                                                       visible=not current_valid)

    def _set_validity(*events):
        if events[0].new != events[0].old:
            validated = validator()
            status_indicator_red.visible = not bool(validated)
            status_indicator_green.visible = bool(validated)

    return pn.Row(status_indicator_green, status_indicator_red,
                  margin=(5, 5, 12, 5),
                  sizing_mode='fixed',
                  align='end'), _set_validity


def _display_keytext(key):
    split_text = key.split('_')
    split_text = [replacements.get(s, s) for s in split_text]
    return ' '.join(split_text)


def _validate_array_string(string, min, max):
    # note min and max are both included as valid
    try:
        string = string.replace(')', '').replace('(', '')
        string = string.replace(']', '').replace('[', '')
        components = [s.strip() for s in string.split(',')]
        assert all(v.isdigit() for v in components)
        int_components = tuple(map(int, components))
        n_components = len(int_components)
        assert n_components >= min and n_components <= max
    except (ValueError, TypeError, AssertionError):
        return False
    return int_components


def _converter_array(key, definition, default=None):
    if default is None:
        default = ''
    input_box = pn.widgets.TextInput(name=_display_keytext(key),
                                     value=str(default),
                                     placeholder='Comma-separated int...')

    min = definition['minItems']
    max = definition['maxItems']

    def validator():
        return _validate_array_string(input_box.value, min, max)

    status_indicator, status_setter = _boolean_status(validator)
    input_box.param.watch(status_setter, 'value')

    def param_cb():
        validated_string = validator()
        return bool(validated_string), validated_string

    return [{key: input_box, 'status': status_indicator}], param_cb


@functools.lru_cache(None)
def _generate_dtypes():
    dtype_codes = ['i', '?', 'u', 'f', 'c']
    byte_lengths = ['', *[(2**i)//8 for i in range(4, 8)]]
    endianness = ['<']
    dtypes = []
    for code in dtype_codes:
        for length in byte_lengths:
            for e in endianness:
                try:
                    dt = np.dtype(f'{e}{code}{length}')
                    dtypes.append(dt.name)
                except TypeError:
                    pass
    return dtypes


def _converter_dtype(key, definition, default=None):
    dtypes = _generate_dtypes()

    if default is None:
        default = ''

    ac_input = pn.widgets.AutocompleteInput(name=_display_keytext(key),
                                         value=str(default),
                                         options=dtypes,
                                         restrict=True)

    def validator():
        if ac_input.value_input in dtypes or ac_input.value in dtypes:
            return True
        return False

    status_indicator, status_setter = _boolean_status(validator)
    ac_input.param.watch(status_setter, 'value_input')
    ac_input.param.watch(status_setter, 'value')

    def param_cb():
        valid = validator()
        return valid, ac_input.value

    return [{key: ac_input, 'status': status_indicator}], param_cb


def _converter_path(key, definition, default=None):
    if default is None:
        default = ''
    input_field = pn.widgets.TextInput(name=_display_keytext(key),
                                       value=str(default))
    choose_button = pn.widgets.Button(name='Choose file',
                               button_type='primary',
                               align='end',
                               width=110,
                               sizing_mode='fixed')
    clear_button = pn.widgets.Button(name='Clear',
                               button_type='primary',
                               align='end',
                               width=110,
                               sizing_mode='fixed')
    autodetect_button = pn.widgets.Button(name='Autodetect',
                                          button_type='primary',
                                          align='end',
                                          width=110,
                                          sizing_mode='fixed')

    def _validate_path():
        try:
            path = pathlib.Path(input_field.value)
            valid = path.is_file()
        except Exception:
            valid = False
        return valid

    status_indicator, status_setter = _boolean_status(_validate_path)
    input_field.param.watch(status_setter, 'value')

    def _clear_input(*events):
        input_field.value = ''

    clear_button.on_click(_clear_input)

    def param_cb():
        valid = _validate_path()
        return valid, input_field.value

    return [{key: input_field, 'status': status_indicator},
            {'dialog': choose_button,
            'clear': clear_button,
            'autodetect': autodetect_button}], param_cb


def _converter_string(key, definition, default=None):
    if default is None:
        default = ''
    input_field = pn.widgets.TextInput(name=_display_keytext(key), value=str(default))

    def param_cb():
        return True, input_field.value

    return [{key: input_field}], param_cb


def _converter_number(key, definition, default=None):
    if default is None:
        default = 0
    spinner = pn.widgets.IntInput(name=_display_keytext(key), value=int(default), step=1, height=55)
    return [{key: spinner}], lambda: (True, spinner.value)


def _converter_enum(key, definition, default=None):
    if default is None:
        default = ''
    select = pn.widgets.Select(name=_display_keytext(key),
                               value=str(default),
                               options=definition['enum'])

    def param_cb():
        valid = select.value in select.options
        return valid, select.value

    return [{key: select}], param_cb


def _converter_const(key, definition, default=None):
    return [], lambda: (True, None)


_converters = {'string': _converter_string,
               'array': _converter_array,
               'number': _converter_number,
               'enum': _converter_enum,
               'const': _converter_const,
               'dtype': _converter_dtype,
               'path': _converter_path}
