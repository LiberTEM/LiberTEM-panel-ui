from __future__ import annotations
import panel as pn


card_css = '''.cardtree {
  border: 0px;
  border-width: 0px;
  border-radius: 0.0rem;
}

.cardtree-header {
  align-items: center;
  background-color: transparent;
  display: inline-flex;
  justify-content: start;
  width: 100%;
  border: 0px;
  border-radius: 0.0rem;
}

.cardtree-button {
  background-color: transparent;
  margin-left: 0.0em;
  align-items: center;
}
'''


def minimal_card(
      title: str,
      *objects: pn.viewable.Layoutable,
      collapsed: bool = False,
      **card_kwargs
    ) -> pn.layout.Card:
    card_header = pn.widgets.StaticText(
        value=title,
        margin=(5, 5),
        align=('center', 'center'),
    )

    return pn.layout.Card(
        *objects,
        header=card_header,
        css_classes=['cardtree'],
        button_css_classes=['cardtree-button'],
        header_css_classes=['cardtree-header'],
        stylesheets=[card_css],
        margin=card_kwargs.pop('margin', (5, 5)),
        collapsed=collapsed,
        **card_kwargs,
    )
