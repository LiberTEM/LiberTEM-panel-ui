import numpy as np
import time
from playwright.sync_api import Page
from libertem_ui.applications.image_utils import fine_adjust


def test_fine_adjust(page: Page):
    circle_image = (np.linalg.norm(np.mgrid[-200: 200, -300: 300], axis=0) < 100)
    circle_image_shifted = np.roll(circle_image, (-50, 80), axis=(0, 1))
    layout, getter = fine_adjust(circle_image, circle_image_shifted)
    port = np.random.randint(8081, 9957)
    show_th = layout.show(open=False, threaded=True, port=int(port))
    try:
        page.goto(f"http://localhost:{port}")
        page.get_by_role("button", name="▷", exact=True).click()
        time.sleep(0.25)
        transform = getter()
        assert transform.translation[0] == -1.0
    finally:
        show_th.stop()
