import numpy as np
import time
from libertem_ui.applications.image_utils import fine_adjust


def test_fine_adjust():
    circle_image = (np.linalg.norm(np.mgrid[-200: 200, -300: 300], axis=0) < 100)
    circle_image_shifted = np.roll(circle_image, (-50, 80), axis=(0, 1))
    layout3, _ = fine_adjust(circle_image, circle_image_shifted)
    show_th = layout3.show(open=False, threaded=True)
    time.sleep(0.3)
    show_th.stop()
