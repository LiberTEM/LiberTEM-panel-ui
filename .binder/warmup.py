import numpy as np
import libertem.api as lt
from libertem.udf.raw import PickUDF
from libertem.udf.sum import SumUDF
from libertem_ui.utils.data import demo_dataset

from libertem_ui.windows.imaging import VirtualDetectorWindow  # noqa

if __name__ == '__main__':
    data = demo_dataset()
    np.save('test_data.npy', data)

    with lt.Context.make_with('inline') as ctx:
        ds = ctx.load('npy', path='test_data.npy')
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[0, 0] = True
        ctx.run_udf(ds, PickUDF(), roi=roi)
        ctx.run_udf(ds, SumUDF())

    with lt.Context.make_with(cpus=1) as ctx:
        pass
