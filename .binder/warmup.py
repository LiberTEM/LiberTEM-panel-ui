import numpy as np
import libertem.api as lt
from libertem.udf.raw import PickUDF
from libertem.web.dataset import prime_numba_cache


if __name__ == '__main__':
    data = np.zeros((8, 8, 8, 8), dtype=np.float32)
    with lt.Context.make_with('inline') as ctx:
        ds_mem = ctx.load('memory', data=data, num_partitions=1)
        prime_numba_cache(ds_mem)
        roi = np.zeros(ds_mem.shape.nav, dtype=bool)
        ds_mem[0, 0] = True
        ctx.run_udf(ds_mem, PickUDF, roi=roi)
