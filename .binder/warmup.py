from pathlib import Path
import numpy as np
import tempfile
import libertem.api as lt
from libertem.web.dataset import prime_numba_cache


if __name__ == '__main__':
    data = np.random.uniform(size=(8, 8, 32, 32)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / 'data.npy'
        np.save(data_path, data)
        with lt.Context.make_with(cpus=2) as ctx:
            ds_mem = ctx.load('memory', data=data, num_partitions=4)
            ds_npy = ctx.load('npy', path=data_path)
            prime_numba_cache(ds_mem)
            prime_numba_cache(ds_npy)
            pick_a = ctx.create_pick_analysis(ds_npy, 0, 0)
            pick_r = ctx.run(pick_a)
