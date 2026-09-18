#!/usr/bin/env python3
"""CPU initializer checks and build routing; not a native GPU qualification."""
import json
import os
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
results = []
env=dict(os.environ)
if Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk').exists():
    env['CPLUS_INCLUDE_PATH']='/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1'
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # Only the RNG state type is needed to compile the actual initializer on the host.
    (tmp/'curand_kernel.h').write_text('struct curandState {};\n')
    (tmp/'swarm_kern.cuh').write_text('#include <const_defs.cuh>\n')
    # Compile the actual physical speed helpers and pair closure on the host.
    collision = (repo/'inc/swarm/_collision.cuh').read_text()
    physical = collision.split('#ifndef CODE_UNIT // physical units', 1)[1].split('// combine query-local relative speeds', 1)[0]
    (tmp/'collision_velocity_check.hpp').write_text('#ifndef CODE_UNIT\n' + physical)
    for model in ('strong_turbulence','weak_turbulence'):
        exe = tmp/model
        subprocess.run(['c++','-std=c++17','-O2','-DMULTISIZE','-DDIFFUSION','-DCOLLISION','-DCOLLISION_QUERY_LOCAL',
                        '-I'+str(tmp),'-I'+str(root/'models'/model),
                        '-I'+str(repo/'inc/swarm'),str(root/'common/check.cpp'),
                        '-o',str(exe)],check=True,env=env)
        result = json.loads(subprocess.check_output([str(exe)],text=True))
        result['model'] = model
        for backend,target in (('rocm','gfx942'),('cuda','sm_80')):
            for search in ('kdtree','morton'):
                plan = subprocess.check_output(['make','-Bn','-C',str(root),f'MODEL={model}',
                     f'GPU_BACKEND={backend}',f'GPU_TARGET={target}',
                     f'COLLISION_SEARCH={search}','CUDA_MATH=precise'],text=True)
                assert str(repo/'val/temp/swarm_coagulation/bin'/model/backend/search/'gamedev') in plan
                assert str(root/'outputs'/model/backend/search) in plan
                assert f'-DCOLLISION_{search.upper()}' in plan and '-DMULTISIZE' in plan
                assert '-DCOLLISION_QUERY_LOCAL' in plan
                assert 'Using header override: '+str(root/'models/../common/_col_chain.cuh') in plan
                assert 'src/swarm/diffusion_pos.cu' in plan
                assert '-DDIFFUSE_CONCENTRATION' in plan
                assert 'src/swarm/dyn_rate_calc.cu' in plan
                ext='cu' if backend=='cuda' else 'hip'
                assert str(root/'models/../common'/('swarm_runtime.'+ext)) in plan
                assert 'lab/collision_fix' not in plan
                other = 'MORTON' if search=='kdtree' else 'KDTREE'
                assert f'-DCOLLISION_{other}' not in plan
                assert '-DCODE_UNIT' not in plan and '-DRADIATION' not in plan
                assert str(root/'models'/model) in plan and str(root/'models/../common') in plan
                result[backend+'_'+search+'_routing_passed'] = True
        results.append(result)
for header in ('kdtree/index_heap.cuh','morton/morton_index.cuh','morton/morton_query.cuh'):
    assert not (root/'common'/header).exists()
    assert (repo/'inc/swarm'/header).is_file()
assert not (root/'common/local_evolve.inc').exists()
assert 'std::fill(randsize, randsize + N_P, INIT_SMIN)' in (root/'common/swarm_runtime.cu').read_text()
assert (root/'common/swarm_runtime.hip').read_text()=='#include \"swarm_runtime.cu\"\n'
print(json.dumps({'checks':results,'native_gpu_execution':False},indent=2))
