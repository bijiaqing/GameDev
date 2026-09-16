#!/usr/bin/env python3
"""CPU initializer checks and build routing; not a native GPU qualification."""
import json
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
results = []
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    # Only the RNG state type is needed to compile the actual initializer on the host.
    (tmp/'curand_kernel.h').write_text('struct curandState {};\n')
    (tmp/'swarm_kern.cuh').write_text('#include <const_defs.cuh>\n')
    # Compile the actual physical speed helpers and pair closure on the host.
    collision = (repo/'inc/comm/swarm/_collision.cuh').read_text()
    physical = collision.split('#ifndef CODE_UNIT // physical units', 1)[1].split('// combine resolved and unresolved relative speeds', 1)[0]
    (tmp/'collision_velocity_check.hpp').write_text('#ifndef CODE_UNIT\n' + physical)
    for model in ('strong_turbulence','weak_turbulence'):
        exe = tmp/model
        subprocess.run(['c++','-std=c++17','-O2','-DMULTISIZE','-DDIFFUSION','-DCOLLISION','-DCOLLISION_QUERY_LOCAL',
                        '-I'+str(tmp),'-I'+str(root/'models'/model),
                        '-I'+str(repo/'inc/comm/swarm'),str(root/'common/check.cpp'),
                        '-o',str(exe)],check=True)
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
                assert 'Using header override: '+str(root/'models/../common/_col_chain.cuh') not in plan
                other = 'MORTON' if search=='kdtree' else 'KDTREE'
                assert f'-DCOLLISION_{other}' not in plan
                assert '-DCODE_UNIT' not in plan and '-DRADIATION' not in plan
                assert str(root/'models'/model) in plan and str(root/'models/../common') in plan
                result[backend+'_'+search+'_routing_passed'] = True
        results.append(result)
print(json.dumps({'checks':results,'native_gpu_execution':False},indent=2))
