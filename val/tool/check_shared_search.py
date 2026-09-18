#!/usr/bin/env python3
"""Check search-header consolidation against retained snapshots (not GPU execution)."""
from pathlib import Path
import re,subprocess,json
repo=Path(__file__).resolve().parents[2]
snapshot=repo/'val/temp/shared_search/before'
assert snapshot.is_dir(), 'Missing pre-consolidation search snapshot'
compat=(repo/'inc/gpu_compat.cuh').read_text()
def normalize(s):
 s=re.sub(r'\b(?:cuda|hip)([A-Z]\w*)',r'gpu\1',s)
 s=re.sub(r'\b(?:cuda|hip)##call','gpu##call',s)
 s=re.sub(r'\b(KDTREE|CUBIT)_(?:CUDA|HIP)_',r'\1_GPU_',s)
 s=re.sub(r'\b(?:cuda|hip)_(vec_t|t)\b',r'gpu_\1',s)
 s=re.sub(r'\b(?:__CUDA_ARCH__|__HIP_DEVICE_COMPILE__)\b','GAMEDEV_GPU_DEVICE',s)
 s=re.sub(r'_morton_(?:cuda|hip)_check','_morton_gpu_check',s)
 s=s.replace('thrust::hip_rocprim::par','GPU_THRUST_DEVICE').replace('thrust::device.on','GPU_THRUST_DEVICE.on').replace('thrust::device,','GPU_THRUST_DEVICE,')
 s=re.sub(r'"(at %s: )?(CUDA|HIP) call',lambda m:'"'+(m[1] or '')+'" GPU_BACKEND_NAME " call',s)
 return s.replace('"fatal cuda error"','"fatal " GPU_BACKEND_NAME " error"').replace('"fatal HIP error"','"fatal " GPU_BACKEND_NAME " error"')
def preprocess(s,defs):
 # Include routing is checked separately by native builds. Here compare actual
 # backend-selected expressions, including error macros exercised below.
 s=compat+'\n'+s
 s=re.sub(r'^\s*#include[^\n]*','',s,flags=re.M)
 for prefix in ('KDTREE','CUBIT'):
  for name,args in [('CHECK','op()'),('CALL','Free(ptr)'),('SYNC_CHECK',''),('CHECK_NOTHROW','op()'),('CALL_NOTHROW','Free(ptr)'),('CHECK2','"test", op()'),('CHECK2_NOTHROW','"test", op()'),('SYNC_CHECK_STREAM','stream')]:
   macro=prefix+'_GPU_'+name
   s+=f'\n#ifdef {macro}\n{macro}({args})\n#endif\n'
 p=subprocess.run(['c++','-E','-P','-x','c++',*['-D'+d for d in defs],'-'],input=s,text=True,capture_output=True,check=True)
 s=re.sub(r'"<stdin>"\s*,\s*\d+', '"<stdin>", 0',p.stdout)
 # Macro diagnostics also contain __LINE__ in other argument positions.
 s=re.sub(r'\b\d+\s*,\s*(cuda|hip)GetErrorString',r'0, \1GetErrorString',s)
 while re.search(r'"([^"\\]*)"\s*"([^"\\]*)"',s):
  s=re.sub(r'"([^"\\]*)"\s*"([^"\\]*)"',lambda m:'"'+m[1]+m[2]+'"',s)
 return re.findall(r'"(?:\\.|[^"\\])*"|\w+|[^\s]',s)
fail=[];count=0
for backend in ('cuda','rocm'):
 for old in (snapshot/f'inc/{backend}/swarm').rglob('*'):
  if old.suffix not in ('.h','.cuh'):continue
  rel=old.relative_to(snapshot/f'inc/{backend}/swarm');new=repo/'inc/swarm'/rel
  for device in (False,True):
   for version in (11000,12000):
    defs=['GAMEDEV_'+backend.upper(),f'CUDART_VERSION={version}']+(['GAMEDEV_GPU_DEVICE'] if device else [])
    a=preprocess(normalize(old.read_text()),defs);b=preprocess(new.read_text(),defs)
    if a!=b:
     i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
     fail.append([str(rel),backend,device,version,a[max(0,i-3):i+8],b[max(0,i-3):i+8]])
    count+=1
assert count==200, f'Incomplete snapshot: {count}'
report={'preprocessed_comparisons':count,'failures':fail,'native_gpu_execution':False}
(repo/'val/temp/shared_search/source_identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'comparisons':count,'failures':len(fail),'examples':fail[:5]},indent=2));assert not fail
