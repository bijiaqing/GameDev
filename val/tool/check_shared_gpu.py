#!/usr/bin/env python3
"""Compare shared sources to the pre-migration snapshot after preprocessing API mappings."""
from pathlib import Path
import subprocess,re,json
repo=Path(__file__).resolve().parents[2]
before=repo/'val/temp/shared_gpu/before'
assert before.is_dir(), 'Migration snapshot missing: val/temp/shared_gpu/before'
compat=(repo/'inc/gpu_compat.cuh').read_text()
def preprocess(s,defines):
 s=re.sub(r'^\s*#include[^\n]*','',s,flags=re.M)
 p=subprocess.run(['c++','-E','-P','-x','c++',*['-D'+d for d in defines],'-'],input=s,text=True,capture_output=True)
 if p.returncode:raise RuntimeError(p.stderr)
 s=p.stdout.replace('hip_fail','cuda_fail').replace('hip_status_','cuda_status_')
 # Error diagnostics stringify the pre-expansion API name and source line.
 s=re.sub(r'"<stdin>"\s*,\s*\d+', '\"<stdin>\", 0',s)
 def diagnostic(m):
  value=m[0].replace('hipHostMalloc','hipMallocHost').replace('hipHostFree','hipFreeHost')
  return re.sub(r'\b(?:cuda|hip|gpu)([A-Z]\w*)',r'api\1',value)
 s=re.sub(r'"(?:\\.|[^"\\])*"',diagnostic,s)
 # Adjacent literals are equivalent; comments/whitespace are immaterial.
 while re.search(r'"([^"\\]*)"\s*"([^"\\]*)"',s):s=re.sub(r'"([^"\\]*)"\s*"([^"\\]*)"',lambda m:'"'+m[1]+m[2]+'"',s)
 return re.findall(r'"(?:\\.|[^"\\])*"|\w+|[^\s]',s)
fail=[];count=0
modes=[[],['DIFFUSION'],['COLLISION','MULTISIZE','COLLISION_KDTREE'],['COLLISION','MULTISIZE','COLLISION_MORTON','BERNOULLI'],['COLLISION','MULTISIZE','COLLISION_MORTON','KNN_CACHE','BERNOULLI','IMPORTGAS','COL_DIAGNOSTICS'],['DIFFUSION','FLUID_BLOCK_SWEEP'],['DIFFUSION','FLUID_BLOCK_SWEEP','CUDA_SYNC_TRACE','HIP_SYNC_TRACE']]
for kind in ['src','inc']:
 for backend in ['cuda','rocm']:
  for p in (before/kind/backend).rglob('*'):
   if not p.is_file():continue
   rel=p.relative_to(before/kind/backend);new=repo/kind/rel
   if new.suffix=='.hip':new=new.with_suffix('.cu')
   for mode in modes:
    defs=['GAMEDEV_'+backend.upper(),'TRANSPORT']+mode
    a=preprocess(p.read_text(),defs);b=preprocess(compat+'\n'+new.read_text(),defs)
    if a!=b:
     first=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
     fail.append([str(rel),backend,mode,a[max(0,first-5):first+10],b[max(0,first-5):first+10]])
    count+=1
assert count == 196, 'Incomplete migration snapshot'
report={'preprocessed_comparisons':count,'failures':fail,'native_gpu_execution':False}
(repo/'val/temp/shared_gpu/source_identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'comparisons':count,'failures':len(fail),'first_failures':fail[:8]},indent=2));assert not fail
