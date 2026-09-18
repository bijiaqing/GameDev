#!/usr/bin/env python3
"""Host oracle checks of production search primitives; no GPU qualification."""
from pathlib import Path
import json,os,re,subprocess
repo=Path(__file__).resolve().parents[2]
out=repo/'val/temp/search_optimizations';out.mkdir(parents=True,exist_ok=True)
env=dict(os.environ);sdk=Path('/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1')
if sdk.exists():env['CPLUS_INCLUDE_PATH']=str(sdk)
def run(name,source,defines=()):
 cpp=out/(name+'.cpp');cpp.write_text(source)
 subprocess.run(['c++','-std=c++20','-O2','-pthread',*['-D'+d for d in defines],str(cpp),'-o',str(out/name)],env=env,check=True)
 subprocess.run([str(out/name)],check=True)
morton_pre=r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cassert>
#include <climits>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include <iostream>
#define __device__
#define __forceinline__ inline
constexpr float CUDART_INF_F=std::numeric_limits<float>::infinity();
thread_local struct {int x;} threadIdx;
constexpr int TEST_BLOCK =
#ifdef PACK_BLOCK
PACK_BLOCK;
#else
64;
#endif
std::barrier sync_threads(TEST_BLOCK);
void __syncthreads(){sync_threads.arrive_and_wait();}
std::atomic<int> any_improves(0);int skips=0,merges=0;
int __syncthreads_or(int flag){
 if(threadIdx.x==0)any_improves.store(0);
 __syncthreads();any_improves.fetch_or(flag);__syncthreads();
 int result=any_improves.load();
 if(threadIdx.x==0){if(result)++merges;else ++skips;}
 __syncthreads();return result;
}

#define __shared__ static
int warpSize=32;bool ballot_flags[TEST_BLOCK];
unsigned long long ballot(int flag){
 ballot_flags[threadIdx.x]=flag;__syncthreads();
 unsigned long long result=0;int start=threadIdx.x/warpSize*warpSize;
 for(int i=0;i<warpSize && start+i<TEST_BLOCK;++i)if(ballot_flags[start+i])result|=1ULL<<i;
 __syncthreads();return result;
}
unsigned long long __ballot(int flag){return ballot(flag);}
unsigned long long __ballot_sync(unsigned int,int flag){return ballot(flag);}
int __popcll(unsigned long long mask){return __builtin_popcountll(mask);}

constexpr float MORTON_INF_F=CUDART_INF_F;
'''
morton_test=r'''
template<int K,int S> void streams(){
 constexpr int CAP=K<S-K?K:S-K;
 for(int width:{32,64})for(int trial=0;trial<4;++trial){
  warpSize=width;std::array<float,S>d;std::array<int,S>ids;
  d.fill(CUDART_INF_F);ids.fill(INT_MAX);
  int batch=0;std::vector<std::pair<float,int>> expected;
  std::mt19937 rng(722+trial);std::array<float,1024>source;
  for(int j=0;j<1024;++j){source[j]=trial==1?float(1024-j):float(rng()%512);if(trial!=0 && j%7!=0 && source[j]<=400)expected.emplace_back(source[j],j);}
  while(expected.size()<K)expected.emplace_back(CUDART_INF_F,INT_MAX);
  std::sort(expected.begin(),expected.end());
  std::vector<std::thread>threads;
  for(int lane=0;lane<64;++lane)threads.emplace_back([&,lane]{threadIdx.x=lane;
   int offset=0;
   while(offset<1024){
    int old=batch,take=std::min(64,std::min(CAP-old,1024-offset));
    float distance=lane<take?source[offset+lane]:CUDART_INF_F;int id=offset+lane;
    bool eligible=lane<take && trial!=0 && id%7!=0 && distance<=400 && _morton_neighbor_less(distance,id,d[K-1],ids[K-1]);
    int count=_morton_pack_tile(distance,id,eligible,d.data()+K+old,ids.data()+K+old);
    if(lane==0)batch=old+count;__syncthreads();offset+=take;
    if(batch==CAP){_morton_pair_merge<K,64,S>(d.data(),ids.data(),batch);if(lane==0)batch=0;__syncthreads();}
   }
   if(batch>0)_morton_pair_merge<K,64,S>(d.data(),ids.data(),batch);
  });
  for(auto &t:threads)t.join();
  for(int j=0;j<K;++j)assert(d[j]==expected[j].first && ids[j]==expected[j].second);
 }
 std::cout<<"8 compacted streams match exact top K (warp widths 32/64)\n";
}

int main(){streams<32,128>();streams<127,256>();streams<128,256>();streams<200,512>();streams<256,512>();streams<257,1024>();streams<512,1024>();streams<600,1024>();}
'''
pack_test=r'''
int main(){
 for(int width:{32,64})for(int trial=0;trial<5;++trial){
  warpSize=width;float d[TEST_BLOCK];int ids[TEST_BLOCK];int counts[TEST_BLOCK];
  std::vector<int> expected;
  for(int i=0;i<TEST_BLOCK;++i)if(trial==0 || (trial!=1 && i%trial==0))expected.push_back(i);
  std::vector<std::thread> threads;
  for(int i=0;i<TEST_BLOCK;++i)threads.emplace_back([&,i]{
   threadIdx.x=i;bool keep=trial==0 || (trial!=1 && i%trial==0);
   counts[i]=_morton_pack_tile<TEST_BLOCK>(float(i),i,keep,d,ids);
  });
  for(auto &t:threads)t.join();
  for(int count:counts)assert(count==int(expected.size()));
  for(int i=0;i<int(expected.size());++i)assert(ids[i]==expected[i] && d[i]==float(expected[i]));
 }
}
'''
heap_pre=r'''
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <iostream>
#include <memory>
#define __shared__ static
struct { int x=0; } threadIdx;
#define __device__
#define __forceinline__ inline
constexpr int COL_IMAGE_COUNT=3;
unsigned int __float_as_uint(float x){return std::bit_cast<unsigned int>(x);}
float __uint_as_float(unsigned int x){return std::bit_cast<float>(x);}
struct Node {int idx_old; int image;};

#include <algorithm>
#include <vector>
#include <numeric>
'''
heap_test=r'''
template<int K> void heaps(){
 std::mt19937 rng(7821);Node nodes[1200];unsigned char active[400];
 for(int i=0;i<1200;++i)nodes[i]={i/3,i%3};
 for(int i=0;i<400;++i)active[i]=i%11!=0;
 constexpr int T=idx_old_heap<K,Node>::threads;
 for(bool dedup:{false,true})for(bool mask:{false,true})for(float radius:{0.f,1.f,10.f}){
  std::unique_ptr<idx_old_heap<K,Node>> heaps[T];
  std::vector<unsigned long long> expected[T];
  for(int lane=0;lane<T;++lane){threadIdx.x=lane;heaps[lane]=std::make_unique<idx_old_heap<K,Node>>(radius,nodes,dedup,mask?active:nullptr);}
  for(int lane=0;lane<T;++lane){
   std::vector<int> ids(1200);std::iota(ids.begin(),ids.end(),0);std::shuffle(ids.begin(),ids.end(),rng);
   std::vector<unsigned long long> best(400,~0ULL);
   for(int id:ids){
    float distance=float(rng()%200)/10.f;
    heaps[lane]->processCandidate(id,distance);
    if((!mask||active[id/3])&&distance<=radius*radius){
     auto key=(static_cast<unsigned long long>(__float_as_uint(distance))<<32)|static_cast<unsigned int>(id);
     if(dedup)best[id/3]=std::min(best[id/3],key);else expected[lane].push_back(key);
    }
   }
   if(dedup)for(auto key:best)if(key!=~0ULL)expected[lane].push_back(key);
   auto empty=(static_cast<unsigned long long>(__float_as_uint(radius*radius))<<32)|0xffffffffU;
   while(expected[lane].size()<K)expected[lane].push_back(empty);
   std::sort(expected[lane].begin(),expected[lane].end());expected[lane].resize(K);
  }
  for(int lane=0;lane<T;++lane){std::vector<unsigned long long> actual;
   for(int j=0;j<K;++j)actual.push_back(heaps[lane]->get_key(j));
   std::sort(actual.begin(),actual.end());assert(actual==expected[lane]);
  }
 }
}
int main(){heaps<1>();heaps<127>();heaps<128>();heaps<200>();heaps<256>();heaps<257>();heaps<512>();heaps<1025>();heaps<2049>();heaps<4096>();}
'''
types=(repo/'inc/swarm/morton/morton_types.cuh').read_text()
a=types.index('bool _morton_neighbor_less');b=types.index('\n}',a)+2
source=(repo/'inc/swarm/morton/morton_index.cuh').read_text()
chunk=source[source.index('// sort shared-memory neighbor pairs'):source.index('// traverse the adaptive hierarchy')]
for backend in ['CUDA','ROCM']:run('morton_'+backend.lower(),morton_pre+types[a:b]+'\n'+chunk+morton_test,['GAMEDEV_'+backend])
for backend in ['CUDA','ROCM']:
 for block in [32,128,256]:run(f'pack_{backend.lower()}_{block}',morton_pre+types[a:b]+'\n'+chunk+pack_test,['GAMEDEV_'+backend,f'PACK_BLOCK={block}'])
source=(repo/'inc/swarm/_collision.cuh').read_text()
helpers='\n'.join(re.findall(r'int _(?:encode_col_neighbor|get_col_idx_old|get_col_image) \([^\n]+',source))
source=(repo/'inc/swarm/kdtree/index_heap.cuh').read_text()
run('heap',heap_pre+helpers+'\n'+source[source.index('// retain'):source.rindex('#endif')]+heap_test)
report={'native_gpu_execution':False,'morton_K':[32,127,128,200,256,257,512,600],'warp_widths':[32,64],'packing_block_sizes':[32,64,128,256],'heap_K':[1,127,128,200,256,257,512,1025,2049,4096]}
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
