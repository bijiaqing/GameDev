#include <cstdlib>
// compare the device hierarchy with an independent serial topology oracle
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <morton/morton_index.cuh>
#include <gpu.cuh>

static std::uint64_t key_of(float3 point, int dim, int depth)
{
    int cells = 1 << depth;
    int xyz[3] = {std::min(cells-1, int(point.x*cells)),
                  std::min(cells-1, int(point.y*cells)), dim == 3 ? std::min(cells-1, int(point.z*cells)) : 0};
    std::uint64_t key = 0;
    for (int bit = 0; bit < depth; ++bit)
        for (int axis = 0; axis < 3; ++axis)
            key |= std::uint64_t((xyz[axis] >> bit) & 1) << (3*bit + axis);
    return key;
}

static void check(morton_index &index, const std::vector<float3> &points, int dim, int target, int depth)
{
    float3 *device = nullptr;
    if (gpuError_t status = gpuMalloc(reinterpret_cast<void **>(&device), sizeof(*device)*(points.size())); status != gpuSuccess)
    {
        std::cerr << "test input allocation" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(device, points.data(), sizeof(*(device))*(points.size()), gpuMemcpyHostToDevice); status != gpuSuccess)
    {
        std::cerr << "test input copy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    index.build(device, int(points.size()), make_float3(0,0,0), 1, dim, target, depth);
    if (gpuError_t status = gpuFree(device); status != gpuSuccess)
    {
        std::cerr << "test input release" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    auto view = index.view();
    std::vector<morton_point> sorted(points.size());
    std::vector<morton_node> actual(index.node_count());
    if (gpuError_t status = gpuMemcpy(sorted.data(), view.dev_point, sizeof(*(sorted.data()))*(sorted.size()), gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "test points copy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    if (gpuError_t status = gpuMemcpy(actual.data(), view.dev_node, sizeof(*(actual.data()))*(actual.size()), gpuMemcpyDeviceToHost); status != gpuSuccess)
    {
        std::cerr << "test nodes copy" << ": " << gpuGetErrorString(status) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    std::vector<int> order(points.size());
    for (int i=0; i<int(order.size()); ++i) order[i]=i;
    std::stable_sort(order.begin(), order.end(), [&](int a,int b){return key_of(points[a],dim,depth)<key_of(points[b],dim,depth);});
    for (int i=0; i<int(order.size()); ++i)
    {
        assert(sorted[i].idx_old == order[i]);
        assert(sorted[i].cartesian.x == points[order[i]].x);
        assert(sorted[i].cartesian.y == points[order[i]].y);
        assert(sorted[i].cartesian.z == points[order[i]].z);
    }
    std::vector<morton_node> expected;
    std::vector<int> leaves;
    std::function<int(int,int,int,float3,float)> split = [&](int begin,int end,int level,float3 lower,float width)
    {
        int id=int(expected.size());
        morton_node node{};
        node.idx_begin=begin; node.count=end-begin; node.lower=lower; node.width=width;
        for (int &child:node.idx_child) child=-1;
        expected.push_back(node);
        if (end-begin <= target || level == depth) {leaves.push_back(end-begin); return id;}
        int cursor=begin, shift=3*(depth-level-1);
        for (int code=0; code<(1<<dim); ++code)
        {
            int first=cursor;
            // independent linear partition, rather than the device binary search
            while (cursor<end && int((key_of(points[order[cursor]],dim,depth)>>shift)&7)==code) ++cursor;
            if (cursor==first) continue;
            float3 child_lower=lower;
            if(code&1) child_lower.x += 0.5f*width;
            if(code&2) child_lower.y += 0.5f*width;
            if(dim==3 && (code&4)) child_lower.z += 0.5f*width;
            int child=split(first,cursor,level+1,child_lower,0.5f*width);
            expected[id].idx_child[code]=child;
            expected[id].child_count++;
        }
        assert(cursor==end);
        return id;
    };
    split(0,int(points.size()),0,make_float3(0,0,0),1);
    assert(actual.size()==expected.size());
    std::vector<bool> visited(actual.size(),false);
    std::function<void(int,int)> compare = [&](int a,int b)
    {
        assert(a>=0 && a<int(actual.size()) && !visited[a]); visited[a]=true;
        auto x=actual[a], y=expected[b];
        assert(x.idx_begin==y.idx_begin && x.count==y.count && x.child_count==y.child_count);
        assert(x.width==y.width && x.lower.x==y.lower.x && x.lower.y==y.lower.y && x.lower.z==y.lower.z);
        for (int code=0;code<8;++code)
        {
            if(y.idx_child[code]<0) assert(x.idx_child[code]==-1);
            else compare(x.idx_child[code],y.idx_child[code]);
        }
    };
    compare(0,0);
    assert(std::all_of(visited.begin(),visited.end(),[](bool x){return x;}));
    auto sizes=index.leaf_counts();
    std::sort(sizes.begin(),sizes.end()); std::sort(leaves.begin(),leaves.end());
    assert(sizes==leaves && index.leaf_count()==int(leaves.size()));
    assert(view.point_count==int(points.size()) && view.node_count==int(expected.size()));
}

int main()
{
    morton_index index;
    std::mt19937 rng(761);
    int cases=0;
    for(int dim:{2,3}) for(int target:{1,128}) for(int depth:{1,5,20})
    {
        std::vector<float3> points;
        for(int i=0;i<513;++i)
            points.push_back(make_float3(float(rng()%4096)/4096, float(rng()%4096)/4096,
                                        dim==3 ? float(rng()%4096)/4096 : 0));
        points.insert(points.end(),300,make_float3(0.5f,0.5f,dim==3 ? 0.5f : 0));
        points.push_back(make_float3(0,0,0)); points.push_back(make_float3(1,1,dim==3 ? 1:0));
        check(index,points,dim,target,depth); ++cases;
        check(index,std::vector<float3>(400,make_float3(0.25f,0.25f,0.25f)),dim,target,depth); ++cases;
        check(index,{make_float3(0,0,0)},dim,target,depth); ++cases;
    }
    index.release(); assert(index.node_count()==0 && index.leaf_count()==0 && index.leaf_counts().empty());
    std::cout << "{\"topology_cases\":" << cases << ",\"passed\":true}\n";
}
