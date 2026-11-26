// Optional Linux profiler for the CUDA runtime calls made by the codec.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

#include <cuda_runtime_api.h>

namespace {
enum Call { DeviceSync, StreamSync, EventSync, Copy, CopyAsync, Graph, Count };
std::atomic<bool> active{false};
std::atomic<uint64_t> totals[Count]{};

template<class Fn>
Fn actual(const char* name) {
    if (auto* sym = dlsym(RTLD_NEXT, name)) return reinterpret_cast<Fn>(sym);
    static void* runtime = dlopen("libcudart.so.13", RTLD_LOCAL | RTLD_NOW);
    auto* sym = runtime ? dlsym(runtime, name) : nullptr;
    if (!sym) std::abort();
    return reinterpret_cast<Fn>(sym);
}

void record(Call call) {
    if (active.load(std::memory_order_relaxed)) totals[call].fetch_add(1, std::memory_order_relaxed);
}
}

extern "C" void pht_count_start() {
    for (auto& counter : totals) counter.store(0, std::memory_order_relaxed);
    active.store(true, std::memory_order_relaxed);
}

extern "C" void pht_count_stop(uint64_t* out) {
    active.store(false, std::memory_order_relaxed);
    for (int i = 0; i < Count; ++i) out[i] = totals[i].load(std::memory_order_relaxed);
}

extern "C" cudaError_t cudaDeviceSynchronize() {
    static const auto function = actual<decltype(&cudaDeviceSynchronize)>("cudaDeviceSynchronize");
    record(DeviceSync);
    return function();
}

extern "C" cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    static const auto function = actual<decltype(&cudaStreamSynchronize)>("cudaStreamSynchronize");
    record(StreamSync);
    return function(stream);
}

extern "C" cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    static const auto function = actual<decltype(&cudaEventSynchronize)>("cudaEventSynchronize");
    record(EventSync);
    return function(event);
}

extern "C" cudaError_t cudaMemcpy(void* to, const void* from, size_t bytes, cudaMemcpyKind kind) {
    static const auto function = actual<decltype(&cudaMemcpy)>("cudaMemcpy");
    record(Copy);
    return function(to, from, bytes, kind);
}

extern "C" cudaError_t cudaMemcpyAsync(void* to, const void* from, size_t bytes, cudaMemcpyKind kind,
                                       cudaStream_t stream) {
    static const auto function = actual<decltype(&cudaMemcpyAsync)>("cudaMemcpyAsync");
    record(CopyAsync);
    return function(to, from, bytes, kind, stream);
}

extern "C" cudaError_t cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
    static const auto function = actual<decltype(&cudaGraphLaunch)>("cudaGraphLaunch");
    record(Graph);
    return function(graph, stream);
}
