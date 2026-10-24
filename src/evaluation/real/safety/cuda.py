"""PyTorch allocator measurements without importing torch for CPU execution."""

import importlib


class CudaMemory:
    """Counters cover PyTorch allocations, excluding the CUDA context and other processes."""

    def __init__(self, device):
        self.device = device
        self.enabled = isinstance(device, int) or str(device).lower().split(":")[0] == "cuda"

    def _cuda(self):
        return importlib.import_module("torch").cuda

    def synchronize(self):
        if self.enabled:
            self._cuda().synchronize(self.device)

    def snapshot(self):
        if not self.enabled:
            return {}
        cuda = self._cuda()
        cuda.synchronize(self.device)
        return {
            "allocated_bytes": cuda.memory_allocated(self.device),
            "reserved_bytes": cuda.memory_reserved(self.device),
            "max_allocated_bytes": cuda.max_memory_allocated(self.device),
            "max_reserved_bytes": cuda.max_memory_reserved(self.device),
        }

    def reset_peak(self):
        if self.enabled:
            cuda = self._cuda()
            cuda.synchronize(self.device)
            cuda.reset_peak_memory_stats(self.device)

    def empty_cache(self):
        if self.enabled:
            cuda = self._cuda()
            with cuda.device(self.device):
                cuda.empty_cache()
