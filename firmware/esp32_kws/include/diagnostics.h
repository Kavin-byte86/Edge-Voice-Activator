/* GroundWatch Hardware Diagnostics & RAM/CPU Measurement Module */
#ifndef DIAGNOSTICS_H_
#define DIAGNOSTICS_H_

#include <cstdint>
#include <cstddef>

struct SystemMemoryStats {
    uint32_t free_heap_bytes;
    uint32_t total_heap_bytes;
    uint32_t min_free_heap_bytes;
    uint32_t max_alloc_block_bytes;
    uint32_t static_global_ram_bytes;
    uint32_t tensor_arena_bytes;
    uint32_t tensor_arena_used_bytes;
    uint32_t ring_buffer_bytes;
    uint32_t dma_audio_bytes;
    uint32_t total_application_ram_bytes;
    uint32_t loop_stack_hwm_bytes;
};

void init_diagnostics();
void update_cpu_measurements();
float get_idle_cpu_percent();
float get_inference_cpu_percent();

SystemMemoryStats get_system_memory_stats();
void print_hardware_benchmark_report(uint32_t feature_extraction_ms, uint32_t inference_ms, float akash_go_confidence);

#endif // DIAGNOSTICS_H_
