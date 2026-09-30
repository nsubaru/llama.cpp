#pragma once
#include "ggml-backend.h"
#include <algorithm>
#include <cstdint>
#include <vector>

struct ggml_sparse_chunk { size_t offset; size_t size; uint64_t handle; };

inline size_t ggml_sparse_size(const std::vector<ggml_sparse_chunk> & chunks) {
    size_t size = 0;
    for (const auto & chunk : chunks) size += chunk.size;
    return size;
}

// New chunks contain only new pages, so a failed growth can release them without touching live data.
template<class Commit, class Release>
bool ggml_sparse_update(std::vector<ggml_sparse_chunk> & chunks, size_t reserved, size_t granularity,
        const ggml_backend_buffer_range * ranges, size_t count, Commit commit, Release release) {
    if (!granularity) return false;
    size_t end = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto & r = ranges[i];
        if (!r.size || r.offset < end || r.offset % granularity || r.size % granularity ||
            r.offset > reserved || r.size > reserved - r.offset) return false;
        end = r.offset + r.size;
    }
    std::sort(chunks.begin(), chunks.end(), [](const auto & a, const auto & b) { return a.offset < b.offset; });
    std::vector<ggml_sparse_chunk> kept, added, removed;
    kept.reserve(chunks.size() + count);
    removed.reserve(chunks.size());
    for (const auto & chunk : chunks) {
        bool keep = false;
        for (size_t i = 0; i < count; ++i) {
            const auto & r = ranges[i];
            if (chunk.offset >= r.offset && chunk.offset + chunk.size <= r.offset + r.size) { keep = true; break; }
            if (chunk.offset < r.offset + r.size && r.offset < chunk.offset + chunk.size) return false;
        }
        (keep ? kept : removed).push_back(chunk);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t pos = ranges[i].offset;
        const size_t limit = pos + ranges[i].size;
        for (const auto & chunk : kept) {
            if (chunk.offset + chunk.size <= pos || chunk.offset >= limit) continue;
            if (pos < chunk.offset) added.push_back({pos, chunk.offset - pos, 0});
            pos = chunk.offset + chunk.size;
        }
        if (pos < limit) added.push_back({pos, limit - pos, 0});
    }
    kept.reserve(kept.size() + added.size());
    size_t committed = 0;
    for (auto & chunk : added) {
        if (!commit(chunk)) {
            while (committed) release(added[--committed]);
            return false;
        }
        ++committed;
    }
    for (const auto & chunk : removed) release(chunk);
    kept.insert(kept.end(), added.begin(), added.end());
    chunks.swap(kept);
    return true;
}
