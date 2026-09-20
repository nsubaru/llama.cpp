#include "ggml.h"
#include "llama.h"
#include "llama-training.h"
#include "../src/llama-adapter.h"
#include "../src/llama-model.h"
#include "ggml-cpu.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <limits>
#include <map>
#include <string>
#include <vector>

constexpr int offset_has_kv      = 1000;
constexpr int offset_has_tensors = 2000;
constexpr int offset_has_data    = 3000;

enum handcrafted_file_type {
    HANDCRAFTED_HEADER_BAD_MAGIC           =  10,
    HANDCRAFTED_HEADER_BAD_VERSION_0       =  15,
    HANDCRAFTED_HEADER_BAD_VERSION_1       =  20,
    HANDCRAFTED_HEADER_BAD_VERSION_FUTURE  =  30,
    HANDCRAFTED_HEADER_BAD_N_TENSORS       =  40,
    HANDCRAFTED_HEADER_BAD_N_KV            =  50,
    HANDCRAFTED_HEADER_EMPTY               = 800,

    HANDCRAFTED_KV_BAD_KEY_SIZE            =  10 + offset_has_kv,
    HANDCRAFTED_KV_EMPTY_KEY               =  15 + offset_has_kv,
    HANDCRAFTED_KV_BAD_TYPE                =  20 + offset_has_kv,
    // HANDCRAFTED_KV_BAD_VALUE_SIZE          =  30 + offset_has_kv, // removed because it can result in allocations > 1 TB (default sanitizer limit)
    HANDCRAFTED_KV_DUPLICATE_KEY           =  40 + offset_has_kv,
    HANDCRAFTED_KV_BAD_ALIGN               =  50 + offset_has_kv,
    HANDCRAFTED_KV_WRONG_TYPE_ALIGN        =  55 + offset_has_kv,
    HANDCRAFTED_KV_SUCCESS                 = 800 + offset_has_kv,

    HANDCRAFTED_TENSORS_BAD_NAME_SIZE      =  10 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_N_DIMS         =  20 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_SHAPE          =  30 + offset_has_tensors,
    HANDCRAFTED_TENSORS_ZERO_DIM           =  35 + offset_has_tensors,
    HANDCRAFTED_TENSORS_NE_TOO_BIG         =  40 + offset_has_tensors,
    HANDCRAFTED_TENSORS_NBYTES_TOO_BIG     =  45 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_TYPE           =  50 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_OFFSET         =  60 + offset_has_tensors,
    HANDCRAFTED_TENSORS_DUPLICATE_NAME     =  70 + offset_has_tensors,
    HANDCRAFTED_TENSORS_BAD_ALIGN          =  75 + offset_has_tensors,
    HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN =  80 + offset_has_tensors,
    HANDCRAFTED_TENSORS_SUCCESS            = 800 + offset_has_tensors,
    HANDCRAFTED_TENSORS_CUSTOM_ALIGN       = 810 + offset_has_tensors,

    HANDCRAFTED_DATA_NOT_ENOUGH_DATA       =  10 + offset_has_data,
    HANDCRAFTED_DATA_BAD_ALIGN             =  15 + offset_has_data,
    HANDCRAFTED_DATA_INCONSISTENT_ALIGN    =  20 + offset_has_data,
    HANDCRAFTED_DATA_MEM_SIZE_OVERFLOW     =  30 + offset_has_data,
    HANDCRAFTED_DATA_SUCCESS               = 800 + offset_has_data,
    HANDCRAFTED_DATA_CUSTOM_ALIGN          = 810 + offset_has_data,
};

static std::string handcrafted_file_type_name(const enum handcrafted_file_type hft) {
    switch (hft) {
        case HANDCRAFTED_HEADER_BAD_MAGIC:           return "HEADER_BAD_MAGIC";
        case HANDCRAFTED_HEADER_BAD_VERSION_0:       return "HEADER_BAD_VERSION_0";
        case HANDCRAFTED_HEADER_BAD_VERSION_1:       return "HEADER_BAD_VERSION_1";
        case HANDCRAFTED_HEADER_BAD_VERSION_FUTURE:  return "HEADER_BAD_VERSION_FUTURE";
        case HANDCRAFTED_HEADER_BAD_N_KV:            return "HEADER_BAD_N_KV";
        case HANDCRAFTED_HEADER_BAD_N_TENSORS:       return "HEADER_BAD_N_TENSORS";
        case HANDCRAFTED_HEADER_EMPTY:               return "HEADER_EMPTY";

        case HANDCRAFTED_KV_BAD_KEY_SIZE:            return "KV_BAD_KEY_SIZE";
        case HANDCRAFTED_KV_EMPTY_KEY:               return "KV_EMPTY_KEY";
        case HANDCRAFTED_KV_BAD_TYPE:                return "KV_BAD_TYPE";
        case HANDCRAFTED_KV_DUPLICATE_KEY:           return "KV_DUPLICATE_KEY";
        case HANDCRAFTED_KV_BAD_ALIGN:               return "KV_BAD_ALIGN";
        case HANDCRAFTED_KV_WRONG_TYPE_ALIGN:        return "KV_WRONG_TYPE_ALIGN";
        case HANDCRAFTED_KV_SUCCESS:                 return "KV_RANDOM_KV";

        case HANDCRAFTED_TENSORS_BAD_NAME_SIZE:      return "TENSORS_BAD_NAME_SIZE";
        case HANDCRAFTED_TENSORS_BAD_N_DIMS:         return "TENSORS_BAD_N_DIMS";
        case HANDCRAFTED_TENSORS_BAD_SHAPE:          return "TENSORS_BAD_SHAPE";
        case HANDCRAFTED_TENSORS_ZERO_DIM:           return "TENSORS_ZERO_DIM";
        case HANDCRAFTED_TENSORS_NE_TOO_BIG:         return "TENSORS_NE_TOO_BIG";
        case HANDCRAFTED_TENSORS_NBYTES_TOO_BIG:     return "TENSORS_NBYTES_TOO_BIG";
        case HANDCRAFTED_TENSORS_BAD_TYPE:           return "TENSORS_BAD_TYPE";
        case HANDCRAFTED_TENSORS_BAD_OFFSET:         return "TENSORS_BAD_OFFSET";
        case HANDCRAFTED_TENSORS_DUPLICATE_NAME:     return "TENSORS_DUPLICATE_NAME";
        case HANDCRAFTED_TENSORS_BAD_ALIGN:          return "TENSORS_BAD_ALIGN";
        case HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN: return "TENSORS_INCONSISTENT_ALIGN";
        case HANDCRAFTED_TENSORS_SUCCESS:            return "TENSORS_SUCCESS";
        case HANDCRAFTED_TENSORS_CUSTOM_ALIGN:       return "TENSORS_CUSTOM_ALIGN";

        case HANDCRAFTED_DATA_NOT_ENOUGH_DATA:       return "DATA_NOT_ENOUGH_DATA";
        case HANDCRAFTED_DATA_BAD_ALIGN:             return "DATA_BAD_ALIGN";
        case HANDCRAFTED_DATA_INCONSISTENT_ALIGN:    return "DATA_INCONSISTENT_ALIGN";
        case HANDCRAFTED_DATA_MEM_SIZE_OVERFLOW:     return "DATA_MEM_SIZE_OVERFLOW";
        case HANDCRAFTED_DATA_SUCCESS:               return "DATA_SUCCESS";
        case HANDCRAFTED_DATA_CUSTOM_ALIGN:          return "DATA_CUSTOM_ALIGN";
    }
    GGML_ABORT("fatal error");
}

static bool expect_context_not_null(const enum handcrafted_file_type hft) {
    if (hft == HANDCRAFTED_TENSORS_ZERO_DIM) {
        return true;
    }
    if (hft < offset_has_kv) {
        return hft >= HANDCRAFTED_HEADER_EMPTY;
    }
    if (hft < offset_has_tensors) {
        return hft >= HANDCRAFTED_KV_SUCCESS;
    }
    if (hft < offset_has_data) {
        return hft >= HANDCRAFTED_TENSORS_SUCCESS;
    }
    return hft >= HANDCRAFTED_DATA_SUCCESS;
}

typedef std::pair<enum ggml_type, std::array<int64_t, GGML_MAX_DIMS>> tensor_config_t;

static std::vector<tensor_config_t> get_tensor_configs(std::mt19937 & rng) {
    std::vector<tensor_config_t> tensor_configs;
    tensor_configs.reserve(100);

    for (int i = 0; i < 100; ++i) {
        const enum ggml_type type = ggml_type(rng() % GGML_TYPE_COUNT);
        if (ggml_type_size(type) == 0) {
            continue;
        }

        std::array<int64_t, GGML_MAX_DIMS> shape = {1, 1, 1, 1};
        shape[0] = (1 + rng() % 10) * ggml_blck_size(type);
        const int n_dims = 1 + rng() % GGML_MAX_DIMS;
        for (int i = 1; i < n_dims; ++i) {
            shape[i] = 1 + rng() % 10;
        }

        tensor_configs.push_back(std::make_pair(type, shape));
    }

    return tensor_configs;
}

static std::vector<std::pair<enum gguf_type, enum gguf_type>> get_kv_types(std::mt19937 rng) {
    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    kv_types.reserve(100);

    for (int i = 0; i < 100; ++i) {
        const gguf_type type = gguf_type(rng() % GGUF_TYPE_COUNT);

        if (type == GGUF_TYPE_ARRAY) {
            const gguf_type type_arr = gguf_type(rng() % GGUF_TYPE_COUNT);
            if (type_arr == GGUF_TYPE_ARRAY) {
                continue;
            }
            kv_types.push_back(std::make_pair(type, type_arr));
            continue;
        }

        kv_types.push_back(std::make_pair(type, gguf_type(-1)));
    }
    std::shuffle(kv_types.begin(), kv_types.end(), rng);

    return kv_types;
}

template <typename T>
static void helper_write(FILE * file, const T & val) {
    GGML_ASSERT(fwrite(&val, 1, sizeof(val), file) == sizeof(val));
}

static void helper_write(FILE * file, const void * data, const size_t nbytes) {
    GGML_ASSERT(fwrite(data, 1, nbytes, file) == nbytes);
}

static std::vector<uint8_t> read_file_to_buffer(FILE * file) {
    GGML_ASSERT(file != nullptr);
    GGML_ASSERT(fseek(file, 0, SEEK_END) == 0);

    const long size = ftell(file);
    GGML_ASSERT(size >= 0);

    rewind(file);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    GGML_ASSERT(fread(data.data(), 1, data.size(), file) == data.size());

    rewind(file);
    return data;
}

struct callback_reader_data {
    const uint8_t * data;
    size_t size;
};

static size_t read_buffer_callback(void * userdata, void * output, uint64_t offset, size_t len) {
    GGML_ASSERT(len > 0);

    const callback_reader_data & reader = *static_cast<callback_reader_data *>(userdata);

    if (offset > reader.size || len > reader.size - offset) {
        return 0;
    }

    const size_t data_offset = static_cast<size_t>(offset);
    const size_t nread = std::min(len, reader.size - data_offset);
    memcpy(static_cast<uint8_t *>(output), reader.data + data_offset, nread);
    return nread;
}

static FILE * get_handcrafted_file(const unsigned int seed, const enum handcrafted_file_type hft, const int extra_bytes = 0) {
    FILE * file = tmpfile();

    if (!file) {
        return file;
    }

    std::mt19937 rng(seed);
    uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;

    if (hft == HANDCRAFTED_HEADER_BAD_MAGIC) {
        const char bad_magic[4] = {'F', 'U', 'G', 'G'};
        helper_write(file, bad_magic, sizeof(bad_magic));
    } else {
        helper_write(file, GGUF_MAGIC, 4);
    }

    if (hft == HANDCRAFTED_HEADER_BAD_VERSION_0) {
        const uint32_t version = 0;
        helper_write(file, version);
    } else if (hft == HANDCRAFTED_HEADER_BAD_VERSION_1) {
        const uint32_t version = 1;
        helper_write(file, version);
    } else if (hft == HANDCRAFTED_HEADER_BAD_VERSION_FUTURE) {
        const uint32_t version = GGUF_VERSION + 1;
        helper_write(file, version);
    } else {
        const uint32_t version = GGUF_VERSION;
        helper_write(file, version);
    }

    std::vector<tensor_config_t> tensor_configs;
    if (hft >= offset_has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }

    if (hft == HANDCRAFTED_DATA_MEM_SIZE_OVERFLOW) {
        tensor_configs.resize(2);

        tensor_configs[0] = { GGML_TYPE_I8, { 0x7FFFFFFFFFFFFFC0, 1, 1, 1 } };
        tensor_configs[1] = { GGML_TYPE_I8, { 0x7FFFFFFFFFFFFFC0, 1, 1, 1 } };
    }

    if (hft == HANDCRAFTED_HEADER_BAD_N_TENSORS) {
        const uint64_t n_tensors = -1;
        helper_write(file, n_tensors);
    } else {
        const uint64_t n_tensors = tensor_configs.size();
        helper_write(file, n_tensors);
    }

    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    if (hft >= offset_has_kv) {
        kv_types = get_kv_types(rng);
    }
    {
        uint64_t n_kv = kv_types.size();
        if (hft == HANDCRAFTED_KV_BAD_ALIGN        || hft == HANDCRAFTED_KV_WRONG_TYPE_ALIGN ||
            hft == HANDCRAFTED_TENSORS_BAD_ALIGN   || hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN ||
            hft == HANDCRAFTED_DATA_BAD_ALIGN      || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN) {

            n_kv += 1;
        } else if (hft == HANDCRAFTED_HEADER_BAD_N_KV) {
            n_kv = -1;
        }
        helper_write(file, n_kv);
    }

    if (hft < offset_has_kv) {
        while (ftell(file) % alignment != 0) {
            const char pad = 0;
            helper_write(file, pad);
        }

        for (int i = 0; i < extra_bytes; ++i) {
            const char tmp = 0;
            helper_write(file, tmp);
        }
        rewind(file);
        return file;
    }

    for (int i = 0; i < int(kv_types.size()); ++i) {
        const enum gguf_type type     = gguf_type(hft == HANDCRAFTED_KV_BAD_TYPE ? GGUF_TYPE_COUNT : kv_types[i].first);
        const enum gguf_type type_arr = gguf_type(hft == HANDCRAFTED_KV_BAD_TYPE ? GGUF_TYPE_COUNT : kv_types[i].second);

        const std::string key = hft == HANDCRAFTED_KV_EMPTY_KEY
            ? ""
            : "my_key_" + std::to_string((hft == HANDCRAFTED_KV_DUPLICATE_KEY ? i/2 : i));

        if (hft == HANDCRAFTED_KV_BAD_KEY_SIZE) {
            const uint64_t n = -1;
            helper_write(file, n);
        } else {
            const uint64_t n = key.length();
            helper_write(file, n);
        }
        helper_write(file, key.data(), key.length());

        {
            const int32_t type32 = int32_t(type);
            helper_write(file, type32);
        }

        uint32_t data[16];
        for (int j = 0; j < 16; ++j) {
            data[j] = rng();
            if (type == GGUF_TYPE_STRING || type_arr == GGUF_TYPE_STRING) {
                data[j] |= 0x01010101; // avoid random null-termination of string
            }
        }

        if (type == GGUF_TYPE_STRING) {
            const uint64_t n = rng() % sizeof(data);
            helper_write(file, n);
            helper_write(file, data, n);
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            {
                const int32_t type32 = int32_t(type_arr);
                helper_write(file, type32);
            }
            if (type_arr == GGUF_TYPE_STRING) {
                const uint64_t nstr = rng() % (16 + 1);
                helper_write(file, nstr);
                for (uint64_t istr = 0; istr < nstr; ++istr) {
                    const uint64_t n = rng() % (sizeof(uint32_t) + 1);
                    helper_write(file, n);
                    helper_write(file, &data[istr], n);
                }
                continue;
            }
            const size_t type_size = gguf_type_size(type_arr);
            const uint64_t n = (rng() % sizeof(data)) / type_size;
            helper_write(file, n);
            helper_write(file, &data, n*type_size);
            continue;
        }

        helper_write(file, data, hft == HANDCRAFTED_KV_BAD_TYPE ? 1 : gguf_type_size(type));
    }

    if (hft == HANDCRAFTED_KV_BAD_ALIGN        || hft == HANDCRAFTED_KV_WRONG_TYPE_ALIGN ||
        hft == HANDCRAFTED_TENSORS_BAD_ALIGN   || hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN ||
        hft == HANDCRAFTED_DATA_BAD_ALIGN      || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN) {

        const uint64_t n = strlen(GGUF_KEY_GENERAL_ALIGNMENT);
        helper_write(file, n);
        helper_write(file, GGUF_KEY_GENERAL_ALIGNMENT, n);

        // HANDCRAFTED_KV_WRONG_TYPE_ALIGN declares general.alignment with a non-UINT32 type,
        // which the loader must reject cleanly instead of aborting on an assertion
        const int32_t type = hft == HANDCRAFTED_KV_WRONG_TYPE_ALIGN ? int32_t(GGUF_TYPE_INT32) : int32_t(GGUF_TYPE_UINT32);
        helper_write(file, type);

        alignment = expect_context_not_null(hft) ? 1 : 13;
        helper_write(file, alignment);
    }

    if (hft < offset_has_tensors) {
        while (ftell(file) % alignment != 0) {
            const char pad = 0;
            helper_write(file, pad);
        }

        for (int i = 0; i < extra_bytes; ++i) {
            const char tmp = 0;
            helper_write(file, tmp);
        }
        rewind(file);
        return file;
    }

    if (hft == HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN || hft == HANDCRAFTED_DATA_INCONSISTENT_ALIGN) {
        alignment = 1;
    }

    uint64_t offset = 0;
    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = hft == HANDCRAFTED_TENSORS_NBYTES_TOO_BIG ? GGML_TYPE_I64 : tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        std::string name = "my_tensor";
        if (hft != HANDCRAFTED_TENSORS_DUPLICATE_NAME) {
            name += "_" + std::to_string(i);
        }
        if (hft == HANDCRAFTED_TENSORS_BAD_NAME_SIZE) {
            name += "_with_a_very_long_name_which_is_longer_than_what_is_allowed_for_ggml_tensors";
            GGML_ASSERT(name.length() >= GGML_MAX_NAME);
        }
        {
            const uint64_t n = name.length();
            helper_write(file, n);
        }
        helper_write(file, name.data(), name.length());

        uint32_t n_dims = (hft == HANDCRAFTED_TENSORS_NE_TOO_BIG || hft == HANDCRAFTED_TENSORS_NBYTES_TOO_BIG) ? 2 : 1;
        for (int i = GGML_MAX_DIMS-1; i >= 1; --i) {
            if (shape[i] != 1) {
                n_dims = i + 1;
                break;
            }
        }
        if (hft == HANDCRAFTED_TENSORS_ZERO_DIM) {
            n_dims = 2;
        }
        if (hft == HANDCRAFTED_TENSORS_BAD_N_DIMS) {
            const uint32_t n_dims_bad = GGML_MAX_DIMS + 1;
            helper_write(file, n_dims_bad);
        } else {
            helper_write(file, n_dims);
        }

        if (hft == HANDCRAFTED_TENSORS_BAD_SHAPE) {
            const int64_t bad_dim = -1;
            for (uint32_t j = 0; j < n_dims; ++j) {
                helper_write(file, bad_dim);
            }
        } else if (hft == HANDCRAFTED_TENSORS_ZERO_DIM) {
            const int64_t zero_shape[2] = { shape[0], 0 };
            helper_write(file, zero_shape, 2*sizeof(int64_t));
        } else if (hft == HANDCRAFTED_TENSORS_NE_TOO_BIG){
            const int64_t big_dim = 4*int64_t(INT32_MAX);
            for (uint32_t j = 0; j < n_dims; ++j) {
                helper_write(file, big_dim);
            }
        } else if (hft == HANDCRAFTED_TENSORS_NBYTES_TOO_BIG){
            const size_t  big_ne  = SIZE_MAX/ggml_type_size(type);
            const int64_t big_dim = GGML_PAD(int64_t(1.01f*std::pow(big_ne, 1.0f/n_dims)) + 1, ggml_blck_size(type));
            for (uint32_t j = 0; j < n_dims; ++j) {
                helper_write(file, big_dim);
            }
        } else {
            helper_write(file, shape.data(), n_dims*sizeof(int64_t));
        }

        {
            const int32_t type32 = hft == HANDCRAFTED_TENSORS_BAD_TYPE ? GGML_TYPE_COUNT : int32_t(type);
            helper_write(file, type32);
        }

        if (hft == HANDCRAFTED_TENSORS_BAD_OFFSET) {
            const uint64_t bad_offset = -1;
            helper_write(file, bad_offset);
        } else {
            helper_write(file, offset);
        }

        int64_t ne = shape[0];
        for (uint32_t i = 1; i < n_dims; ++i) {
            ne *= shape[i];
        }
        if (hft == HANDCRAFTED_TENSORS_ZERO_DIM) {
            ne = 0;
        }

        offset += GGML_PAD(ggml_row_size(type, ne), (uint64_t) alignment);
    }

    while (ftell(file) % alignment != 0) {
        const char pad = 0;
        helper_write(file, pad);
    }

    if (hft >= offset_has_data) {
        rng.seed(seed + 1);
        uint64_t nbytes = offset;
        if (hft == HANDCRAFTED_DATA_NOT_ENOUGH_DATA) {
            nbytes -= 1;
        }
        if (hft == HANDCRAFTED_DATA_MEM_SIZE_OVERFLOW) {
            nbytes = 32;
        }
        for (uint64_t i = 0; i < nbytes; ++i) {
            const uint8_t random_byte = i % 256;
            helper_write(file, random_byte);
        }
    }

    for (int i = 0; i < extra_bytes; ++i) {
        const char tmp = 0;
        helper_write(file, tmp);
    }
    rewind(file);
    return file;
}

static bool handcrafted_check_header(const gguf_context * gguf_ctx, const unsigned int seed, const bool has_kv, const bool has_tensors, const bool alignment_defined) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs;
    if (has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }
    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types;
    if (has_kv) {
        kv_types = get_kv_types(rng);
    }

    bool ok = true;

    if (gguf_get_version(gguf_ctx) != GGUF_VERSION) {
        ok = false;
    }
    if (gguf_get_n_tensors(gguf_ctx) != int(tensor_configs.size())) {
        ok = false;
    }
    if (gguf_get_n_kv(gguf_ctx) != int(alignment_defined ? kv_types.size() + 1 : kv_types.size())) {
        ok = false;
    }

    return ok;
}

static bool handcrafted_check_kv(const gguf_context * gguf_ctx, const unsigned int seed, const bool has_tensors, const bool alignment_defined) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs;
    if (has_tensors) {
        tensor_configs = get_tensor_configs(rng);
    }

    std::vector<std::pair<enum gguf_type, enum gguf_type>> kv_types = get_kv_types(rng);

    bool ok = true;

    for (int i = 0; i < int(kv_types.size()); ++i) {
        const enum gguf_type type     = gguf_type(kv_types[i].first);
        const enum gguf_type type_arr = gguf_type(kv_types[i].second);

        const std::string key = "my_key_" + std::to_string(i);

        uint32_t data[16];
        for (int j = 0; j < 16; ++j) {
            data[j] = rng();
            if (type == GGUF_TYPE_STRING || type_arr == GGUF_TYPE_STRING) {
                data[j] |= 0x01010101; // avoid random null-termination of string
            }
        }

        const char * data8 = reinterpret_cast<const char *>(data);
        const int id = gguf_find_key(gguf_ctx, key.c_str());

        if (type == GGUF_TYPE_STRING) {
            const char * str = gguf_get_val_str(gguf_ctx, id);
            const uint64_t n = strlen(str);
            const uint64_t n_expected = rng() % sizeof(data);
            if (n != n_expected) {
                ok = false;
                continue;
            }
            if (!std::equal(str, str + n, data8)) {
                ok = false;
            }
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            const size_t type_size = gguf_type_size(type_arr);
            const uint64_t arr_n = gguf_get_arr_n(gguf_ctx, id);

            if (type_arr == GGUF_TYPE_STRING) {
                const uint64_t nstr_expected = rng() % (16 + 1);
                if (arr_n != nstr_expected) {
                    ok = false;
                    continue;
                }
                for (uint64_t istr = 0; istr < nstr_expected; ++istr) {
                    const char * str = gguf_get_arr_str(gguf_ctx, id, istr);
                    const uint64_t n = strlen(str);
                    const uint64_t n_expected = rng() % (sizeof(uint32_t) + 1);

                    if (n != n_expected) {
                        ok = false;
                        continue;
                    }
                    const char * str_expected = reinterpret_cast<const char *>(&data[istr]);
                    if (strncmp(str, str_expected, n) != 0) {
                        ok = false;
                        continue;
                    }
                }
                continue;
            }

            const uint64_t arr_n_expected = (rng() % sizeof(data)) / type_size;
            if (arr_n != arr_n_expected) {
                ok = false;
                continue;
            }

            const char * data_gguf = reinterpret_cast<const char *>(gguf_get_arr_data(gguf_ctx, id));

            if (type_arr == GGUF_TYPE_BOOL) {
                for (size_t arr_i = 0; arr_i < arr_n; ++arr_i) {
                    if (bool(data8[arr_i]) != bool(data_gguf[arr_i])) {
                        ok = false;
                    }
                }
                continue;
            }

            if (!std::equal(data8, data8 + arr_n*type_size, data_gguf)) {
                ok = false;
            }
            continue;
        }

        const char * data_gguf = reinterpret_cast<const char *>(gguf_get_val_data(gguf_ctx, id));

        if (type == GGUF_TYPE_BOOL) {
            if (bool(*data8) != bool(*data_gguf)) {
                ok = false;
            }
            continue;
        }

        if (!std::equal(data8, data8 + gguf_type_size(type), data_gguf)) {
            ok = false;
        }
    }

    const uint32_t expected_alignment = alignment_defined ? 1 : GGUF_DEFAULT_ALIGNMENT;
    if (gguf_get_alignment(gguf_ctx) != expected_alignment) {
        ok = false;
    }

    return ok;
}

static bool handcrafted_check_tensors(const gguf_context * gguf_ctx, const unsigned int seed) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs = get_tensor_configs(rng);

    // Call get_kv_types to get the same RNG state:
    get_kv_types(rng);

    bool ok = true;

    const int id_alignment = gguf_find_key(gguf_ctx, GGUF_KEY_GENERAL_ALIGNMENT);
    const uint32_t alignment = id_alignment >= 0 ? gguf_get_val_u32(gguf_ctx, id_alignment) : GGUF_DEFAULT_ALIGNMENT;

    uint64_t expected_offset = 0;
    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        const std::string name = "my_tensor_" + std::to_string(i);
        const int id = gguf_find_tensor(gguf_ctx, name.c_str());

        if (id >= 0) {
            if (std::string(gguf_get_tensor_name(gguf_ctx, id)) != name) {
                ok = false;
            }

            if (gguf_get_tensor_type(gguf_ctx, id) != type) {
                ok = false;
            }

            const int64_t * ne = gguf_get_tensor_ne(gguf_ctx, id);
            for (int j = 0; j < GGML_MAX_DIMS; ++j) {
                if (ne[j] != shape[j]) {
                    ok = false;
                }
            }
        } else {
            ok = false;
            continue;
        }

        const size_t offset = gguf_get_tensor_offset(gguf_ctx, id);

        if (offset != expected_offset) {
            ok = false;
        }

        int64_t ne = shape[0];
        for (size_t j = 1; j < GGML_MAX_DIMS; ++j) {
            ne *= shape[j];
        }
        expected_offset += GGML_PAD(ggml_row_size(type, ne), alignment);
    }

    return ok;
}

static bool handcrafted_check_tensor_data(const gguf_context * gguf_ctx, const unsigned int seed, FILE * file) {
    if (!gguf_ctx) {
        return false;
    }

    std::mt19937 rng(seed);

    std::vector<tensor_config_t> tensor_configs = get_tensor_configs(rng);

    bool ok = true;

    for (int i = 0; i < int(tensor_configs.size()); ++i) {
        const ggml_type                          type  = tensor_configs[i].first;
        const std::array<int64_t, GGML_MAX_DIMS> shape = tensor_configs[i].second;

        int64_t ne = shape[0];
        for (size_t j = 1; j < GGML_MAX_DIMS; ++j) {
            ne *= shape[j];
        }
        const size_t size = ggml_row_size(type, ne);

        const std::string name = "my_tensor_" + std::to_string(i);
        const size_t offset = gguf_get_tensor_offset(gguf_ctx, gguf_find_tensor(gguf_ctx, name.c_str()));

        std::vector<uint8_t> data(size);
        GGML_ASSERT(fseek(file, gguf_get_data_offset(gguf_ctx) + offset, SEEK_SET) == 0);
        GGML_ASSERT(fread(data.data(), 1, data.size(), file) == data.size());

        for (size_t j = 0; j < size; ++j) {
            const uint8_t expected_byte = (j + offset) % 256;
            if (data[j] != expected_byte) {
                ok = false;
            }
        }
    }

    return ok;
}

static std::pair<int, int> test_handcrafted_file(const unsigned int seed) {
    int npass = 0;
    int ntest = 0;

    const std::vector<handcrafted_file_type> hfts = {
        HANDCRAFTED_HEADER_BAD_MAGIC,
        HANDCRAFTED_HEADER_BAD_VERSION_0,
        HANDCRAFTED_HEADER_BAD_VERSION_1,
        HANDCRAFTED_HEADER_BAD_VERSION_FUTURE,
        HANDCRAFTED_HEADER_BAD_N_KV,
        HANDCRAFTED_HEADER_BAD_N_TENSORS,
        HANDCRAFTED_HEADER_EMPTY,

        HANDCRAFTED_KV_BAD_KEY_SIZE,
        HANDCRAFTED_KV_EMPTY_KEY,
        HANDCRAFTED_KV_BAD_TYPE,
        HANDCRAFTED_KV_DUPLICATE_KEY,
        HANDCRAFTED_KV_BAD_ALIGN,
        HANDCRAFTED_KV_WRONG_TYPE_ALIGN,
        HANDCRAFTED_KV_SUCCESS,

        HANDCRAFTED_TENSORS_BAD_NAME_SIZE,
        HANDCRAFTED_TENSORS_BAD_N_DIMS,
        HANDCRAFTED_TENSORS_BAD_SHAPE,
        HANDCRAFTED_TENSORS_ZERO_DIM,
        HANDCRAFTED_TENSORS_NE_TOO_BIG,
        HANDCRAFTED_TENSORS_NBYTES_TOO_BIG,
        HANDCRAFTED_TENSORS_BAD_TYPE,
        HANDCRAFTED_TENSORS_BAD_OFFSET,
        HANDCRAFTED_TENSORS_DUPLICATE_NAME,
        HANDCRAFTED_TENSORS_BAD_ALIGN,
        HANDCRAFTED_TENSORS_INCONSISTENT_ALIGN,
        HANDCRAFTED_TENSORS_SUCCESS,
        HANDCRAFTED_TENSORS_CUSTOM_ALIGN,

        HANDCRAFTED_DATA_NOT_ENOUGH_DATA,
        HANDCRAFTED_DATA_BAD_ALIGN,
        HANDCRAFTED_DATA_INCONSISTENT_ALIGN,
        HANDCRAFTED_DATA_MEM_SIZE_OVERFLOW,
        HANDCRAFTED_DATA_SUCCESS,
        HANDCRAFTED_DATA_CUSTOM_ALIGN,
    };

    for (enum handcrafted_file_type hft : hfts) {
        printf("%s: handcrafted_file_type=%s\n", __func__, handcrafted_file_type_name(hft).c_str());
        FILE * file = get_handcrafted_file(seed, hft);

#ifdef _WIN32
        if (!file) {
            printf("failed to create tmpfile(), needs elevated privileges on Windows");
            printf("skipping tests");
            continue;
        }
#else
        GGML_ASSERT(file);
#endif // _WIN32

        struct ggml_context * ctx = nullptr;
        struct gguf_init_params gguf_params = {
            /*no_alloc =*/ false,
            /*ctx      =*/ hft >= offset_has_data ? &ctx : nullptr,
        };

        struct gguf_context * gguf_ctx = gguf_init_from_file_ptr(file, gguf_params);

        if (expect_context_not_null(hft)) {
            printf("%s:   - context_not_null: ", __func__);
        } else {
            printf("%s:   - context_null: ", __func__);
        }
        if (bool(gguf_ctx) == expect_context_not_null(hft)) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;

        if (hft >= offset_has_data && !expect_context_not_null(hft)) {
            printf("%s:   - no_dangling_ggml_context_pointer: ", __func__);
            if (ctx) {
                printf("\033[1;31mFAIL\033[0m\n");
            } else {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            }
            ntest++;
        }

        const bool alignment_defined = hft == HANDCRAFTED_TENSORS_CUSTOM_ALIGN || hft == HANDCRAFTED_DATA_CUSTOM_ALIGN;

        if (expect_context_not_null(hft)) {
            printf("%s:   - check_header: ", __func__);
            if (handcrafted_check_header(gguf_ctx, seed, hft >= offset_has_kv, hft >= offset_has_tensors, alignment_defined)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        if (expect_context_not_null(hft) && hft >= offset_has_kv) {
            printf("%s:   - check_kv: ", __func__);
            if (handcrafted_check_kv(gguf_ctx, seed, hft >= offset_has_tensors, alignment_defined)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        // HANDCRAFTED_TENSORS_ZERO_DIM deliberately mangles the tensor shapes to 0 elements,
        // so only assert that it loads without crashing; skip the exact-geometry comparison.
        if (expect_context_not_null(hft) && hft >= offset_has_tensors && hft != HANDCRAFTED_TENSORS_ZERO_DIM) {
            printf("%s:   - check_tensors: ", __func__);
            if (handcrafted_check_tensors(gguf_ctx, seed)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        if (expect_context_not_null(hft) && hft >= offset_has_data) {
            printf("%s:   - check_tensor_data: ", __func__);
            if (handcrafted_check_tensor_data(gguf_ctx, seed, file)) {
                printf("\033[1;32mOK\033[0m\n");
                npass++;
            } else {
                printf("\033[1;31mFAIL\033[0m\n");
            }
            ntest++;
        }

        fclose(file);
        if (gguf_ctx) {
            ggml_free(ctx);
            gguf_free(gguf_ctx);
        }
        printf("\n");
    }


    return std::make_pair(npass, ntest);
}

struct random_gguf_context_result {
    struct gguf_context * gguf_ctx;
    struct ggml_context * ctx;
    ggml_backend_buffer_t buffer;
};

static struct random_gguf_context_result get_random_gguf_context(ggml_backend_t backend, const unsigned int seed) {
    std::mt19937 rng(seed);

    struct gguf_context * gguf_ctx = gguf_init_empty();

    for (int i = 0; i < 256; ++i) {
        const std::string key = "my_key_" + std::to_string(rng() % 1024);
        const enum gguf_type type = gguf_type(rng() % GGUF_TYPE_COUNT);

        switch (type) {
            case GGUF_TYPE_UINT8:   gguf_set_val_u8  (gguf_ctx, key.c_str(), rng() % (1 <<  7));             break;
            case GGUF_TYPE_INT8:    gguf_set_val_i8  (gguf_ctx, key.c_str(), rng() % (1 <<  7) - (1 <<  6)); break;
            case GGUF_TYPE_UINT16:  gguf_set_val_u16 (gguf_ctx, key.c_str(), rng() % (1 << 15));             break;
            case GGUF_TYPE_INT16:   gguf_set_val_i16 (gguf_ctx, key.c_str(), rng() % (1 << 15) - (1 << 14)); break;
            case GGUF_TYPE_UINT32:  gguf_set_val_u32 (gguf_ctx, key.c_str(), rng());                         break;
            case GGUF_TYPE_INT32:   gguf_set_val_i32 (gguf_ctx, key.c_str(), rng()             - (1 << 30)); break;
            case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (gguf_ctx, key.c_str(), rng() % 1024      - 512);       break;
            case GGUF_TYPE_BOOL:    gguf_set_val_bool(gguf_ctx, key.c_str(), rng() % 2 == 0);                break;
            case GGUF_TYPE_STRING:  gguf_set_val_str (gguf_ctx, key.c_str(), std::to_string(rng()).c_str()); break;
            case GGUF_TYPE_UINT64:  gguf_set_val_u64 (gguf_ctx, key.c_str(), rng());                         break;
            case GGUF_TYPE_INT64:   gguf_set_val_i64 (gguf_ctx, key.c_str(), rng()             - (1 << 30)); break;
            case GGUF_TYPE_FLOAT64: gguf_set_val_f32 (gguf_ctx, key.c_str(), rng() % 1024      - 512);       break;
            case GGUF_TYPE_ARRAY: {
                const enum gguf_type type_arr = gguf_type(rng() % GGUF_TYPE_COUNT);
                const uint64_t ne = rng() % 1024;

                switch (type_arr) {
                    case GGUF_TYPE_UINT8:
                    case GGUF_TYPE_INT8:
                    case GGUF_TYPE_UINT16:
                    case GGUF_TYPE_INT16:
                    case GGUF_TYPE_UINT32:
                    case GGUF_TYPE_INT32:
                    case GGUF_TYPE_FLOAT32:
                    case GGUF_TYPE_BOOL:
                    case GGUF_TYPE_UINT64:
                    case GGUF_TYPE_INT64:
                    case GGUF_TYPE_FLOAT64: {
                        const size_t nbytes = ne*gguf_type_size(type_arr);
                        std::vector<uint32_t> random_data((nbytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
                        for (size_t j = 0; j < random_data.size(); ++j) {
                            random_data[j] = rng();
                            if (type_arr == GGUF_TYPE_BOOL) {
                                random_data[j] &= 0x01010101; // the sanitizer complains if booleans are not 0 or 1
                            }
                        }
                        gguf_set_arr_data(gguf_ctx, key.c_str(), type_arr, random_data.data(), ne);
                    } break;
                    case GGUF_TYPE_STRING: {
                        std::vector<std::string>  data_cpp(ne);
                        std::vector<const char *> data_c(ne);
                        for (size_t j = 0; j < data_cpp.size(); ++j) {
                            data_cpp[j] = std::to_string(rng());
                            data_c[j]   = data_cpp[j].c_str();
                        }
                        gguf_set_arr_str(gguf_ctx, key.c_str(), data_c.data(), ne);
                    } break;
                    case GGUF_TYPE_ARRAY: {
                        break; // not supported
                    }
                    case GGUF_TYPE_COUNT:
                    default: {
                        GGML_ABORT("fatal error");
                    }
                }
            } break;
            case GGUF_TYPE_COUNT:
            default: {
                GGML_ABORT("fatal error");
            }
        }
    }

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 256*ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(ggml_params);

    for (int i = 0; i < 256; ++i) {
        const std::string name = "my_tensor_" + std::to_string(i);
        const enum ggml_type type = ggml_type(rng() % GGML_TYPE_COUNT);
        const size_t type_size = ggml_type_size(type);

        if (type_size == 0) {
            continue;
        }

        const int n_dims = 1 + rng() % GGML_MAX_DIMS;
        int64_t ne[GGML_MAX_DIMS];
        ne[0] = (1 + rng() % 10) * ggml_blck_size(type);
        for (int j = 1; j < n_dims; ++j) {
            ne[j] = 1 + rng() % 10;
        }

        struct ggml_tensor * tensor = ggml_new_tensor(ctx, type, n_dims, ne);
        ggml_set_name(tensor, name.c_str());
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (struct ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
        const size_t nbytes = ggml_nbytes(t);
        std::vector<uint32_t> random_data((nbytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        for (size_t j = 0; j < random_data.size(); ++j) {
            random_data[j] = rng();
        }
        ggml_backend_tensor_set(t, random_data.data(), 0, nbytes);

        gguf_add_tensor(gguf_ctx, t);
    }

    return {gguf_ctx, ctx, buf};
}

static bool all_kv_in_other(const gguf_context * ctx, const gguf_context * other) {
    bool ok = true;

    const int n_kv = gguf_get_n_kv(ctx);
    for (int id = 0; id < n_kv; ++id) {
        const char * name = gguf_get_key(ctx, id);

        const int idx_other = gguf_find_key(other, name);
        if (idx_other < 0) {
            ok = false;
            continue;
        }

        const gguf_type type = gguf_get_kv_type(ctx, id);
        if (type != gguf_get_kv_type(other, idx_other)) {
            ok = false;
            continue;
        }

        if (type == GGUF_TYPE_ARRAY) {
            const size_t arr_n = gguf_get_arr_n(ctx, id);
            if (arr_n != gguf_get_arr_n(other, idx_other)) {
                ok = false;
                continue;
            }

            const gguf_type type_arr = gguf_get_arr_type(ctx, id);
            if (type_arr != gguf_get_arr_type(other, idx_other)) {
                ok = false;
                continue;
            }

            if (type_arr == GGUF_TYPE_BOOL) {
                const int8_t * data       = reinterpret_cast<const int8_t *>(gguf_get_arr_data(ctx,   id));
                const int8_t * data_other = reinterpret_cast<const int8_t *>(gguf_get_arr_data(other, idx_other));
                for (size_t arr_i = 0; arr_i < arr_n; ++arr_i) {
                    if (bool(data[arr_i]) != bool(data_other[arr_i])) {
                        ok = false;
                    }
                }
                continue;
            }

            if (type_arr == GGUF_TYPE_STRING) {
                for (size_t arr_i = 0; arr_i < arr_n; ++arr_i) {
                    const std::string str       = gguf_get_arr_str(ctx,   id,       arr_i);
                    const std::string str_other = gguf_get_arr_str(other, idx_other, arr_i);
                    if (str != str_other) {
                        ok = false;
                    }
                }
                continue;
            }

            const int8_t * data       = reinterpret_cast<const int8_t *>(gguf_get_arr_data(ctx,   id));
            const int8_t * data_other = reinterpret_cast<const int8_t *>(gguf_get_arr_data(other, idx_other));
            if (!std::equal(data, data + arr_n*gguf_type_size(type_arr), data_other)) {
                ok = false;
            }
            continue;
        }

        if (type == GGUF_TYPE_STRING) {
            const std::string str       = gguf_get_val_str(ctx,   id);
            const std::string str_other = gguf_get_val_str(other, idx_other);
            if (str != str_other) {
                ok = false;
            }
            continue;
        }

        const char * data       = reinterpret_cast<const char *>(gguf_get_val_data(ctx,   id));
        const char * data_other = reinterpret_cast<const char *>(gguf_get_val_data(other, idx_other));
        if (!std::equal(data, data + gguf_type_size(type), data_other)) {
            ok = false;
        }
    }

    return ok;
}

static bool all_tensors_in_other(const gguf_context * ctx, const gguf_context * other) {
    bool ok = true;

    const int n_tensors = gguf_get_n_tensors(ctx);
    for (int id = 0; id < n_tensors; ++id) {
        const std::string name = gguf_get_tensor_name(ctx, id);

        const int idx_other = gguf_find_tensor(other, name.c_str());
        if (id != idx_other) {
            ok = false;
            if (idx_other < 0) {
                continue;
            }
        }

        const ggml_type type = gguf_get_tensor_type(ctx, id);
        if (type != gguf_get_tensor_type(other, id)) {
            ok = false;
        }

        const size_t offset = gguf_get_tensor_offset(ctx, id);
        if (offset != gguf_get_tensor_offset(other, id)) {
            ok = false;
        }
    }

    return ok;
}

static bool same_tensor_data(const struct ggml_context * orig, const struct ggml_context * read) {
    bool ok = true;

    struct ggml_tensor * t_orig = ggml_get_first_tensor(orig);
    struct ggml_tensor * t_read = ggml_get_first_tensor(read);

    if (std::string(t_read->name) != "GGUF tensor data binary blob") {
        return false;
    }
    t_read = ggml_get_next_tensor(read, t_read);

    while (t_orig) {
        if (!t_read) {
            ok = false;
            break;
        }

        const size_t nbytes = ggml_nbytes(t_orig);
        if (ggml_nbytes(t_read) != nbytes) {
            ok = false;
            break;
        }
        std::vector<char> data_orig(nbytes);
        ggml_backend_tensor_get(t_orig, data_orig.data(), 0, nbytes);
        if (!std::equal(data_orig.data(), data_orig.data() + nbytes, reinterpret_cast<const char *>(t_read->data))) {
            ok = false;
        }

        t_orig = ggml_get_next_tensor(orig, t_orig);
        t_read = ggml_get_next_tensor(read, t_read);
    }
    if (t_read) {
        ok = false;
    }

    return ok;
}

enum roundtrip_read_mode {
    ROUNDTRIP_READ_MODE_FILE,
    ROUNDTRIP_READ_MODE_FILE_OFFSET, // GGUF embedded after some bytes of a bigger file
    ROUNDTRIP_READ_MODE_BUFFER,
    ROUNDTRIP_READ_MODE_CALLBACK,
};

static const char * roundtrip_read_mode_name(const roundtrip_read_mode mode) {
    switch (mode) {
        case ROUNDTRIP_READ_MODE_FILE:        return "file";
        case ROUNDTRIP_READ_MODE_FILE_OFFSET: return "file_offset";
        case ROUNDTRIP_READ_MODE_BUFFER:      return "buffer";
        case ROUNDTRIP_READ_MODE_CALLBACK:    return "callback";
    }

    GGML_ABORT("fatal error");
}

static std::pair<int, int> test_roundtrip(
        ggml_backend_dev_t dev, const unsigned int seed, const bool only_meta,
        const roundtrip_read_mode read_mode) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("%s: device=%s, backend=%s, only_meta=%s, read_mode=%s\n",
        __func__, ggml_backend_dev_description(dev), ggml_backend_name(backend),
        only_meta ? "yes" : "no", roundtrip_read_mode_name(read_mode));

    int npass = 0;
    int ntest = 0;

    struct gguf_context * gguf_ctx_0;
    struct ggml_context * ctx_0;
    ggml_backend_buffer_t bbuf;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed);
        gguf_ctx_0 = result.gguf_ctx;
        ctx_0      = result.ctx;
        bbuf       = result.buffer;
    }

    FILE * file = tmpfile();

#ifdef _WIN32
    if (!file) {
        printf("failed to create tmpfile(), needs elevated privileges on Windows");
        printf("skipping tests");
        return std::make_pair(0, 0);
    }
#else
    GGML_ASSERT(file);
#endif // _WIN32

    // not a multiple of any alignment, so the data section padding must be relative to the GGUF start
    const long prefix = read_mode == ROUNDTRIP_READ_MODE_FILE_OFFSET ? 7 : 0;
    for (long i = 0; i < prefix; ++i) {
        fputc(0xAB, file);
    }

    gguf_write_to_file_ptr(gguf_ctx_0, file, only_meta);
    rewind(file);

    struct ggml_context * ctx_1 = nullptr;
    struct gguf_init_params gguf_params = {
        /*no_alloc =*/ false,
        /*ctx      =*/ only_meta ? nullptr : &ctx_1,
    };
    struct gguf_context * gguf_ctx_1 = nullptr;
    const std::vector<uint8_t> data = read_mode == ROUNDTRIP_READ_MODE_FILE
        ? std::vector<uint8_t>()
        : read_file_to_buffer(file);

    if (read_mode == ROUNDTRIP_READ_MODE_BUFFER) {
        gguf_ctx_1 = gguf_init_from_buffer(data.data(), data.size(), gguf_params);
    } else if (read_mode == ROUNDTRIP_READ_MODE_CALLBACK) {
        callback_reader_data reader = {
            /*.data = */ data.data(),
            /*.size = */ data.size(),
        };
        gguf_ctx_1 = gguf_init_from_callback(read_buffer_callback, &reader, 4096, 4ull << 30 /* 4GB */, gguf_params);
    } else {
        GGML_ASSERT(fseek(file, prefix, SEEK_SET) == 0);
        gguf_ctx_1 = gguf_init_from_file_ptr(file, gguf_params);
    }

    printf("%s: same_version: ", __func__);
    if (gguf_get_version(gguf_ctx_0) == gguf_get_version(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: same_n_kv: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: same_n_tensors: ", __func__);
    if (gguf_get_n_tensors(gguf_ctx_0) == gguf_get_n_tensors(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_orig_kv_in_read: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_read_kv_in_orig: ", __func__);
    if (all_kv_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_orig_tensors_in_read: ", __func__);
    if (all_tensors_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_read_tensors_in_orig: ", __func__);
    if (all_tensors_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    if (!only_meta) {
        printf("%s: same_tensor_data: ", __func__);
        if (same_tensor_data(ctx_0, ctx_1)) {
            printf("\033[1;32mOK\033[0m\n");
            npass++;
        } else {
            printf("\033[1;31mFAIL\033[0m\n");
        }
        ntest++;
    }

    ggml_backend_buffer_free(bbuf);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    gguf_free(gguf_ctx_0);
    gguf_free(gguf_ctx_1);
    ggml_backend_free(backend);
    fclose(file);

    printf("\n");
    return std::make_pair(npass, ntest);
}

static std::pair<int, int> test_gguf_set_kv(ggml_backend_dev_t dev, const unsigned int seed) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    printf("%s: device=%s, backend=%s\n", __func__, ggml_backend_dev_description(dev), ggml_backend_name(backend));

    int npass = 0;
    int ntest = 0;

    struct gguf_context * gguf_ctx_0;
    struct ggml_context * ctx_0;
    ggml_backend_buffer_t bbuf_0;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed);
        gguf_ctx_0 = result.gguf_ctx;
        ctx_0      = result.ctx;
        bbuf_0     = result.buffer;
    }

    struct gguf_context * gguf_ctx_1;
    struct ggml_context * ctx_1;
    ggml_backend_buffer_t bbuf_1;
    {
        struct random_gguf_context_result result = get_random_gguf_context(backend, seed + 1);
        gguf_ctx_1 = result.gguf_ctx;
        ctx_1      = result.ctx;
        bbuf_1     = result.buffer;
    }

    struct gguf_context * gguf_ctx_2 = gguf_init_empty();

    gguf_set_kv(gguf_ctx_1, gguf_ctx_0);
    gguf_set_kv(gguf_ctx_2, gguf_ctx_0);

    printf("%s: same_n_kv: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_2)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_0_in_1: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_0_in_2: ", __func__);
    if (all_kv_in_other(gguf_ctx_0, gguf_ctx_2)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    gguf_set_kv(gguf_ctx_0, gguf_ctx_1);

    printf("%s: same_n_kv_after_double_copy: ", __func__);
    if (gguf_get_n_kv(gguf_ctx_0) == gguf_get_n_kv(gguf_ctx_1)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    printf("%s: all_kv_1_in_0_after_double_copy: ", __func__);
    if (all_kv_in_other(gguf_ctx_1, gguf_ctx_0)) {
        printf("\033[1;32mOK\033[0m\n");
        npass++;
    } else {
        printf("\033[1;31mFAIL\033[0m\n");
    }
    ntest++;

    ggml_backend_buffer_free(bbuf_0);
    ggml_backend_buffer_free(bbuf_1);
    ggml_free(ctx_0);
    ggml_free(ctx_1);
    gguf_free(gguf_ctx_0);
    gguf_free(gguf_ctx_1);
    gguf_free(gguf_ctx_2);
    ggml_backend_free(backend);

    printf("\n");
    return std::make_pair(npass, ntest);
}


// Expert descriptors and copying must remain independent of model allocation and weight residency.
static std::pair<int, int> test_expert_parts() {
    int pass = 0, total = 0;
    auto check = [&](bool value, const char * name) {
        ++total;
        if (value) ++pass;
        else fprintf(stderr, "expert paging: %s failed\n", name);
    };
    for (bool fused : {false, true}) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
            ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
            gguf_context * meta = gguf_init_empty();
            gguf_set_val_str(meta, "general.architecture", "qwen3next");
            gguf_set_val_u32(meta, "qwen3next.expert_count", 3);
            std::vector<const char *> names = fused
                ? std::vector<const char *>{"blk.0.ffn_gate_up_exps.weight", "blk.0.ffn_down_exps.weight"}
                : std::vector<const char *>{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
            size_t bytes = 0;
            for (const auto * name : names) {
                auto * tensor = ggml_new_tensor_3d(ctx, type, 256, 32, 3);
                ggml_set_name(tensor, name);
                gguf_add_tensor(meta, tensor);
                bytes += GGML_PAD(ggml_nbytes(tensor), 32);
            }
            const size_t header = gguf_get_meta_size(meta);
            std::vector<uint8_t> source(header+bytes, 0);
            gguf_get_meta_data(meta, source.data());
            std::vector<llama_model_tensor_part> parts(names.size());
            check(llama_model_describe_expert_parts(source.data(), source.size(), nullptr, 0) == (int)names.size(), "bank count");
            check(llama_model_describe_expert_parts(source.data(), source.size(), parts.data(), parts.size()) == (int)names.size(), "bank metadata");
            check(llama_model_describe_expert_parts(source.data(), source.size()-1, nullptr, 0) == -1, "truncated bank");
            for (size_t i=0; i<parts.size(); ++i) {
                const auto & part = parts[i];
                check(std::string(part.name) == names[i] && part.ne[2] == 3 && part.type == type, "preserved layout");
                std::vector<uint8_t> dest(part.size, 0x5a);
                check(!llama_model_copy_expert_part(source.data(), source.size(), &part, dest.data(), dest.size(),
                    [](float, void *) { return false; }, nullptr), "cancel before payload read");
                check(dest.front() == 0x5a && dest.back() == 0x5a, "cancellation leaves destination untouched");
                check(llama_model_copy_expert_part(source.data(), source.size(), &part, dest.data(), dest.size(), nullptr, nullptr), "copy bank");
                check(std::memcmp(dest.data(), source.data()+part.offset, part.size) == 0, "exact quantized copy");
                check(!llama_model_copy_expert_part(source.data(), source.size(), &part, dest.data(), dest.size()-1, nullptr, nullptr), "short destination");
                check(!llama_model_copy_expert_part(source.data(), source.size(), &part, source.data()+part.offset, part.size, nullptr, nullptr), "overlapping destination");
                auto invalid = part;
                ++invalid.ne[1];
                check(!llama_model_copy_expert_part(source.data(), source.size(), &invalid, dest.data(), dest.size(), nullptr, nullptr), "changed dimensions");
                invalid = part;
                invalid.offset = UINT64_MAX;
                check(!llama_model_copy_expert_part(source.data(), source.size(), &invalid, dest.data(), dest.size(), nullptr, nullptr), "invalid source range");
            }
            std::vector<llama_model_tensor_view> views;
            for (const auto & part : parts) views.push_back({part.name, source.data()+part.offset, (size_t)part.size});
            auto params = llama_model_default_params();
            check(llama_model_load_from_expert_views(source.data(), source.size(), views.data(), views.size()-1, params) == nullptr, "missing view");
            auto saved = views[0];
            views[0].data = nullptr;
            check(llama_model_load_from_expert_views(source.data(), source.size(), views.data(), views.size(), params) == nullptr, "null view");
            views[0] = saved;
            --views[0].size;
            check(llama_model_load_from_expert_views(source.data(), source.size(), views.data(), views.size(), params) == nullptr, "wrong view size");
            views[0] = saved;
            views[1].name = views[0].name;
            check(llama_model_load_from_expert_views(source.data(), source.size(), views.data(), views.size(), params) == nullptr, "duplicate view");
            gguf_free(meta);
            ggml_free(ctx);
        }
    }

    {
        ggml_context * ctx = ggml_init({1024*1024, nullptr, true});
        gguf_context * meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "qwen3next");
        gguf_set_val_u32(meta, "qwen3next.expert_count", 2);
        auto * tensor = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1024, 1024, 2);
        ggml_set_name(tensor, "blk.0.ffn_up_exps.weight");
        gguf_add_tensor(meta, tensor);
        const size_t header = gguf_get_meta_size(meta);
        std::vector<uint8_t> source(header+ggml_nbytes(tensor), 0);
        gguf_get_meta_data(meta, source.data());
        llama_model_tensor_part part;
        check(llama_model_describe_expert_parts(source.data(), source.size(), &part, 1) == 1, "large bank metadata");
        std::vector<uint8_t> dest(part.size, 0x5a);
        int callbacks = 0;
        check(!llama_model_copy_expert_part(source.data(), source.size(), &part, dest.data(), dest.size(),
            [](float, void * value) { return ++*static_cast<int *>(value) == 1; }, &callbacks), "cancel between chunks");
        check(callbacks == 2 && dest.front() == 0 && dest.back() == 0x5a, "bounded partial copy");
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(source.data()+part.offset, &invalid, sizeof(invalid));
        check(!llama_model_copy_expert_part(source.data(), source.size(), &part, dest.data(), dest.size(), nullptr, nullptr), "reject invalid expert payload");
        gguf_free(meta);
        ggml_free(ctx);
    }

    // MUL_MAT_ID must touch only the routed expert. Other complete expert ranges
    // are inaccessible, which catches eager reads instead of relying on RSS measurements.
#if defined(_WIN32)
    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);
    const size_t expert_bytes = system_info.dwPageSize;
#else
    const size_t expert_bytes = static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    const size_t expert_elements = expert_bytes / sizeof(float);
    const int64_t width = static_cast<int64_t>(expert_elements / 4);
#if defined(_WIN32)
    auto * bank = static_cast<float *>(VirtualAlloc(nullptr, 3*expert_bytes, MEM_RESERVE|MEM_COMMIT, PAGE_READWRITE));
#else
    auto * bank = static_cast<float *>(mmap(nullptr, 3*expert_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
    if (bank == MAP_FAILED) bank = nullptr;
#endif
    check(bank != nullptr, "protected bank allocation");
    if (bank != nullptr) {
        std::fill(bank+expert_elements, bank+2*expert_elements, 1.0f);
#if defined(_WIN32)
        DWORD old;
        check(VirtualProtect(bank, expert_bytes, PAGE_NOACCESS, &old) != 0, "protect first expert");
        check(VirtualProtect(bank+2*expert_elements, expert_bytes, PAGE_NOACCESS, &old) != 0, "protect last expert");
#else
        check(mprotect(bank, expert_bytes, PROT_NONE) == 0, "protect first expert");
        check(mprotect(bank+2*expert_elements, expert_bytes, PROT_NONE) == 0, "protect last expert");
#endif
        ggml_context * ctx = ggml_init({4*1024*1024, nullptr, false});
        auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 4, 3);
        weights->data = bank;
        auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
        std::fill(static_cast<float *>(input->data), static_cast<float *>(input->data)+width, 1.0f);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
        *static_cast<int32_t *>(ids->data) = 1;
        auto * output = ggml_mul_mat_id(ctx, weights, input, ids);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        check(ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS, "selected expert compute");
        check(static_cast<float *>(output->data)[0] == static_cast<float>(width), "selected expert result");
        ggml_set_param(input);
        auto * loss = ggml_sum(ctx, output);
        ggml_set_loss(loss);
        auto * backward = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(backward, loss);
        ggml_build_backward_expand(ctx, backward, nullptr);
        ggml_graph_reset(backward);
        check(ggml_graph_compute_with_ctx(ctx, backward, 2) == GGML_STATUS_SUCCESS, "selected expert input gradient");
        auto * dx = ggml_graph_get_grad(backward, input);
        check(static_cast<float *>(dx->data)[0] == 4.0f, "selected expert gradient value");
        check(ggml_graph_get_grad(backward, weights) == nullptr, "no frozen expert gradient");
        check(bank[expert_elements] == 1.0f, "base remains immutable");
        ggml_free(ctx);
#if defined(_WIN32)
        VirtualFree(bank, 0, MEM_RELEASE);
#else
        munmap(bank, 3*expert_bytes);
#endif
    }
    fprintf(stdout, "expert paging: %d/%d passed\n", pass, total);
    return {pass, total};
}

static std::pair<int, int> test_quantized_expert_gradients() {
    int pass = 0, total = 0;
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
      for (bool routed : {false, true}) {
        auto * ctx = ggml_init({4*1024*1024, nullptr, false});
        auto * weights = ggml_new_tensor_3d(ctx, type, 256, 3, 4);
        std::vector<float> source(256*3*4);
        for (size_t i = 0; i < source.size(); ++i) source[i] = sinf((float) i * 0.07f);
        ggml_quantize_chunk(type, source.data(), weights->data, 0, 12, 256, nullptr);
        std::vector<uint8_t> original(ggml_nbytes(weights));
        memcpy(original.data(), weights->data, original.size());
        auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 1, 2);
        ggml_set_f32(input, 1.0f);
        ggml_set_param(input);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, 2);
        int32_t routes[] = {1, 3, 3, 0};
        memcpy(ids->data, routes, sizeof(routes));
        auto * product = routed ? ggml_mul_mat_id(ctx, weights, input, ids) :
            ggml_mul_mat(ctx, ggml_view_2d(ctx, weights, 256, 3, weights->nb[1], 0), ggml_reshape_2d(ctx, input, 256, 2));
        auto * loss = ggml_sum(ctx, product);
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
        auto * dx = ggml_graph_get_grad(graph, input);
        float row[256];
        for (int token = 0; token < 2; ++token) {
            std::vector<float> expected(256, 0.0f);
            for (int route = 0; route < (routed ? 2 : 1); ++route) {
                for (int channel = 0; channel < 3; ++channel) {
                    const char * data = (const char *) weights->data + (routed ? routes[token*2+route] : 0)*weights->nb[2] + channel*weights->nb[1];
                    if (type == GGML_TYPE_F32) memcpy(row, data, sizeof(row));
                    else ggml_get_type_traits(type)->to_float(data, row, 256);
                    for (int j = 0; j < 256; ++j) expected[j] += row[j];
                }
            }
            for (int j = 0; j < 256; ++j) ok &= fabsf(((float *) dx->data)[token*256+j] - expected[j]) < 1e-5f;
        }
        ok &= memcmp(original.data(), weights->data, original.size()) == 0;
        ok &= ggml_graph_get_grad(graph, weights) == nullptr;
        ++total;
        pass += ok;
        fprintf(stdout, "frozen %s gradients %s: %s\n", routed ? "expert" : "matrix", ggml_type_name(type), ok ? "OK" : "FAIL");
        ggml_free(ctx);
      }
    }
    return {pass, total};
}

static std::pair<int, int> test_sparse_loss() {
    int pass = 0, total = 0;
    for (int mode = 0; mode < 3; ++mode) {
        auto * ctx = ggml_init({4*1024*1024, nullptr, false});
        auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3);
        auto * labels = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        float values[] = {0.2f, -0.7f, 1.1f, 0.4f, -0.2f, 9, -4, 3, 6, 8, -0.1f, 0.5f, 0.3f, -0.4f, 0.2f};
        int32_t ids[] = {2, -1, 0};
        if (mode == 1) std::fill(ids, ids+3, -1);
        if (mode == 2) for (float & value : values) value = 10000.0f;
        memcpy(logits->data, values, sizeof(values));
        memcpy(labels->data, ids, sizeof(ids));
        ggml_set_param(logits);
        auto * loss = ggml_cross_entropy_loss_sparse(ctx, logits, labels);
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS;
        auto * dx = ggml_graph_get_grad(graph, logits);
        float gradient[15]; memcpy(gradient, dx->data, sizeof(gradient));
        const float original_loss = *(float *) loss->data;
        if (mode == 1) ok &= original_loss == 0;
        if (mode == 2) ok &= fabsf(original_loss - logf(5.0f)) < 1e-6f;
        for (int i = 0; i < 15; ++i) {
            if (ids[i/5] == -1) ok &= gradient[i] == 0;
            const float eps = mode == 2 ? 0.125f : 0.01f;
            ((float *) logits->data)[i] = values[i]+eps;
            ggml_graph_compute_with_ctx(ctx, graph, 3);
            const float plus = *(float *) loss->data;
            ((float *) logits->data)[i] = values[i]-eps;
            ggml_graph_compute_with_ctx(ctx, graph, 3);
            const float minus = *(float *) loss->data;
            ((float *) logits->data)[i] = values[i];
            ok &= fabsf((plus-minus)/(2*eps) - gradient[i]) < 3e-4f;
        }
        if (mode == 0) {
            for (int i = 0; i < 15; ++i) ((float *) logits->data)[i] -= 0.1f*gradient[i];
            ggml_graph_compute_with_ctx(ctx, graph, 3);
            ok &= *(float *) loss->data < original_loss;
        }
        ++total; pass += ok;
        fprintf(stdout, "sparse masked loss mode %d: %s\n", mode, ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    return {pass, total};
}

static std::pair<int, int> test_routing_gradients() {
    int pass = 0, total = 0;
    for (const auto bounds : {std::pair<float, float>{-1.0f, 1.0f}, {-INFINITY, 1.0f}, {-1.0f, INFINITY}}) {
        auto * ctx = ggml_init({1024*1024, nullptr, false});
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
        const float values[] = {-2, -1, -0.5f, 0, 0.5f, 1, 2};
        memcpy(input->data, values, sizeof(values));
        ggml_set_param(input);
        auto * loss = ggml_sum(ctx, ggml_clamp(ctx, input, bounds.first, bounds.second));
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
        const auto * dx = (const float *) ggml_graph_get_grad(graph, input)->data;
        for (int i = 0; i < 7; ++i) ok &= dx[i] == (values[i] > bounds.first && values[i] < bounds.second ? 1.0f : 0.0f);
        ++total; pass += ok;
        fprintf(stdout, "clamp gradients [%g,%g]: %s\n", bounds.first, bounds.second, ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    {
        auto * ctx = ggml_init({4*1024*1024, nullptr, false});
        auto * owner = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 7, 2, 2);
        for (int64_t i = 0; i < ggml_nelements(owner); ++i) ((float *) owner->data)[i] = sinf((float) i);
        ggml_set_param(owner);
        auto * input = ggml_view_4d(ctx, owner, 3, 5, 2, 2, owner->nb[1], owner->nb[2], owner->nb[3], owner->nb[0] + owner->nb[1]);
        auto * ids_owner = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 6, 2, 2);
        auto * ids = ggml_view_3d(ctx, ids_owner, 4, 2, 2, ids_owner->nb[1], ids_owner->nb[2], sizeof(int32_t));
        const int32_t selected[] = {0, 2, 2, 4};
        for (int batch = 0; batch < 4; ++batch) {
            for (int j = 0; j < 4; ++j) ((int32_t *) ids_owner->data)[batch*6 + j + 1] = (selected[j] + batch) % 5;
        }
        auto * loss = ggml_sum(ctx, ggml_sqr(ctx, ggml_get_rows(ctx, input, ids)));
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
        auto * dx = ggml_graph_get_grad(graph, owner);
        std::vector<float> reference(ggml_nelements(owner), 0.0f);
        for (int batch = 0; batch < 4; ++batch) for (int j = 0; j < 4; ++j) for (int x = 0; x < 3; ++x) {
            const int index = batch*28 + ((selected[j] + batch) % 5 + 1)*4 + x + 1;
            reference[index] += 2.0f*((float *) owner->data)[index];
        }
        for (size_t i = 0; i < reference.size(); ++i) ok &= reference[i] == ((float *) dx->data)[i];
        ok &= ggml_graph_compute_with_ctx(ctx, graph, 4) == GGML_STATUS_SUCCESS;
        for (size_t i = 0; i < reference.size(); ++i) ok &= reference[i] == ((float *) dx->data)[i];
        ok &= ggml_graph_get_grad(graph, ids_owner) == nullptr;
        ++total; pass += ok;
        fprintf(stdout, "batched gather gradients with strided views and duplicate indices: %s\n", ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    {
        auto * ctx = ggml_init({8*1024*1024, nullptr, false});
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
        auto * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
        auto * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 4);
        auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 4);
        auto * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 6, 4);
        auto * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 6, 4);
        auto * down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 6, 4, 4);
        int offset = 0;
        for (auto * tensor : {input, a, b, up, gate, down}) {
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) ((float *) tensor->data)[i] = 0.5f*sinf((float) (++offset)*0.37f);
        }
        for (int64_t i = 0; i < ggml_nelements(input); ++i) ((float *) input->data)[i] += 0.75f;
        for (int i = 0; i < 16; ++i) ((float *) router->data)[i] = 0.2f*(i/4 - 1) + 0.03f*(i%4);
        ggml_set_param(a);
        ggml_set_param(b);
        std::vector<std::vector<uint8_t>> frozen;
        for (auto * tensor : {router, up, gate, down}) {
            frozen.emplace_back(ggml_nbytes(tensor));
            memcpy(frozen.back().data(), tensor->data, frozen.back().size());
        }
        auto * adapted = ggml_add(ctx, input, ggml_mul_mat(ctx, b, ggml_mul_mat(ctx, a, input)));
        auto * logits = ggml_mul_mat(ctx, router, adapted);
        auto * probabilities = ggml_soft_max(ctx, logits);
        auto * ids = ggml_argsort_top_k(ctx, probabilities, 2);
        auto * selected = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1, 4, 3), ids);
        selected = ggml_reshape_2d(ctx, selected, 2, 3);
        auto * denominator = ggml_clamp(ctx, ggml_sum_rows(ctx, selected), 6.103515625e-5f, INFINITY);
        auto * weights = ggml_reshape_3d(ctx, ggml_div(ctx, selected, denominator), 1, 2, 3);
        auto * x = ggml_reshape_3d(ctx, adapted, 4, 1, 3);
        auto * hidden = ggml_swiglu_split(ctx, ggml_mul_mat_id(ctx, gate, x, ids), ggml_mul_mat_id(ctx, up, x, ids));
        auto * experts = ggml_mul(ctx, ggml_mul_mat_id(ctx, down, hidden, ids), weights);
        auto * output = ggml_add(ctx,
            ggml_view_2d(ctx, experts, 4, 3, experts->nb[2], 0),
            ggml_view_2d(ctx, experts, 4, 3, experts->nb[2], experts->nb[1]));
        auto * loss = ggml_sum(ctx, ggml_sqr(ctx, output));
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS;
        float router_gradient = 0;
        auto * dlogits = ggml_graph_get_grad(graph, logits);
        for (int64_t i = 0; i < ggml_nelements(dlogits); ++i) router_gradient += fabsf(((float *) dlogits->data)[i]);
        ok &= router_gradient > 1e-5f;
        if (router_gradient <= 1e-5f) fprintf(stderr, "routing gradient too small: %g\n", router_gradient);
        std::vector<std::vector<float>> gradients;
        for (auto * param : {a, b}) {
            gradients.emplace_back(ggml_nelements(param));
            memcpy(gradients.back().data(), ggml_graph_get_grad(graph, param)->data, ggml_nbytes(param));
        }
        // Independent double-precision evaluation avoids cancellation in F32 finite differences.
        auto reference_loss = [&](ggml_tensor * parameter, int64_t element, double perturbation) {
            auto value = [&](ggml_tensor * t, int64_t i) {
                return (double) ((float *) t->data)[i] + (t == parameter && i == element ? perturbation : 0.0);
            };
            double result = 0;
            for (int token = 0; token < 3; ++token) {
                double low_rank[2] = {}, adapted_ref[4] = {}, probabilities_ref[4] = {};
                for (int r = 0; r < 2; ++r) for (int c = 0; c < 4; ++c) low_rank[r] += value(a, r*4+c)*value(input, token*4+c);
                for (int c = 0; c < 4; ++c) {
                    adapted_ref[c] = value(input, token*4+c);
                    for (int r = 0; r < 2; ++r) adapted_ref[c] += value(b, c*2+r)*low_rank[r];
                }
                double sum = 0;
                for (int e = 0; e < 4; ++e) {
                    double score = 0;
                    for (int c = 0; c < 4; ++c) score += value(router, e*4+c)*adapted_ref[c];
                    probabilities_ref[e] = exp(score);
                    sum += probabilities_ref[e];
                }
                for (double & p : probabilities_ref) p /= sum;
                int routes[] = {0, 1, 2, 3};
                std::sort(routes, routes+4, [&](int l, int r) { return probabilities_ref[l] > probabilities_ref[r]; });
                const double normalizer = std::max(probabilities_ref[routes[0]] + probabilities_ref[routes[1]], 6.103515625e-5);
                double output_ref[4] = {};
                for (int j = 0; j < 2; ++j) {
                    const int e = routes[j];
                    for (int f = 0; f < 6; ++f) {
                        double up_ref = 0, gate_ref = 0;
                        for (int c = 0; c < 4; ++c) {
                            up_ref += value(up, e*24+f*4+c)*adapted_ref[c];
                            gate_ref += value(gate, e*24+f*4+c)*adapted_ref[c];
                        }
                        const double hidden_ref = up_ref*gate_ref/(1.0+exp(-gate_ref));
                        for (int c = 0; c < 4; ++c) output_ref[c] += value(down, e*24+c*6+f)*hidden_ref*probabilities_ref[e]/normalizer;
                    }
                }
                for (double v : output_ref) result += v*v;
            }
            return result;
        };
        ok &= fabs(reference_loss(nullptr, 0, 0) - *(float *) loss->data) < 1e-6;
        int parameter = 0;
        for (auto * param : {a, b}) {
            for (int64_t i = 0; i < ggml_nelements(param); ++i) {
                const double eps = 1e-4;
                const double numerical = (reference_loss(param, i, eps) - reference_loss(param, i, -eps))/(2*eps);
                const float analytical = gradients[parameter][i];
                if (fabs(numerical - analytical) >= 1e-6) {
                    fprintf(stderr, "routing derivative parameter=%d element=%zu: numerical=%g analytical=%g\n", parameter, (size_t) i, numerical, analytical);
                    ok = false;
                }
            }
            ++parameter;
        }
        int tensor_index = 0;
        for (auto * tensor : {router, up, gate, down}) {
            ok &= ggml_graph_get_grad(graph, tensor) == nullptr;
            ok &= memcmp(frozen[tensor_index].data(), tensor->data, ggml_nbytes(tensor)) == 0;
            ++tensor_index;
        }
        ok &= ggml_graph_get_grad(graph, ids) == nullptr;
        ++total; pass += ok;
        fprintf(stdout, "adapter gradients through frozen experts and normalized routing probabilities: %s\n", ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    return {pass, total};
}

static std::pair<int, int> test_attention_training() {
    int pass = 0, total = 0;
    for (int32_t n_past : {-1, 0, 2}) {
        auto * ctx = ggml_init({8*1024*1024, nullptr, false});
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 3, 4, 2);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 5, 2, 2);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, 5, 2, 2);
        for (auto * input : {q, k, v}) {
            for (int64_t i = 0; i < ggml_nelements(input); ++i) ((float *) input->data)[i] = sinf((float) i + input->ne[0]);
            ggml_set_param(input);
        }
        auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 5, 3);
        for (int query = 0; query < 3; ++query) {
            for (int key = 0; key < 5; ++key) ((float *) mask->data)[query*5 + key] = n_past < 0 || key <= query + n_past ? 0.0f : -INFINITY;
        }
        auto expand_heads = [&](ggml_tensor * input) {
            auto * grouped = ggml_reshape_4d(ctx, input, input->ne[0], input->ne[1], 1, input->ne[2]*input->ne[3]);
            auto * repeated = ggml_repeat_4d(ctx, grouped, input->ne[0], input->ne[1], q->ne[2]/input->ne[2], grouped->ne[3]);
            return ggml_reshape_4d(ctx, repeated, input->ne[0], input->ne[1], q->ne[2], q->ne[3]);
        };
        auto * scores = ggml_mul_mat(ctx, expand_heads(k), q);
        auto * probabilities = ggml_soft_max_ext(ctx, scores, mask, 0.5f, 0.0f);
        auto * dense = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, expand_heads(v))), probabilities);
        dense = ggml_cont(ctx, ggml_permute(ctx, dense, 0, 2, 1, 3));
        auto * bounded = ggml_attention_training(ctx, q, k, v, 0.5f, n_past);
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, dense);
        ggml_build_forward_expand(graph, bounded);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 1) == GGML_STATUS_SUCCESS;
        std::vector<float> single(ggml_nelements(bounded));
        memcpy(single.data(), bounded->data, ggml_nbytes(bounded));
        for (size_t i = 0; i < single.size(); ++i) ok &= fabsf(single[i] - ((float *) dense->data)[i]) < 1e-6f;
        ok &= ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS;
        ok &= memcmp(single.data(), bounded->data, ggml_nbytes(bounded)) == 0;
        auto backward = [&](ggml_tensor * output) {
            auto * loss = ggml_sum(ctx, ggml_sqr(ctx, output));
            ggml_set_loss(loss);
            auto * back = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
            ggml_build_forward_expand(back, loss);
            ggml_build_backward_expand(ctx, back, nullptr);
            ggml_graph_reset(back);
            ok &= ggml_graph_compute_with_ctx(ctx, back, 3) == GGML_STATUS_SUCCESS;
            return back;
        };
        auto * dense_back = backward(dense);
        auto * bounded_back = backward(bounded);
        for (auto * input : {q, k, v}) {
            const auto * reference = (const float *) ggml_graph_get_grad(dense_back, input)->data;
            const auto * actual = (const float *) ggml_graph_get_grad(bounded_back, input)->data;
            for (int64_t i = 0; i < ggml_nelements(input); ++i) ok &= fabsf(reference[i] - actual[i]) < 2e-6f;
        }
        ++total; pass += ok;
        fprintf(stdout, "training attention dense reference n_past=%d: %s\n", n_past, ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    {
        // A dense two-head score tensor alone needs 32 MiB at this sequence length.
        const int64_t tokens = 2048;
        auto * ctx = ggml_init({8*1024*1024, nullptr, false});
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, tokens, 2, 1);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, tokens, 1, 1);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 3, tokens, 1, 1);
        for (auto * input : {q, k, v}) {
            ggml_set_f32(input, 0.25f);
            ggml_set_param(input);
        }
        auto * output = ggml_attention_training(ctx, q, k, v, 0.5f, 0);
        auto * loss = ggml_sum(ctx, output);
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 3) == GGML_STATUS_SUCCESS;
        ok &= *(float *) loss->data == (float) (tokens*2*3)*0.25f;
        const auto * dv = (const float *) ggml_graph_get_grad(graph, v)->data;
        double first = 0.0;
        for (int64_t i = 1; i <= tokens; ++i) first += 2.0/i;
        ok &= fabs(dv[0] - first) < 1e-4;
        ok &= fabsf(dv[3*(tokens - 1)] - 2.0f/tokens) < 1e-6f;
        for (auto * input : {q, k}) {
            const auto * gradient = (const float *) ggml_graph_get_grad(graph, input)->data;
            for (int64_t i = 0; i < ggml_nelements(input); ++i) ok &= fabsf(gradient[i]) < 1e-6f;
        }
        ++total; pass += ok;
        fprintf(stdout, "training attention 2048 tokens with 8 MiB arena: %s\n", ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    return {pass, total};
}

static std::pair<int, int> test_l2_norm_gradients() {
    int pass = 0;
    for (bool clamped : {false, true}) {
        auto * ctx = ggml_init({1024*1024, nullptr, false});
        auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
        ((float *) input->data)[0] = clamped ? 0.0f : 3.0f;
        ((float *) input->data)[1] = clamped ? 0.0f : 4.0f;
        ggml_set_param(input);
        auto * loss = ggml_sum(ctx, ggml_l2_norm(ctx, input, 0.1f));
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
        const auto * derivative = (const float *) ggml_graph_get_grad(graph, input)->data;
        ok &= fabsf(derivative[0] - (clamped ? 10.0f : 0.032f)) < 1e-6f;
        ok &= fabsf(derivative[1] - (clamped ? 10.0f : -0.024f)) < 1e-6f;
        pass += ok;
        fprintf(stdout, "L2 gradient clamped=%d: %s\n", clamped, ok ? "OK" : "FAIL");
        ggml_free(ctx);
    }
    return {pass, 2};
}

static std::pair<int, int> test_concat_gradients() {
    int pass = 0;
    for (int axis = 0; axis < 4; ++axis) {
        auto * ctx = ggml_init({1024*1024, nullptr, false});
        int64_t shape[] = {2, 3, 2, 2};
        auto * a = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, shape);
        shape[axis] += 1;
        auto * b = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, shape);
        for (auto * p : {a, b}) {
            ggml_set_param(p);
            for (int64_t i = 0; i < ggml_nelements(p); ++i) ((float *) p->data)[i] = sinf(i*0.17f);
        }
        auto * joined = ggml_cont(ctx, ggml_transpose(ctx, ggml_concat(ctx, a, b, axis)));
        auto * loss = ggml_add(ctx, ggml_sum(ctx, ggml_sqr(ctx, joined)), ggml_sum(ctx, a));
        ggml_set_loss(loss);
        auto * graph = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
        ggml_build_forward_expand(graph, loss);
        ggml_build_backward_expand(ctx, graph, nullptr);
        ggml_graph_reset(graph);
        bool ok = ggml_graph_compute_with_ctx(ctx, graph, 2) == GGML_STATUS_SUCCESS;
        for (auto * p : {a, b}) {
            auto * gradient = ggml_graph_get_grad(graph, p);
            for (int64_t i = 0; i < ggml_nelements(p); ++i) ok &= fabsf(((float *) gradient->data)[i] -
                (2*((float *) p->data)[i] + (p == a ? 1 : 0))) < 1e-6f;
        }
        fprintf(stdout, "concat gradient axis %d with strided incoming gradient: %s\n", axis, ok ? "OK" : "FAIL");
        pass += ok;
        ggml_free(ctx);
    }
    return {pass, 4};
}

static std::pair<int, int> test_qwen_training() {
    int pass = 0, total = 0;
    auto check = [&](bool value, const char * label) {
        ++total; pass += value;
        fprintf(stdout, "Qwen training %s: %s\n", label, value ? "OK" : "FAIL");
    };
    for (bool mixed : {false, true}) {
        const int embd = mixed ? 256 : 32;
        auto * ctx = ggml_init({1024*1024, nullptr, true});
        auto * meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "qwen3next");
        gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
        for (auto setting : {std::pair<const char *, uint32_t>{"vocab_size", 8}, {"context_length", 64},
                {"embedding_length", (uint32_t) embd}, {"block_count", 3}, {"feed_forward_length", (uint32_t) embd},
                {"expert_count", 4}, {"expert_used_count", 2}, {"expert_feed_forward_length", (uint32_t) embd},
                {"expert_shared_feed_forward_length", (uint32_t) embd}, {"attention.head_count", 2}, {"attention.head_count_kv", 1},
                {"attention.key_length", 16}, {"attention.value_length", 16}, {"rope.dimension_count", 16},
                {"ssm.conv_kernel", 4}, {"ssm.inner_size", 16}, {"ssm.state_size", 8}, {"ssm.time_step_rank", 2}, {"ssm.group_count", 1}}) {
            gguf_set_val_u32(meta, (std::string("qwen3next.") + setting.first).c_str(), setting.second);
        }
        const bool recurrent[] = {false, true, false};
        gguf_set_arr_data(meta, "qwen3next.attention.recurrent_layers", GGUF_TYPE_BOOL, recurrent, 3);
        gguf_set_val_f32(meta, "qwen3next.attention.layer_norm_rms_epsilon", 1e-5f);
        auto tensor = [&](const std::string & name, int64_t x, int64_t y = 1, int64_t z = 1) {
            ggml_type type = GGML_TYPE_F32;
            if (mixed && x == 256 && y > 1) {
                const ggml_type formats[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_BF16};
                type = formats[gguf_get_n_tensors(meta) % 5];
            }
            auto * t = ggml_new_tensor_3d(ctx, type, x, y, z);
            ggml_set_name(t, name.c_str());
            gguf_add_tensor(meta, t);
        };
        tensor("token_embd.weight", embd, 8);
        tensor("output.weight", embd, 8);
        tensor("output_norm.weight", embd);
        std::vector<std::string> targets;
        for (int il = 0; il < 3; ++il) {
            const std::string prefix = "blk." + std::to_string(il) + ".";
            tensor(prefix + "attn_norm.weight", embd);
            tensor(prefix + "post_attention_norm.weight", embd);
            if (il == 1) {
                if (mixed) {
                    tensor(prefix + "ssm_in.weight", embd, 48);
                } else {
                    tensor(prefix + "attn_qkv.weight", embd, 32);
                    tensor(prefix + "attn_gate.weight", embd, 16);
                }
                tensor(prefix + "ssm_conv1d.weight", 4, 32);
                tensor(prefix + "ssm_dt.bias", 2);
                tensor(prefix + "ssm_a", 2);
                tensor(prefix + "ssm_ba.weight", embd, 4);
                tensor(prefix + "ssm_norm.weight", 8);
                tensor(prefix + "ssm_out.weight", 16, embd);
                targets.emplace_back(prefix + "ssm_out.weight");
            } else {
                tensor(prefix + "attn_q.weight", embd, 64);
                tensor(prefix + "attn_k.weight", embd, 16);
                tensor(prefix + "attn_v.weight", embd, 16);
                tensor(prefix + "attn_output.weight", 32, embd);
                tensor(prefix + "attn_q_norm.weight", 16);
                tensor(prefix + "attn_k_norm.weight", 16);
                targets.emplace_back(prefix + "attn_output.weight");
            }
            tensor(prefix + "ffn_gate_inp.weight", embd, 4);
            for (const char * name : {"ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight"}) tensor(prefix + name, embd, embd, 4);
            tensor(prefix + "ffn_gate_inp_shexp.weight", embd);
            for (const char * name : {"ffn_up_shexp.weight", "ffn_gate_shexp.weight", "ffn_down_shexp.weight"}) {
                tensor(prefix + name, embd, embd);
                targets.emplace_back(prefix + name);
            }
        }
        const size_t header = gguf_get_meta_size(meta);
        size_t size = header;
        for (int64_t i = 0; i < gguf_get_n_tensors(meta); ++i) size = std::max(size, header + gguf_get_tensor_offset(meta, i) + gguf_get_tensor_size(meta, i));
        std::vector<uint8_t> base(size);
        gguf_get_meta_data(meta, base.data());
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            const std::string name = t->name;
            const int64_t index = gguf_find_tensor(meta, t->name);
            std::vector<float> values(ggml_nelements(t));
            for (size_t i = 0; i < values.size(); ++i) values[i] = name.find("norm") != std::string::npos ? 1.0f :
                name.find("ssm_a") != std::string::npos ? -0.5f : (float) (sin(i*0.17 + index*0.63)*0.6/sqrt((double) t->ne[0]));
            ggml_quantize_chunk(t->type, values.data(), base.data()+header+gguf_get_tensor_offset(meta, index), 0, ggml_nrows(t), t->ne[0], nullptr);
        }
        gguf_free(meta);
        ggml_free(ctx);
        auto params = llama_model_default_params();
        params.n_gpu_layers = 0;
        params.use_extra_bufts = false;
        auto * model = llama_model_load_from_buffer_view(base.data(), base.size(), params);
        check(model != nullptr, mixed ? "mixed-format mapped fixture" : "F32 mapped fixture");
        if (!model) continue;
        const auto original = base;
        std::vector<const char *> names;
        for (const auto & name : targets) names.push_back(name.c_str());
        auto * adapter = llama_adapter_lora_init_training(model, names.data(), names.size(), 2, 4.0f, 20260920);
        check(adapter != nullptr, "initialize all target kinds");
        if (!adapter) { llama_model_free(model); continue; }
        for (const auto & entry : adapter->ab_map) {
            for (int64_t i = 0; i < ggml_nelements(entry.second.b); ++i) ((float *) entry.second.b->data)[i] = 0.01f*sinf((float) i);
        }
        llama_adapter_training_params config = {4, 2, 128*1024*1024};
        char error[256];
        auto * training = llama_adapter_training_init(model, adapter, config, error, sizeof(error));
        check(training != nullptr, "build complete causal graph");
        if (!training) {
            fprintf(stderr, "training init: %s\n", error);
            llama_adapter_lora_free(adapter); llama_model_free(model); continue;
        }
        const size_t workspace = llama_adapter_training_workspace_size(training);
        check(workspace > 0 && workspace <= config.workspace_budget_bytes, "workspace accounting");
        auto too_small = config;
        too_small.workspace_budget_bytes = workspace - 1;
        auto * rejected = llama_adapter_training_init(model, adapter, too_small, error, sizeof(error));
        check(rejected == nullptr && strstr(error, "budget"), "reject insufficient workspace");
        llama_adapter_training_free(rejected);
        const auto architecture = model->arch;
        model->arch = LLM_ARCH_LLAMA;
        rejected = llama_adapter_training_init(model, adapter, config, error, sizeof(error));
        model->arch = architecture;
        check(rejected == nullptr && strstr(error, "Qwen3Next"), "reject unsupported architecture");
        llama_adapter_training_free(rejected);
        const char * unsupported_target = "blk.0.ffn_gate_inp.weight";
        auto * unsupported = llama_adapter_lora_init_training(model, &unsupported_target, 1, 2, 4.0f, 0);
        rejected = llama_adapter_training_init(model, unsupported, config, error, sizeof(error));
        check(unsupported && !rejected && strstr(error, "target"), "reject unsupported adapter target");
        llama_adapter_training_free(rejected);
        llama_adapter_lora_free(unsupported);
        llama_token tokens[] = {1, 3, 2, 4}, labels[] = {3, -1, 4, 5};
        float loss = 0;
        check(llama_adapter_training_compute(training, tokens, labels, &loss, nullptr, nullptr), "full forward and backward");
        if (*llama_adapter_training_error(training)) fprintf(stderr, "%s\n", llama_adapter_training_error(training));
        const float first_loss = loss;
        auto * first = adapter->ab_map.at(targets.front()).a;
        std::vector<float> gradient(ggml_nelements(first));
        check(llama_adapter_training_gradient(training, first->name, gradient.data(), gradient.size()) == gradient.size() &&
            std::any_of(gradient.begin(), gradient.end(), [](float g) { return fabsf(g) > 1e-7f; }), "gradient reaches adapter before recurrent layer");
        if (!mixed) {
            bool close = true;
            for (size_t i : {size_t(0), size_t(11), size_t(31), size_t(57)}) {
                const float value = ((float *) first->data)[i];
                const float eps = 0.01f;
                float plus = NAN, minus = NAN;
                ((float *) first->data)[i] = value + eps;
                close &= llama_adapter_training_compute(training, tokens, labels, &plus, nullptr, nullptr);
                ((float *) first->data)[i] = value - eps;
                close &= llama_adapter_training_compute(training, tokens, labels, &minus, nullptr, nullptr);
                ((float *) first->data)[i] = value;
                close &= std::isfinite(plus) && std::isfinite(minus) && fabsf((plus - minus)/(2*eps) - gradient[i]) < 3e-4f;
            }
            check(close, "full-graph finite differences");
        }
        auto inference_params = llama_context_default_params();
        inference_params.n_ctx = 64; inference_params.n_batch = 4; inference_params.n_ubatch = 4;
        inference_params.n_threads = 2; inference_params.n_threads_batch = 2;
        inference_params.offload_kqv = false; inference_params.op_offload = false;
        inference_params.type_k = GGML_TYPE_F32; inference_params.type_v = GGML_TYPE_F32;
        inference_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto inference_loss = [&](llama_adapter_lora * selected_adapter) {
            auto * inference = llama_init_from_model(model, inference_params);
            if (!inference) return std::numeric_limits<double>::quiet_NaN();
            float scale = 1;
            bool success = llama_set_adapters_lora(inference, &selected_adapter, 1, &scale) == 0;
            auto batch = llama_batch_init(4, 0, 1);
            batch.n_tokens = 4;
            for (int i = 0; i < 4; ++i) {
                batch.token[i] = tokens[i]; batch.pos[i] = i; batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0; batch.logits[i] = 1;
            }
            success &= llama_decode(inference, batch) == 0;
            double expected_loss = 0;
            for (int i = 0; success && i < 4; ++i) if (labels[i] >= 0) {
                const float * logits = llama_get_logits_ith(inference, i);
                const float max = *std::max_element(logits, logits+8);
                double sum = 0;
                for (int j = 0; j < 8; ++j) sum += exp((double) logits[j] - max);
                expected_loss += log(sum) + max - logits[labels[i]];
            }
            llama_batch_free(batch);
            llama_free(inference);
            return success ? expected_loss/3 : std::numeric_limits<double>::quiet_NaN();
        };
        check(fabs(inference_loss(adapter) - first_loss) < 2e-5, "loss matches inference with F32 caches");
        check(!llama_adapter_training_compute(training, tokens, labels, &loss, [](void *) { return true; }, nullptr) &&
            llama_adapter_training_gradient(training, first->name, nullptr, 0) == 0, "cancellation invalidates gradients");
        int abort_count = 0;
        check(!llama_adapter_training_compute(training, tokens, labels, &loss, [](void * count) { return ++*(int *) count >= 4; }, &abort_count) &&
            abort_count >= 4 && llama_adapter_training_gradient(training, first->name, nullptr, 0) == 0, "cancellation during graph execution");
        llama_token invalid_labels[] = {-1, -1, -1, -1};
        check(!llama_adapter_training_compute(training, tokens, invalid_labels, &loss, nullptr, nullptr), "reject entirely masked sequence");
        invalid_labels[0] = 8;
        check(!llama_adapter_training_compute(training, tokens, invalid_labels, &loss, nullptr, nullptr), "reject invalid target token");
        check(llama_adapter_training_compute(training, tokens, labels, &loss, nullptr, nullptr) && loss == first_loss, "repeat starts with fresh recurrent state");
        float guard = 123.0f;
        check(llama_adapter_training_gradient(training, first->name, &guard, 1) == gradient.size() && guard == 123.0f, "undersized gradient destination untouched");
        auto * recomputed = llama_adapter_training_init_checkpointed(model, adapter, config, nullptr, error, sizeof(error));
        check(recomputed != nullptr, "preflight layer recomputation");
        if (!recomputed) fprintf(stderr, "recomputation: %s\n", error);
        auto equivalent = [&](llama_adapter_training * candidate) {
            float candidate_loss = NAN;
            bool ok = candidate && llama_adapter_training_compute(candidate, tokens, labels, &candidate_loss, nullptr, nullptr);
            if (candidate && !ok) fprintf(stderr, "recomputed compute: %s\n", llama_adapter_training_error(candidate));
            ok &= fabsf(candidate_loss-first_loss) < 2e-6f;
            for (const auto & entry : adapter->ab_map) for (auto * p : {entry.second.a, entry.second.b}) {
                std::vector<float> reference(ggml_nelements(p)), actual(reference.size());
                ok &= llama_adapter_training_gradient(training, p->name, reference.data(), reference.size()) == reference.size();
                ok &= llama_adapter_training_gradient(candidate, p->name, actual.data(), actual.size()) == actual.size();
                for (size_t i = 0; i < reference.size(); ++i) ok &= fabsf(reference[i]-actual[i]) < 2e-5f;
            }
            return ok;
        };
        check(equivalent(recomputed), "recomputed loss and every adapter gradient match full graph");
        const size_t recomputed_workspace = llama_adapter_training_workspace_size(recomputed);
        fprintf(stdout, "Qwen %s charged workspace: full=%zu recomputed=%zu bytes\n", mixed ? "mixed" : "F32", workspace, recomputed_workspace);
        check(recomputed_workspace > 0 && recomputed_workspace < workspace, "recomputation reduces charged workspace");
        check(llama_adapter_training_checkpoint_size(training) == 0 &&
            llama_adapter_training_checkpoint_size(recomputed) == 2*embd*4*sizeof(float), "preflight reports layer-boundary payload capacity");
        auto constrained = config;
        constrained.workspace_budget_bytes = recomputed_workspace;
        rejected = llama_adapter_training_init(model, adapter, constrained, error, sizeof(error));
        check(!rejected && strstr(error, "budget"), "full graph rejected under recomputation budget");
        llama_adapter_training_free(rejected);
        auto * bounded = llama_adapter_training_init_checkpointed(model, adapter, constrained, nullptr, error, sizeof(error));
        check(equivalent(bounded), "complete sequence fits recomputation budget");
        llama_adapter_training_free(bounded);
        struct boundary_store {
            std::map<uint32_t, FILE *> files;
            size_t written = 0, read = 0;
            bool fail_write = false, fail_read = false, fail_clear = false;
        } store;
        llama_adapter_checkpoint_store disk = {&store,
            [](void * data, uint32_t boundary, const void * bytes, size_t size) {
                auto & s = *(boundary_store *) data;
                if (s.fail_write) return false;
                FILE * file = tmpfile();
                if (!file) return false;
                s.files.emplace(boundary, file);
                const bool ok = fwrite(bytes, 1, size, file) == size && fflush(file) == 0;
                if (ok) s.written += size;
                return ok;
            },
            [](void * data, uint32_t boundary, void * bytes, size_t size) {
                auto & s = *(boundary_store *) data;
                const auto found = s.files.find(boundary);
                if (s.fail_read || found == s.files.end() || fseek(found->second, 0, SEEK_SET) != 0) return false;
                const bool ok = fread(bytes, 1, size, found->second) == size;
                if (ok) s.read += size;
                return ok;
            },
            [](void * data) {
                auto & s = *(boundary_store *) data;
                const bool had_files = !s.files.empty();
                for (const auto & file : s.files) fclose(file.second);
                s.files.clear();
                return !s.fail_clear || !had_files;
            }};
        auto * spilled = llama_adapter_training_init_checkpointed(model, adapter, config, &disk, error, sizeof(error));
        check(equivalent(spilled) && store.written == 2*embd*4*sizeof(float) && store.read == store.written && store.files.empty(),
            "disk boundaries reproduce gradients and are cleaned up");
        check(llama_adapter_training_workspace_size(spilled) + 2*embd*4*sizeof(float) == recomputed_workspace,
            "external boundaries excluded from charged RAM");
        store.fail_write = true;
        check(!llama_adapter_training_compute(spilled, tokens, labels, &loss, nullptr, nullptr) && store.files.empty() &&
            llama_adapter_training_gradient(spilled, first->name, nullptr, 0) == 0, "checkpoint write failure invalidates gradients");
        store.fail_write = false;
        store.fail_read = true;
        check(!llama_adapter_training_compute(spilled, tokens, labels, &loss, nullptr, nullptr) && store.files.empty() &&
            llama_adapter_training_gradient(spilled, first->name, nullptr, 0) == 0, "checkpoint read failure cleans partial run");
        store.fail_read = false;
        check(!llama_adapter_training_compute(spilled, tokens, labels, &loss, [](void * data) {
            return !((boundary_store *) data)->files.empty();
        }, &store) && store.files.empty(), "cancellation releases written layer checkpoints");
        store.fail_clear = true;
        check(!llama_adapter_training_compute(spilled, tokens, labels, &loss, nullptr, nullptr) && store.files.empty() &&
            llama_adapter_training_gradient(spilled, first->name, nullptr, 0) == 0, "post-compute cleanup failure invalidates gradients");
        store.fail_clear = false;
        check(equivalent(spilled), "fresh checkpointed run recovers after failed attempts");
        llama_adapter_training_free(spilled);
        llama_adapter_training_free(recomputed);
        const char * late_target = "blk.2.attn_output.weight";
        auto * late_adapter = llama_adapter_lora_init_training(model, &late_target, 1, 2, 4, 1);
        auto * late_full = llama_adapter_training_init(model, late_adapter, config, error, sizeof(error));
        auto * late_recomputed = llama_adapter_training_init_checkpointed(model, late_adapter, config, nullptr, error, sizeof(error));
        float late_loss = NAN, late_recomputed_loss = NAN;
        bool late_ok = late_adapter && late_full && late_recomputed &&
            llama_adapter_training_compute(late_full, tokens, labels, &late_loss, nullptr, nullptr) &&
            llama_adapter_training_compute(late_recomputed, tokens, labels, &late_recomputed_loss, nullptr, nullptr) && late_loss == late_recomputed_loss;
        if (late_adapter) for (auto * p : {late_adapter->ab_map.at(late_target).a, late_adapter->ab_map.at(late_target).b}) {
            std::vector<float> a(ggml_nelements(p)), b(a.size());
            late_ok &= llama_adapter_training_gradient(late_full, p->name, a.data(), a.size()) == a.size();
            late_ok &= llama_adapter_training_gradient(late_recomputed, p->name, b.data(), b.size()) == b.size();
            for (size_t i = 0; i < a.size(); ++i) late_ok &= fabsf(a[i]-b[i]) < 2e-5f;
        }
        check(late_ok, "recompute handles entirely frozen prefix layers");
        llama_adapter_training_free(late_full);
        llama_adapter_training_free(late_recomputed);
        llama_adapter_lora_free(late_adapter);
        const char * binding = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        auto settings = llama_adapter_optimizer_default_params();
        settings.gradient_clip = 0.1f;
        settings.epsilon = 1e-4f;
        auto small_state = settings;
        small_state.state_budget_bytes = 1;
        auto * bad_optimizer = llama_adapter_optimizer_init(adapter, small_state, binding, error, sizeof(error));
        check(!bad_optimizer && strstr(error, "budget"), "optimizer budget preflight");
        llama_adapter_optimizer_free(bad_optimizer);
        auto * optimizer = llama_adapter_optimizer_init(adapter, settings, binding, error, sizeof(error));
        check(optimizer != nullptr, "initialize AdamW");
        if (!optimizer) {
            llama_adapter_training_free(training); llama_adapter_lora_free(adapter); llama_model_free(model); continue;
        }
        auto save = [&](llama_adapter_optimizer * selected) {
            std::vector<uint8_t> bytes(llama_adapter_optimizer_save(selected, nullptr, 0));
            if (bytes.empty() || llama_adapter_optimizer_save(selected, bytes.data(), bytes.size()) != bytes.size()) bytes.clear();
            return bytes;
        };
        struct expected_parameter { ggml_tensor * tensor; std::vector<float> values; std::vector<double> gradient; };
        std::map<std::string, expected_parameter> expected;
        for (const auto & entry : adapter->ab_map) for (auto * p : {entry.second.a, entry.second.b}) {
            expected.emplace(p->name, expected_parameter{p, std::vector<float>((float *) p->data, (float *) p->data+ggml_nelements(p)),
                std::vector<double>(ggml_nelements(p))});
        }
        llama_token short_labels[] = {-1, -1, -1, 5};
        bool accumulated = true;
        for (const auto * selected : {labels, short_labels}) {
            accumulated &= llama_adapter_training_compute(training, tokens, selected, &loss, nullptr, nullptr);
            for (auto & p : expected) {
                std::vector<float> g(p.second.gradient.size());
                accumulated &= llama_adapter_training_gradient(training, p.first.c_str(), g.data(), g.size()) == g.size();
                for (size_t i = 0; i < g.size(); ++i) p.second.gradient[i] += g[i]*(selected == labels ? 3.0 : 1.0)/4.0;
            }
            accumulated &= llama_adapter_optimizer_accumulate(optimizer, training);
            accumulated &= !llama_adapter_optimizer_accumulate(optimizer, training);
        }
        check(accumulated && llama_adapter_optimizer_pending_tokens(optimizer) == 4, "token-weighted accumulation consumes each compute once");
        const auto pending = save(optimizer);
        check(!pending.empty(), "checkpoint includes pending accumulation");
        if (pending.empty()) {
            llama_adapter_optimizer_free(optimizer); llama_adapter_training_free(training); llama_adapter_lora_free(adapter); llama_model_free(model); continue;
        }
        std::vector<uint8_t> undersized(pending.size() - 1, 0x5a);
        check(llama_adapter_optimizer_save(optimizer, undersized.data(), undersized.size()) == pending.size() &&
            std::all_of(undersized.begin(), undersized.end(), [](uint8_t b) { return b == 0x5a; }), "undersized checkpoint destination untouched");
        check(!llama_adapter_optimizer_step(optimizer, NAN) && save(optimizer) == pending, "invalid step preserves checkpoint");
        auto & last_parameter = expected.rbegin()->second;
        auto * last_value = (float *) last_parameter.tensor->data + last_parameter.values.size() - 1;
        const float original_last = *last_value;
        *last_value = std::numeric_limits<float>::max();
        const bool overflow_rejected = !llama_adapter_optimizer_step(optimizer, std::numeric_limits<float>::max());
        *last_value = original_last;
        check(overflow_rejected && save(optimizer) == pending, "late nonfinite candidate commits no parameters or state");
        auto * clone = llama_adapter_lora_init_training(model, names.data(), names.size(), 2, 4.0f, 17);
        auto * resumed = llama_adapter_optimizer_init(clone, settings, binding, error, sizeof(error));
        auto * resumed_training = llama_adapter_training_init(model, clone, config, error, sizeof(error));
        check(clone && resumed && resumed_training, "create independent resume context");
        if (!resumed || !resumed_training || pending.empty()) {
            llama_adapter_optimizer_free(resumed); llama_adapter_training_free(resumed_training); llama_adapter_lora_free(clone);
            llama_adapter_optimizer_free(optimizer); llama_adapter_training_free(training); llama_adapter_lora_free(adapter); llama_model_free(model); continue;
        }
        const auto before_restore = save(resumed);
        auto corrupted = pending;
        auto * checkpoint_meta = gguf_init_from_buffer(pending.data(), pending.size(), {true, nullptr});
        const int64_t last_tensor = gguf_get_n_tensors(checkpoint_meta) - 1;
        const size_t last_offset = gguf_get_data_offset(checkpoint_meta) + gguf_get_tensor_offset(checkpoint_meta, last_tensor) + gguf_get_tensor_size(checkpoint_meta, last_tensor) - sizeof(float);
        const float invalid = NAN;
        memcpy(corrupted.data()+last_offset, &invalid, sizeof(invalid));
        check(!llama_adapter_optimizer_load(resumed, corrupted.data(), corrupted.size()) && save(resumed) == before_restore, "late corrupt payload restore is atomic");
        check(!llama_adapter_optimizer_load(resumed, pending.data(), last_offset) && save(resumed) == before_restore, "truncated checkpoint rejected");
        gguf_set_val_str(checkpoint_meta, "optimizer.binding.sha256", "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
        corrupted = pending;
        gguf_get_meta_data(checkpoint_meta, corrupted.data());
        check(!llama_adapter_optimizer_load(resumed, corrupted.data(), corrupted.size()) && save(resumed) == before_restore, "different manifest binding rejected");
        gguf_set_val_str(checkpoint_meta, "optimizer.binding.sha256", binding);
        gguf_set_val_f32(checkpoint_meta, "optimizer.beta1", 0.5f);
        gguf_get_meta_data(checkpoint_meta, corrupted.data());
        check(!llama_adapter_optimizer_load(resumed, corrupted.data(), corrupted.size()) && save(resumed) == before_restore, "different AdamW settings rejected");
        gguf_free(checkpoint_meta);
        bool resumed_ok = llama_adapter_training_compute(resumed_training, tokens, labels, &loss, nullptr, nullptr);
        resumed_ok &= llama_adapter_optimizer_load(resumed, pending.data(), pending.size());
        check(resumed_ok && save(resumed) == pending && llama_adapter_optimizer_pending_tokens(resumed) == 4 &&
            !llama_adapter_optimizer_accumulate(resumed, resumed_training), "restore exact weights and state and invalidate old gradients");
        double squared_norm = 0;
        for (const auto & p : expected) for (double g : p.second.gradient) squared_norm += g*g;
        const double clipping = std::min(1.0, settings.gradient_clip/sqrt(squared_norm));
        const float rate = 0.005f;
        bool trained = llama_adapter_training_compute(training, tokens, labels, &loss, nullptr, nullptr);
        trained &= llama_adapter_optimizer_step(optimizer, rate) && llama_adapter_optimizer_step(resumed, rate);
        check(trained && llama_adapter_training_gradient(training, first->name, nullptr, 0) == 0 &&
            !llama_adapter_optimizer_accumulate(optimizer, training), "committed step invalidates unconsumed gradients");
        bool numerical = true;
        for (const auto & p : expected) for (size_t i = 0; i < p.second.values.size(); ++i) {
            const double g = p.second.gradient[i]*clipping;
            const double reference = p.second.values[i]*(1.0-rate*settings.weight_decay) - rate*g/(fabs(g)+settings.epsilon);
            numerical &= fabs(((const float *) p.second.tensor->data)[i] - reference) < 2e-7;
        }
        check(numerical && clipping < 1.0, "AdamW matches independent token-weighted clipped update");
        check(save(resumed) == save(optimizer), "pending-gradient resume matches first committed step");
        for (int step = 1; step < 30; ++step) {
            for (const auto * selected : {labels, short_labels}) {
                trained &= llama_adapter_training_compute(training, tokens, selected, &loss, nullptr, nullptr);
                trained &= llama_adapter_optimizer_accumulate(optimizer, training);
                trained &= llama_adapter_training_compute(resumed_training, tokens, selected, &loss, nullptr, nullptr);
                trained &= llama_adapter_optimizer_accumulate(resumed, resumed_training);
            }
            trained &= llama_adapter_optimizer_step(optimizer, rate) && llama_adapter_optimizer_step(resumed, rate);
        }
        check(trained && llama_adapter_optimizer_step_count(optimizer) == 30 && llama_adapter_optimizer_pending_tokens(optimizer) == 0 &&
            save(optimizer) == save(resumed), "resumed and uninterrupted AdamW checkpoints match exactly");
        const auto completed = save(optimizer);
        check(llama_adapter_optimizer_load(resumed, completed.data(), completed.size()) && save(resumed) == completed, "completed-step checkpoint roundtrip");
        llama_adapter_optimizer_free(resumed);
        llama_adapter_training_free(resumed_training);
        llama_adapter_lora_free(clone);
        llama_adapter_optimizer_free(optimizer);
        trained &= llama_adapter_training_compute(training, tokens, labels, &loss, nullptr, nullptr);
        fprintf(stdout, "Qwen %s loss: %.6f -> %.6f\n", mixed ? "mixed" : "F32", first_loss, loss);
        check(trained && loss < first_loss - 0.02f, "adapter-only loss reduction");
        const size_t export_size = llama_adapter_lora_export_buffer(adapter, nullptr, 0);
        std::vector<uint8_t> exported(export_size);
        check(export_size && llama_adapter_lora_export_buffer(adapter, exported.data(), export_size) == export_size, "export trained Qwen fixture adapter");
        auto * imported = llama_adapter_lora_init_from_buffer(model, exported.data(), exported.size());
        check(imported && fabs(inference_loss(imported) - loss) < 1e-3, "reimported F16 adapter preserves trained loss");
        llama_adapter_lora_free(imported);
        check(base == original, "mapped base bytes unchanged");
        llama_adapter_training_free(training);
        llama_adapter_lora_free(adapter);
        llama_model_free(model);
    }
    return {pass, total};
}

static bool test_adapter_optimizer(llama_model * model, llama_adapter_lora * adapter) {
    auto * ctx = ggml_init({4*1024*1024, nullptr, false});
    auto * weight = model->layers[0].wo;
    const auto pair = adapter->ab_map.at(weight->name);
    std::vector<uint8_t> original(ggml_nbytes(weight));
    ggml_backend_tensor_get(weight, original.data(), 0, original.size());
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 1);
    for (int j = 0; j < 32; ++j) ((float *) input->data)[j] = cosf(j*0.3f);
    auto * labels = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    *(int32_t *) labels->data = 3;
    auto * base = ggml_mul_mat(ctx, weight, input);
    auto * delta = ggml_scale(ctx, ggml_mul_mat(ctx, pair.b, ggml_mul_mat(ctx, pair.a, input)), pair.get_scale(adapter->alpha, 1.0f));
    auto * loss = ggml_cross_entropy_loss_sparse(ctx, ggml_add(ctx, base, delta), labels);
    ggml_set_loss(loss);
    auto * backward = ggml_new_graph_custom(ctx, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(backward, loss);
    ggml_build_backward_expand(ctx, backward, nullptr);
    auto * optimizer = ggml_graph_dup(ctx, backward, true);
    auto * configuration = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
    for (auto * parameter : {pair.a, pair.b}) {
        auto * m = ggml_dup_tensor(ctx, parameter);
        auto * v = ggml_dup_tensor(ctx, parameter);
        ggml_set_f32(m, 0);
        ggml_set_f32(v, 0);
        ggml_build_forward_expand(optimizer, ggml_opt_step_adamw(ctx, parameter,
            ggml_graph_get_grad(backward, parameter), m, v, configuration));
    }
    float first_loss = 0;
    bool ok = true;
    for (int step = 1; step <= 80; ++step) {
        float settings[] = {0.03f, 0.9f, 0.999f, 1e-8f, 0.01f, 1.0f/(1.0f-powf(0.9f, (float) step)), 1.0f/(1.0f-powf(0.999f, (float) step))};
        memcpy(configuration->data, settings, sizeof(settings));
        ggml_graph_reset(backward);
        ok &= ggml_graph_compute_with_ctx(ctx, optimizer, 2) == GGML_STATUS_SUCCESS;
        if (step == 1) first_loss = *(float *) loss->data;
    }
    const float last_loss = *(float *) loss->data;
    ok &= std::isfinite(last_loss) && last_loss < first_loss*0.05f;
    ok &= ggml_graph_get_grad(backward, weight) == nullptr;
    std::vector<uint8_t> after(original.size());
    ggml_backend_tensor_get(weight, after.data(), 0, after.size());
    ok &= after == original;
    fprintf(stdout, "adapter matrix overfit: %.6f -> %.6f: %s\n", first_loss, last_loss, ok ? "OK" : "FAIL");
    ggml_free(ctx);
    return ok;
}

static std::pair<int, int> test_adapter_buffer() {
    int pass = 0, total = 0;
    auto check = [&](bool value, const char * label) {
        ++total;
        pass += value;
        if (!value) fprintf(stderr, "adapter buffer: %s failed\n", label);
    };
    auto pack = [](gguf_context * meta, ggml_context * ctx) {
        size_t size = gguf_get_meta_size(meta);
        for (auto * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            size = std::max(size, gguf_get_meta_size(meta) + gguf_get_tensor_offset(meta, gguf_find_tensor(meta, t->name)) + ggml_nbytes(t));
        }
        std::vector<uint8_t> bytes(size, 0);
        gguf_get_meta_data(meta, bytes.data());
        return bytes;
    };
    auto * ctx = ggml_init({1024*1024, nullptr, true});
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    gguf_set_val_u32(meta, "llama.vocab_size", 8);
    gguf_set_val_u32(meta, "llama.context_length", 32);
    gguf_set_val_u32(meta, "llama.embedding_length", 32);
    gguf_set_val_u32(meta, "llama.block_count", 1);
    gguf_set_val_u32(meta, "llama.feed_forward_length", 32);
    gguf_set_val_u32(meta, "llama.attention.head_count", 1);
    gguf_set_val_u32(meta, "llama.attention.head_count_kv", 1);
    gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    auto tensor = [&](const char * name, int64_t x, int64_t y) {
        auto * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, x, y);
        ggml_set_name(t, name);
        gguf_add_tensor(meta, t);
    };
    tensor("token_embd.weight", 32, 8);
    tensor("output_norm.weight", 32, 1);
    tensor("blk.0.attn_norm.weight", 32, 1);
    tensor("blk.0.ffn_norm.weight", 32, 1);
    for (const char * name : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"}) {
        tensor((std::string("blk.0.") + name + ".weight").c_str(), 32, 32);
    }
    auto base = pack(meta, ctx);
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_buffer_view(base.data(), base.size(), params);
    check(model != nullptr, "tiny mapped base");
    gguf_free(meta);
    ggml_free(ctx);
    if (model) {
        const char * target = "blk.0.attn_output.weight";
        auto * initialized = llama_adapter_lora_init_training(model, &target, 1, 2, 4.0f, 20260920);
        check(initialized != nullptr, "initialize training parameters");
        if (initialized) {
            const auto pair = initialized->ab_map.at(target);
            check(std::all_of((float *) pair.b->data, (float *) pair.b->data + ggml_nelements(pair.b), [](float value) { return value == 0.0f; }), "initial neutral B");
            const size_t size = llama_adapter_lora_export_buffer(initialized, nullptr, 0);
            std::vector<uint8_t> bytes(size, 0x7f);
            check(size > 0, "export size");
            check(llama_adapter_lora_export_buffer(initialized, bytes.data(), size-1) == size && bytes[0] == 0x7f, "undersized buffer unchanged");
            check(llama_adapter_lora_export_buffer(initialized, bytes.data(), size) == size, "F16 export");
            auto * same = llama_adapter_lora_init_training(model, &target, 1, 2, 4.0f, 20260920);
            std::vector<uint8_t> repeated(size);
            check(same && llama_adapter_lora_export_buffer(same, repeated.data(), size) == size && repeated == bytes, "deterministic initialization");
            llama_adapter_lora_free(same);
            check(test_adapter_optimizer(model, initialized), "adapter-only AdamW overfit");
            check(llama_adapter_lora_export_buffer(initialized, bytes.data(), size) == size && bytes != repeated, "trained export differs");
            auto * imported = llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size());
            check(imported != nullptr, "trained F16 import");
            if (imported) {
                float f32[64];
                ggml_fp16_t f16[64];
                ggml_backend_tensor_get(imported->ab_map.at(target).b, f16, 0, sizeof(f16));
                ggml_fp16_to_fp32_row(f16, f32, 64);
                bool close = true;
                for (int j = 0; j < 64; ++j) close &= fabsf(f32[j] - ((float *) pair.b->data)[j]) < 0.001f;
                check(close, "export preserves learned weights");
            }
            llama_adapter_lora_free(imported);
            ((float *) pair.b->data)[0] = std::numeric_limits<float>::infinity();
            check(llama_adapter_lora_export_buffer(initialized, bytes.data(), size) == 0, "nonfinite export rejected");
            llama_adapter_lora_free(initialized);
        }
        const char * duplicate[] = {target, target};
        check(!llama_adapter_lora_init_training(model, duplicate, 2, 2, 4.0f, 1), "duplicate targets rejected");
        const char * unknown = "missing.weight";
        check(!llama_adapter_lora_init_training(model, &unknown, 1, 2, 4.0f, 1), "unknown target rejected");
        ctx = ggml_init({1024*1024, nullptr, true});
        meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "llama");
        gguf_set_val_str(meta, "general.type", "adapter");
        gguf_set_val_str(meta, "adapter.type", "lora");
        gguf_set_val_f32(meta, "adapter.lora.alpha", 4.0f);
        gguf_set_val_u64(meta, "adapter.training.optimizer_steps", 42);
        tensor("blk.0.attn_output.weight.lora_a", 32, 2);
        tensor("blk.0.attn_output.weight.lora_b", 2, 32);
        auto bytes = pack(meta, ctx);
        auto * adapter = llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size());
        check(adapter != nullptr, "valid borrowed adapter");
        if (adapter) {
            std::vector<uint8_t> exported(llama_adapter_lora_export_buffer(adapter, nullptr, 0));
            check(llama_adapter_lora_export_buffer(adapter, exported.data(), exported.size()) == exported.size(), "imported export");
            auto * metadata = gguf_init_from_buffer(exported.data(), exported.size(), {true, nullptr});
            const int64_t key = metadata ? gguf_find_key(metadata, "adapter.training.optimizer_steps") : -1;
            check(key >= 0 && gguf_get_val_u64(metadata, key) == 42, "typed provenance survives export");
            gguf_free(metadata);
        }
        llama_adapter_lora_free(adapter);
        check(!llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size()-1), "truncated payload");
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        std::memcpy(bytes.data()+gguf_get_meta_size(meta), &invalid, sizeof(invalid));
        check(!llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size()), "non-finite weights");
        gguf_set_val_str(meta, "adapter.lora.alpha", "invalid");
        bytes = pack(meta, ctx);
        check(!llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size()), "wrong metadata type");
        gguf_set_val_f32(meta, "adapter.lora.alpha", 4.0f);
        gguf_set_val_str(meta, "general.architecture", "qwen3next");
        bytes = pack(meta, ctx);
        check(!llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size()), "wrong architecture");
        gguf_set_val_str(meta, "general.architecture", "llama");
        tensor("output_norm.weight", 32, 1);
        bytes = pack(meta, ctx);
        check(!llama_adapter_lora_init_from_buffer(model, bytes.data(), bytes.size()), "unsupported tensor");
        check(!llama_adapter_lora_init_from_buffer(model, nullptr, 0), "empty buffer");
        gguf_free(meta);
        ggml_free(ctx);
        llama_model_free(model);
    }
    fprintf(stdout, "adapter buffer: %d/%d passed\n", pass, total);
    return {pass, total};
}

static void print_usage() {
    printf("usage: test-gguf [seed]\n");
    printf("  if no seed is unspecified then a random seed is used\n");
}

int main(int argc, char ** argv) {
    if (argc > 2) {
        print_usage();
        return 1;
    }

    std::random_device rd;
    const unsigned int seed = argc < 2 ? rd() : std::stoi(argv[1]);

    // Initialize ggml backends early so the prints aren't interleaved with the test results:
    ggml_backend_dev_count();
    fprintf(stderr, "\n");

    int npass = 0;
    int ntest = 0;
    {
        auto result = test_attention_training();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_l2_norm_gradients();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_sparse_loss();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_quantized_expert_gradients();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_routing_gradients();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_adapter_buffer();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_concat_gradients();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_qwen_training();
        npass += result.first;
        ntest += result.second;
    }
    {
        auto result = test_expert_parts();
        npass += result.first;
        ntest += result.second;
    }
    {
        std::pair<int, int> result = test_handcrafted_file(seed);
        npass += result.first;
        ntest += result.second;
    }

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        for (bool only_meta : {true, false}) {
            std::pair<int, int> result = test_roundtrip(dev, seed, only_meta, ROUNDTRIP_READ_MODE_FILE);
            npass += result.first;
            ntest += result.second;
        }
        {
            std::pair<int, int> result = test_roundtrip(dev, seed, /*only_meta=*/false, ROUNDTRIP_READ_MODE_FILE_OFFSET);
            npass += result.first;
            ntest += result.second;
        }
        {
            std::pair<int, int> result = test_roundtrip(dev, seed, /*only_meta=*/false, ROUNDTRIP_READ_MODE_BUFFER);
            npass += result.first;
            ntest += result.second;
        }
        {
            std::pair<int, int> result = test_roundtrip(dev, seed, /*only_meta=*/false, ROUNDTRIP_READ_MODE_CALLBACK);
            npass += result.first;
            ntest += result.second;
        }

        {
            std::pair<int, int> result = test_gguf_set_kv(dev, seed);
            npass += result.first;
            ntest += result.second;
        }
    }

    printf("%d/%d tests passed\n", npass, ntest);
    if (npass != ntest) {
        printf("\033[1;31mFAIL\033[0m\n");
        return 1;
    }
    printf("\033[1;32mOK\033[0m\n");
    return 0;
}
