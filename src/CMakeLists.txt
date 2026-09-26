llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

# Serialized state is a disposable cache, valid only for this exact implementation and toolchain.
# Reconfigure on source edits too: a Git commit alone does not identify local fork changes.
file(GLOB_RECURSE LLAMA_STATE_ID_INPUTS CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.h"
    "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/ggml/include/*.h"
    "${PROJECT_SOURCE_DIR}/ggml/src/*.cpp" "${PROJECT_SOURCE_DIR}/ggml/src/*.h"
    "${PROJECT_SOURCE_DIR}/ggml/src/*.c" "${PROJECT_SOURCE_DIR}/ggml/src/*.cu"
    "${PROJECT_SOURCE_DIR}/ggml/src/*.cuh")
list(SORT LLAMA_STATE_ID_INPUTS)
set(LLAMA_STATE_ID_MATERIAL "${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_SIZEOF_VOID_P};${CMAKE_CXX_FLAGS};${CMAKE_CUDA_ARCHITECTURES};${GGML_CUDA};${GGML_NATIVE};${GGML_AVX512}")
foreach(LLAMA_STATE_ID_FILE IN LISTS LLAMA_STATE_ID_INPUTS)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${LLAMA_STATE_ID_FILE}")
    file(SHA256 "${LLAMA_STATE_ID_FILE}" LLAMA_STATE_ID_HASH)
    string(APPEND LLAMA_STATE_ID_MATERIAL ";${LLAMA_STATE_ID_HASH}")
endforeach()
string(SHA256 LLAMA_STATE_BUILD_ID "${LLAMA_STATE_ID_MATERIAL}")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/llama-state-build-id-$<CONFIG>.h"
    CONTENT "#define LLAMA_STATE_BUILD_ID \"${LLAMA_STATE_BUILD_ID}:$<CONFIG>\"\n")
target_compile_definitions(llama PRIVATE LLAMA_STATE_BUILD_ID_HEADER="llama-state-build-id-$<CONFIG>.h")

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
