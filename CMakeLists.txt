cmake_minimum_required(VERSION 3.13)
project(videomae_wasm)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set Torch_DIR to help find LibTorch
set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")

# Find LibTorch package
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Emscripten specific settings
set(CMAKE_EXECUTABLE_SUFFIX ".js")

add_executable(videomae_wasm videomae_wrapper.cpp)
target_link_libraries(videomae_wasm "${TORCH_LIBRARIES}")

# Emscripten specific flags
set_target_properties(videomae_wasm PROPERTIES
    LINK_FLAGS "-s WASM=1 \
                -s ALLOW_MEMORY_GROWTH=1 \
                -s EXPORTED_FUNCTIONS=['_malloc','_free','_create_inference','_destroy_inference','_run_inference'] \
                -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
                -s TOTAL_MEMORY=1GB \
                --preload-file model@/model \
                --preload-file kinetics400.csv@/kinetics400.csv \
                -O3"
)