cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(HashedEmbeddingBag)

add_compile_definitions(CPP_ONLY)
add_compile_definitions(CUDA_HAS_FP16=1)
add_compile_definitions(__CUDA_NO_HALF_OPERATORS__)
add_compile_definitions(__CUDA_NO_HALF2_OPERATORS__)
add_compile_definitions(__CUDA_NO_HALF_CONVERSIONS__)
set(USE_CUDA ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

torch_cuda_based_add_library(hashed_embedding_bag_kernel hashed_embedding_bag_kernel.cu hashed_embedding_bag_kernel.cuh)

target_link_libraries(hashed_embedding_bag_kernel  "${TORCH_LIBRARIES}")
target_link_libraries(hashed_embedding_bag_kernel  ${PYTHON_LIBRARIES})

cuda_add_executable(HashedEmbeddingBag_test_cpp HashedEmbeddingBag_test_cpp.cpp)

target_link_libraries(HashedEmbeddingBag_test_cpp  "${TORCH_LIBRARIES}")
target_link_libraries(HashedEmbeddingBag_test_cpp  ${PYTHON_LIBRARIES})
target_link_libraries(HashedEmbeddingBag_test_cpp ${GTEST_LIBRARIES})
target_link_libraries(HashedEmbeddingBag_test_cpp hashed_embedding_bag_kernel)
set_property(TARGET HashedEmbeddingBag_test_cpp   PROPERTY CXX_STANDARD 17)

include_directories(".")
cuda_add_executable(HashedEmbeddingBag_benchmark benchmark/HashedEmbeddingBag_benchmark.cpp)

target_link_libraries(HashedEmbeddingBag_benchmark  "${TORCH_LIBRARIES}")
target_link_libraries(HashedEmbeddingBag_benchmark  ${PYTHON_LIBRARIES})
target_link_libraries(HashedEmbeddingBag_benchmark hashed_embedding_bag_kernel)
set_property(TARGET HashedEmbeddingBag_benchmark   PROPERTY CXX_STANDARD 17)


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET HashedEmbeddingBag_test_cpp
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:HashedEmbeddingBag_test_cpp>)
    add_custom_command(TARGET HashedEmbeddingBag_benchmark
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:HashedEmbeddingBag_benchmark>)
endif (MSVC)
