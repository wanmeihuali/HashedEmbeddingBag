cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(HashedEmbeddingBag)

add_compile_definitions(CPP_ONLY)
find_package(PythonLibs REQUIRED)
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(${PYTHON_INCLUDE_DIRS})

cuda_add_library(hashed_embedding_bag_kernel hashed_embedding_bag_kernel.cu hashed_embedding_bag_kernel.cuh)

target_link_libraries(hashed_embedding_bag_kernel  "${TORCH_LIBRARIES}")
target_link_libraries(hashed_embedding_bag_kernel  ${PYTHON_LIBRARIES})

cuda_add_executable(HashedEmbeddingBag_test_cpp HashedEmbeddingBag_test_cpp.cpp)

target_link_libraries(HashedEmbeddingBag_test_cpp  "${TORCH_LIBRARIES}")
target_link_libraries(HashedEmbeddingBag_test_cpp  ${PYTHON_LIBRARIES})
target_link_libraries(HashedEmbeddingBag_test_cpp ${GTEST_LIBRARIES})
target_link_libraries(HashedEmbeddingBag_test_cpp hashed_embedding_bag_kernel)
set_property(TARGET HashedEmbeddingBag_test_cpp   PROPERTY CXX_STANDARD 17)

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
endif (MSVC)
