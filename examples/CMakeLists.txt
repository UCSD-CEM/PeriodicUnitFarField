add_executable(example_host_device example_host_device.cu)
#link OpenMP
if(OpenMP_CXX_FOUND)
    target_include_directories(example_host_device PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(example_host_device PUBLIC OpenMP::OpenMP_CXX)
endif()