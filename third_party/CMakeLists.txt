find_package(Vulkan REQUIRED)

add_library(vkbootstrap STATIC)
add_library(glm INTERFACE)
add_library(stb INTERFACE)

add_subdirectory(fastgltf)
add_subdirectory(fmt EXCLUDE_FROM_ALL)
add_subdirectory(SDL EXCLUDE_FROM_ALL)
add_subdirectory(VulkanMemoryAllocator)
add_subdirectory(volk)

target_sources(vkbootstrap PRIVATE 
  vkbootstrap/VkBootstrap.h
  vkbootstrap/VkBootstrap.cpp
  )

target_include_directories(vkbootstrap PUBLIC vkbootstrap)
target_link_libraries(vkbootstrap PUBLIC Vulkan::Vulkan $<$<BOOL:UNIX>:${CMAKE_DL_LIBS}>)
set_property(TARGET vkbootstrap PROPERTY CXX_STANDARD 20)

#glm is header only lib so we only need the include path
target_include_directories(glm INTERFACE glm)

add_library(imgui STATIC)

target_include_directories(imgui PUBLIC imgui)

target_sources(imgui PRIVATE 
  imgui/imgui.h
  imgui/imgui.cpp

  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui_tables.cpp
  imgui/backends/imgui_impl_vulkan.cpp
  imgui/backends/imgui_impl_sdl2.cpp
  )

target_link_libraries(imgui PUBLIC Vulkan::Vulkan SDL2::SDL2 volk)

target_include_directories(stb INTERFACE stb)
