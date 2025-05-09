# Add all the subdirectories of source files
file(GLOB_RECURSE SRC_CORE
	"Core/*.h"
	"Core/*.hpp"
	"Core/*.cpp"
)

# Add the rest of the source files in the src folder
file(GLOB SRC 
	"*.h"
	"*.hpp"
	"*.cpp"
)

# Create executable with all found sources
add_executable(engine ${SRC} ${SRC_CORE})

# Add filters for the solution
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC_CORE})

set_property(TARGET engine PROPERTY CXX_STANDARD 20)
target_compile_definitions(engine PUBLIC GLM_FORCE_DEPTH_ZERO_TO_ONE)
target_include_directories(engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(engine PUBLIC vma glm Vulkan::Vulkan fmt::fmt stb_image SDL2::SDL2 vkbootstrap imgui fastgltf::fastgltf)

target_precompile_headers(engine PUBLIC <optional> <vector> <memory> <string> <vector> <unordered_map> <glm/mat4x4.hpp>  <glm/vec4.hpp> <vulkan/vulkan.h>)

add_custom_command(TARGET engine POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:engine> $<TARGET_FILE_DIR:engine>
  COMMAND_EXPAND_LISTS
  )