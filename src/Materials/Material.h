#pragma once

#include <vulkan/vulkan.h>

/*
	This file will only hold some useful common structs about materials.
*/

// Material Data
// Describes the type of the pass. Current 2 supported types: Opaque, Transparent
enum class MaterialPass : uint8_t
{
    Opaque,
    Transparent,
    Other
};

struct MaterialInstance
{
    VkDescriptorSet materialSet;
    MaterialPass passType;
};