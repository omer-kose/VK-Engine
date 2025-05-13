#pragma once

#include <vulkan/vulkan.h>

/*
	This file will only hold some useful common structs about materials.

    In general Material Framework has 3 components in this engine:
    1- Material classes that declares the resources of the materials and only holds descriptor layout for that type of material. Therefore, for all the instances it is shared and static.
    2- The actual material instance that hold the resources and descriptor sets defined for that material type. Material classes builds them and return. All the resources allocated, must be cleared out by user of the 
    material instance (for example meshes). They are generally held in smart pointers so deallocation happens automatically.
    3- The pipeline which is defined by a corresponding Pass.

    For example, as for GLTF Metallic workflow there is:
    1- GLTFMetallicMaterial which declares the resources and holds the static descriptor layout
    2- MaterialInstance that is common for all types of materials which defines whether the material is transparent or opaque and a descriptor set to be bound.
    3- GLTFMetallicPass, a static class, which defines the pipeline and in the Execute function it uses the MaterialInstance in the RenderObject to bind it and draw.
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
