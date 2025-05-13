#pragma once

#include <Core/vk_types.h>
#include <Core/vk_descriptors.h>

#include "Material.h"

class VulkanEngine;

// PBR Metallic Material follows the GLTF format
// TODO: Consider this moving to a materials.h file
struct GLTFMetallicRoughnessMaterial
{
	VkDescriptorSetLayout materialLayout;

	// CPU representation of the MaterialConstants uniform buffer
	struct MaterialConstants
	{
		glm::vec4 colorFactors;
		glm::vec4 metalRoughnessFactors;
		// padding to complete the uniform buffer to 256 bytes (most GPUs expect a minimum alignment of 256 bytes for uniform buffers)
		glm::vec4 extra[14];
	};

	struct MaterialResources
	{
		AllocatedImage colorImage;
		VkSampler colorSampler;
		AllocatedImage metalRoughnessImage;
		VkSampler metalRoughnessSampler;
		VkBuffer dataBuffer; // Handle to the buffer holding MaterialConstants data
		uint32_t dataBufferOffset; // Multiple materials in a GLTF files will be stored in a single buffer, so the actual data for the specific material instance is fetched with this offset
	};

	DescriptorWriter writer;

	void buildMaterialLayout(VulkanEngine* engine);

	// This struct only stores the pipelines and the layouts. The material resources are allocated outside. Allocator must clean them properly.
	void clearResources(VkDevice device);

	MaterialInstance createInstance(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
};