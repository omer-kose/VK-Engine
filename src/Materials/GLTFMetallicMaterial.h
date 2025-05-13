#pragma once

#include <Core/vk_types.h>
#include <Core/vk_descriptors.h>

#include "Material.h"

class VulkanEngine;

// PBR Metallic Material follows the GLTF format
class GLTFMetallicRoughnessMaterial
{
public:
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
		uint32_t dataBufferOffset; // Multiple materials in a GLTF file will be stored in a single buffer, so the actual data for the specific material instance is fetched with this offset
	};

	static void BuildMaterialLayout(VulkanEngine* engine);

	// This static class only stores material layout. The material resources are allocated outside per material instance. Allocator-side must clean them properly.
	static void ClearMaterialLayout(VkDevice device);

	static MaterialInstance CreateInstance(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator);
private:
	static VkDescriptorSetLayout MaterialLayout;
	static DescriptorWriter Writer;
};