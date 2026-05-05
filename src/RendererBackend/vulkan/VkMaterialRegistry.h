#pragma once

#include <vector>
#include <RendererBackend/vulkan/vk_types.h>
#include <RendererBackend/vulkan/vk_descriptors.h>

namespace SK::Asset
{
	struct AssetRegistry;
}

namespace SK::Material
{
	struct MaterialRegistry;
}

namespace SK::VkRendererBackend
{
	struct State;
	struct VkAssetRegistry;

	struct GPUPBRMaterialData
	{
		float baseColorFactor[4] = { 1.f, 1.f, 1.f, 1.f };
		float metallicFactor = 1.f;
		float roughnessFactor = 1.f;
		uint32_t textureIndices[5] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX };
	};

	struct VkMaterialRegistry
	{
		/*
			For textures and materials, bindless descriptors are used. VkMaterialRegistry creates and manages a descriptor set for materials and textures of the materials.
		*/
		VkDescriptorSetLayout resourceDescriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorSet resourceDescriptorSet = VK_NULL_HANDLE;
		DescriptorAllocator resourceDescriptorAllocator;
		AllocatedBuffer materialBuffer;
		std::vector<GPUPBRMaterialData> materialData;
	};
}