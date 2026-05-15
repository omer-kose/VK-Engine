#pragma once

#include <vector>
#include <RendererBackend/Vulkan/vk_types.h>
#include <RendererBackend/Vulkan/vk_descriptors.h>

#include <MaterialSystem/MaterialInfo.h>

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

	struct VkMaterialRegistry
	{
		/*
			For textures and materials, bindless descriptors are used. VkMaterialRegistry creates and manages a descriptor set for materials and textures of the materials.

			Any other textures in the engine that are not related to materials, will be handled via their own descriptors in the renderers they used.
		*/
		VkDescriptorSetLayout resourceDescriptorSetLayout = VK_NULL_HANDLE;
		VkDescriptorSet resourceDescriptorSet = VK_NULL_HANDLE;
		DescriptorAllocator resourceDescriptorAllocator;
		AllocatedBuffer pbrMaterialBuffer;
	};

	void buildMaterialRegistry(State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, SK::Material::MaterialRegistry* materialRegistry, VkAssetRegistry* vkAssetRegistry, VkMaterialRegistry* vkMaterialRegistry);
	void clearMaterialRegistry(State* vkRendererBackend, VkMaterialRegistry* vkMaterialRegistry);
}