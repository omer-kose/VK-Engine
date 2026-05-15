#include "VkMaterialRegistry.h"

#include <RendererBackend/Vulkan/VkAssetRegistry.h>
#include <RendererBackend/Vulkan/vk_renderer.h>

#include <AssetSystem/AssetRegistry.h>
#include <MaterialSystem/MaterialRegistry.h>

void SK::VkRendererBackend::buildMaterialRegistry(State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, SK::Material::MaterialRegistry* materialRegistry, VkAssetRegistry* vkAssetRegistry, VkMaterialRegistry* vkMaterialRegistry)
{
	std::vector<SK::Material::PBRData> pbrMaterialData;
	pbrMaterialData.reserve(materialRegistry->instances.size());

	// Gather up PBRData into a single buffer
	for (const auto& materialInstance : materialRegistry->instances)
	{
		SK::Material::PBRData gpuPbrMaterialData{};
		const SK::Material::PBRData* pbrData = &materialInstance.materialData;
		for (int i = 0; i < 4; ++i)
		{
			gpuPbrMaterialData.baseColorFactor[i] = pbrData->baseColorFactor[i];
		}

		gpuPbrMaterialData.metallicFactor = pbrData->metallicFactor;
		gpuPbrMaterialData.roughnessFactor = pbrData->roughnessFactor;

		gpuPbrMaterialData.baseColorTexture = pbrData->baseColorTexture;
		gpuPbrMaterialData.metallicRoughnessTexture = pbrData->metallicRoughnessTexture;
		gpuPbrMaterialData.normalTexture = pbrData->normalTexture;
		gpuPbrMaterialData.emissiveTexture = pbrData->emissiveTexture;

		pbrMaterialData.push_back(gpuPbrMaterialData);
	}

	// Create and upload the GPU buffer that will hold all the material data in the asset system.
	const size_t pbrMaterialBufferSize = pbrMaterialData.size() * sizeof(SK::Material::PBRData);
	assert(pbrMaterialBufferSize >= 0);
	vkMaterialRegistry->pbrMaterialBuffer = SK::VkRendererBackend::createAndUploadGPUBuffer(vkRendererBackend, pbrMaterialBufferSize,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, static_cast<void*>(pbrMaterialData.data()));

	DescriptorLayoutBuilder builder;
	// Binding 0 Material Buffer
	builder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	// Binding 1 all the Textures in the Asset System 
	// TODO: For now, each texture has its own sampler which is a waste. Consider binding samplers seperately later
	builder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, assetRegistry->textures.size());
	vkMaterialRegistry->resourceDescriptorSetLayout = builder.build(vkRendererBackend->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

	std::vector<DescriptorAllocator::PoolSize> poolSizes = { 
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, static_cast<uint32_t>(vkAssetRegistry->textures.size()) }
	};

	// Descriptor Allocator will exactly allocate a single descriptor set holding all the material data and textures related to them.
	vkMaterialRegistry->resourceDescriptorAllocator.initPool(vkRendererBackend->device, 1, poolSizes);
	vkMaterialRegistry->resourceDescriptorSet = vkMaterialRegistry->resourceDescriptorAllocator.allocate(vkRendererBackend->device, vkMaterialRegistry->resourceDescriptorSetLayout);

	// Write the descriptors
	DescriptorWriter writer;
	writer.writeBuffer(0, vkMaterialRegistry->pbrMaterialBuffer.buffer, pbrMaterialBufferSize, 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
	std::vector<VkDescriptorImageInfo> textureImageInfo(vkAssetRegistry->textures.size());
	for (size_t i = 0; i < vkAssetRegistry->textures.size(); ++i)
	{
		const SK::VkRendererBackend::VkAssetRegistry::GPUTexture& gpuTexture = vkAssetRegistry->textures[i];
		textureImageInfo[i] = VkDescriptorImageInfo{ .sampler = gpuTexture.sampler, .imageView = gpuTexture.image.imageView, .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
	}
	writer.writeImages(1, textureImageInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	writer.updateSet(vkRendererBackend->device, vkMaterialRegistry->resourceDescriptorSet);
}

void SK::VkRendererBackend::clearMaterialRegistry(State* vkRendererBackend, VkMaterialRegistry* vkMaterialRegistry)
{
	if (vkMaterialRegistry->resourceDescriptorSetLayout != VK_NULL_HANDLE)
	{
		vkDestroyDescriptorSetLayout(vkRendererBackend->device, vkMaterialRegistry->resourceDescriptorSetLayout, nullptr);
		vkMaterialRegistry->resourceDescriptorSetLayout = VK_NULL_HANDLE;
	}

	vkMaterialRegistry->resourceDescriptorAllocator.destroyPool(vkRendererBackend->device);

	if (vkMaterialRegistry->pbrMaterialBuffer.buffer != VK_NULL_HANDLE)
	{
		destroyBuffer(vkRendererBackend, vkMaterialRegistry->pbrMaterialBuffer);
		vkMaterialRegistry->pbrMaterialBuffer = {};
	}

	vkMaterialRegistry->resourceDescriptorSet = VK_NULL_HANDLE;
}
