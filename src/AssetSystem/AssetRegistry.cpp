#include "AssetRegistry.h"

#include <RendererBackend/vulkan/vk_renderer.h>

void SK::Asset::clearAssetRegistry(SK::VkRendererBackend::State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry)
{
	// Clear the meshes
	for(Mesh& mesh : assetRegistry->meshes)
	{
		SK::VkRendererBackend::destroyBuffer(vkRendererBackend, mesh.meshBuffers.vertexBuffer);
		SK::VkRendererBackend::destroyBuffer(vkRendererBackend, mesh.meshBuffers.indexBuffer);
	}

	// Clear the texture images
	for(Texture& texture : assetRegistry->textures)
	{
		// If a texture cannot be loaded for some reason, it is defaulted to error image from the backend. In that case, don't destroy backend owned resource
		if(texture.image.image != vkRendererBackend->errorCheckerboardImage.image)
		{
			SK::VkRendererBackend::destroyImage(vkRendererBackend, texture.image);
		}
	}

	// Clear the samplers
	for(VkSampler& sampler : assetRegistry->samplers)
	{
		vkDestroySampler(vkRendererBackend->device, sampler, nullptr);
	}

	// Clear all the containers
	assetRegistry->meshes.clear();
	assetRegistry->textures.clear();
	assetRegistry->samplers.clear();

	assetRegistry->meshIndexByName.clear();
	assetRegistry->textureIndexByName.clear();
}
