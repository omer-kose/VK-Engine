#include "VkAssetBuilder.h"

#include <RendererBackend/vulkan/vk_renderer.h>
#include <RendererBackend/vulkan/VkAssetRegistry.h>
#include <AssetSystem/AssetRegistry.h>

void SK::VkRendererBackend::buildGPUAssets(State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, VkAssetRegistry* vkAssetRegistry)
{
    // Meshes
    for(auto& mesh : assetRegistry->meshes)
    {
        VkAssetRegistry::GPUMesh gpuMesh{};
        gpuMesh.name = mesh.name;

        gpuMesh.meshBuffers = SK::VkRendererBackend::uploadMesh(vkRendererBackend, mesh.vertices, mesh.indices);

        uint32_t idx = static_cast<uint32_t>(vkAssetRegistry->meshes.size());
        vkAssetRegistry->meshIndexByName[gpuMesh.name] = idx;
        vkAssetRegistry->meshes.push_back(std::move(gpuMesh));
    }

    // Textures (placeholder)
    for(const auto& texture : assetRegistry->textures)
    {
        VkAssetRegistry::GPUTexture gpuTexture{};
        gpuTexture.name = texture.name;

        // Create image from tex.image.data
        gpuTexture.image = SK::VkRendererBackend::createImage(vkRendererBackend, (void*)texture.image.data.data(), VkExtent3D{texture.image.width, texture.image.height, 1}, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, texture.description.mipMapped);
        
        // TODO: Create sampler based on tex.desc

        uint32_t idx = static_cast<uint32_t>(assetRegistry->textures.size());
        vkAssetRegistry->textureIndexByName[gpuTexture.name] = idx;
        vkAssetRegistry->textures.push_back(std::move(gpuTexture));
    }
}

void SK::VkRendererBackend::clearGPUAssets(State* vkRendererBackend, VkAssetRegistry* vkAssetRegistry)
{
    for(auto& mesh : vkAssetRegistry->meshes)
    {
        SK::VkRendererBackend::destroyBuffer(vkRendererBackend, mesh.meshBuffers.vertexBuffer);
        SK::VkRendererBackend::destroyBuffer(vkRendererBackend, mesh.meshBuffers.indexBuffer);
    }

    for(auto& texture : vkAssetRegistry->textures)
    {
        SK::VkRendererBackend::destroyImage(vkRendererBackend, texture.image);
        vkDestroySampler(vkRendererBackend->device, texture.sampler, nullptr);
    }

    vkAssetRegistry->meshes.clear();
    vkAssetRegistry->textures.clear();
    vkAssetRegistry->meshIndexByName.clear();
    vkAssetRegistry->textureIndexByName.clear();
}
