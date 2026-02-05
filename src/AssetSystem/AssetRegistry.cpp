#include "AssetRegistry.h"

#include <RendererBackend/vulkan/vk_renderer.h>

void SK::Asset::registerImported(AssetRegistry* assetRegistry, ImportedAsset&& importedAsset)
{
    for(auto& mesh : importedAsset.meshes)
    {
        uint32_t idx = static_cast<uint32_t>(assetRegistry->meshes.size());
        assetRegistry->meshIndexByName[mesh.name] = idx;
        assetRegistry->meshes.push_back(std::move(mesh));
    }

    for(auto& tex : importedAsset.textures)
    {
        uint32_t idx = static_cast<uint32_t>(assetRegistry->textures.size());
        assetRegistry->textureIndexByName[tex.name] = idx;
        assetRegistry->textures.push_back(std::move(tex));
    }
}

void SK::Asset::discardCPUMeshData(AssetRegistry* assetRegistry)
{
    for(auto& mesh : assetRegistry->meshes)
    {
        if(mesh.retention == CPURetention::DropAfterUpload)
        {
            mesh.vertices.clear();
            mesh.indices.clear();
            mesh.vertices.shrink_to_fit();
            mesh.indices.shrink_to_fit();
        }
    }
}

void SK::Asset::discardCPUTextureData(AssetRegistry* assetRegistry)
{
    for(auto& tex : assetRegistry->textures)
    {
        if(tex.retention == CPURetention::DropAfterUpload)
        {
            tex.image.data.clear();
            tex.image.data.shrink_to_fit();
        }
    }
}

void SK::Asset::clearAssetRegistry(SK::Asset::AssetRegistry* assetRegistry)
{

}
