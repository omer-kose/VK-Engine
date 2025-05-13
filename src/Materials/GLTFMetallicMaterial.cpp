#include "GLTFMetallicMaterial.h"

#include <Core/vk_engine.h>

VkDescriptorSetLayout GLTFMetallicRoughnessMaterial::MaterialLayout = VK_NULL_HANDLE;
DescriptorWriter GLTFMetallicRoughnessMaterial::Writer = {};

void GLTFMetallicRoughnessMaterial::BuildMaterialLayout(VulkanEngine* engine)
{
    // Layout of the GLTF Metallic Roughness Material set
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    MaterialLayout = layoutBuilder.build(engine->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
}

void GLTFMetallicRoughnessMaterial::ClearMaterialLayout(VkDevice device)
{
    vkDestroyDescriptorSetLayout(device, MaterialLayout, nullptr);
}

MaterialInstance GLTFMetallicRoughnessMaterial::CreateInstance(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.materialSet = descriptorAllocator.allocate(device, MaterialLayout);

    Writer.clear();
    Writer.writeBuffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    Writer.writeImage(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.writeImage(2, resources.metalRoughnessImage.imageView, resources.metalRoughnessSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    Writer.updateSet(device, matData.materialSet);

    return matData;
}
