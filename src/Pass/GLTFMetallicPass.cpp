#include "GLTFMetallicPass.h"

#include <Core/vk_engine.h>

void GLTFMetallicPass::execute(VulkanEngine* engine, VkCommandBuffer& cmd, DrawContext* ctx)
{
    std::vector<uint32_t> opaqueDraws;
    opaqueDraws.reserve(ctx->opaqueSurfaces.size());

    for(uint32_t i = 0; i < ctx->opaqueSurfaces.size(); ++i)
    {
        opaqueDraws.push_back(i);
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](const auto& iA, const auto& iB) {
        const RenderObject& A = ctx->opaqueSurfaces[iA];
        const RenderObject& B = ctx->opaqueSurfaces[iB];
        if(A.materialInstance == B.materialInstance)
        {
            return A.indexBuffer < B.indexBuffer;
        }
        else
        {
            return A.materialInstance < B.materialInstance;
        }

        });

    // Keep track of states to avoid unnecessary rebindings
    MaterialPipeline* lastPipeline = nullptr;
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const RenderObject& robj) {
        if(robj.materialInstance != lastMaterial)
        {
            lastMaterial = robj.materialInstance;
            // if the material pipeline is changed bind the new pipeline as well as global scene descriptor set
            if(lastPipeline != robj.materialInstance->materialPipeline)
            {
                lastPipeline = robj.materialInstance->materialPipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, robj.materialInstance->materialPipeline->pipeline);
                VkDescriptorSet sceneDescriptorSet = engine->getSceneBufferDescriptorSet();
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, robj.materialInstance->materialPipeline->layout, 0, 1, &sceneDescriptorSet, 0, nullptr);

                // Set dynamic viewport and scissor again in case of an override (all of the material pipelines use dynamic states so setting them once after a bind is actually enough)
                engine->setViewport(cmd);
                engine->setScissor(cmd);
            }

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, robj.materialInstance->materialPipeline->layout, 1, 1, &robj.materialInstance->materialSet, 0, nullptr);
        }

        GPUDrawPushConstants pushConstants;
        pushConstants.vertexBufferAddress = robj.vertexBufferAddress;
        pushConstants.worldMatrix = robj.transform;
        vkCmdPushConstants(cmd, robj.materialInstance->materialPipeline->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

        if(lastIndexBuffer != robj.indexBuffer)
        {
            lastIndexBuffer = robj.indexBuffer;
            vkCmdBindIndexBuffer(cmd, robj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        vkCmdDrawIndexed(cmd, robj.indexCount, 1, robj.firstIndex, 0, 0);
        };

    for(uint32_t idx : opaqueDraws)
    {
        draw(ctx->opaqueSurfaces[idx]);
    }

    for(const RenderObject& robj : ctx->transparentSurfaces)
    {
        draw(robj);
    }
}

void GLTFMetallicPass::update()
{
}
