#include "ForwardRenderer.h"

#include <RendererBackend/Vulkan/VkRendererBackend.h>
#include <RendererBackend/Vulkan/VkPipelines.h>
#include <RendererBackend/Vulkan/VkInitializers.h>

#include <RendererBackend/Vulkan/VkAssetRegistry.h>
#include <RendererBackend/Vulkan/VkMaterialRegistry.h>

#include <Renderer/DrawContext.h>

#include <chrono>
#include <algorithm>

void SK::ForwardRenderer::init(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, SK::VkRendererBackend::VkMaterialRegistry* vkMaterialRegistry)
{
    // Load the shaders
    const char* vertexShaderPath = "../../shaders/glsl/forward/forward_vert.spv";
    VkShaderModule vertexShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, vertexShaderPath);
    if(!vertexShader)
    {
        fmt::println("Error when building the forward vertex shader");
    }

    const char* fragmentShaderPath = "../../shaders/glsl/forward/forward_frag.spv";
    VkShaderModule fragmentShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, fragmentShaderPath);;
    if(!fragmentShader)
    {
        fmt::println("Error when building the forward fragment shader");
    }

    // Set push constant range
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // Mesh pipeline layout
    SK::VkRendererBackend::PipelineLayoutKey layoutKey;
    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Bindless Resources Descriptor Set
    layoutKey.setLayouts = { vkRendererBackend->gpuSceneDataDescriptorLayout, vkMaterialRegistry->resourceDescriptorSetLayout };
    layoutKey.pushConstantRanges = { pushConstantRange };

    forwardRenderer->pipelineLayout = SK::VkRendererBackend::getOrCreatePipelineLayout(vkRendererBackend, layoutKey);

    // Build the pipeline keys and retrieve the pipelines from the vkRendererBackend backend
    size_t vertHash = std::hash<std::string>{}(vertexShaderPath);
    size_t fragHash = std::hash<std::string>{}(fragmentShaderPath);

    SK::VkRendererBackend::PipelineKey opaqueKey = {};
    opaqueKey.vertShader = vertHash;
    opaqueKey.fragShader = fragHash;
    opaqueKey.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    opaqueKey.polygonMode = VK_POLYGON_MODE_FILL;
    opaqueKey.cullMode = VK_CULL_MODE_NONE;
    opaqueKey.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    opaqueKey.depthTest = true;
    opaqueKey.depthWrite = true;
    opaqueKey.depthCompare = VK_COMPARE_OP_LESS_OR_EQUAL;
    opaqueKey.blending = false;
    opaqueKey.colorFormat = vkRendererBackend->drawImage.imageFormat;
    opaqueKey.depthFormat = vkRendererBackend->depthImage.imageFormat;
    opaqueKey.layout = forwardRenderer->pipelineLayout;

    forwardRenderer->opaquePipeline = SK::VkRendererBackend::getOrCreatePipeline(vkRendererBackend, opaqueKey);

    SK::VkRendererBackend::PipelineKey transparentKey = opaqueKey;
    transparentKey.blending = true;
    transparentKey.depthWrite = false;
    transparentKey.depthTest = false;

    forwardRenderer->transparentPipeline = SK::VkRendererBackend::getOrCreatePipeline(vkRendererBackend, transparentKey);
}

void SK::ForwardRenderer::draw(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, SK::VkRendererBackend::VkAssetRegistry* vkAssetRegistry, SK::VkRendererBackend::VkMaterialRegistry* vkMaterialRegistry, const SK::Renderer::DrawContext& ctx)
{
    VkCommandBuffer cmd = vkRendererBackend->currentCmdBuffer;

    // Begin a renderpass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = SK::VkInit::attachment_info(vkRendererBackend->drawImage.imageView, &vkRendererBackend->colorAttachmentClearValue, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = SK::VkInit::depth_attachment_info(vkRendererBackend->depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = SK::VkInit::rendering_info(vkRendererBackend->drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    auto start = std::chrono::system_clock::now();

    std::vector<uint32_t> opaqueDraws;
    opaqueDraws.reserve(ctx.opaque.size());

    for(uint32_t i = 0; i < ctx.opaque.size(); ++i)
    {
        opaqueDraws.push_back(i);
    }

    // sort the opaque surfaces by mesh index to minimize index buffer bindings
    std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](const auto& iA, const auto& iB) {
        const SK::Renderer::DrawPacket& A = ctx.opaque[iA];
        const SK::Renderer::DrawPacket& B = ctx.opaque[iB];
        return A.meshIndex < B.meshIndex;
    });

    // Keep track of states to avoid unnecessary rebindings
    uint32_t lastMeshIndex = UINT_MAX;

    auto bindPipelineAndBindlessResources = [&](VkPipeline pipeline) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDescriptorSet gpuSceneDescriptorSet = SK::VkRendererBackend::fetchCurrentSceneBufferDescriptorSet(vkRendererBackend);;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardRenderer->pipelineLayout, 0, 1, &gpuSceneDescriptorSet, 0, nullptr);
        // Bind the Material + Texture resource descriptors once (bindless resources)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardRenderer->pipelineLayout, 1, 1, &vkMaterialRegistry->resourceDescriptorSet, 0, nullptr);

        // Set dynamic viewport and scissor again in case of an override (setting them once while binding the pipeline is enough)
        SK::VkRendererBackend::setViewport(vkRendererBackend, cmd);
        SK::VkRendererBackend::setScissor(vkRendererBackend, cmd);
    };

    auto draw = [&](const SK::Renderer::DrawPacket& packet) {
        PushConstants pushConstants;
        pushConstants.vertexBufferAddress = vkAssetRegistry->meshes[packet.meshIndex].meshBuffers.vertexBufferAddress;
        pushConstants.worldMatrix = packet.worldTransform;
        pushConstants.materialIndex = packet.materialIndex;
        vkCmdPushConstants(cmd, forwardRenderer->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstants), &pushConstants);

        if(lastMeshIndex != packet.meshIndex)
        {
            lastMeshIndex = packet.meshIndex;
            vkCmdBindIndexBuffer(cmd, vkAssetRegistry->meshes[packet.meshIndex].meshBuffers.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        }

        vkCmdDrawIndexed(cmd, packet.indexCount, 1, packet.startIndex, 0, 0);
    };

    bindPipelineAndBindlessResources(forwardRenderer->opaquePipeline);
    for(uint32_t idx : opaqueDraws)
    {
        draw(ctx.opaque[idx]);
    }

    bindPipelineAndBindlessResources(forwardRenderer->transparentPipeline);
    for(const SK::Renderer::DrawPacket& packet : ctx.transparent)
    {
        draw(packet);
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    vkRendererBackend->stats.geometryDrawRecordTime = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void SK::ForwardRenderer::shutdown(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend)
{

}