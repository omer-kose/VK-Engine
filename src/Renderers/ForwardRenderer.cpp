#include "ForwardRenderer.h"

#include <RendererBackend/vulkan/vk_renderer.h>
#include <RendererBackend/vulkan/vk_pipelines.h>
#include <RendererBackend/vulkan/vk_initializers.h>


void SK::ForwardRenderer::init(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend)
{
    // Load the shaders
    VkShaderModule meshVertexShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, "../../shaders/glsl/gltf_metallic/mesh_vert.spv");
    if(!meshVertexShader)
    {
        fmt::println("Error when building the mesh vertex shader");
    }

    VkShaderModule meshFragmentShader = SK::VkRendererBackend::getOrLoadShader(vkRendererBackend, "../../shaders/glsl/gltf_metallic/mesh_frag.spv");;
    if(!meshFragmentShader)
    {
        fmt::println("Error when building the mesh fragment shader");
    }

    // Set push constant range
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(GPUDrawPushConstants);
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // Set descriptor sets
    // Material set (set 1)
    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    VkDescriptorSetLayout materialLayout = layoutBuilder.build(vkRendererBackend->device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    // Mesh pipeline layout
    SK::VkRendererBackend::PipelineLayoutKey layoutKey;
    // 2 sets: 0 -> Scene Descriptor Set, 1 -> Material Descriptor Set
    layoutKey.setLayouts = { vkRendererBackend->gpuSceneDataDescriptorLayout, materialLayout };
    layoutKey.pushConstantRanges = { pushConstantRange };

    forwardRenderer->pipelineLayout = SK::VkRendererBackend::getOrCreatePipelineLayout(vkRendererBackend, layoutKey);

    // Build the pipeline keys and retrieve the pipelines from the vkRendererBackend backend
    size_t vertHash = std::hash<std::string>{}("../../shaders/glsl/gltf_metallic/mesh_vert.spv");
    size_t fragHash = std::hash<std::string>{}("../../shaders/glsl/gltf_metallic/mesh_frag.spv");

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

    // Descriptor Set Layout is not needed as Material descriptors will be created while getting instanced.
    vkDestroyDescriptorSetLayout(vkRendererBackend->device, materialLayout, nullptr);
}

void SK::ForwardRenderer::draw(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, const SK::VkRendererBackend::DrawContext& ctx)
{
    VkCommandBuffer cmd = vkRendererBackend->currentCmdBuffer;

    // Begin a renderpass connected to the draw image
    VkRenderingAttachmentInfo colorAttachment = SK::VkInit::attachment_info(vkRendererBackend->drawImage.imageView, &vkRendererBackend->colorAttachmentClearValue, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo depthAttachment = SK::VkInit::depth_attachment_info(vkRendererBackend->depthImage.imageView, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL);

    VkRenderingInfo renderInfo = SK::VkInit::rendering_info(vkRendererBackend->drawExtent, &colorAttachment, &depthAttachment);
    vkCmdBeginRendering(cmd, &renderInfo);

    auto start = std::chrono::system_clock::now();

    std::vector<uint32_t> opaqueDraws;
    opaqueDraws.reserve(ctx.opaqueGLTFSurfaces.size());

    for(uint32_t i = 0; i < ctx.opaqueGLTFSurfaces.size(); ++i)
    {
        opaqueDraws.push_back(i);
    }

    // sort the opaque surfaces by material and mesh
    std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](const auto& iA, const auto& iB) {
        const SK::VkRendererBackend::RenderObject& A = ctx.opaqueGLTFSurfaces[iA];
        const SK::VkRendererBackend::RenderObject& B = ctx.opaqueGLTFSurfaces[iB];
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
    MaterialInstance* lastMaterial = nullptr;
    VkBuffer lastIndexBuffer = VK_NULL_HANDLE;

    auto draw = [&](const SK::VkRendererBackend::RenderObject& robj, VkPipeline pipeline) {
        if(robj.materialInstance != lastMaterial)
        {
            lastMaterial = robj.materialInstance;
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkDescriptorSet gpuSceneDescriptorSet = SK::VkRendererBackend::fetchCurrentSceneBufferDescriptorSet(vkRendererBackend);;
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardRenderer->pipelineLayout, 0, 1, &gpuSceneDescriptorSet, 0, nullptr);

            // Set dynamic viewport and scissor again in case of an override (all of the material pipelines use dynamic states so setting them once after a bind is actually enough)
            SK::VkRendererBackend::setViewport(vkRendererBackend, cmd);
            SK::VkRendererBackend::setScissor(vkRendererBackend, cmd);

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, forwardRenderer->pipelineLayout, 1, 1, &robj.materialInstance->materialSet, 0, nullptr);
        }

        GPUDrawPushConstants pushConstants;
        pushConstants.vertexBufferAddress = robj.vertexBufferAddress;
        pushConstants.worldMatrix = robj.transform;
        vkCmdPushConstants(cmd, forwardRenderer->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GPUDrawPushConstants), &pushConstants);

        if(lastIndexBuffer != robj.indexBuffer)
        {
            lastIndexBuffer = robj.indexBuffer;
            vkCmdBindIndexBuffer(cmd, robj.indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        }

        vkCmdDrawIndexed(cmd, robj.indexCount, 1, robj.firstIndex, 0, 0);
    };

    for(uint32_t idx : opaqueDraws)
    {
        draw(ctx.opaqueGLTFSurfaces[idx], forwardRenderer->opaquePipeline);
    }

    for(const SK::VkRendererBackend::RenderObject& robj : ctx.transparentGLTFSurfaces)
    {
        draw(robj, forwardRenderer->transparentPipeline);
    }

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    vkRendererBackend->stats.geometryDrawRecordTime = elapsed.count() / 1000.f;

    vkCmdEndRendering(cmd);
}

void SK::ForwardRenderer::shutdown(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend)
{

}