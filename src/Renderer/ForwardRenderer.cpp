#include "ForwardRenderer.h"

#include <RendererBackend/Vulkan/VkRendererBackend.h>
#include <RendererBackend/Vulkan/VkPipelines.h>
#include <RendererBackend/Vulkan/VkInitializers.h>

#include <RendererBackend/Vulkan/VkAssetRegistry.h>
#include <RendererBackend/Vulkan/VkMaterialRegistry.h>

#include <Renderer/DrawContext.h>

#include <chrono>
#include <algorithm>

void SK::ForwardRenderer::createResources(SK::Renderer::RenderContext* renderContext, Resources* resources)
{
	SK::Renderer::GraphicsPipelineDesc opaqueDesc{};
	opaqueDesc.debugName = "Forward Opaque";
	opaqueDesc.vertexShaderPath = "../../shaders/glsl/forward/forward_vert.spv";
	opaqueDesc.fragmentShaderPath = "../../shaders/glsl/forward/forward_frag.spv";
	opaqueDesc.topology = SK::Renderer::PrimitiveTopology::TriangleList;
	opaqueDesc.polygonMode = SK::Renderer::PolygonMode::Fill;
	opaqueDesc.cullMode = SK::Renderer::CullMode::Back;
	opaqueDesc.frontFace = SK::Renderer::FrontFace::CounterClockwise;
	opaqueDesc.depthTest = true;
	opaqueDesc.depthWrite = true;
	opaqueDesc.depthCompare = SK::Renderer::CompareOp::LessEqual;
	opaqueDesc.blending = false;
	opaqueDesc.pushConstantSize = sizeof(PushConstants);
	opaqueDesc.pushConstantStages = SK::Renderer::ShaderStageFlagBits::VertexShader | SK::Renderer::ShaderStageFlagBits::FragmentShader;
	opaqueDesc.usesSceneResources = true;
	opaqueDesc.usesMaterialResources = true;

	resources->opaquePipeline = SK::Renderer::getGraphicsPipeline(renderContext, opaqueDesc);

	SK::Renderer::GraphicsPipelineDesc transparentDesc = opaqueDesc;
	transparentDesc.debugName = "Forward Transparent";
	transparentDesc.depthTest = false;
	transparentDesc.depthWrite = false;
	transparentDesc.blending = true;

	resources->transparentPipeline = SK::Renderer::getGraphicsPipeline(renderContext, transparentDesc);
}

void SK::ForwardRenderer::draw(SK::Renderer::RenderContext* renderContext, const Resources& resources, const Input& input)
{
	SK::Renderer::beginMainRendering(renderContext);

	std::vector<uint32_t> opaqueDraws;
	opaqueDraws.reserve(input.drawContext->opaque.size());

	for (uint32_t i = 0; i < input.drawContext->opaque.size(); ++i)
	{
		opaqueDraws.push_back(i);
	}

	std::sort(opaqueDraws.begin(), opaqueDraws.end(), [&](uint32_t iA, uint32_t iB) {
		const SK::Renderer::DrawPacket& a = input.drawContext->opaque[iA];
		const SK::Renderer::DrawPacket& b = input.drawContext->opaque[iB];

		return a.meshIndex < b.meshIndex;
		});

	uint32_t lastMeshIndex = UINT32_MAX;

	auto bindCommonResources = [&]() {
		SK::Renderer::bindSceneResources(renderContext);
		SK::Renderer::bindMaterialResources(renderContext);
	};

	auto drawPacket = [&](const SK::Renderer::DrawPacket& packet) {
		PushConstants pushConstants{};
		pushConstants.worldMatrix = packet.worldTransform;
		pushConstants.materialIndex = packet.materialIndex;
		pushConstants.vertexBufferAddress = SK::Renderer::getVertexBufferDeviceAddress(renderContext, packet.meshIndex);

		SK::Renderer::pushConstants(renderContext, SK::Renderer::ShaderStageFlagBits::VertexShader | SK::Renderer::ShaderStageFlagBits::FragmentShader, 0, sizeof(PushConstants), &pushConstants);

		if (lastMeshIndex != packet.meshIndex)
		{
			lastMeshIndex = packet.meshIndex;
			SK::Renderer::bindIndexBuffer(renderContext, packet.meshIndex, SK::Renderer::IndexType::Uint32);
		}

		SK::Renderer::drawIndexed(renderContext, packet.indexCount, 1, packet.startIndex, 0, 0);
	};

	SK::Renderer::bindPipeline(renderContext, resources.opaquePipeline);
	bindCommonResources();

	for (uint32_t drawIndex : opaqueDraws)
	{
		drawPacket(input.drawContext->opaque[drawIndex]);
	}

	SK::Renderer::bindPipeline(renderContext, resources.transparentPipeline);
	bindCommonResources();

	for (const SK::Renderer::DrawPacket& packet : input.drawContext->transparent)
	{
		drawPacket(packet);
	}

	SK::Renderer::endRendering(renderContext);
}
