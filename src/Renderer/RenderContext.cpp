#include "RenderContext.h"

#include <cassert>

SK::Renderer::PipelineHandle SK::Renderer::getGraphicsPipeline(RenderContext* renderContext, const GraphicsPipelineDesc& desc)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->getGraphicsPipeline != nullptr);

	return renderContext->api->getGraphicsPipeline(renderContext, desc);
}

SK::Renderer::PipelineHandle SK::Renderer::getComputePipeline(RenderContext* renderContext, const ComputePipelineDesc& desc)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->getComputePipeline != nullptr);

	return renderContext->api->getComputePipeline(renderContext, desc);
}

SK::Renderer::BufferDeviceAddress SK::Renderer::getVertexBufferDeviceAddress(RenderContext* renderContext, size_t meshIndex)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->getVertexBufferDeviceAddress != nullptr);

	return renderContext->api->getVertexBufferDeviceAddress(renderContext, meshIndex);
}

SK::Renderer::BufferHandle SK::Renderer::createBuffer(RenderContext* renderContext, const BufferDesc& desc)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->createBuffer != nullptr);
	
	return renderContext->api->createBuffer(renderContext, desc);
}

SK::Renderer::TextureHandle SK::Renderer::createTexture(RenderContext* renderContext, const TextureDesc& desc)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->createTexture != nullptr);

	return renderContext->api->createTexture(renderContext, desc);
}

void SK::Renderer::beginMainRendering(RenderContext* renderContext)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->beginMainRendering != nullptr);

	renderContext->api->beginMainRendering(renderContext);
}

void SK::Renderer::endRendering(RenderContext* renderContext)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->endRendering != nullptr);

	renderContext->api->endRendering(renderContext);
}

void SK::Renderer::bindPipeline(RenderContext* renderContext, PipelineHandle pipeline)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->bindPipeline != nullptr);
	assert(pipeline.id != INVALID_HANDLE);

	renderContext->api->bindPipeline(renderContext, pipeline);
}

void SK::Renderer::bindSceneResources(RenderContext* renderContext)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->bindSceneResources != nullptr);

	renderContext->api->bindSceneResources(renderContext);
}

void SK::Renderer::bindMaterialResources(RenderContext* renderContext)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->bindMaterialResources != nullptr);

	renderContext->api->bindMaterialResources(renderContext);
}

void SK::Renderer::pushConstants(RenderContext* renderContext, ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->pushConstants != nullptr);
	assert(data != nullptr || size == 0);

	renderContext->api->pushConstants(renderContext, stages, offset, size, data);
}

void SK::Renderer::bindIndexBuffer(RenderContext* renderContext, size_t meshIndex, IndexType indexType)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->bindIndexBuffer != nullptr);

	renderContext->api->bindIndexBuffer(renderContext, meshIndex, indexType);
}

void SK::Renderer::drawIndexed(RenderContext* renderContext, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->drawIndexed != nullptr);

	renderContext->api->drawIndexed(renderContext, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
}

void SK::Renderer::dispatch(RenderContext* renderContext, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ)
{
	assert(renderContext != nullptr);
	assert(renderContext->api != nullptr);
	assert(renderContext->api->dispatch != nullptr);

	renderContext->api->dispatch(renderContext, groupCountX, groupCountY, groupCountZ);
}