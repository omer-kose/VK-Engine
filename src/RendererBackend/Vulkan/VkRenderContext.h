#pragma once

#include <Renderer/RenderContext.h>

#include <RendererBackend/Vulkan/VkTypes.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace SK::VkRendererBackend
{
	struct State;
	struct VkSceneResources;

	struct PipelineRecord
	{
		VkPipeline pipeline = VK_NULL_HANDLE;
		VkPipelineLayout layout = VK_NULL_HANDLE;
		SK::Renderer::PipelineKind kind = SK::Renderer::PipelineKind::Graphics;
	};

	/*
		VkRenderContext provides a context for the backend. The engine frontend (RenderContext) will use the functionality provided by the backend via VkRenderContext bridge.
	*/
	struct VkRenderContext
	{
		State* vkRendererBackend = nullptr;
		VkSceneResources* sceneResources = nullptr;

		// VkRenderContext does not own actual handles. It just caches the handles for functionality. The creation and cleaning of the handles are always done by the Renderer Backend.
		std::vector<PipelineRecord> pipelines;
		std::unordered_map<size_t, uint64_t> pipelineIndexByHash;

		// Cache the current pipeline information for later use when a pipeline is bound.
		SK::Renderer::PipelineKind currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
		VkPipelineLayout currentPipelineLayout = VK_NULL_HANDLE;
	};

	void initVkRenderContext(VkRenderContext* vkRenderContext, State* vkRendererBackend, VkSceneResources* vkSceneResources);
	SK::Renderer::RenderContext makeRenderContext(VkRenderContext* vkRenderContext);
	void clearVkRenderContext(VkRenderContext* vkRenderContext);
}