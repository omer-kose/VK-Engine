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

	struct BufferRecord
	{
		AllocatedBuffer buffer;
		const char* debugName = nullptr;
	};

	struct TextureRecord
	{
		AllocatedImage image;
		uint32_t samplerIndex; // The number of permutations for samplers is quite limited in real usage scenarios, so a 32 bit index is a bit overkill.
		const char* debugName = nullptr;
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
		// TODO: This is for reusing the pipelines while retrieving them in frontend renderers. However, the retrievers also store those pipeline handles so this might be an unnecessary book-keeping.
		std::unordered_map<size_t, uint64_t> pipelineIndexByHash;

		// Cache the current pipeline information for later use when a pipeline is bound.
		SK::Renderer::PipelineKind currentPipelineKind = SK::Renderer::PipelineKind::Graphics;
		VkPipelineLayout currentPipelineLayout = VK_NULL_HANDLE;

		std::vector<BufferRecord> buffers;
		std::vector<TextureRecord> textures;
		std::vector<VkSampler> samplers;
		// Sampler cache 
		std::unordered_map<size_t, uint32_t> samplerIndexByHash;
	};

	void initVkRenderContext(VkRenderContext* vkRenderContext, State* vkRendererBackend, VkSceneResources* vkSceneResources);
	SK::Renderer::RenderContext makeRenderContext(VkRenderContext* vkRenderContext);
	void clearVkRenderContext(VkRenderContext* vkRenderContext);
}