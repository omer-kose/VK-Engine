#pragma once
#include <RendererBackend/vulkan/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRendererBackend
{
	struct State;
	struct DrawContext;
};

namespace SK::ForwardRenderer
{
	struct State
	{
		VkPipelineLayout pipelineLayout; // both transparent and opaque objects use the same pipeline layout
		VkPipeline opaquePipeline;
		VkPipeline transparentPipeline;
	};

	void init(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend);
	void draw(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, const SK::VkRendererBackend::DrawContext& ctx);
	void shutdown(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend);
};