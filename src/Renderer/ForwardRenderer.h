#pragma once
#include <RendererBackend/vulkan/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRendererBackend
{
	struct State;
	struct VkAssetRegistry;
	struct VkMaterialRegistry;
};

namespace SK::Renderer
{
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

	// Push constants for mesh draws
	struct PushConstants
	{
		glm::mat4 worldMatrix;
		VkDeviceAddress vertexBufferAddress;
		uint32_t materialIndex;
	};

	void init(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, SK::VkRendererBackend::VkMaterialRegistry* vkMaterialRegistry);
	void draw(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend, SK::VkRendererBackend::VkAssetRegistry* vkAssetRegistry, SK::VkRendererBackend::VkMaterialRegistry* vkMaterialRegistry, const SK::Renderer::DrawContext & ctx);
	void shutdown(State* forwardRenderer, SK::VkRendererBackend::State* vkRendererBackend);
};