#pragma once
#include <Core/vk_types.h>

// Forward declare with the namespace
namespace SK::VkRenderer
{
	struct Renderer;
	struct RenderObject;
};

class GLTFMetallicPass
{
public:
	static void Init(SK::VkRenderer::Renderer* renderer);
	static void Execute(SK::VkRenderer::Renderer* renderer, VkCommandBuffer& cmd);
	static void Update();
	static void ClearResources(SK::VkRenderer::Renderer* renderer);
private:
	static VkPipeline OpaquePipeline;
	static VkPipeline TransparentPipeline;
	static VkPipelineLayout PipelineLayout; // both transparent and opaque objects use the same pipeline layout
};