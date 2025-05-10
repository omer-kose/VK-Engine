#pragma once
#include "PassBase.h"

struct RenderObject;

class GLTFMetallicPass : public GraphicsPassBase
{
public:
	virtual void init(VulkanEngine* engine) override;
	virtual void execute(VulkanEngine* engine, VkCommandBuffer& cmd) override;
	virtual void update() override;
	virtual void clearResources(VulkanEngine* engine) override;
private:
	VkPipeline opaquePipeline;
	VkPipeline transparentPipeline;
	VkPipelineLayout pipelineLayout; // both transparent and opaque objects use the same pipeline layout
};