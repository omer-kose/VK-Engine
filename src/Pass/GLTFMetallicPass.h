#pragma once
#include "PassBase.h"

struct RenderObject;

class GLTFMetallicPass : public GraphicsPassBase
{
public:
	virtual void execute(VulkanEngine* engine, VkCommandBuffer& cmd, DrawContext* ctx) override;
	virtual void update() override;
};