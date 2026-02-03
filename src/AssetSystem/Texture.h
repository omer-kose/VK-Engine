#pragma once

#include <RendererBackend/vulkan/vk_types.h>

#include <string>

namespace SK::Asset
{
	struct Texture
	{
		AllocatedImage image;
		VkSampler sampler;
		std::string name;
	};
};