#pragma once
#include <Core/vk_types.h>

class VulkanEngine;

/*
	Each type of pass will have its own base class (might not be necessary but keeping it flexible for the time being. I cannot foresee what structure I will have in the future)
	Each pass will store its own needed parameters for shaders and will define a pipeline (thus all the necessary other constructs like descriptor layouts etc.)	
*/

/*
	Base class for Graphics Passes.

	Some graphics passes need the actual list of renderables (such as a standard forward pass) and some don't (such as a light pass in a deferred pipeline or a skybox pass). 
	There are two possible simple solutions to this in my mind:
	1- Execute just takes the engine and calls some utils such as engine->drawRenderObjects(cmd). All it has to do is to simply bind its own stuff and use engine for the basic draw call by calling it 
	2- Pass an optional DrawContext* (which stores all the renderables of the current frame). 

	First option is much simpler. However, it has no control over the order of encoding the draw instructions into the command buffer. For example, while rendering a whole scene with a material pass
	to minimize the state changes, the draw calls are sorted. When the option 1 is used, it cannot be used as engine common draw call will just draw without any particular order.

	Second option is more flexible. If the pass has no need for any objects to draw it can be passed as nullptr and the pass won't use any information. As for passes which will use objects to draw,
	they can have their own index list to reorder draw calls to minimize state changes. So, I am going with the second option.
*/
class GraphicsPassBase
{
public:
	virtual void execute(VulkanEngine* engine, VkCommandBuffer& cmd, DrawContext* ctx) = 0;
	virtual void update() = 0;
	virtual ~GraphicsPassBase() = default;
};