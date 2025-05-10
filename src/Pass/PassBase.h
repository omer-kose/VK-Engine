#pragma once
#include <Core/vk_types.h>

class VulkanEngine;

/*
	The term pass is an abstraction, these are not actual render passes, the engine uses dynamic rendering.

	Each type of pass will have its own base class (might not be necessary but keeping it flexible for the time being. I cannot foresee what structure I will have in the future)
	Each pass will store its own needed variables for shaders and will define a pipeline and a pipeline layout.

	Passes can:
	1- define their pipeline and pipeline layout
	2- define any pass-specific descriptor set they will use
	3- Fetch the draw context from the engine pointer and read it
	4- Call any engine utility

	If a pass, need per-mesh specific descriptor set (such as material descriptor sets), they can fetch it from the MaterialInstance pointer in the RenderObject. 

	TODO: Consider making these Pass classes static. All the Passes should have the same behaviour. There is no need to instantiate them and hold them as a engine member variable.

	For example:
	GLTFMaterialPass::Init(engine)
	GLTFMaterialPass::Execute(engine, cmd)


	Even though some passes will hold some variables like textures (like a skybox or shadow pass), the pass is done once per frame for all the objects in the DrawContext. So, the resources are used for that sole pass.
	No need for instancing.

	Start the static logic here, if it works well, consider making material types static classes as well.
*/

/*
	Abstract Base class for Graphics Passes. 
*/
class GraphicsPassBase
{
public:
	virtual void init(VulkanEngine* engine) = 0;
	virtual void execute(VulkanEngine* engine, VkCommandBuffer& cmd) = 0;
	virtual void update() = 0;
	virtual void clearResources(VulkanEngine* engine) = 0; // This instead of destructor because passes contains VkPipelines where the order of destruction matters. So, when to call clearResources should be decided by me.
	virtual ~GraphicsPassBase() = default;
};