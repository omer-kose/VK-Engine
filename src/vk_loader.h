#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"

#include <unordered_map>
#include <filesystem>

class VulkanEngine;

// Any non-specific GLTFMaterial Instance
struct GLTFMaterialInstance
{
	MaterialInstance instance;
};

// GLTF Primitive (Surface)
struct GeoSurface
{
	// starting index and 
	uint32_t startIndex;
	uint32_t count;
	
	// Each surface has its own material instance (different parts of a mesh can have different materials)
	std::shared_ptr<GLTFMaterialInstance> materialInstance; 
};

// GLTF Mesh Asset
struct MeshAsset
{
	std::string name;
	// Each mesh consists of one or more surfaces 
	std::vector<GeoSurface> surfaces;
	GPUMeshBuffers meshBuffers;
};

struct LoadedGLTF : public IRenderable
{
	// Storage for all the data given in the gltf file
	std::unordered_map<std::string, std::shared_ptr<MeshAsset>> meshes;
	std::unordered_map<std::string, std::shared_ptr<SceneNode>> sceneNodes;
	std::unordered_map<std::string, AllocatedImage> textures;
	std::unordered_map<std::string, std::shared_ptr<GLTFMaterialInstance>> materialInstances;

	// Nodes that don't have a parent, for iterating through the file in tree order.
	std::vector<std::shared_ptr<SceneNode>> topNodes;

	std::vector<VkSampler> samplers;

	DescriptorAllocatorGrowable descriptorAllocator;

	// All the MaterialConstants data is held in a single buffer contiguously
	AllocatedBuffer materialDataBuffer;

	VulkanEngine* engine;

	~LoadedGLTF() { clearAll(); };

	virtual void registerDraw(const glm::mat4& topMatrix, DrawContext& ctx);

private:
	void clearAll();
};

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath);

// Aside from debugging this is not used as it is only used to load the meshes directly. 
std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);