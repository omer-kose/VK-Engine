#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"

#include <unordered_map>
#include <filesystem>

// Material types
#include <Materials/GLTFMetallicMaterial.h>

class VulkanEngine;

// Bounds of a geometry. It both stores radius and extents. So, depending on the situation, a bounding box or a bounding sphere can be used
struct Bounds
{
	glm::vec3 origin; // origin of the bound
	float sphereRadius; // radius of the sphere
	glm::vec3 extents; // half edge lengths of the bounding box 
};

/* GLTF START */

// GLTF Primitive (Surface)
struct GLTFGeoSurface
{
	// starting index and 
	uint32_t startIndex;
	uint32_t count;
	
	Bounds bounds;

	// Each surface has its own material instance (different parts of a mesh can have different materials)
	std::shared_ptr<MaterialInstance> materialInstance; 
};

struct GLTFMeshAsset
{
	std::string name;
	// Each mesh consists of one or more surfaces 
	std::vector<GLTFGeoSurface> surfaces;
	GPUMeshBuffers meshBuffers;
};

struct GLTFMeshNode : public GLTFSceneNode
{
	std::shared_ptr<GLTFMeshAsset> mesh;

	// Creates a Render Object and adds all the surfaces in the mesh into the context's opaqueGLTFSurfaces
	virtual void registerDraw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

struct LoadedGLTF : public IRenderable
{
	// Storage for all the data given in the gltf file
	std::unordered_map<std::string, std::shared_ptr<GLTFMeshAsset>> meshes;
	std::unordered_map<std::string, std::shared_ptr<GLTFSceneNode>> sceneNodes;
	std::unordered_map<std::string, AllocatedImage> textures;
	std::unordered_map<std::string, std::shared_ptr<MaterialInstance>> materialInstances;

	// Nodes that don't have a parent, for iterating through the file in tree order.
	std::vector<std::shared_ptr<GLTFSceneNode>> topNodes;

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
std::optional<std::vector<std::shared_ptr<GLTFMeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath);

/* GLTF END */