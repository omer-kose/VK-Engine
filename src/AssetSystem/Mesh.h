#pragma once

#include <RendererBackend/vulkan/vk_types.h>

#include <string>

namespace SK::Asset
{
	// Bounds of a mesh geometry. It both stores radius and extents. So, depending on the situation, a bounding box or a bounding sphere can be used
	struct MeshBounds
	{
		glm::vec3 origin; // origin of the bound
		float sphereRadius; // radius of the sphere
		glm::vec3 extents; // half edge lengths of the bounding box 
	};

	struct SubMesh
	{
		// starting index and count into the index buffer
		uint32_t startIndex;
		uint32_t count;

		MeshBounds bounds;

		// TODO: Add material instance pointer. Each submesh will have its own material
	};

	struct Mesh
	{
		GPUMeshBuffers meshBuffers;
		// A mesh can consist of multiple sub meshes. The size of this array is always >=1 
		std::vector<SubMesh> subMeshes;
		std::string name;
	};
};