#pragma once

#include <string>
#include <vector>
#include <cstdint>

// By the end of implementation, Asset System should be totally independent of vk_types.h. The types like Vertex etc. are not Vulkan specific but for now they stay
#include <RendererBackend/vulkan/vk_types.h>

/*
	Asset System is API-agnostic. It only holds raw data + metadata. It follows retention policy. Unnecessary CPU data will be discarded after uploaded to the GPU.
*/

namespace SK::Asset
{
	enum class CPURetention : uint8_t
	{
		DropAfterUpload = 0,
		KeepOnCPU
	};

	enum class TextureFilter : uint8_t
	{
		NEAREST = 0,
		LINEAR
	};

	enum class TextureMipmapMode : uint8_t
	{
		NEAREST = 0,
		LINEAR
	};

	enum class TextureAddressMode : uint8_t
	{
		REPEAT = 0, // wrap
		CLAMP_TO_EDGE = 1
	};

	struct RawImage
	{
		std::vector<uint8_t> data;
		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t channels = 0;
	};

	struct TextureDescription
	{
		bool mipmapped = false;
		TextureFilter minFilter = TextureFilter::LINEAR;
		TextureFilter magFilter = TextureFilter::LINEAR;
		TextureMipmapMode mipmapMode = TextureMipmapMode::LINEAR;
		TextureAddressMode addressMode = TextureAddressMode::REPEAT;
	};

	struct RawTexture
	{
		RawImage image;
		TextureDescription description;
		CPURetention retention = CPURetention::DropAfterUpload;
		std::string name;
	};

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
		uint32_t startIndex = 0;
		uint32_t count = 0;
		MeshBounds bounds;
		uint32_t materialIndex = UINT32_MAX; // optional, filled later
	};

	struct RawMesh
	{
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		std::vector<SubMesh> subMeshes;
		CPURetention retention = CPURetention::DropAfterUpload;
		std::string name;
	};
}