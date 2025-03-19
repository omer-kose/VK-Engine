#include <vk_loader.h>

#include "stb_image.h"
#include <iostream>

#include "vk_engine.h"
#include "vk_initializers.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
{
	std::cout << "Loading GLTF: " << filePath << std::endl;

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(filePath);

	constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

	fastgltf::Asset asset;
	fastgltf::Parser parser{};
	// loadBinaryGLTF requires the parent path tgo find relative paths
	auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
	if(load)
	{
		asset = std::move(load.get());
	}
	else
	{
		fmt::print("Failed to load gltf: {} \n", fastgltf::to_underlying(load.error()));
		return {};
	}

	std::vector<std::shared_ptr<MeshAsset>> meshes;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for(fastgltf::Mesh& mesh : asset.meshes)
	{
		MeshAsset newMesh;
		newMesh.name = mesh.name;
		
		vertices.clear();
		indices.clear();

		for(auto&& p : mesh.primitives)
		{
			GeoSurface newSurface;
			newSurface.startIndex = (uint32_t)indices.size();
			newSurface.count = (uint32_t)asset.accessors[p.indicesAccessor.value()].count;

			size_t initialVertex = vertices.size();
			
			// load indices
			{
				fastgltf::Accessor& indexAccessor = asset.accessors[p.indicesAccessor.value()];
				indices.reserve(indices.size() + indexAccessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(asset, indexAccessor, [&](std::uint32_t idx){
					indices.push_back(initialVertex + idx);
				});
			}

			// load vertex positions (which will always exist)
			{
				fastgltf::Accessor& posAccessor = asset.accessors[p.findAttribute("POSITION")->second];
				vertices.resize(vertices.size() + posAccessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor, 
					[&](glm::vec3 v, size_t index) {
						Vertex newVertex;
						newVertex.position = v;
						// default the other attributes
						newVertex.normal = glm::vec3(1.0f, 0.0f, 0.0f);
						newVertex.color = glm::vec4(1.0f);
						newVertex.uv_x = 0.0f;
						newVertex.uv_y = 0.0f;
						vertices[initialVertex + index] = newVertex;
				});
			}

			// The remaining attributes may or may not exist
			// load vertex normals
			auto normals = p.findAttribute("NORMAL");
			if(normals != p.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[normals->second],
					[&](glm::vec3 n, size_t index) {
						vertices[initialVertex + index].normal = n;
				});	
			}

			// load UVs
			auto uv = p.findAttribute("TEXCOORD_0");
			if(uv != p.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[uv->second],
					[&](glm::vec2 uv, size_t index) {
						vertices[initialVertex + index].uv_x = uv.x;
						vertices[initialVertex + index].uv_y = uv.y;
					});
			}

			// load vertex colors
			auto colors = p.findAttribute("COLOR_0");
			if(colors != p.attributes.end())
			{
				fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[colors->second],
					[&](glm::vec4 c, size_t index) {
						vertices[initialVertex + index].color = c;
					});
			}

			newMesh.surfaces.push_back(newSurface);
		}
		
		// With OverrideColors flag true, override vertex colors with normals which is useful for debugging
		constexpr bool OverrideColors = false;
		if(OverrideColors)
		{
			for(Vertex& v : vertices)
			{
				v.color = glm::vec4(v.normal, 1.0f);
			}
		}

		newMesh.meshBuffers = engine->uploadMesh(vertices, indices);
		
		meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
	}

	return meshes;
}
