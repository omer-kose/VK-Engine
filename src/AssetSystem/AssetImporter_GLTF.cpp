#include "AssetImporter_GLTF.h"

#include <fmt/core.h>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

#include <RendererBackend/vulkan/vk_renderer.h>
#include "AssetRegistry.h"

#include "stb_image.h"

// TODO: Rename this back once I get rid of vk_loader
std::optional<AllocatedImage> loadImageTemp(SK::VkRendererBackend::State* vkRendererBackend, fastgltf::Asset& asset, fastgltf::Image& image)
{
	AllocatedImage newImage = {};

	int width, height, nrChannels;

	std::visit(fastgltf::visitor{
		[](auto& arg) {},
		[&](fastgltf::sources::URI& filePath) {
			assert(filePath.fileByteOffset == 0); // offsets are not supported with stbi
			assert(filePath.uri.isLocalPath()); // only supporting loading local files

			const std::string path(filePath.uri.path().begin(), filePath.uri.path().end());
			unsigned char* data = stbi_load(path.c_str(), &width, &height, &nrChannels, 4);
			if(data)
			{
				VkExtent3D imageSize;
				imageSize.width = width;
				imageSize.height = height;
				imageSize.depth = 1;

				newImage = SK::VkRendererBackend::createImage(vkRendererBackend, data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

				stbi_image_free(data);
			}
		},
		[&](fastgltf::sources::Vector& vector) {
			unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
			if(data)
			{
				VkExtent3D imageSize;
				imageSize.width = width;
				imageSize.height = height;
				imageSize.depth = 1;

				newImage = SK::VkRendererBackend::createImage(vkRendererBackend, data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

				stbi_image_free(data);
			}
		},
		[&](fastgltf::sources::BufferView& view) {
			auto& bufferView = asset.bufferViews[view.bufferViewIndex];
			auto& buffer = asset.buffers[bufferView.bufferIndex];

			std::visit(fastgltf::visitor{
				[](auto& args) {},
				[&](fastgltf::sources::Vector& vector) { // only VectorWithMime is processed as during the load LoadExternalBuffers is specified meaning all the external buffers are already loaded into vector
					unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
					if(data)
					{
						VkExtent3D imageSize;
						imageSize.width = width;
						imageSize.height = height;
						imageSize.depth = 1;

						newImage = SK::VkRendererBackend::createImage(vkRendererBackend, data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

						stbi_image_free(data);
					}
				}
			}, buffer.data);
		}
		}, image.data);

	// check if any of the attempts to load the data is failed
	if(newImage.image == VK_NULL_HANDLE)
	{
		return {};
	}
	else
	{
		return newImage;
	}
}

// TODO: Rename this back once I get rid of vk_loader
VkFilter extractFilterTemp(fastgltf::Filter filter)
{
	switch(filter)
	{
		// nearest samplers
		case fastgltf::Filter::Nearest:
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::NearestMipMapLinear:
			return VK_FILTER_NEAREST;

		//linear samplers
		case fastgltf::Filter::Linear:
		case fastgltf::Filter::LinearMipMapNearest:
		case fastgltf::Filter::LinearMipMapLinear:
		default:
			return VK_FILTER_LINEAR;
	}
}

// TODO: Rename this back once I get rid of vk_loader
VkSamplerMipmapMode extractMipmapModeTemp(fastgltf::Filter filter)
{
	switch(filter)
	{
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::LinearMipMapNearest:
			return VK_SAMPLER_MIPMAP_MODE_NEAREST;

		case fastgltf::Filter::NearestMipMapLinear:
		case fastgltf::Filter::LinearMipMapLinear:
		default:
			return VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
}

//bool SK::Asset::loadGLTFData(SK::VkRendererBackend::State* vkRendererBackend, SK::Asset::AssetRegistry* assetRegistry, std::string_view filePath)
//{
//	fmt::println("Loading GLTF: {}", filePath);
//	std::string_view gltfFileName = filePath.substr(filePath.find_last_of("/") + 1);
//	// Get rid of the extension name in the file name
//	gltfFileName = gltfFileName.substr(0, gltfFileName.find_last_of("."));
//
//	fastgltf::Parser parser{};
//
//	constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;
//
//	fastgltf::GltfDataBuffer data;
//	data.loadFromFile(filePath);
//
//	fastgltf::Asset asset;
//
//	std::filesystem::path path = filePath;
//
//	auto type = fastgltf::determineGltfFileType(&data);
//	if(type == fastgltf::GltfType::glTF)
//	{
//		auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
//		if(load)
//		{
//			asset = std::move(load.get());
//		}
//		else
//		{
//			fmt::println("Failed to load GLTF: {}", fastgltf::to_underlying(load.error()));
//			return false;
//		}
//	}
//	else if(type == fastgltf::GltfType::GLB)
//	{
//		auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
//		if(load)
//		{
//			asset = std::move(load.get());
//		}
//		else
//		{
//			fmt::println("Failed to load GLB: {}", fastgltf::to_underlying(load.error()));
//			return false;
//		}
//	}
//	else
//	{
//		fmt::println("Failed to determine GLTF container");
//		return false;
//	}
//
//	// Load the textures
//	std::vector<VkSampler> samplers;
//	samplers.reserve(asset.samplers.size());
//	for(const fastgltf::Sampler& gltfSampler : asset.samplers)
//	{
//		VkSamplerCreateInfo samplerInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr };
//		samplerInfo.minLod = 0;
//		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
//
//		samplerInfo.minFilter = extractFilterTemp(gltfSampler.minFilter.value_or(fastgltf::Filter::Nearest));
//		samplerInfo.magFilter = extractFilterTemp(gltfSampler.magFilter.value_or(fastgltf::Filter::Nearest));
//
//		samplerInfo.mipmapMode = extractMipmapModeTemp(gltfSampler.minFilter.value_or(fastgltf::Filter::Nearest));
//
//		VkSampler sampler;
//		vkCreateSampler(vkRendererBackend->device, &samplerInfo, nullptr, &sampler);
//
//		samplers.push_back(sampler);
//		assetRegistry->samplers.push_back(sampler);
//	}
//	
//	for(size_t i = 0; i < asset.textures.size(); ++i)
//	{
//		SK::Asset::Texture texture;
//
//		const fastgltf::Texture& gltfTexture = asset.textures[i];
//		
//		// Name
//		if(!gltfTexture.name.empty())
//		{
//			texture.name = gltfTexture.name;
//		}
//		else
//		{
//			texture.name = fmt::format("gltf_{}_texture_{}", gltfFileName, i);
//		}
//
//		// Sampler
//		if(gltfTexture.samplerIndex.has_value())
//		{
//			texture.sampler = samplers[gltfTexture.samplerIndex.value()];
//		}
//		else
//		{
//			// Default nearest sampler from renderer backend
//			texture.sampler = vkRendererBackend->defaultSamplerNearest;
//		}
//
//		if(gltfTexture.imageIndex.has_value())
//		{
//			fastgltf::Image& gltfImage = asset.images[gltfTexture.imageIndex.value()];
//			auto loadedImage = loadImageTemp(vkRendererBackend, asset, gltfImage);
//			texture.image = loadedImage.value();
//			if(!loadedImage.has_value())
//			{
//				fmt::println("Failed to image for texture {}", texture.name);
//				// Failed to load, assign the default error texture
//				texture.image = vkRendererBackend->errorCheckerboardImage;
//			}
//		}
//		else
//		{
//			fmt::println("Texture {} has no image", texture.name);
//			// Failed to load, assign the default error texture
//			texture.image = vkRendererBackend->errorCheckerboardImage;
//		}
//
//		// Register in asset registry
//		uint32_t textureIndex = static_cast<uint32_t>(assetRegistry->textures.size());
//		assetRegistry->textureIndexByName[texture.name] = textureIndex;
//		assetRegistry->textures.push_back(std::move(texture));
//	}
//
//
//    // Load the meshes
//    // Temporary CPU buffers reused for each mesh
//    std::vector<uint32_t> indices;
//    std::vector<Vertex> vertices;
//
//    for(size_t i = 0; i < asset.meshes.size(); ++i)
//    {
//		fastgltf::Mesh& gltfMesh = asset.meshes[i];
//
//        SK::Asset::Mesh mesh{};
//		if(!gltfMesh.name.empty())
//		{
//			mesh.name = gltfMesh.name;
//		}
//		else
//		{
//			mesh.name = fmt::format("gltf_{}_mesh_{}", gltfFileName, i);
//		}
//		
//        indices.clear();
//        vertices.clear();
//		mesh.subMeshes.clear();
//
//        for(auto& primitive : gltfMesh.primitives)
//        {
//            SK::Asset::SubMesh subMesh{};
//
//            subMesh.startIndex = static_cast<uint32_t>(indices.size());
//
//            const size_t initialVertex = vertices.size();
//
//            // Load indices
//            {
//                // For any GLTF mesh, index buffer must exist
//                if(primitive.indicesAccessor.has_value())
//                {
//                    fastgltf::Accessor& indexAccessor = asset.accessors[primitive.indicesAccessor.value()];
//                    subMesh.count = static_cast<uint32_t>(indexAccessor.count);
//
//                    // TODO: This can be optimized by computing the total index count before the primitive loop
//                    indices.reserve(indices.size() + indexAccessor.count);
//
//                    fastgltf::iterateAccessor<uint32_t>(asset, indexAccessor, [&](uint32_t idx) {
//                        indices.push_back(idx + static_cast<uint32_t>(initialVertex));
//                    });
//                }
//                else
//                {
//                    fmt::println("Index accessor doesn't exist for a primitive in mesh: {}", mesh.name);
//                    return false;
//                }
//            }
//
//            // Load positions
//            {
//                auto posIt = primitive.findAttribute("POSITION");
//                // For any GLTF mesh, position attribute must exist
//                if(posIt != primitive.attributes.end())
//                {
//                    fastgltf::Accessor& posAccessor = asset.accessors[posIt->second];
//
//                    const size_t vertexOffset = vertices.size();
//                    // TODO: This can be optimized by computing the total vertex count before the primitive loop
//                    vertices.resize(vertices.size() + posAccessor.count);
//
//                    fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor, [&](glm::vec3 v, size_t index) {
//                        Vertex& out = vertices[vertexOffset + index];
//                        out.position = v;
//                        // Zero initialize other fields
//                        out.normal = { 0, 0, 0 };
//                        out.color = glm::vec4{ 0.0f };
//                        out.uv_x = 0;
//                        out.uv_y = 0;
//                    });
//                }
//                else
//                {
//                    fmt::println("Position attribute doesn't exist for a primitive in mesh: {}", mesh.name);
//                    return false;
//                }
//            }
//
//            // Load normals
//            if(auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end())
//            {
//                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[it->second], [&](glm::vec3 v, size_t index) {
//                     vertices[initialVertex + index].normal = v;
//                });
//            }
//
//            // Load UVs
//            if(auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end())
//            {
//                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[it->second], [&](glm::vec2 v, size_t index) {
//                    vertices[initialVertex + index].uv_x = v.x;
//                    vertices[initialVertex + index].uv_y = v.y;
//                });
//            }
//
//            // Load colors
//            if(auto it = primitive.findAttribute("COLOR_0"); it != primitive.attributes.end())
//            {
//                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[it->second], [&](glm::vec4 v, size_t index) {
//                    vertices[initialVertex + index].color = v;
//                });
//            }
//
//            // TODO: Assign a material to the submesh later
//
//            // Compute bounds
//            {
//                glm::vec3 minPos = vertices[initialVertex].position;
//                glm::vec3 maxPos = minPos;
//
//                // TODO: This loop will change a bit when the vertex array is initialized before the primitive loop. 
//                for(size_t i = initialVertex; i < vertices.size(); ++i)
//                {
//                    minPos = glm::min(minPos, vertices[i].position);
//                    maxPos = glm::max(maxPos, vertices[i].position);
//                }
//
//                subMesh.bounds.origin = (minPos + maxPos) * 0.5f;
//                subMesh.bounds.extents = (maxPos - minPos) * 0.5f;
//                subMesh.bounds.sphereRadius = glm::length(subMesh.bounds.extents);
//            }
//
//			mesh.subMeshes.push_back(subMesh);
//        }
//
//        // Upload to GPU
//		mesh.meshBuffers = SK::VkRendererBackend::uploadMesh(vkRendererBackend, vertices, indices);
//
//        // Register in AssetRegistry
//        uint32_t meshIndex = static_cast<uint32_t>(assetRegistry->meshes.size());
//        assetRegistry->meshIndexByName[mesh.name] = meshIndex;
//        assetRegistry->meshes.push_back(std::move(mesh));
//    }
//
//	return true;
//}