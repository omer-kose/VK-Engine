#include "AssetImporter_GLTF.h"

#include <fmt/core.h>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

#include "AssetRegistry.h"

#include "stb_image.h"

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <glm/gtx/quaternion.hpp>

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

static bool hasMipmapFilter(fastgltf::Filter filter)
{
	switch(filter)
	{
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::LinearMipMapNearest:
		case fastgltf::Filter::NearestMipMapLinear:
		case fastgltf::Filter::LinearMipMapLinear:
			return true;
		default:
			return false;
	}
}

static std::optional<SK::Asset::RawImage> loadRawImageFromGLTF(const std::filesystem::path& basePath, fastgltf::Asset& asset, fastgltf::Image& image)
{
    int width = 0, height = 0, channels = 0;

    SK::Asset::RawImage out{};

    auto tryAssign = [&](unsigned char* data, int w, int h, int c) 
    {
        if(!data)
        {
            return false;
        }

        out.width = static_cast<uint32_t>(w);
        out.height = static_cast<uint32_t>(h);
        out.channels = 4; // force RGBA
        out.data.assign(data, data + (w * h * 4));
        stbi_image_free(data);
        return true;
    };

    bool loaded = false;

    std::visit(fastgltf::visitor
    {
        [](auto&) {},
        [&](fastgltf::sources::URI& filePath) 
        {
            assert(filePath.fileByteOffset == 0);
            assert(filePath.uri.isLocalPath());
            std::filesystem::path fullPath = basePath / filePath.uri.path();
            unsigned char* data = stbi_load(fullPath.string().c_str(), &width, &height, &channels, 4);
            loaded = tryAssign(data, width, height, channels);
        },
        [&](fastgltf::sources::Vector& vec) 
        {
            unsigned char* data = stbi_load_from_memory(vec.bytes.data(), static_cast<int>(vec.bytes.size()), &width, &height, &channels, 4);
            loaded = tryAssign(data, width, height, channels);
        },
        [&](fastgltf::sources::BufferView& view) 
        {
            auto& bufferView = asset.bufferViews[view.bufferViewIndex];
            auto& buffer = asset.buffers[bufferView.bufferIndex];

            std::visit(fastgltf::visitor
            {
                [](auto&) {},
                [&](fastgltf::sources::Vector& vec) 
                {
                    unsigned char* data = stbi_load_from_memory(vec.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &channels, 4);
                    loaded = tryAssign(data, width, height, channels);
                }
            }, buffer.data);
        }
    }, image.data);

    if(!loaded)
    {
        return {};
    }

    return out;
}

bool SK::Asset::importGLTF(std::string_view filePath, ImportedAsset* outAsset)
{
    if(!outAsset)
    {
        return false;
    }

    outAsset->meshes.clear();
    outAsset->textures.clear();
    outAsset->gltfScene.reset();

    fmt::println("Loading GLTF: {}", filePath);

    std::filesystem::path gltfPath = filePath;
    std::string_view gltfFileName = filePath.substr(filePath.find_last_of("/") + 1);
    gltfFileName = gltfFileName.substr(0, gltfFileName.find_last_of("."));

    fastgltf::Parser parser{};
    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    fastgltf::Asset asset;

    auto type = fastgltf::determineGltfFileType(&data);
    if(type == fastgltf::GltfType::glTF)
    {
        auto load = parser.loadGLTF(&data, gltfPath.parent_path(), gltfOptions);
        if(!load)
        {
            fmt::println("Failed to load GLTF: {}", fastgltf::to_underlying(load.error()));
            return false;
        }
        asset = std::move(load.get());
    }
    else if(type == fastgltf::GltfType::GLB)
    {
        auto load = parser.loadBinaryGLTF(&data, gltfPath.parent_path(), gltfOptions);
        if(!load)
        {
            fmt::println("Failed to load GLB: {}", fastgltf::to_underlying(load.error()));
            return false;
        }
        asset = std::move(load.get());
    }
    else
    {
        fmt::println("Failed to determine GLTF container");
        return false;
    }

    // Textures
    outAsset->textures.reserve(asset.textures.size());

    for(size_t i = 0; i < asset.textures.size(); ++i)
    {
        const fastgltf::Texture& gltfTexture = asset.textures[i];

        RawTexture texture{};
        texture.name = gltfTexture.name.empty() ? fmt::format("gltf_{}_texture_{}", gltfFileName, i) : gltfTexture.name.c_str();

        // Mipmap hint
        if(gltfTexture.samplerIndex.has_value())
        {
            const fastgltf::Sampler& sampler = asset.samplers[gltfTexture.samplerIndex.value()];
            if(sampler.minFilter.has_value())
            {
                texture.description.mipMapped = hasMipmapFilter(sampler.minFilter.value());
            }
        }

        if(gltfTexture.imageIndex.has_value())
        {
            fastgltf::Image& gltfImage = asset.images[gltfTexture.imageIndex.value()];
            auto img = loadRawImageFromGLTF(gltfPath.parent_path(), asset, gltfImage);
            if(img.has_value())
            {
                texture.image = std::move(img.value());
            }
            else
            {
                fmt::println("Failed to load image for texture: {}", texture.name);
            }
        }
        else
        {
            fmt::println("Texture {} has no image", texture.name);
        }

        outAsset->textures.push_back(std::move(texture));
    }

    // Meshes
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    outAsset->meshes.reserve(asset.meshes.size());

    for(size_t i = 0; i < asset.meshes.size(); ++i)
    {
        fastgltf::Mesh& gltfMesh = asset.meshes[i];

        RawMesh mesh{};
        mesh.name = gltfMesh.name.empty() ? fmt::format("gltf_{}_mesh_{}", gltfFileName, i) : gltfMesh.name.c_str();

        indices.clear();
        vertices.clear();
        mesh.subMeshes.clear();

        for(auto& primitive : gltfMesh.primitives)
        {
            SubMesh subMesh{};
            subMesh.startIndex = static_cast<uint32_t>(indices.size());

            const size_t initialVertex = vertices.size();

            // Indices
            if(primitive.indicesAccessor.has_value())
            {
                fastgltf::Accessor& indexAccessor = asset.accessors[primitive.indicesAccessor.value()];
                subMesh.count = static_cast<uint32_t>(indexAccessor.count);

                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<uint32_t>(asset, indexAccessor, [&](uint32_t idx) 
                {
                    indices.push_back(idx + static_cast<uint32_t>(initialVertex));
                });
            }
            else
            {
                fmt::println("Index accessor missing in mesh: {}", mesh.name);
                return false;
            }

            // Positions
            auto posIt = primitive.findAttribute("POSITION");
            if(posIt != primitive.attributes.end())
            {
                fastgltf::Accessor& posAccessor = asset.accessors[posIt->second];

                const size_t vertexOffset = vertices.size();
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor,
                    [&](glm::vec3 v, size_t index) 
                    {
                        Vertex& out = vertices[vertexOffset + index];
                        out.position = v;
                        out.normal = { 0, 0, 0 };
                        out.color = glm::vec4{ 0.0f };
                        out.uv_x = 0;
                        out.uv_y = 0;
                    });
            }
            else
            {
                fmt::println("Position attribute missing in mesh: {}", mesh.name);
                return false;
            }

            // Normals
            if(auto it = primitive.findAttribute("NORMAL"); it != primitive.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[it->second],
                    [&](glm::vec3 v, size_t index) 
                    {
                        vertices[initialVertex + index].normal = v;
                    });
            }

            // UVs
            if(auto it = primitive.findAttribute("TEXCOORD_0"); it != primitive.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[it->second],
                    [&](glm::vec2 v, size_t index) 
                    {
                        vertices[initialVertex + index].uv_x = v.x;
                        vertices[initialVertex + index].uv_y = v.y;
                    });
            }

            // Colors
            if(auto it = primitive.findAttribute("COLOR_0"); it != primitive.attributes.end())
            {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[it->second],
                    [&](glm::vec4 v, size_t index) 
                    {
                        vertices[initialVertex + index].color = v;
                    });
            }

            // Bounds
            glm::vec3 minPos = vertices[initialVertex].position;
            glm::vec3 maxPos = minPos;

            for(size_t v = initialVertex; v < vertices.size(); ++v)
            {
                minPos = glm::min(minPos, vertices[v].position);
                maxPos = glm::max(maxPos, vertices[v].position);
            }

            subMesh.bounds.origin = (minPos + maxPos) * 0.5f;
            subMesh.bounds.extents = (maxPos - minPos) * 0.5f;
            subMesh.bounds.sphereRadius = glm::length(subMesh.bounds.extents);

            mesh.subMeshes.push_back(subMesh);
        }

        mesh.vertices = std::move(vertices);
        mesh.indices = std::move(indices);

        outAsset->meshes.push_back(std::move(mesh));
    }

    // GLTF Scene
    if(!asset.nodes.empty())
    {
        outAsset->gltfScene.emplace();
        GLTFScene& scene = outAsset->gltfScene.value();

        scene.nodes.resize(asset.nodes.size());

        // Fill nodes + transforms
        for(size_t i = 0; i < asset.nodes.size(); ++i)
        {
            fastgltf::Node& n = asset.nodes[i];
            GLTFSceneNode& node = scene.nodes[i];

            node.meshIndex = n.meshIndex.has_value() ? static_cast<int>(n.meshIndex.value()) : -1;

            std::visit(fastgltf::visitor
            {
                [&](fastgltf::Node::TransformMatrix matrix) 
                {
                    memcpy(&node.localTransform, &matrix, sizeof(matrix));
                },
                [&](fastgltf::Node::TRS transform) 
                {
                    glm::vec3 t(transform.translation[0], transform.translation[1], transform.translation[2]);
                    glm::quat r(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
                    glm::vec3 s(transform.scale[0], transform.scale[1], transform.scale[2]);

                    glm::mat4 tm = glm::translate(glm::mat4(1.0f), t);
                    glm::mat4 rm = glm::toMat4(r);
                    glm::mat4 sm = glm::scale(glm::mat4(1.0f), s);

                    node.localTransform = tm * rm * sm;
                }
            }, n.transform);

            node.children.reserve(n.children.size());
            for(auto c : n.children)
            {
                node.children.push_back(static_cast<int>(c));
                scene.nodes[c].parent = static_cast<int>(i);
            }
        }

        // Roots
        for(size_t i = 0; i < scene.nodes.size(); ++i)
        {
            if(scene.nodes[i].parent == -1)
            {
                scene.rootNodes.push_back(static_cast<int>(i));
            }
        }
    }

    return true;
}
