#include <Core/vk_loader.h>

#include "stb_image.h"
#include <iostream>

#include "vk_engine.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include <glm/gtx/quaternion.hpp>

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>


std::optional<AllocatedImage> loadImage(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image)
{
	AllocatedImage newImage = {};

	int width, height, nrChannels;

	std::visit(fastgltf::visitor{
		[](auto& arg){},
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

				newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

				stbi_image_free(data);
			}
		},
		[&](fastgltf::sources::Vector& vector){
			unsigned char* data = stbi_load_from_memory(vector.bytes.data(), static_cast<int>(vector.bytes.size()), &width, &height, &nrChannels, 4);
			if(data)
			{
				VkExtent3D imageSize;
				imageSize.width = width;
				imageSize.height = height;
				imageSize.depth = 1;

				newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

				stbi_image_free(data);	
			}
		},
		[&](fastgltf::sources::BufferView& view){
			auto& bufferView = asset.bufferViews[view.bufferViewIndex];
			auto& buffer = asset.buffers[bufferView.bufferIndex];

			std::visit(fastgltf::visitor{
				[](auto& args){}, 
				[&](fastgltf::sources::Vector& vector){ // only VectorWithMime is processed as during the load LoadExternalBuffers is specified meaning all the external buffers are already loaded into vector
					unsigned char* data = stbi_load_from_memory(vector.bytes.data() + bufferView.byteOffset, static_cast<int>(bufferView.byteLength), &width, &height, &nrChannels, 4);
					if(data)
					{
						VkExtent3D imageSize;
						imageSize.width = width;
						imageSize.height = height;
						imageSize.depth = 1;

						newImage = engine->createImage(data, imageSize, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_SAMPLED_BIT, false);

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

VkFilter extractFilter(fastgltf::Filter filter)
{
	switch(filter) 
	{
		// nearest samplers
		case fastgltf::Filter::Nearest:
		case fastgltf::Filter::NearestMipMapNearest:
		case fastgltf::Filter::NearestMipMapLinear:
			return VK_FILTER_NEAREST;

		// linear samplers
		case fastgltf::Filter::Linear:
		case fastgltf::Filter::LinearMipMapNearest:
		case fastgltf::Filter::LinearMipMapLinear:
		default:
			return VK_FILTER_LINEAR;
	}
}

VkSamplerMipmapMode extractMipmapMode(fastgltf::Filter filter)
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


std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath)
{
	fmt::print("Loading GLTF: {}", filePath);

	std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
	scene->engine = engine;

	fastgltf::Parser parser{};

	constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

	fastgltf::GltfDataBuffer data;
	data.loadFromFile(filePath);

	fastgltf::Asset asset;
	
	std::filesystem::path path = filePath;

	auto type = fastgltf::determineGltfFileType(&data);
	if(type == fastgltf::GltfType::glTF)
	{
		auto load = parser.loadGLTF(&data, path.parent_path(), gltfOptions);
		if(load)
		{
			asset = std::move(load.get());
		}
		else
		{
			std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
			return {};
		}
	}
	else if(type == fastgltf::GltfType::GLB)
	{
		auto load = parser.loadBinaryGLTF(&data, path.parent_path(), gltfOptions);
		if(load)
		{
			asset = std::move(load.get());
		}
		else
		{
			std::cerr << "Failed to load GLB: " << fastgltf::to_underlying(load.error()) << std::endl;
			return {};
		}
	}
	else
	{
		std::cerr << "Failed to determine glTF container" << std::endl;
		return {};
	}

	// Here, the the descriptors to be used can be estimated accurately
	std::vector<DescriptorAllocatorGrowable::PoolSize> poolSizes = {
		{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
		{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3},
		{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
	};

	scene->descriptorAllocator.init(engine->device, asset.materials.size(), poolSizes);

	// Load samplers
	for(fastgltf::Sampler& sampler : asset.samplers)
	{
		VkSamplerCreateInfo samplerInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr };
		samplerInfo.minLod = 0;
		samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

		samplerInfo.minFilter = extractFilter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));
		samplerInfo.magFilter = extractFilter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));

		samplerInfo.mipmapMode = extractMipmapMode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

		VkSampler newSampler;
		vkCreateSampler(engine->device, &samplerInfo, nullptr, &newSampler);

		scene->samplers.push_back(newSampler);
	}

	// temporal arrays for all the objects to use while creating the GLTF data
	std::vector<std::shared_ptr<GLTFMeshAsset>> meshes;
	std::vector<std::shared_ptr<GLTFSceneNode>> sceneNodes;
	std::vector<AllocatedImage> textures;
	std::vector<std::shared_ptr<MaterialInstance>> materialInstances;

	// Load the textures
	for(fastgltf::Image& image : asset.images)
	{
		std::optional<AllocatedImage> img = loadImage(engine, asset, image);

		if(img.has_value())
		{
			textures.push_back(img.value());
			scene->textures[image.name.c_str()] = img.value();
		}
		else
		{
			// failed to load, default to error image but give an error
			textures.push_back(engine->errorCheckerboardImage);
			fmt::print("Failed to load gltf texture: {} \n", image.name);
		}
	}

	scene->materialDataBuffer = engine->createBuffer(asset.materials.size() * sizeof(GLTFMetallicRoughnessMaterial::MaterialConstants), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
	int dataIndex = 0;
	GLTFMetallicRoughnessMaterial::MaterialConstants* pMaterialDataBuffer = (GLTFMetallicRoughnessMaterial::MaterialConstants*)scene->materialDataBuffer.allocInfo.pMappedData;

	// Load the materials
	for(fastgltf::Material& mat : asset.materials)
	{
		std::shared_ptr<MaterialInstance> newMat = std::make_shared<MaterialInstance>();
		materialInstances.push_back(newMat);
		scene->materialInstances[mat.name.c_str()] = newMat;

		GLTFMetallicRoughnessMaterial::MaterialConstants constants;
		constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
		constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
		constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
		constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

		constants.metalRoughnessFactors.x = mat.pbrData.metallicFactor;
		constants.metalRoughnessFactors.y = mat.pbrData.roughnessFactor;
		// Write material constants data to buffer
		pMaterialDataBuffer[dataIndex] = constants;

		MaterialPass passType = MaterialPass::Opaque;
		if(mat.alphaMode == fastgltf::AlphaMode::Blend)
		{
			passType = MaterialPass::Transparent;
		}

		GLTFMetallicRoughnessMaterial::MaterialResources materialResources;
		// Default the material textures
		materialResources.colorImage = engine->whiteImage;
		materialResources.colorSampler = engine->defaultSamplerLinear;
		materialResources.metalRoughnessImage = engine->whiteImage;
		materialResources.metalRoughnessSampler = engine->defaultSamplerLinear;

		// Set the uniform buffer for the material data
		materialResources.dataBuffer = scene->materialDataBuffer.buffer;
		materialResources.dataBufferOffset = dataIndex * sizeof(GLTFMetallicRoughnessMaterial::MaterialConstants);
		// Grab textures from gltf file
		if(mat.pbrData.baseColorTexture.has_value())
		{
			size_t imgIdx = asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
			size_t samplerIdx = asset.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

			materialResources.colorImage = textures[imgIdx];
			materialResources.colorSampler = scene->samplers[samplerIdx];
		}

		// create the material instance
		*newMat = GLTFMetallicRoughnessMaterial::CreateInstance(engine->device, passType, materialResources, scene->descriptorAllocator);

		++dataIndex;
	}

	// Load the meshes
	// use the same vectors for all meshes so that the memory doesnt reallocate as often
	std::vector<uint32_t> indices;
	std::vector<Vertex> vertices;

	for(fastgltf::Mesh& mesh : asset.meshes) 
	{
		std::shared_ptr<GLTFMeshAsset> newmesh = std::make_shared<GLTFMeshAsset>();
		meshes.push_back(newmesh);
		scene->meshes[mesh.name.c_str()] = newmesh;
		newmesh->name = mesh.name;

		// clear the mesh arrays each mesh, we dont want to merge them by error
		indices.clear();
		vertices.clear();

		for(auto&& p : mesh.primitives) 
		{
			GLTFGeoSurface newSurface;
			newSurface.startIndex = (uint32_t)indices.size();
			newSurface.count = (uint32_t)asset.accessors[p.indicesAccessor.value()].count;

			size_t initial_vtx = vertices.size();

			// load indices
			{
				fastgltf::Accessor& indexaccessor = asset.accessors[p.indicesAccessor.value()];
				indices.reserve(indices.size() + indexaccessor.count);

				fastgltf::iterateAccessor<std::uint32_t>(asset, indexaccessor,
					[&](std::uint32_t idx) {
						indices.push_back(idx + initial_vtx);
					});
			}

			// load vertex positions
			{
				fastgltf::Accessor& posAccessor = asset.accessors[p.findAttribute("POSITION")->second];
				vertices.resize(vertices.size() + posAccessor.count);

				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, posAccessor,
					[&](glm::vec3 v, size_t index) {
						Vertex newvtx;
						newvtx.position = v;
						newvtx.normal = { 1, 0, 0 };
						newvtx.color = glm::vec4{ 1.f };
						newvtx.uv_x = 0;
						newvtx.uv_y = 0;
						vertices[initial_vtx + index] = newvtx;
					});
			}

			// load vertex normals
			auto normals = p.findAttribute("NORMAL");
			if(normals != p.attributes.end()) 
			{

				fastgltf::iterateAccessorWithIndex<glm::vec3>(asset, asset.accessors[normals->second],
					[&](glm::vec3 v, size_t index) {
						vertices[initial_vtx + index].normal = v;
					});
			}

			// load UVs
			auto uv = p.findAttribute("TEXCOORD_0");
			if(uv != p.attributes.end()) 
			{

				fastgltf::iterateAccessorWithIndex<glm::vec2>(asset, asset.accessors[uv->second],
					[&](glm::vec2 v, size_t index) {
						vertices[initial_vtx + index].uv_x = v.x;
						vertices[initial_vtx + index].uv_y = v.y;
					});
			}

			// load vertex colors
			auto colors = p.findAttribute("COLOR_0");
			if(colors != p.attributes.end()) 
			{

				fastgltf::iterateAccessorWithIndex<glm::vec4>(asset, asset.accessors[colors->second],
					[&](glm::vec4 v, size_t index) {
						vertices[initial_vtx + index].color = v;
					});
			}

			if(p.materialIndex.has_value()) 
			{
				newSurface.materialInstance = materialInstances[p.materialIndex.value()];
			}
			else 
			{
				newSurface.materialInstance = materialInstances[0];
			}

			// Compute the bounds of the surface
			glm::vec3 minPos = vertices[initial_vtx].position;
			glm::vec3 maxPos = vertices[initial_vtx].position;
			// find min/max bounds
			for(int i = initial_vtx; i < vertices.size(); ++i)
			{
				minPos = glm::min(minPos, vertices[i].position);
				maxPos = glm::max(minPos, vertices[i].position);
			}

			newSurface.bounds.origin = (minPos + maxPos) / 2.0f;
			newSurface.bounds.extents = (maxPos - minPos) / 2.0f; // half lengths of the bounding box
			newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

			newmesh->surfaces.push_back(newSurface);
		}

		newmesh->meshBuffers = engine->uploadMesh(vertices, indices);
	}

	// Load all the nodes
	for(fastgltf::Node& gltfNode : asset.nodes)
	{
		std::shared_ptr<GLTFSceneNode> newSceneNode;

		if(gltfNode.meshIndex.has_value())
		{
			newSceneNode = std::make_shared<GLTFMeshNode>();
			static_cast<GLTFMeshNode*>(newSceneNode.get())->mesh = meshes[gltfNode.meshIndex.value()];
		}
		else
		{
			newSceneNode = std::make_shared<GLTFSceneNode>();
		}

		sceneNodes.push_back(newSceneNode);
		scene->sceneNodes[gltfNode.name.c_str()] = newSceneNode;

		// Calculate the local transform of the scene node
		std::visit(fastgltf::visitor{
			[&](fastgltf::Node::TransformMatrix matrix) {
				memcpy(&newSceneNode->localTransform, &matrix, sizeof(matrix));
			},
			[&](fastgltf::Node::TRS transform) {
				glm::vec3 t = glm::vec3(transform.translation[0], transform.translation[1], transform.translation[2]);
				glm::quat r = glm::quat(transform.rotation[3], transform.rotation[0], transform.rotation[1], transform.rotation[2]);
				glm::vec3 s = glm::vec3(transform.scale[0], transform.scale[1], transform.scale[2]);

				glm::mat4 tm = glm::translate(glm::mat4(1.0f), t);
				glm::mat4 rm = glm::toMat4(r);
				glm::mat4 sm = glm::scale(glm::mat4(1.0f), s);

				newSceneNode->localTransform = tm * rm * sm;
			}
		}, gltfNode.transform);
	}

	// Build the scene graph hierarchy
	for(int i = 0; i < asset.nodes.size(); ++i)
	{
		fastgltf::Node& gltfNode = asset.nodes[i];
		std::shared_ptr<GLTFSceneNode>& sceneNode = sceneNodes[i];

		for(auto& c : gltfNode.children)
		{
			sceneNode->children.push_back(sceneNodes[c]);
			sceneNodes[c]->parent = sceneNode;
		}
	}

	// Find the top nodes with no parents
	for(std::shared_ptr<GLTFSceneNode>& sceneNode : sceneNodes)
	{
		if(sceneNode->parent.lock() == nullptr)
		{
			scene->topNodes.push_back(sceneNode);
			sceneNode->refreshTransform(glm::mat4(1.0f));
		}
	}

	return scene;
}

std::optional<std::vector<std::shared_ptr<GLTFMeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
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

	std::vector<std::shared_ptr<GLTFMeshAsset>> meshes;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	for(fastgltf::Mesh& mesh : asset.meshes)
	{
		GLTFMeshAsset newMesh;
		newMesh.name = mesh.name;
		
		vertices.clear();
		indices.clear();

		for(auto&& p : mesh.primitives)
		{
			GLTFGeoSurface newSurface;
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
		
		meshes.emplace_back(std::make_shared<GLTFMeshAsset>(std::move(newMesh)));
	}

	return meshes;
}

void LoadedGLTF::registerDraw(const glm::mat4& topMatrix, DrawContext& ctx)
{
	for(auto& n : topNodes)
	{
		n->registerDraw(topMatrix, ctx);
	}
}

void LoadedGLTF::clearAll()
{
	VkDevice device = engine->device;

	descriptorAllocator.destroyPools(device);
	
	engine->destroyBuffer(materialDataBuffer);

	for(auto& [k, v] : meshes)
	{
		engine->destroyBuffer(v->meshBuffers.vertexBuffer);
		engine->destroyBuffer(v->meshBuffers.indexBuffer);
	}

	for(auto& [k, v] : textures)
	{
		if(v.image == engine->errorCheckerboardImage.image)
		{
			// Don't destroy the default images of the engine
			continue;
		}

		engine->destroyImage(v);
	}

	for(auto& sampler : samplers)
	{
		vkDestroySampler(device, sampler, nullptr);
	}
}

void GLTFMeshNode::registerDraw(const glm::mat4& topMatrix, DrawContext& ctx)
{
	// Instead of directly using the worldTransform of the Mesh, it is multiplied with the topMatrix given. This allows drawing the same mesh multiple times with a different transform
	// without altering its worldTransform field.
	glm::mat4 nodeMatrix = topMatrix * worldTransform;

	for(auto& s : mesh->surfaces)
	{
		RenderObject robj;
		robj.indexCount = s.count;
		robj.firstIndex = s.startIndex;
		robj.indexBuffer = mesh->meshBuffers.indexBuffer.buffer;
		robj.materialInstance = (MaterialInstance*)s.materialInstance.get();

		robj.bounds = s.bounds;

		robj.transform = nodeMatrix;
		robj.vertexBufferAddress = mesh->meshBuffers.vertexBufferAddress;

		ctx.opaqueGLTFSurfaces.push_back(robj);
	}

	// Recurse down the scene node
	GLTFSceneNode::registerDraw(topMatrix, ctx);
}
