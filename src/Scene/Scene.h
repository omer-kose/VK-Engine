#pragma once

#include <AssetSystem/AssetRegistry.h>
#include <MaterialSystem/MaterialRegistry.h>
#include <Renderer/DrawContext.h>
#include <Renderer/GlobalGPUTypes.h>
#include <camera.h>
#include "MeshInstance.h"

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <string_view>
#include <vector>

namespace SK::Scene
{
	struct State
	{
		Camera camera;

		SK::Asset::AssetRegistry assetRegistry;
		SK::Material::MaterialRegistry materialRegistry;

		std::vector<MeshInstance> meshInstances;

		SK::Renderer::DrawContext drawContext;
		SK::Renderer::GPUSceneData gpuSceneData;

		float fov = 70.0f;
		// TODO: Consider moving to infinite far plane
		float nearPlane = 0.1f;
		float farPlane = 10000.0f;

		glm::vec4 ambientColor = glm::vec4(0.1f);
		glm::vec4 sunlightDirection = glm::vec4(0.0f, 1.0f, 0.5f, 1.0f);
		glm::vec4 sunlightColor = glm::vec4(1.0f);
	};

	void setCameraProperties(State* scene, const glm::vec3& position, float pitch, float yaw);
	void setProjectionProperties(State* scene, float fov, float nearPlane, float farPlane);
	void setGlobalLightingProperties(State* scene, const glm::vec4& ambientColor, const glm::vec4& sunlightDirection, const glm::vec4& sunlightColor);
	void updateGPUSceneData(State* scene, uint32_t viewportWidth, uint32_t viewportHeight);
	void updateCamera(State* scene);

	// Only one GLTF scene is active in a scene.
	bool loadGLTFScene(State* scene, std::string_view filePath, const glm::mat4& sceneWorldTransform = glm::mat4(1.0f));

	void buildDrawContext(State* scene);

	void clear(State* scene);
};
