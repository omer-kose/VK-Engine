#pragma once

#include <vector>
#include <AssetSystem/AssetRegistry.h>
#include <MaterialSystem/MaterialRegistry.h>
#include <Scene/MeshInstance.h>
#include "DrawContext.h"

namespace SK::Renderer
{
	// build draw packets from given mesh instances and records them into the given DrawContext
	void buildDrawPacketsFromMeshInstances(SK::Asset::AssetRegistry* assetRegistry, SK::Material::MaterialRegistry* materialRegistry, const std::vector<SK::Scene::MeshInstance>& instances, DrawContext* outCtx);
}