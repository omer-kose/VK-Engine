#pragma once

#include <vector>
#include <AssetSystem/AssetRegistry.h>
#include <Scene/MeshInstance.h>
#include "DrawContext.h"

namespace SK::Renderer
{
	// build draw packets from given mesh instances and records them into the given DrawContext
	void buildDrawPacketsFromMeshInstances(SK::Asset::AssetRegistry* assetRegistry, const std::vector<SK::Scene::MeshInstance>& instances, DrawContext* outCtx);
}