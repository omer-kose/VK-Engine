#include "DrawPacketBuilder.h"

void SK::Renderer::buildDrawPacketsFromMeshInstances(SK::Asset::AssetRegistry* assetRegistry, SK::Material::MaterialRegistry* materialRegistry, const std::vector<SK::Scene::MeshInstance>& instances, DrawContext* outCtx)
{
	outCtx->clear();

	for(const auto& inst : instances)
	{
		const auto& mesh = assetRegistry->meshes[inst.meshIndex];

		for(const auto& subMesh : mesh.subMeshes)
		{
			DrawPacket packet{};
			packet.meshIndex = inst.meshIndex;
			packet.startIndex = subMesh.startIndex;
			packet.indexCount = subMesh.indexCount;
			packet.bounds = subMesh.bounds;
			packet.worldTransform = inst.worldTransform;
			packet.materialIndex = subMesh.materialIndex;

			// If the submesh has no valid material assigned, it will be skipped.
			if(packet.materialIndex != SK::Material::INVALID_MATERIAL)
			{
				// TODO: For now everything is going to the opaque list. Later, do alphaMode split with material registry lookup.
				outCtx->opaque.push_back(packet);
			}
		}
	}
}
