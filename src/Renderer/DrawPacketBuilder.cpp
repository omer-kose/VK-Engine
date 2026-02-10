#include "DrawPacketBuilder.h"

void SK::Renderer::buildDrawPacketsFromMeshInstances(SK::Asset::AssetRegistry* assetRegistry, const std::vector<SK::Scene::MeshInstance>& instances, DrawContext* outCtx)
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

			// TODO: For now pushing everything to the opaque list
			outCtx->opaque.push_back(packet);
		}
	}
}
