#version 450

#extension GL_GOOGLE_include_directive : require

#include "input_structures.h"

layout (location = 0) out vec3 normal;
layout (location = 1) out vec3 colorFactor;
layout (location = 2) out vec2 uv;

void main()
{
	Vertex v = pushConstants.vertexBuffer.vertices[gl_VertexIndex];
	vec4 position = vec4(v.position, 1.0f);
	gl_Position = sceneData[pushConstants.frameIndex].viewproj * pushConstants.worldMatrix * position;

	normal = (transpose(inverse(pushConstants.worldMatrix)) * vec4(v.normal, 0.0f)).xyz; // TODO: Pass the inverse transpose from the CPU side don't recompute it per vertex.
	PBRData pbrData = pbrMaterials[pushConstants.materialIndex];
	colorFactor = pbrData.baseColorFactor.xyz;
	uv.x = v.uv_x;
	uv.y = v.uv_y;
}