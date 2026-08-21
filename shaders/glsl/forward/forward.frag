#version 450

#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#include "input_structures.h"

layout (location = 0) in vec3 normal;
layout (location = 1) in vec3 colorFactor;
layout (location = 2) in vec2 uv;

layout (location = 0) out vec4 fragColor;

void main() 
{
	float lightValue = max(dot(normal, sceneData[pushConstants.frameIndex].sunlightDirection.xyz), 0.1f);
	
	PBRData pbrData = pbrMaterials[pushConstants.materialIndex];
	vec3 color = colorFactor * texture(sampler2D(nonuniformEXT(textures[pbrData.baseColorTexture]), samplers[pbrData.baseColorTextureSampler]), uv).xyz;
	vec3 ambient = color *  sceneData[pushConstants.frameIndex].ambientColor.xyz;

	fragColor = vec4(color * lightValue * sceneData[pushConstants.frameIndex].sunlightColor.w + ambient, 1.0f);
}