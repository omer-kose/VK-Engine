
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

struct Vertex
{
	vec3 position;
	float uv_x;
	vec3 normal;
	float uv_y;
	vec4 tangent;
};

layout(buffer_reference, std430) readonly buffer VertexBuffer
{
	Vertex vertices[];
};

layout(push_constant, scalar) uniform PushConstants
{
	mat4 worldMatrix;
	VertexBuffer vertexBuffer;
	uint materialIndex; // Index into the PBRMaterialData buffer
} pushConstants;

layout(set = 0, binding = 0) uniform SceneData
{
	mat4 view;
	mat4 proj;
	mat4 viewproj;
	vec4 ambientColor;
	vec4 sunlightDirection; //w for sun power
	vec4 sunlightColor;
} sceneData;

struct PBRData
{
	vec4 baseColorFactor;
	float metallicFactor;
	float roughnessFactor;
	// Texture ids
	uint baseColorTexture;
	uint metallicRoughnessTexture;
	uint normalTexture;
	uint emissiveTexture;
	// Sampler ids
	uint8_t baseColorTextureSampler;
	uint8_t metallicRoughnessTextureSampler;
	uint8_t normalTextureSampler;
	uint8_t emissiveTextureSampler;
};

// Bindless Material + Texture resources
layout(set = 1, binding = 0, scalar) readonly buffer PBRMaterials
{
	PBRData pbrMaterials[];
};

layout(set = 1, binding = 1) uniform sampler2D textures[];