#pragma once

#include <array>
#include <cstdint>
#include <variant>

/*
	For now, only PBR materials are supported.
*/

namespace SK::Material
{
	enum class AlphaMode : uint8_t
	{
		Opaque = 0,
		Transparent
	};

	static constexpr uint32_t INVALID_TEXTURE = UINT32_MAX;
	static constexpr uint32_t INVALID_MATERIAL = UINT32_MAX;

	// Using scalar layout for the material buffer. 1-to-1 matching with what will be stored on the GPU side. 
	struct PBRData
	{
		float baseColorFactor[4] = { 1.f, 1.f, 1.f, 1.f };
		float metallicFactor = 1.f;
		float roughnessFactor = 1.f;
		// Texture ids
		uint32_t baseColorTexture;
		uint32_t metallicRoughnessTexture;
		uint32_t normalTexture;
		uint32_t emissiveTexture;
	};

	struct Instance
	{
		// Defaulted to Opaque PBR Material
		AlphaMode alphaMode = AlphaMode::Opaque;

		PBRData materialData = PBRData{};
	};
}