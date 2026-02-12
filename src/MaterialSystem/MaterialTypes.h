#pragma once

#include <array>
#include <cstdint>
#include <variant>

namespace SK::Material
{
	enum class Type : uint8_t
	{
		PBR = 0 // metallic-roughness workflow
	};

	enum class AlphaMode : uint8_t
	{
		Opaque = 0,
		Transparent
	};

	enum class TextureSlot : uint8_t
	{
		BaseColor = 0,
		MetallicRoughness, // metalness and roughness are packed together in a single texture
		Normal,
		Occlusion,
		Emissive,
		Count
	};

	static constexpr uint32_t INVALID_TEXTURE = UINT32_MAX;
	static constexpr uint32_t INVALID_MATERIAL = UINT32_MAX;

	// TODO: Think about the layout. Either will be aligned (alignas(16)) wrt GPU spec or I will directly use scalar layout for all the buffers.
	struct PBRData
	{
		float baseColorFactor[4] = { 1.f, 1.f, 1.f, 1.f };
		float metallicFactor = 1.f;
		float roughnessFactor = 1.f;
	};

	using MaterialParams = std::variant<PBRData>;

	struct Instance
	{
		// Defaulted to Opaque PBR Material
		Type type = Type::PBR;
		AlphaMode alphaMode = AlphaMode::Opaque;

		std::array<uint32_t, static_cast<size_t>(TextureSlot::Count)> textureIndices = { INVALID_TEXTURE, INVALID_TEXTURE, INVALID_TEXTURE, INVALID_TEXTURE, INVALID_TEXTURE };

		MaterialParams params = PBRData{};
	};
}