//------------------------------------------------------------------------------------
// Settings
//------------------------------------------------------------------------------------
#define MAX_REFLECTION_RAY_MARCH_STEP		0.02f
#define RAY_MARH_BIAS						0.001f
#define SPEC_INTENSITY_TO_REFLECTANCE		0.5f

//------------------------------------------------------------------------------------
// Constants
//------------------------------------------------------------------------------------
cbuffer SSRConstants : register(b0)
{
	float2 BufferSize;
	float2 DitherTilingFactor;
	int NumDepthMips;
};

cbuffer CSOffsetConstants : register(b1)
{
	float4		MinZ_MaxZRatio;
	float4		CameraPosition;
	float4x4	InvViewProjectionMatrix;
	float4x4	ViewProjectionMatrix;
	float4x4	ScreenToWorldMatrix;
};

//------------------------------------------------------------------------------------
// Textures
//------------------------------------------------------------------------------------
Texture2D DitherTex;
Texture2D HZBTex;
Texture2D Gbuf_DiffuseTex;
Texture2D Gbuf_WorldNormalTex;
Texture2D Gbuf_SpecularTex;

//------------------------------------------------------------------------------------
// Samplers
//------------------------------------------------------------------------------------
SamplerState PointSamp;

//------------------------------------------------------------------------------------
// Hepler Functions
//------------------------------------------------------------------------------------
float ConvertFromDeviceZ(float DeviceZ)
{
	#define Z_PRECISION	0.001f
	DeviceZ = min(DeviceZ, 1 - Z_PRECISION);
	return 1.f / (DeviceZ * MinZ_MaxZRatio[2] - MinZ_MaxZRatio[3]);
}

/** Return whether screen space reflections are supported. Bit index 1 of 2bit mask */
bool IsReflectionSupported(float MaskValue)
{
	/* 2-bit mask */
	#define MASK_COMRESSION_FACTOR (3.f)		

	int Mask = (int)(MaskValue * MASK_COMRESSION_FACTOR);
	return (Mask & 0x2) > 0;
}

/** Return wheter this pixel belongs to a mover. Bit index 3 of misc GBuffer mask */
bool IsMoverPixel(float MiscGBufferMaskValue)
{
	/* 8-bit mask */
	#define MISC_GBUFFER_MASK_COMPRESSION_FACTOR 255.f				

	int MiscGBufferMask = (int)(MiscGBufferMaskValue * MISC_GBUFFER_MASK_COMPRESSION_FACTOR);
	return (MiscGBufferMask & 0x8) > 0;
}

float DecodeSpecularPower(float4 SpecularColorAndPower)
{
	float Bias = 128.0 / 255.0;
	float p = (SpecularColorAndPower.a - Bias) * 255.0 / 127.0;
	return p*p*500;
}

//------------------------------------------------------------------------------------
// HZB Traversal 
//------------------------------------------------------------------------------------
void StepThroughCell(inout float3 RaySample, float3 RayDir, int MipLevel)
{
	// Size of current mip 
	int2 MipSize = int2(BufferSize) >> MipLevel;

	// UV converted to index in the mip
	float2 MipCellIndex = RaySample.xy * float2(MipSize);

	// The cell boundary UV in the direction of the ray
	float2 BoundaryUV;
	BoundaryUV.x = RayDir.x > 0 ? ceil(MipCellIndex.x) / float(MipSize.x) : floor(MipCellIndex.x) / float(MipSize.x);
	BoundaryUV.y = RayDir.y > 0 ? ceil(MipCellIndex.y) / float(MipSize.y) : floor(MipCellIndex.y) / float(MipSize.y);

	// Result of 2 ray-line intersections where we intersect the ray with the boundary cell UVs
	float2 ParameterUV;
	ParameterUV.x = (BoundaryUV.x - RaySample.x) / RayDir.x;
	ParameterUV.y = (BoundaryUV.y - RaySample.y) / RayDir.y;

	float CellStepOffset = 0.05;

	// Pick the cell intersection that is closer, and march to that cell
	if (abs(ParameterUV.x) < abs(ParameterUV.y))
	{
		RaySample += (ParameterUV.x + CellStepOffset) * RayDir;
	}
	else
	{
		RaySample += (ParameterUV.y + CellStepOffset) * RayDir;
	}
}

//------------------------------------------------------------------------------------
// HZB Raymarch 
//------------------------------------------------------------------------------------
void Raymarch(
	float2 InUV,
	float4 ScreenSpacePos,
	float ReflectionRayMarchStep,
	float3 WorldPosition,
	float3 WorldNormal,
	float3 CameraVector,
	float4x4 ViewProjectionMat,
	out float2 ReflectionUV,
	out float Attenuation,
	out float FilterSize)
{
	ReflectionUV = 0;
	Attenuation = 0;
	FilterSize = 0;

	// Compute world space reflection vector
	float3 ReflectionVector = reflect(CameraVector, WorldNormal.xyz);

	// This will check the direction of the reflection vector with the view direction,
	// and if they are pointing in the same direction, it will drown out those reflections 
	// since we are limited to pixels visible on screen. Attenuate reflections for angles between 
	// 60 degrees and 75 degrees, and drop all contribution beyond the (-60,60)  degree range
	float CameraFacingReflectionAttenuation = 1 - smoothstep(0.25, 0.5, dot(-CameraVector, ReflectionVector));

	// Reject if the reflection vector is pointing back at the viewer.
	[branch]
	if (CameraFacingReflectionAttenuation <= 0)
		return;

	// Compute second sreen space point so that we can get the SS reflection vector
	float4 ScreenSpaceReflectionPoint = mul(ViewProjectionMat, float4(10.f*ReflectionVector + WorldPosition, 1.f));
	ScreenSpaceReflectionPoint /= ScreenSpaceReflectionPoint.w;
	ScreenSpaceReflectionPoint.xy = ScreenSpaceReflectionPoint.xy * float2(0.5, -0.5) + float2(0.5, 0.5);

	// Compute the sreen space reflection vector as the difference of the two screen space points
	float3 ScreenSpaceReflectionVec = normalize(ScreenSpaceReflectionPoint.xyz - ScreenSpacePos.xyz);

	// Dithered offset for raymarching to prevent banding artifacts
	float DitherOffset = DitherTex.SampleLevel(PointSamp, InUV * DitherTilingFactor, 0).r * 0.01f + RAY_MARH_BIAS;

	float3 RaySample = ScreenSpacePos.xyz + DitherOffset * ScreenSpaceReflectionVec;
	float2 UVSamplingAttenuation = float2(0.0, 0.0);

	int MipLevel = 0;
	int IterCount = 0;
	while (	MipLevel > -1 && 
			MipLevel < (NumDepthMips - 1) && 
			IterCount < 20)
	{
		StepThroughCell(RaySample, ScreenSpaceReflectionVec, MipLevel);

		UVSamplingAttenuation = smoothstep(0.05, 0.1, RaySample.xy) * (1 - smoothstep(0.95, 1, RaySample.xy));
		UVSamplingAttenuation.x *= UVSamplingAttenuation.y;

		if (UVSamplingAttenuation.x > 0)
		{
			float ZBufferValue = HZBTex.SampleLevel(PointSamp, RaySample.xy, MipLevel).r;

			if (RaySample.z < ZBufferValue)
			{
				MipLevel++;
			}
			else
			{
				float t = (RaySample.z - ZBufferValue) / ScreenSpaceReflectionVec.z;
				RaySample -= ScreenSpaceReflectionVec * t;
				MipLevel--;
			}

			IterCount++;
		}
		else
		{
			break;
		}
	}


	[branch]
	if (MipLevel == -1)
	{
		// Screenspace UV of the reflected color
		ReflectionUV = RaySample.xy;

		float BitMaskValue = Gbuf_DiffuseTex.SampleLevel(PointSamp, ReflectionUV, 0).a;

		// Bail out if reflected pixel is from a mover primitive, and the normal is not upward facing.
		// This is to restrict movers to floor reflections only since wall reflections create a lot of artifacting. 
		[branch]
		if (IsMoverPixel(BitMaskValue) &&
			/* dot(WorldNormal, float3(0,0,1)) < cos(5 degrees) */
			WorldNormal.z < 0.996f)
		{
			return;
		}

		// Use gloss value to figure out how blurry the reflections should be
		// Gloss values of 50 or more result in mirror reflections. Mapping or 
		// Filter = -0.6*Gloss + 31 was obtained by solving linear equation that 
		// maps Gloss=10 to FilterSize=25, and Gloss=50 to FilterSize=1
		float4 SpecularGBufferVal = Gbuf_SpecularTex.SampleLevel(PointSamp, InUV, 0);
		float Gloss = DecodeSpecularPower(SpecularGBufferVal);
		FilterSize = max(1, -0.6*Gloss + 31);

		// Use specular intensity to figure out the reflectance 
		float Reflectance = SPEC_INTENSITY_TO_REFLECTANCE * SpecularGBufferVal.r;

		// This will check the direction of the normal of the reflection sample with the
		// direction of the reflection vector, and if they are pointing in the same direction,
		// it will drown out those reflections since backward facing pixels are not available 
		// for screen space reflection. Attenuate reflections for angles between 90 degrees 
		// and 100 degrees, and drop all contribution beyond the (-100,100)  degree range
		float4 ReflectionNormalColor = Gbuf_WorldNormalTex.SampleLevel(PointSamp, ReflectionUV, 0);
		float4 ReflectionNormal = ReflectionNormalColor * float4(2, 2, 2, 1) - float4(1, 1, 1, 0);
		float DirectionBasedAttenuation = smoothstep(-0.17, 0.0, dot(ReflectionNormal.xyz, -ReflectionVector));

		// Attenuate any reflection color from the foreground. The GBuffer normal color for foreground objects is (0,0,1)
		float ForegroundAttenuation = step(0.0001f, ReflectionNormalColor.r * ReflectionNormalColor.g);

		// Range based attenuation parameter to gracefully fade out reflections at the edge of ray march range
		float RangeBasedAttenuation = 1.0 - smoothstep(0.5, 1.0, 3.f*length(ReflectionUV - ScreenSpacePos.xy));

		// Use material parameters and normal direction to figure out reflection contribution
		Attenuation = Reflectance * DirectionBasedAttenuation * RangeBasedAttenuation * CameraFacingReflectionAttenuation * UVSamplingAttenuation.x * ForegroundAttenuation;
	}
}

//------------------------------------------------------------------------------------
// Main Entry Point
//------------------------------------------------------------------------------------
RWTexture2D<float4> ReflectionTexOut : register(u0);

[numthreads(32, 32, 1)]
void main(uint3 DispatchID : SV_DispatchThreadID)
{
	uint2 PixelID = uint2(DispatchID.x, DispatchID.y);
	float2 PixelUV = asfloat(PixelID) / BufferSize;
	float2 NDCPos = float2(2.f,-2.f) * PixelUV + float2(-1.f,1.f);

	// Initialize to 0 as some of the code paths might not write to O/P
	ReflectionTexOut[PixelID] = float4(0, 0, 0, 0);

	float3 ScreenVector = mul(ScreenToWorldMatrix, float4(NDCPos, 1, 0)).xyz;

	// Compute world position
	float DeviceZ = HZBTex[PixelID].r;
	float LinearZ = ConvertFromDeviceZ(DeviceZ);
	float3 WorldPosition = ScreenVector * LinearZ;

	// Needed to compute world space reflection vector
	float3 CameraVector = normalize(ScreenVector);
	float4 WorldNormal = Gbuf_WorldNormalTex[PixelID] * float4(2, 2, 2, 1) - float4(1, 1, 1, 0);

	// Skip if reflection is disabled
	[branch]
		if (!IsReflectionSupported(WorldNormal.w))
			return;

	// ScreenSpacePos --> (texcoord.xy, device_z)
	float4 ScreenSpacePos = float4(PixelUV, DeviceZ, 1.f);

	float2 OutReflectionUV;
	float OutAttenuation, OutFilterSize;

	Raymarch(PixelUV, ScreenSpacePos, MAX_REFLECTION_RAY_MARCH_STEP, WorldPosition, WorldNormal, CameraVector, ViewProjectionMatrix, OutReflectionUV, OutAttenuation, OutFilterSize);

	ReflectionTexOut[PixelID] = float4(OutReflectionUV, OutAttenuation, OutFilterSize);
}