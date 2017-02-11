float2 NormalXY;

float4 main() : SV_Target
{
	float NormalZ = sqrt(1.f - dot(NormalXY,NormalXY));
	return float4(NormalXY, NormalZ, 0.0);
}