Shader "Custom/URP/HeightDisplacement"
{
    Properties
    {
        [Header(Textures)]
        _MainTex ("Optical Satellite Texture (RGB)", 2D) = "white" {}
        _HeightMap ("16-bit Heightmap (DSM)", 2D) = "black" {}

        [Header(Elevation Calibration)]
        _HeightScale ("Height Scale / Elevation Range (m)", Float) = 50.0
        _MinElevation ("Minimum Elevation (m)", Float) = 0.0
        _BaseColor ("Color Tint", Color) = (1, 1, 1, 1)

        [Header(Surface Properties)]
        _Smoothness ("Smoothness", Range(0, 1)) = 0.2
        _Metallic ("Metallic", Range(0, 1)) = 0.0
    }

    SubShader
    {
        Tags
        {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "Queue" = "Geometry"
        }
        LOD 200

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float3 normalOS     : NORMAL;
                float4 tangentOS    : TANGENT;
                float2 uv           : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS   : SV_POSITION;
                float3 positionWS   : TEXCOORD0;
                float3 normalWS     : TEXCOORD1;
                float2 uv           : TEXCOORD2;
            };

            TEXTURE2D(_MainTex);        SAMPLER(sampler_MainTex);
            TEXTURE2D(_HeightMap);      SAMPLER(sampler_HeightMap);

            CBUFFER_START(UnityPerMaterial)
                float4 _MainTex_ST;
                float4 _HeightMap_ST;
                half4  _BaseColor;
                float  _HeightScale;
                float  _MinElevation;
                half   _Smoothness;
                half   _Metallic;
            CBUFFER_END

            Varyings vert(Attributes input)
            {
                Varyings output = (Varyings)0;

                output.uv = TRANSFORM_TEX(input.uv, _MainTex);

                // Sample 16-bit Heightmap in vertex shader at highest LOD level
                float normalizedHeight = SAMPLE_TEXTURE2D_LOD(_HeightMap, sampler_HeightMap, output.uv, 0).r;

                // Compute vertical displacement along local Y-axis or vertex normal
                float displacement = normalizedHeight * _HeightScale;

                float3 displacedPositionOS = input.positionOS.xyz;
                displacedPositionOS.y += displacement;

                // Transform position to World and Clip Space
                VertexPositionInputs vertexInput = GetVertexPositionInputs(displacedPositionOS);
                output.positionCS = vertexInput.positionCS;
                output.positionWS = vertexInput.positionWS;

                // Transform Normal to World Space
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);
                output.normalWS = normalInput.normalWS;

                return output;
            }

            half4 frag(Varyings input) : SV_Target
            {
                // Sample optical satellite RGB texture
                half4 albedo = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, input.uv) * _BaseColor;

                // Directional lighting calculation for URP
                Light mainLight = GetMainLight(TransformWorldToShadowCoord(input.positionWS));
                float3 normalWS = normalize(input.normalWS);
                
                half NdotL = saturate(dot(normalWS, mainLight.direction));
                half3 lighting = mainLight.color * (NdotL * mainLight.distanceAttenuation * mainLight.shadowAttenuation);
                
                // Add ambient lighting
                half3 ambient = SampleSH(normalWS);
                half3 finalColor = albedo.rgb * (lighting + ambient);

                return half4(finalColor, albedo.a);
            }
            ENDHLSL
        }
    }
    FallBack "Hidden/Universal Render Pipeline/FallbackError"
}
