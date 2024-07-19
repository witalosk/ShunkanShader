Shader "Unlit/Day03"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "Common.hlsl"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }
            
            float sin01(float x)
            {
                return sin(x) * 0.5 + 0.5;
            }

            float2 clampm1p1(float2 x)
            {
                return saturate(x * 0.5 + 0.5) * 2.0 - 1.0;
            }
            
            float clampm1p1(float x)
            {
                return saturate(x * 0.5 + 0.5) * 2.0 - 1.0;
            }

            float Stepping(float x, float a)
            {
                return floor(x * a) / a;
            }

            float AirPlane(float2 uv)
            {
                uv.x = abs(uv.x);
                float plane = 0.0;

                float2 uv1 = uv * 2;
                uv1.x *= 3;
                plane = 1 - SdRoundedBox(uv1, float2(0.01, 1), 0.1) > 0 ? 1 : plane;

                float2 uv2 = uv * 2;
                uv2.y += 1.45;
                uv2.y *= 2.5;
                plane = SdEquilateralTriangle(uv2, 1) < 0 ? 1 : plane;

                float2 uv3 = uv * 2;
                uv3.y -= 0.4;
                uv3.y *= 3;
                uv3.x *= 0.5;
                plane = SdEquilateralTriangle(uv3, 1) < 0 ? 1 : plane;
                

                return plane;
            }

            float4 Book(float2 uv, float t, float3 bg)
            {
                float3 col = 0.0;
                float alpha = 1.0;
                float mask = smoothstep(saturate(1 - SdRoundedBox(uv, float2(0.3, 0.5), -0.8)), 0.0, 0.02);
                alpha = mask;

                // sky
                col = bg;
                col = lerp(col, 0.8, Stepping(pow(FractalNoise(float3(float2(uv.x * 0.2 + t * 0.05, uv.y * 0.6), t * 0.1), 5) * 0.5 + 0.5, 8), 6));
                col = lerp(col, 1.0, Stepping(pow(FractalNoise(float3(float2(uv.x * 0.2 + t * 0.3, uv.y * 0.6), t * 0.1), 5) * 0.5 + 0.5, 3), 6));

                // airplane
                float2 apuv = uv;
                apuv = RotateVector(apuv, sin(t));
                apuv.y = 1.6 + apuv.y - (t * 2) % 5;
                col += AirPlane((apuv * 7));
                

                col *= mask;

                // tomegu
                if (uv.x < -1.15 && uv.x > -1.32 && abs(uv.y) < 1.25)
                {
                    col = abs(uv.y + 100) % 0.1 < 0.05 ? 0.03 : col;
                    alpha = abs(uv.y + 100) % 0.1 < 0.05 ? 1 : alpha;
                }

                // band
                if (uv.x < 1.1 && uv.x > 0.95 && abs(uv.y) < 1.51)
                {
                    col = 0.03;
                    alpha = 1;
                }

                return float4(col, alpha);
            }

            float4 frag (v2f input) : SV_Target
            {
                float2 uv0 = input.uv * 2.0 - 1.0;
                uv0.x *= 10.0 / 6.0;
                float2 uv = uv0;
                
                float3 col = float3(0.2, 0.1, 0.0) * 0.1;
                col *= Pcg01((uv.y + 100) * 80);

                float2 p1 = uv;
                p1.x -= 0.5;
                float4 b1 = Book(RotateVector(p1 * 2.0, 0.1), _Time, float3(0.6, 0.1, 0.02) * 0.8);
                col = b1.w > 0.0001 ? b1 : col;

                uv.x += 0.3;
                float4 b3 = Book(RotateVector(uv * 2.0, -0.05), _Time * 0.9 + 789.78987, float3(0.0, 0.6352941176, 0.8392156863) * 0.8);
                col = b3.w > 0.0001 ? b3 : col;

                col *= (1 - length(uv0 * 0.5));
                return float4(col, 1.0);
            }
            ENDHLSL
        }
    }
}
