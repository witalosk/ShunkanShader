Shader "Unlit/MadTracer"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags
        {
            "RenderType"="Opaque"
        }
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

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            struct Surface
            {
                float distace;
                float roughness;
                float color;
            };

			            

			float3 scol;

			void dmin(inout float3 d, float x, float y, float z)
			{
				if(x<d.x) d=float3(x,y,z);
			}

			// 3D noise function (IQ, Shane)
			float noise(float3 p)
			{
				float3 ip=floor(p);
				p-=ip;
				float3 s=float3(7, 157, 113);
				float4 h=float4(0., s.yz, s.y+s.z)+dot(ip, s);
				p=p*p*(3.-2.*p);
				h=lerp(frac(sin(h)*43758.5), frac(sin(h+s.x)*43758.5), p.x);
				h.xy=lerp(h.xz, h.yw, p.y);
				return lerp(h.x, h.y, p.z);
			}

			// hemisphere hash function based on a hash by Slerpy
			float3 hashHs(float3 n, float seed)
			{
				float a = frac(sin(seed)*43758.5)*2.-1.;
				float b = 6.283*frac(sin(seed)*41758.5)*2.-1.;
				float c=sqrt(1.-a*a);
				float3 r=float3(c*cos(b), a, c*sin(b));
				return r;
			}

			float box(float2 p)
			{
				p=abs(p); return max(p.x, p.y);
			}

			void pR(inout float2 p, float a)
			{
				p = cos(a)*p+sin(a)*float2(p.y, -p.x);
			}


			float3 map(float3 p)
			{

				float3 q;
				float3 d = float2(0, 1.).yxx;
				float floornoise = .8*noise(3.*p+2.3*_Time.y)+0.1*noise(20.*p+2.2*_Time.y);
				dmin(d, min(5.-p.z, 1.5+p.y), 0.1+0.3*step(fmod(4.*p.z, 1.), .5), .0); 
				dmin(d, length(p+float3(0., 0., 1.9+sin(_Time.y)))-.500, .99, 1.); 		  
				q=p; pR(q.xy, 0.6*_Time.y);

				dmin(d, length(q+float3(0, 0., 1.9+sin(_Time.y)))-.445-0.09*sin(43.*q.x-q.y+10.*_Time.y), 1., 0.1);
				if( _Time.y>24. )p.y-=0.1*_Time.y-2.4; 
				q = abs(p-round(p-.5)-.5);
				if( _Time.y>24. )p.y+=0.1*_Time.y-2.4;
				float g = min(min(box(q.xy), box(q.xz)), box(q.yz))-.05;
				float c = min(.6-abs(p.x+p.z), .45-abs(p.y));
				if (_Time.y>12.) dmin(d, max(g, c), .1, 0.5); //lattice (by Slerpy)

				if( _Time.y>18. )dmin(d, box(p.zx+float2(2, 2))-.5, 1., .4); 
				if( _Time.y>17.3)dmin(d, box(p.zx+float2(2,-2))-.5, 1.,-.4); 
				return d;

			}


			float3 normal(float3 p)
			{
				float m = map(p).x;
				float2 e = float2(0,.05);
				return normalize(m-float3(map(p - e.yxx).x, map(p - e.xyx).x, map(p - e.xxy).x));
			}


			void madtracer(in float3 ro1, in float3 rd1, in float seed)
			{
				scol = 0.0;	
				float t = 0., t2 = 0.;
				float3 m1, m2, rd2, ro2, nor2;
				float3 roold=ro1;
				float3 rdold=rd1;
				m1.x=0.;
				for( int i = 0; i < 140; i++ )
				{

					seed = frac(seed+_Time.y*float(i+1)+.1);
					ro1=lerp(roold, hashHs(ro1, seed), 0.002);				// antialiasing
					rd1=lerp(rdold, hashHs(rdold, seed), 0.06*m1.x);			// antialiasing
					m1 = map(ro1+rd1*t);
					t+=m1.z!=0. ? 0.25*abs(m1.x)+0.0008 : 0.25*m1.x;
					ro2 = ro1 + rd1*t;
					nor2 = normal(ro2); 									// normal of new origin
					seed = frac(seed+_Time.y*float(i+2)+.1);
			        rd2 = lerp(reflect(rd1, nor2), hashHs(nor2, seed), m1.y);// reflect depending on material
					m2 = map(ro2+rd2*t2);
					t2+=m2.z!=0. ? 0.25*abs(m2.x) : 0.25*m2.x;
					scol+=.007*(float3(1.+m2.z, 1., 1.-m2.z)*step(1., m2.y)+float3(1.+m1.z, 1., 1.-m1.z)*step(1., m1.y));
				}
			}

            float4 frag(v2f input) : SV_Target
            {
                float2 uv = input.uv;
                float2 fragCoord = uv * _ScreenParams.xy;
                float4 col = 0.0;

           
				// borders
				if( uv.y>.1&&uv.y<0.9 )
				{
					float seed = sin(fragCoord.x + fragCoord.y)*sin(fragCoord.x - fragCoord.y);
					// float3 bufa= texture(iChannel0, uv).xyz;

					// camera
					float3 ro, rd;
					float2 uv2 = (2.*fragCoord.xy-_ScreenParams.xy)/_ScreenParams.xy.x;
					ro = float3(0, 0,-5);
					rd = normalize(float3(uv2, 1));
					// rotate scene
			        if (_Time.y>12.)
			        {
						pR(rd.xz, .5*-sin(.17*_Time.y));
						pR(rd.yz, .5*sin(.19*_Time.y));
						pR(rd.xy, .4*-cos(.15*_Time.y));
			        }
					// render    
					madtracer(ro, rd, seed);

					float fade =min(3.*abs(sin((3.1415*(_Time.y-12.)/24.))), 1.);
					// fragColor =clamp(float4(0.7*scol+0.7*bufa, 0.)*fade, 0., 1.); // with blur
					col =clamp(float4(0.7*scol, 0.)*fade, 0., 1.);
				}

                return col;
            }
            ENDHLSL
        }
    }
}