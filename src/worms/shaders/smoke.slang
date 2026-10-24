// Stage 0: compute creates the same GPU records consumed by the vertex shader.
struct Creature
{
    float4 position_shape; // Pixel center.xy, ellipse radii.zw.
    float4 tint;           // Body RGB, heading in radians.
};

StructuredBuffer<Creature> creatures;
float2 resolution;

uint hash_uint(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float random_unit(uint value)
{
    return float(hash_uint(value) & 0x00ffffffu) / 16777216.0;
}

float3 body_color(uint index)
{
    switch (index % 5)
    {
    case 0: return float3(0.25, 0.80, 0.67);
    case 1: return float3(0.94, 0.52, 0.33);
    case 2: return float3(0.52, 0.55, 0.94);
    case 3: return float3(0.76, 0.56, 0.89);
    default: return float3(0.92, 0.76, 0.35);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize(
    uint3 thread_id : SV_DispatchThreadID,
    uniform uint count,
    uniform uint seed,
    uniform float time_seconds,
    uniform float2 resolution,
    RWStructuredBuffer<Creature> output_creatures)
{
    uint index = thread_id.x;
    if (index >= count)
        return;

    uint key = hash_uint(seed ^ (index * 747796405u));
    float2 jitter = float2(random_unit(key), random_unit(key + 1u)) - 0.5;
    float2 center;
    float2 radii;
    float heading;

    if (count <= 5)
    {
        // A loose five-creature composition inspired by the supplied reference.
        const float2 centers[5] = {
            float2(0.31, 0.65), float2(0.76, 0.28), float2(0.49, 0.42),
            float2(0.58, 0.82), float2(0.13, 0.84)
        };
        const float sizes[5] = {0.105, 0.140, 0.073, 0.082, 0.060};
        const float headings[5] = {-0.32, 2.10, 0.72, -1.20, 0.38};
        center = (centers[index] + jitter * 0.025) * resolution;
        float scale = min(resolution.x, resolution.y) * sizes[index];
        radii = float2(scale, scale * 0.61);
        heading = headings[index];
    }
    else
    {
        uint columns = uint(ceil(sqrt(float(count) * resolution.x / resolution.y)));
        uint rows = (count + columns - 1u) / columns;
        float2 cell = resolution / float2(columns, rows);
        center = (float2(index % columns, index / columns) + 0.5 + jitter * 0.20) * cell;
        float scale = min(cell.x, cell.y) * (0.27 + random_unit(key + 2u) * 0.05);
        radii = float2(scale, scale * 0.63);
        heading = random_unit(key + 3u) * 6.2831853;
    }

    float phase = random_unit(key + 4u) * 6.2831853;
    center += float2(sin(time_seconds * 0.65 + phase), cos(time_seconds * 0.55 + phase))
        * radii.y * 0.12;
    heading += sin(time_seconds * 0.8 + phase) * 0.10;

    Creature creature;
    creature.position_shape = float4(center, radii);
    creature.tint = float4(body_color(index), heading);
    output_creatures[index] = creature;
}

struct VertexOutput
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    nointerpolation float3 color : COLOR0;
    nointerpolation float2 radii : TEXCOORD1;
};

[shader("vertex")]
VertexOutput vertex_main(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    const float2 corners[6] = {
        float2(-1, -1), float2(1, -1), float2(1, 1),
        float2(-1, -1), float2(1, 1), float2(-1, 1)
    };
    Creature creature = creatures[instance_id];
    float2 local = corners[vertex_id] * 1.38;
    float2 offset = local * creature.position_shape.zw;
    float angle = creature.tint.w;
    float c = cos(angle);
    float s = sin(angle);
    float2 pixel = creature.position_shape.xy
        + float2(c * offset.x - s * offset.y, s * offset.x + c * offset.y);

    VertexOutput output;
    output.position = float4(pixel.x / resolution.x * 2.0 - 1.0,
                             1.0 - pixel.y / resolution.y * 2.0, 0.0, 1.0);
    output.local = local;
    output.color = creature.tint.rgb;
    output.radii = creature.position_shape.zw;
    return output;
}

float segment_distance(float2 p, float2 start, float2 end)
{
    float2 direction = end - start;
    float fraction = saturate(dot(p - start, direction) / dot(direction, direction));
    return length(p - start - fraction * direction);
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target
{
    float2 p = input.local;
    float body_distance = length(p) - 1.0;
    // Two short bent limbs stay inside the padded quad.
    float2 limb_p = float2(p.x, abs(p.y));
    float limb_distance = min(
        segment_distance(limb_p, float2(-0.42, 0.76), float2(-0.70, 1.20)),
        segment_distance(limb_p, float2(-0.70, 1.20), float2(-0.94, 1.22))) - 0.055;
    float distance = min(body_distance, limb_distance);
    float edge_width = max(fwidth(distance), 0.001);
    if (distance > edge_width)
        discard;

    float depth = sqrt(saturate(1.0 - dot(p, p)));
    float light = saturate(dot(normalize(float3(p * 0.65, depth)),
                               normalize(float3(-0.35, -0.55, 0.85))));
    float rim = smoothstep(0.78, 0.98, length(p));
    float3 color = input.color * (0.50 + light * 0.43);
    color += input.color * depth * 0.15;
    color = lerp(color, lerp(input.color, float3(0.85, 1.0, 0.96), 0.48), rim * 0.70);
    float highlight = exp(-dot((p - float2(-0.24, -0.35)) * float2(2.5, 4.0),
                              (p - float2(-0.24, -0.35)) * float2(2.5, 4.0)));
    color += float3(0.72, 0.90, 0.88) * highlight * 0.18;
    color = lerp(input.color * 0.73, color, 1.0 - smoothstep(-edge_width, edge_width, body_distance));

    // Eyes face along the local positive X axis; their radii are round in pixels.
    float2 eye_offset = (float2(p.x, abs(p.y)) - float2(0.49, 0.30))
        * float2(input.radii.x / input.radii.y, 1.0);
    float eye_distance = length(eye_offset) - 0.15;
    float eye_width = max(fwidth(eye_distance), 0.001);
    float eye = 1.0 - smoothstep(-eye_width, eye_width, eye_distance);
    color = lerp(color, float3(0.025, 0.075, 0.085), eye);
    float glint = 1.0 - smoothstep(0.018, 0.05, length(eye_offset - float2(-0.035, -0.04)));
    color = lerp(color, float3(0.65, 0.85, 0.82), glint * eye * 0.65);

    // Opaque edge coverage matches the smoke renderer's fixed clear color.
    float coverage = 1.0 - smoothstep(-edge_width, edge_width, distance);
    return float4(lerp(float3(0.015, 0.045, 0.055), color, coverage), 1.0);
}
