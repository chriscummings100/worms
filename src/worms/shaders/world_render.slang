import world;

StructuredBuffer<Creature> creatures;
StructuredBuffer<World> worlds;
uint creatures_per_world;
uint world_index;
float2 resolution;
float2 offset;
float scale;

static const float2 CORNERS[6] = {
    float2(-1, -1), float2(1, -1), float2(1, 1),
    float2(-1, -1), float2(1, 1), float2(-1, 1)
};

float pixels_per_unit()
{
    return min(resolution.x, resolution.y) / WORLD_SIZE * scale;
}

struct BackgroundVertex
{
    float4 position : SV_Position;
    float2 world_position : TEXCOORD0;
};

[shader("vertex")]
BackgroundVertex background_vertex(uint vertex_id : SV_VertexID)
{
    float2 corner = CORNERS[vertex_id];
    BackgroundVertex output;
    output.position = float4(corner.x, -corner.y, 0.0, 1.0);
    output.world_position = corner * resolution * 0.5 / pixels_per_unit() + WORLD_SIZE * 0.5 + offset;
    return output;
}

[shader("fragment")]
float4 background_fragment(BackgroundVertex input) : SV_Target
{
    float2 uv = input.world_position / WORLD_SIZE;
    if (any(uv < 0.0) || any(uv > 1.0))
        return float4(0.009, 0.018, 0.024, 1.0);
    float glow = exp(-dot(uv - float2(0.35, 0.3), uv - float2(0.35, 0.3)) * 3.0);
    float3 tint = float3(0.010, 0.026, 0.028) + float3(0.012, 0.027, 0.027) * glow;
    tint += float3(0.002, 0.001, 0.004) * float(world_index % 3);
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float border = 1.0 - smoothstep(0.0, 1.5 / (WORLD_SIZE * pixels_per_unit()), edge);
    return float4(lerp(tint, float3(0.09, 0.18, 0.19), border * 0.55), 1.0);
}

struct CreatureVertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    nointerpolation float3 color : COLOR0;
    nointerpolation float2 radii : TEXCOORD1;
    float2 world_position : TEXCOORD2;
};

[shader("vertex")]
CreatureVertex creature_vertex(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    uint local_index = instance_id / 9u;
    Creature creature = creatures[world_index * creatures_per_world + local_index];
    CreatureVertex output = (CreatureVertex)0;
    if (!creature.alive || local_index >= uint(worlds[world_index].num_creatures))
    {
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    // Draw neighboring images of the torus; the fragment shader clips them to
    // the world, so bodies remain whole while crossing either edge or a corner.
    uint copy_index = instance_id % 9u;
    float2 wrap_offset = float2(int(copy_index % 3u) - 1, int(copy_index / 3u) - 1) * WORLD_SIZE;
    float2 local = CORNERS[vertex_id] * 1.38;
    float2 shape = local * creature.radii;
    float2 direction = normalize(creature.velocity);
    float2 position = creature.position + wrap_offset + float2(
        direction.x * shape.x - direction.y * shape.y,
        direction.y * shape.x + direction.x * shape.y);
    float2 ndc = (position - WORLD_SIZE * 0.5 - offset) * pixels_per_unit() * 2.0 / resolution;
    output.position = float4(ndc.x, -ndc.y, 0.0, 1.0);
    output.local = local;
    output.color = creature.color;
    output.radii = creature.radii;
    output.world_position = position;
    return output;
}

float segment_distance(float2 p, float2 start, float2 end)
{
    float2 direction = end - start;
    float fraction = saturate(dot(p - start, direction) / dot(direction, direction));
    return length(p - start - fraction * direction);
}

[shader("fragment")]
float4 creature_fragment(CreatureVertex input) : SV_Target
{
    if (any(input.world_position < 0.0) || any(input.world_position >= WORLD_SIZE))
        discard;
    float2 p = input.local;
    float body_distance = length(p) - 1.0;
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
    float2 highlight_position = (p - float2(-0.24, -0.35)) * float2(2.5, 4.0);
    color += float3(0.72, 0.90, 0.88) * exp(-dot(highlight_position, highlight_position)) * 0.18;
    color = lerp(input.color * 0.73, color, 1.0 - smoothstep(-edge_width, edge_width, body_distance));

    float2 eye_offset = (float2(p.x, abs(p.y)) - float2(0.49, 0.30))
        * float2(input.radii.x / input.radii.y, 1.0);
    float eye_distance = length(eye_offset) - 0.15;
    float eye_width = max(fwidth(eye_distance), 0.001);
    float eye = 1.0 - smoothstep(-eye_width, eye_width, eye_distance);
    color = lerp(color, float3(0.025, 0.075, 0.085), eye);
    float glint = 1.0 - smoothstep(0.018, 0.05, length(eye_offset - float2(-0.035, -0.04)));
    color = lerp(color, float3(0.65, 0.85, 0.82), glint * eye * 0.65);

    float coverage = 1.0 - smoothstep(-edge_width, edge_width, distance);
    return float4(color * coverage, coverage);
}
