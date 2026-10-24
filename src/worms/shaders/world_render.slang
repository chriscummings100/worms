import world;
import vision_types;

StructuredBuffer<Creature> creatures;
StructuredBuffer<Food> food;
StructuredBuffer<World> worlds;
uint creatures_per_world;
uint food_capacity;
uint world_index;
float2 resolution;
float2 offset;
float scale;
int selected;
int64_t selected_guid;
float2 follow_position;
bool follow;
bool wrap_view;
StructuredBuffer<VisionSlot> vision_history;
StructuredBuffer<VisionFrame> vision_frames;
uint total_creatures;
uint vision_frame_index;

static const float2 CORNERS[6] = {
    float2(-1, -1), float2(1, -1), float2(1, 1),
    float2(-1, -1), float2(1, 1), float2(-1, 1)
};

float pixels_per_unit()
{
    return min(resolution.x, resolution.y) / WORLD_SIZE * scale;
}

bool valid_selection()
{
    if (selected < 0 || uint(selected) >= creatures_per_world)
        return false;
    Creature tracked = creatures[world_index * creatures_per_world + uint(selected)];
    return tracked.alive && (selected_guid == 0 || tracked.guid == selected_guid);
}

float2 camera_center()
{
    float2 center = follow ? follow_position : WORLD_SIZE * 0.5;
    if (follow && valid_selection())
        center = creatures[world_index * creatures_per_world + uint(selected)].position;
    return center + offset;
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
    output.world_position = corner * resolution * 0.5 / pixels_per_unit() + camera_center();
    return output;
}

[shader("fragment")]
float4 background_fragment(BackgroundVertex input) : SV_Target
{
    float2 uv = input.world_position / WORLD_SIZE;
    if (!(wrap_view || follow) && (any(uv < 0.0) || any(uv > 1.0)))
        return float4(0.009, 0.018, 0.024, 1.0);
    float glow = exp(-dot(uv - float2(0.35, 0.3), uv - float2(0.35, 0.3)) * 3.0);
    // The followed camera crosses torus seams. A periodic tint has no jump there.
    if (wrap_view || follow)
        glow = exp((dot(cos((uv - float2(0.35, 0.3)) * 6.2831853), float2(1.0)) - 2.0) * 0.5);
    float3 tint = float3(0.010, 0.026, 0.028) + float3(0.012, 0.027, 0.027) * glow;
    tint += float3(0.002, 0.001, 0.004) * float(world_index % 3);
    float edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
    float border = (wrap_view || follow) ? 0.0 : 1.0 - smoothstep(0.0, 1.5 / (WORLD_SIZE * pixels_per_unit()), edge);
    return float4(lerp(tint, float3(0.09, 0.18, 0.19), border * 0.55), 1.0);
}

struct VisionVertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    float2 world_position : TEXCOORD1;
    nointerpolation uint slot_base : TEXCOORD2;
    nointerpolation uint kind : TEXCOORD3;
    nointerpolation float marker_radius : TEXCOORD4;
};

[shader("vertex")]
VisionVertex vision_vertex(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    VisionVertex output = (VisionVertex)0;
    output.position = float4(2.0, 2.0, 0.0, 1.0);
    if (!valid_selection())
        return output;
    uint creature_index = world_index * creatures_per_world + uint(selected);
    uint frame_index = vision_frame_index * total_creatures + creature_index;
    VisionFrame frame = vision_frames[frame_index];
    if (frame.tick < 0)
        return output;
    output.slot_base = frame_index * VISION_SLOTS;

    // One fan and one visible-object marker per segment, with nine torus images.
    // Separate marker quads keep their outlines intact across segment edges.
    uint element = instance_id / 9u;
    float2 local;
    if (element == 0u)
    {
        local = CORNERS[vertex_id] * (VISION_RANGE + 2.0 / pixels_per_unit());
        output.local = local;
    }
    else
    {
        uint segment = element - 1u;
        VisionSlot slot = vision_history[output.slot_base + segment];
        output.kind = slot.kind;
        if (slot.kind == VISION_NONE || slot.distance < 0.0)
            return output;
        output.marker_radius = slot.kind == VISION_FOOD
            ? max(7.0, FOOD_RADIUS * pixels_per_unit() + 6.0)
            : max(4.5, slot.size * 0.5 * pixels_per_unit() + 3.0);
        output.local = CORNERS[vertex_id] * (output.marker_radius + 2.0);
        local = float2(cos(slot.bearing), sin(slot.bearing)) * slot.distance
            + output.local / pixels_per_unit();
    }

    uint copy_index = instance_id % 9u;
    float2 wrap_offset = float2(int(copy_index % 3u) - 1, int(copy_index / 3u) - 1) * WORLD_SIZE;
    float2 center = camera_center();
    float2 relative = frame.position - center;
    if (wrap_view || follow)
        relative -= WORLD_SIZE * floor(relative / WORLD_SIZE + 0.5);
    float2 direction = float2(cos(frame.heading), sin(frame.heading));
    float2 position = center + relative + wrap_offset + float2(
        direction.x * local.x - direction.y * local.y,
        direction.y * local.x + direction.x * local.y);
    float2 ndc = (position - center) * pixels_per_unit() * 2.0 / resolution;
    output.position = float4(ndc.x, -ndc.y, 0.0, 1.0);
    output.world_position = position;
    return output;
}

[shader("fragment")]
float4 vision_fragment(VisionVertex input) : SV_Target
{
    if (!(wrap_view || follow) && (any(input.world_position < 0.0) || any(input.world_position >= WORLD_SIZE)))
        discard;
    if (input.kind != 0u)
    {
        float edge = abs(length(input.local) - input.marker_radius);
        float coverage = 1.0 - smoothstep(0.45, 1.45, edge);
        float3 color = input.kind == VISION_FOOD ? float3(0.67, 0.91, 0.37) : float3(1.0, 0.63, 0.35);
        float alpha = coverage * 0.62;
        return float4(color * alpha, alpha);
    }

    float radius = length(input.local);
    float bearing = atan2(input.local.y, input.local.x);
    float half_fov = VISION_FOV * 0.5;
    float segment_angle = VISION_FOV / float(VISION_SLOTS);
    float segment_coordinate = (bearing + half_fov) / segment_angle;
    uint segment = uint(clamp(floor(segment_coordinate), 0.0, float(VISION_SLOTS - 1u)));
    VisionSlot slot = vision_history[input.slot_base + segment];
    float pixels = pixels_per_unit();
    float angular_edge = max(abs(bearing) - half_fov, 0.0) * radius * pixels;
    float arc = (1.0 - smoothstep(0.35, 1.35, abs(radius - VISION_RANGE) * pixels))
        * (1.0 - smoothstep(0.0, 1.0, angular_edge));

    float boundary = clamp(round(segment_coordinate), 0.0, float(VISION_SLOTS))
        * segment_angle - half_fov;
    float2 ray = float2(cos(boundary), sin(boundary));
    float distance_to_ray = length(input.local - ray * clamp(dot(input.local, ray), 0.0, VISION_RANGE));
    float lines = (1.0 - smoothstep(0.25, 1.25, distance_to_ray * pixels))
        * smoothstep(3.0, 10.0, radius * pixels);
    float fill = abs(bearing) <= half_fov && radius <= VISION_RANGE ? 1.0 : 0.0;
    float fill_alpha = fill * (slot.kind == VISION_NONE ? 0.012 : 0.030)
        * lerp(1.0, 0.6, saturate(radius / VISION_RANGE));
    float3 color = slot.kind == VISION_FOOD ? float3(0.44, 0.70, 0.27)
        : slot.kind == VISION_CREATURE ? float3(0.70, 0.42, 0.22) : float3(0.21, 0.52, 0.56);
    float line_alpha = max(lines * 0.10, arc * 0.26);
    float alpha = fill_alpha + line_alpha * (1.0 - fill_alpha);
    float3 premultiplied = color * fill_alpha
        + float3(0.47, 0.75, 0.71) * line_alpha * (1.0 - fill_alpha);
    return float4(premultiplied, alpha);
}

struct FoodVertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    float2 world_position : TEXCOORD1;
};

[shader("vertex")]
FoodVertex food_vertex(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    uint local_index = instance_id / 9u;
    FoodVertex output = (FoodVertex)0;
    if (local_index >= food_capacity)
    {
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    Food item = food[world_index * food_capacity + local_index];
    if (item.energy == 0u)
    {
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    uint copy_index = instance_id % 9u;
    float2 wrap_offset = float2(int(copy_index % 3u) - 1, int(copy_index / 3u) - 1) * WORLD_SIZE;
    float2 center = camera_center();
    float2 relative = item.position - center;
    if (wrap_view || follow)
        relative -= WORLD_SIZE * floor(relative / WORLD_SIZE + 0.5);
    // A small screen-space floor keeps clusters visible in the overview. This
    // affects only drawing; the food's world-space radius stays FOOD_RADIUS.
    float radius = max(FOOD_RADIUS, 0.7 / pixels_per_unit());
    float2 local = CORNERS[vertex_id] * 2.4;
    float2 position = center + relative + wrap_offset + local * radius;
    float2 ndc = (position - center) * pixels_per_unit() * 2.0 / resolution;
    output.position = float4(ndc.x, -ndc.y, 0.0, 1.0);
    output.local = local;
    output.world_position = position;
    return output;
}

[shader("fragment")]
float4 food_fragment(FoodVertex input) : SV_Target
{
    if (!(wrap_view || follow) && (any(input.world_position < 0.0) || any(input.world_position >= WORLD_SIZE)))
        discard;
    float radius = length(input.local);
    if (radius >= 2.4)
        discard;
    float edge_width = max(fwidth(radius), 0.001);
    float core = 1.0 - smoothstep(1.0 - edge_width, 1.0 + edge_width, radius);
    float halo = exp(-radius * radius * 1.2) * 0.16 * (1.0 - smoothstep(1.6, 2.4, radius));
    float highlight = exp(-dot(input.local + float2(0.3, 0.35), input.local + float2(0.3, 0.35)) * 3.0);
    float3 color = float3(0.48, 0.78, 0.10) + float3(0.32, 0.19, 0.18) * highlight;
    float coverage = core + halo * (1.0 - core);
    return float4(color * core + float3(0.40, 0.72, 0.08) * halo * (1.0 - core), coverage);
}

struct CreatureVertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    nointerpolation float3 color : COLOR0;
    nointerpolation float2 radii : TEXCOORD1;
    float2 world_position : TEXCOORD2;
    nointerpolation uint is_selected : TEXCOORD3;
};

[shader("vertex")]
CreatureVertex creature_vertex(uint vertex_id : SV_VertexID, uint instance_id : SV_InstanceID)
{
    uint local_index = instance_id / 9u;
    Creature creature = creatures[world_index * creatures_per_world + local_index];
    CreatureVertex output = (CreatureVertex)0;
    if (!creature.alive)
    {
        output.position = float4(2.0, 2.0, 0.0, 1.0);
        return output;
    }
    // Draw neighboring images of the torus. A followed camera wraps around its
    // current center, so the selected creature and its neighbors cross seams smoothly.
    uint copy_index = instance_id % 9u;
    float2 wrap_offset = float2(int(copy_index % 3u) - 1, int(copy_index / 3u) - 1) * WORLD_SIZE;
    output.is_selected = int(local_index) == selected && valid_selection() ? 1u : 0u;
    float2 extent = float2(1.38);
    if (output.is_selected != 0u)
        extent = max(extent, 1.0 + 3.0 / (creature.radii * pixels_per_unit()));
    float2 local = CORNERS[vertex_id] * extent;
    float2 shape = local * creature.radii;
    float2 direction = float2(cos(creature.heading), sin(creature.heading));
    float2 center = camera_center();
    float2 relative = creature.position - center;
    if (wrap_view || follow)
        relative -= WORLD_SIZE * floor(relative / WORLD_SIZE + 0.5);
    float2 position = center + relative + wrap_offset + float2(
        direction.x * shape.x - direction.y * shape.y,
        direction.y * shape.x + direction.x * shape.y);
    float2 ndc = (position - center) * pixels_per_unit() * 2.0 / resolution;
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
    if (!(wrap_view || follow) && (any(input.world_position < 0.0) || any(input.world_position >= WORLD_SIZE)))
        discard;
    float2 p = input.local;
    float body_distance = length(p) - 1.0;
    float2 limb_p = float2(p.x, abs(p.y));
    float limb_distance = min(
        segment_distance(limb_p, float2(-0.42, 0.76), float2(-0.70, 1.20)),
        segment_distance(limb_p, float2(-0.70, 1.20), float2(-0.94, 1.22))) - 0.055;
    float distance = min(body_distance, limb_distance);
    float edge_width = max(fwidth(distance), 0.001);
    float outline_width = max(fwidth(body_distance), 0.001);
    float outline = input.is_selected != 0u
        ? 1.0 - smoothstep(outline_width, 2.5 * outline_width, abs(body_distance)) : 0.0;
    if (distance > edge_width && outline <= 0.0)
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
    color = lerp(color, float3(1.0, 0.84, 0.30), outline);
    coverage = max(coverage, outline);
    return float4(color * coverage, coverage);
}
