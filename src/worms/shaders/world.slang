// Each world occupies a fixed block in g_creatures. All creatures are alive in
// this initial motion demo, so the live records already form a compact block.
public struct Creature
{
    public int64_t guid;
    public float2 position;
    public bool alive;
    public int desired_action;
    public int desired_target;
    public float2 velocity;
    public float2 radii;
    public float3 color;
}

public struct World
{
    public int num_creatures;
}

public static const float WORLD_SIZE = 256.0;
public static const float TIME_STEP = 1.0 / 60.0;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
uint num_worlds;
uint creatures_per_world;
uint seed;

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
void initialize(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index < num_worlds)
    {
        World world;
        world.num_creatures = int(creatures_per_world);
        g_worlds[index] = world;
    }
    if (index >= num_worlds * creatures_per_world)
        return;

    uint world_index = index / creatures_per_world;
    uint local_index = index % creatures_per_world;
    // A world's sequence does not change when the total number of worlds changes.
    uint key = hash_uint(seed ^ hash_uint(world_index + 1u) ^ hash_uint(local_index + 123u));
    uint columns = uint(ceil(sqrt(float(creatures_per_world))));
    uint rows = (creatures_per_world + columns - 1u) / columns;
    float2 cell = WORLD_SIZE / float2(columns, rows);
    float2 jitter = float2(random_unit(key), random_unit(key + 1u)) - 0.5;
    float angle = random_unit(key + 2u) * 6.2831853;
    float radius = min(12.0, min(cell.x, cell.y) * 0.28) * (0.8 + random_unit(key + 3u) * 0.4);

    Creature creature;
    creature.guid = int64_t(index) + 1;
    creature.position = (float2(local_index % columns, local_index / columns) + 0.5 + jitter * 0.6) * cell;
    creature.alive = true;
    creature.desired_action = 0;
    creature.desired_target = -1;
    creature.velocity = float2(cos(angle), sin(angle)) * (8.0 + random_unit(key + 4u) * 8.0);
    creature.radii = float2(radius, radius * 0.63);
    creature.color = body_color(local_index + world_index);
    g_creatures[index] = creature;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void step(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;
    Creature creature = g_creatures[index];
    if (!creature.alive)
        return;
    creature.position += creature.velocity * TIME_STEP;
    creature.position -= WORLD_SIZE * floor(creature.position / WORLD_SIZE);
    g_creatures[index] = creature;
}
