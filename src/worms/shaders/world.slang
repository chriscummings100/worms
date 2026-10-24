import brain_types;

// Each world occupies a fixed block in g_creatures. Dead slots can be reused.
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
    public float heading;
    public float max_speed;
    public uint64_t energy;
    public uint64_t split_energy;
    public int64_t parent_guid;
    public uint generation;
    public uint birth_tick;
    public uint age_start_tick; // Birth or most recent successful split.
    public float energy_remainder;
}

public struct World
{
    public int num_creatures;
    public uint num_food;
    public uint food_cursor;
    public uint food_spawn_serial;
    public float max_radius;
    public uint births;
    public uint deaths;
}

// One shared allocation, with a fixed-capacity block of food slots per world.
public struct Food
{
    public float2 position;
    public uint energy;
}

// One world unit is one metre. Size means full body length, excluding limbs.
public static const float WORLD_SIZE = 512.0;
public static const float CREATURE_LENGTH = 1.0;
public static const float FOOD_RADIUS = 0.18;
public static const float TIME_STEP = 1.0 / 60.0;
public static const uint ENERGY_PER_UNIT = 1024u;
public static const uint INITIAL_ENERGY = 100u * ENERGY_PER_UNIT;
public static const float GRID_CELL_SIZE = 4.0;
public static const uint GRID_SIZE = 128u;
public static const uint GRID_CELL_COUNT = GRID_SIZE * GRID_SIZE;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
StructuredBuffer<float> brain_activations;
uint num_worlds;
uint creatures_per_world;
uint initial_creatures;
uint initial_food;
uint seed;

public uint hash_uint(uint value)
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
        World world = (World)0;
        world.num_creatures = int(initial_creatures);
        world.num_food = initial_food;
        world.food_cursor = initial_food;
        world.food_spawn_serial = initial_food;
        world.max_radius = 0.0;
        g_worlds[index] = world;
    }
    if (index >= num_worlds * creatures_per_world)
        return;

    uint world_index = index / creatures_per_world;
    uint local_index = index % creatures_per_world;
    Creature creature = (Creature)0;
    if (local_index >= initial_creatures)
    {
        g_creatures[index] = creature;
        return;
    }
    // A world's sequence does not change when the total number of worlds changes.
    uint key = hash_uint(seed ^ hash_uint(world_index + 1u) ^ hash_uint(local_index + 123u));
    uint columns = uint(ceil(sqrt(float(initial_creatures))));
    uint rows = (initial_creatures + columns - 1u) / columns;
    float2 cell = WORLD_SIZE / float2(columns, rows);
    float2 jitter = float2(random_unit(key), random_unit(key + 1u)) - 0.5;
    float angle = random_unit(key + 2u) * 6.2831853;
    float radius = 0.5 * CREATURE_LENGTH * (0.8 + random_unit(key + 3u) * 0.4);

    creature.guid = int64_t(index) + 1;
    creature.position = (float2(local_index % columns, local_index / columns) + 0.5 + jitter * 0.6) * cell;
    creature.alive = true;
    creature.desired_action = 0;
    creature.desired_target = -1;
    creature.max_speed = 8.0 + random_unit(key + 4u) * 8.0;
    creature.velocity = float2(cos(angle), sin(angle)) * (0.5 * creature.max_speed);
    creature.radii = float2(radius, radius * 0.63);
    creature.color = body_color(local_index + world_index);
    creature.heading = angle;
    creature.energy = max(uint64_t(1), uint64_t(round(float(INITIAL_ENERGY) * (2.0 * radius))));
    creature.split_energy = 2u * creature.energy;
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
    uint output_base = index * BRAIN_ACTIVATIONS + BRAIN_INPUTS + BRAIN_HIDDEN;
    creature.heading += brain_activations[output_base] * MAX_TURN_RATE * TIME_STEP;
    creature.heading -= 6.283185307179586 * floor((creature.heading + 3.141592653589793) / 6.283185307179586);
    creature.velocity = float2(cos(creature.heading), sin(creature.heading))
        * creature.max_speed * brain_activations[output_base + 1u];
    creature.position += creature.velocity * TIME_STEP;
    creature.position -= WORLD_SIZE * floor(creature.position / WORLD_SIZE);
    g_creatures[index] = creature;
}
