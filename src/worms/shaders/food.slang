import world;

RWStructuredBuffer<Food> g_food;
RWStructuredBuffer<World> g_worlds;
uint num_worlds;
uint food_capacity;
uint food_energy;
uint initial_food;
uint target_food;
uint seed;
uint spawn_budget;

uint food_hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}

float food_random(uint value)
{
    return float(food_hash(value) & 0x00ffffffu) / 16777216.0;
}

float2 gradient(int2 lattice, uint period, uint world_seed)
{
    // Both noise octaves tile exactly at the world's wrapping boundaries.
    uint2 wrapped = uint2((lattice % int(period) + int(period)) % int(period));
    uint index = food_hash(world_seed ^ food_hash(wrapped.x + wrapped.y * period));
    switch (index & 7u)
    {
    case 0: return float2(1.0, 0.0);
    case 1: return float2(-1.0, 0.0);
    case 2: return float2(0.0, 1.0);
    case 3: return float2(0.0, -1.0);
    case 4: return float2(0.70710678, 0.70710678);
    case 5: return float2(-0.70710678, 0.70710678);
    case 6: return float2(0.70710678, -0.70710678);
    default: return float2(-0.70710678, -0.70710678);
    }
}

float perlin(float2 position, uint period, uint world_seed)
{
    float2 point = position * float(period) / WORLD_SIZE;
    int2 cell = int2(floor(point));
    float2 local = point - floor(point);
    float2 fade = local * local * local * (local * (local * 6.0 - 15.0) + 10.0);
    float a = dot(gradient(cell, period, world_seed), local);
    float b = dot(gradient(cell + int2(1, 0), period, world_seed), local - float2(1, 0));
    float c = dot(gradient(cell + int2(0, 1), period, world_seed), local - float2(0, 1));
    float d = dot(gradient(cell + int2(1, 1), period, world_seed), local - float2(1, 1));
    return lerp(lerp(a, b, fade.x), lerp(c, d, fade.x), fade.y);
}

float food_density(float2 position, uint world_seed)
{
    float noise = 0.7 * perlin(position, 8u, world_seed)
        + 0.3 * perlin(position, 16u, world_seed ^ 0x9e3779b9u);
    float density = saturate(0.5 + 1.6 * noise);
    // Fourth power creates distinct feeding grounds with sparse gaps.
    return 0.005 + 0.995 * density * density * density * density;
}

float2 food_position(uint world_index, uint sequence)
{
    uint world_seed = food_hash(seed ^ food_hash(world_index + 1u));
    uint key = food_hash(world_seed ^ food_hash(sequence + 123u));
    float2 best_position = 0.0;
    float best_density = -1.0;
    for (uint attempt = 0; attempt < 128u; ++attempt)
    {
        uint candidate = key + attempt * 3u;
        float2 position = WORLD_SIZE * float2(food_random(candidate), food_random(candidate + 1u));
        float density = food_density(position, world_seed);
        if (food_random(candidate + 2u) < density)
            return position;
        if (density > best_density)
        {
            best_position = position;
            best_density = density;
        }
    }
    // Bound GPU work even for an unlucky sequence; retain its densest candidate.
    return best_position;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_food(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * food_capacity)
        return;

    uint world_index = index / food_capacity;
    uint local_index = index % food_capacity;
    bool active = local_index < initial_food;
    Food food;
    food.position = active ? food_position(world_index, local_index) : float2(0.0);
    food.energy = active ? food_energy : 0u;
    g_food[index] = food;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void grow_food(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds || food_capacity == 0u)
        return;
    World world = g_worlds[world_index];
    uint goal = min(target_food, food_capacity);
    uint count = min(spawn_budget, goal - min(world.num_food, goal));
    uint spawned = 0u;
    // One owner per world reuses empty slots without moving surviving food.
    // A spawn sequence, independent of slot reuse, keeps new locations fresh.
    uint cursor = world.food_cursor % food_capacity;
    for (uint scanned = 0u; scanned < food_capacity && spawned < count; ++scanned)
    {
        uint index = world_index * food_capacity + cursor;
        cursor = (cursor + 1u) % food_capacity;
        if (g_food[index].energy != 0u)
            continue;
        Food food;
        food.position = food_position(world_index, world.food_spawn_serial);
        food.energy = food_energy;
        g_food[index] = food;
        ++world.food_spawn_serial;
        ++spawned;
    }
    world.num_food += spawned;
    world.food_cursor = cursor;
    g_worlds[world_index] = world;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void set_food_energy(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * food_capacity)
        return;
    if (g_food[index].energy != 0u)
        g_food[index].energy = food_energy;
}
