import world;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
RWStructuredBuffer<Food> g_food;
RWStructuredBuffer<int> grid_heads;
RWStructuredBuffer<int> grid_next;
RWStructuredBuffer<uint> food_credits;
uint num_worlds;
uint creatures_per_world;
uint food_capacity;
uint seed;
uint tick;
uint food_age_start;
uint food_age_fade;

uint2 cell_at(float2 position)
{
    position -= WORLD_SIZE * floor(position / WORLD_SIZE);
    return uint2(floor(position / GRID_CELL_SIZE));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index < num_worlds * GRID_CELL_COUNT)
        grid_heads[index] = -1;
}

// A single owner builds each world's lists. Worlds run independently; list
// insertion needs no atomics and has no per-cell occupancy limit.
[shader("compute")]
[numthreads(1, 1, 1)]
void build_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds)
        return;

    uint creature_base = world_index * creatures_per_world;
    uint grid_base = world_index * GRID_CELL_COUNT;
    float max_radius = 0.0;
    for (uint local_index = 0; local_index < creatures_per_world; ++local_index)
    {
        uint index = creature_base + local_index;
        grid_next[index] = -1;
        food_credits[index] = 0u;
        Creature creature = g_creatures[index];
        if (!creature.alive)
            continue;

        uint2 cell = cell_at(creature.position);
        uint head = grid_base + cell.y * GRID_SIZE + cell.x;
        grid_next[index] = grid_heads[head];
        grid_heads[head] = int(index);
        max_radius = max(max_radius, max(creature.radii.x, creature.radii.y));
    }
    g_worlds[world_index].max_radius = max_radius;
}

uint eater_priority(uint world_index, uint food_index, uint creature_index)
{
    uint key = hash_uint(seed ^ hash_uint(world_index + 1u));
    key = hash_uint(key ^ hash_uint(tick + 0x9e3779b9u));
    key = hash_uint(key ^ hash_uint(food_index + 0x85ebca6bu));
    return hash_uint(key ^ hash_uint(creature_index + 0xc2b2ae35u));
}

// Each food slot owns its removal and selects precisely one eater. The grid
// and creature positions are immutable throughout this dispatch.
[shader("compute")]
[numthreads(64, 1, 1)]
void collect_food(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * food_capacity)
        return;
    Food food = g_food[index];
    if (food.energy == 0u)
        return;

    uint world_index = index / food_capacity;
    uint grid_base = world_index * GRID_CELL_COUNT;
    uint creature_base = world_index * creatures_per_world;
    int2 food_cell = int2(cell_at(food.position));
    // Usually a 3 x 3 stencil. Larger bodies expand it; at most one complete
    // circuit of each wrapped axis is visited, so no candidate is duplicated.
    int reach = int(min(ceil((g_worlds[world_index].max_radius + FOOD_RADIUS)
        / GRID_CELL_SIZE), float(GRID_SIZE)));
    int count = min(2 * reach + 1, int(GRID_SIZE));
    int2 first_cell = food_cell - reach;
    int eater = -1;
    uint best_priority = 0xffffffffu;

    for (int y = 0; y < count; ++y)
    {
        for (int x = 0; x < count; ++x)
        {
            int2 cell = (first_cell + int2(x, y) + int(GRID_SIZE)) % int(GRID_SIZE);
            int creature_index = grid_heads[grid_base + uint(cell.y) * GRID_SIZE + uint(cell.x)];
            while (creature_index >= 0)
            {
                Creature creature = g_creatures[creature_index];
                float2 delta = creature.position - food.position;
                delta -= WORLD_SIZE * floor(delta / WORLD_SIZE + 0.5);
                float radius = max(creature.radii.x, creature.radii.y) + FOOD_RADIUS;
                if (dot(delta, delta) <= radius * radius)
                {
                    uint priority = eater_priority(world_index, index % food_capacity,
                        uint(creature_index) - creature_base);
                    if (eater < 0 || priority < best_priority
                        || (priority == best_priority && creature_index < eater))
                    {
                        eater = creature_index;
                        best_priority = priority;
                    }
                }
                creature_index = grid_next[creature_index];
            }
        }
    }

    if (eater >= 0)
    {
        g_food[index].energy = 0u;
        InterlockedAdd(food_credits[eater], food.energy);
        InterlockedAdd(g_worlds[world_index].num_food, 0xffffffffu);
    }
}

// A separate pass avoids overlapping credits racing with energy writes. The
// full bounded food allocation fits in a uint credit; lifetime energy is 64-bit.
[shader("compute")]
[numthreads(64, 1, 1)]
void apply_food(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;
    if (food_credits[index] == 0u)
        return;

    uint age = tick - g_creatures[index].age_start_tick;
    if (age >= food_age_start && age - food_age_start >= food_age_fade)
        return; // No intake at the end of life, including a zero-length fade.
    uint64_t credit = uint64_t(food_credits[index]);
    if (age > food_age_start)
    {
        uint remaining = food_age_fade - min(age - food_age_start, food_age_fade);
        // Round once after summing all food this tick; widen before multiplying.
        credit = credit * uint64_t(remaining) / uint64_t(food_age_fade);
    }
    g_creatures[index].energy += credit;
}
