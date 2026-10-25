import world;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
RWStructuredBuffer<Food> g_food;
RWStructuredBuffer<int> grid_heads;
RWStructuredBuffer<int> grid_next;
RWStructuredBuffer<uint> food_credits;
RWStructuredBuffer<uint64_t> food_energy_totals;
StructuredBuffer<int> food_grid_heads;
StructuredBuffer<int> food_grid_next;
RWStructuredBuffer<uint64_t> food_contact_winners;
RWStructuredBuffer<uint> food_contact_queue;
RWStructuredBuffer<uint> food_contact_counts;
RWStructuredBuffer<uint> food_contact_overflow;
RWStructuredBuffer<int64_t> contact_guids;
uint num_worlds;
uint creatures_per_world;
uint food_capacity;
uint seed;
uint tick;
uint food_age_start;
uint food_age_fade;
groupshared float radius_reduction[64];

uint2 cell_at(float2 position)
{
    position -= WORLD_SIZE * floor(position / WORLD_SIZE);
    return uint2(floor(position / GRID_CELL_SIZE));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_grid(uint3 thread_id : SV_DispatchThreadID, uint3 local_id : SV_GroupThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * GRID_CELL_COUNT)
        return;
    grid_heads[index] = -1;

    // The first clearing group in each world also reduces its maximum radius.
    // GRID_CELL_COUNT is a multiple of 64, so this branch is uniform per group.
    // Creature positions and radii stay immutable until the contact pass ends.
    // This keeps the existing float field without float atomics or another pass.
    if (index % GRID_CELL_COUNT < 64u)
    {
        uint world_index = index / GRID_CELL_COUNT;
        uint creature_base = world_index * creatures_per_world;
        float radius = 0.0;
        for (uint local_index = local_id.x; local_index < creatures_per_world; local_index += 64u)
        {
            Creature creature = g_creatures[creature_base + local_index];
            if (creature.alive)
                radius = max(radius, max(creature.radii.x, creature.radii.y));
        }
        radius_reduction[local_id.x] = radius;
        GroupMemoryBarrierWithGroupSync();
        for (uint stride = 32u; stride > 0u; stride >>= 1u)
        {
            if (local_id.x < stride)
                radius_reduction[local_id.x] = max(radius_reduction[local_id.x],
                    radius_reduction[local_id.x + stride]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (local_id.x == 0u)
            g_worlds[world_index].max_radius = radius_reduction[0];
    }
}

// Every slot has one writer. Consumers run only after this dispatch completes,
// so publishing a head before its next link is written is safe.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;

    grid_next[index] = -1;
    food_credits[index] = 0u;
    Creature creature = g_creatures[index];
    if (!creature.alive)
        return;

    uint world_index = index / creatures_per_world;
    uint grid_base = world_index * GRID_CELL_COUNT;
    uint2 cell = cell_at(creature.position);
    uint head = grid_base + cell.y * GRID_SIZE + cell.x;
    int previous;
    InterlockedExchange(grid_heads[head], int(index), previous);
    grid_next[index] = previous;
}

uint eater_priority(uint world_index, uint food_index, uint creature_index)
{
    uint key = hash_uint(seed ^ hash_uint(world_index + 1u));
    key = hash_uint(key ^ hash_uint(tick + 0x9e3779b9u));
    key = hash_uint(key ^ hash_uint(food_index + 0x85ebca6bu));
    return hash_uint(key ^ hash_uint(creature_index + 0xc2b2ae35u));
}

// A first claim appends each food slot exactly once, so every world's queue is
// bounded by food_capacity even if every creature touches every food piece.
// Queue contents beyond the current count are scratch and need not be cleared.
[shader("compute")]
[numthreads(64, 1, 1)]
void clear_food_contacts(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index < num_worlds * food_capacity)
        food_contact_winners[index] = ~uint64_t(0);
    if (index < num_worlds)
    {
        food_contact_counts[index] = 0u;
        food_contact_overflow[index] = 0u;
    }
}

// One group per creature, with lanes walking different nearby cells. The usual
// nine-cell stencil fits in one iteration; larger bodies stride over all cells
// they can touch, visiting each wrapped cell at most once. All grid links and
// creature/food positions remain immutable until detection and sensing finish.
[shader("compute")]
[numthreads(16, 1, 1)]
void detect_food_contacts(uint3 group_id : SV_GroupID, uint3 local_id : SV_GroupThreadID)
{
    uint index = group_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;
    Creature creature = g_creatures[index];
    if (local_id.x == 0u)
        contact_guids[index] = creature.guid;
    if (!creature.alive || food_capacity == 0u)
        return;

    uint world_index = index / creatures_per_world;
    uint creature_base = world_index * creatures_per_world;
    uint food_base = world_index * food_capacity;
    uint grid_base = world_index * GRID_CELL_COUNT;
    float radius = max(creature.radii.x, creature.radii.y) + FOOD_RADIUS;
    int reach = int(min(ceil(radius / GRID_CELL_SIZE), float(GRID_SIZE)));
    uint count = uint(min(2 * reach + 1, int(GRID_SIZE)));
    int2 first_cell = int2(cell_at(creature.position)) - reach;

    for (uint local_cell = local_id.x; local_cell < count * count; local_cell += 16u)
    {
        int2 offset = int2(local_cell % count, local_cell / count);
        int2 cell = (first_cell + offset + int(GRID_SIZE)) % int(GRID_SIZE);
        int food_index = food_grid_heads[grid_base + uint(cell.y) * GRID_SIZE + uint(cell.x)];
        while (food_index >= 0)
        {
            Food food = g_food[food_index];
            float2 delta = creature.position - food.position;
            delta -= WORLD_SIZE * floor(delta / WORLD_SIZE + 0.5);
            if (food.energy != 0u && dot(delta, delta) <= radius * radius)
            {
                uint priority = eater_priority(world_index, uint(food_index) - food_base,
                    index - creature_base);
                // Both priority and tie-breaking index are reduced atomically.
                // The append order never chooses the eater.
                uint64_t candidate = (uint64_t(priority) << 32u) | uint64_t(index);
                uint64_t previous;
                InterlockedMin(food_contact_winners[food_index], candidate, previous);
                if (previous == ~uint64_t(0))
                {
                    uint action_index;
                    InterlockedAdd(food_contact_counts[world_index], 1u, action_index);
                    if (action_index < food_capacity)
                        food_contact_queue[food_base + action_index] = uint(food_index);
                    else
                        InterlockedOr(food_contact_overflow[world_index], 1u);
                }
            }
            food_index = food_grid_next[food_index];
        }
    }
}

// Sparse actions resolve serially inside each world, with worlds independent.
// This dispatch must follow detection: the first claimant may have queued an
// entry before a later claimant supplied the final winning priority.
[shader("compute")]
[numthreads(1, 1, 1)]
void resolve_food_contacts(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds)
        return;
    uint creature_base = world_index * creatures_per_world;
    uint food_base = world_index * food_capacity;
    uint count = min(food_contact_counts[world_index], food_capacity);
    for (uint action_index = 0u; action_index < count; ++action_index)
    {
        uint food_index = food_contact_queue[food_base + action_index];
        uint64_t winner = food_contact_winners[food_index];
        if (winner == ~uint64_t(0))
            continue;
        uint eater = uint(winner & uint64_t(0xffffffffu));
        if (eater < creature_base || eater >= creature_base + creatures_per_world)
            continue;
        Creature creature = g_creatures[eater];
        Food food = g_food[food_index];
        // Structural resolution may invalidate queued actions. A reused slot
        // cannot collect food on behalf of the creature detected here earlier.
        if (!creature.alive || creature.guid != contact_guids[eater] || food.energy == 0u)
            continue;

        g_food[food_index].energy = 0u;
        food_credits[eater] += food.energy;
        g_worlds[world_index].num_food -= 1u;
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
    // Record intake once, after age scaling, before lifecycle can reuse a slot.
    // Cohort totals survive deaths; splitting never transfers these counters.
    uint cohort = g_creatures[index].lineage;
    InterlockedAdd(food_energy_totals[3u * (index / creatures_per_world) + cohort], credit);
}
