import world;
import brain_types;

StructuredBuffer<Creature> current_creatures;
StructuredBuffer<Creature> reference_creatures;
StructuredBuffer<half> current_weights;
StructuredBuffer<half> reference_weights;
StructuredBuffer<uint> sample_order;
RWStructuredBuffer<Creature> trial_creatures;
RWStructuredBuffer<World> trial_worlds;
RWStructuredBuffer<half> trial_weights;
RWStructuredBuffer<Food> trial_food;
RWStructuredBuffer<uint> counts;
uint source_count;
uint trial_capacity;
uint sample_size;
uint layout_seed;
uint food_capacity;

float unit_random(uint value)
{
    return float(hash_uint(value) & 0x00ffffffu) / 16777216.0;
}

void place_creature(uint source_index, uint sample_index,
                    uint cohort, uint cohort_size, uint world_index)
{
    // Swap the cohorts between identical starting layouts. Each sampled brain
    // gets both starting positions, independent of its live age, body or energy.
    uint position_group = world_index == 0u ? cohort : 1u - cohort;
    uint local_index = position_group * cohort_size + sample_index;
    uint index = world_index * trial_capacity + local_index;
    uint key = hash_uint(layout_seed ^ hash_uint(local_index + 123u));
    uint columns = uint(ceil(sqrt(float(2u * cohort_size))));
    uint rows = (2u * cohort_size + columns - 1u) / columns;
    float2 cell = WORLD_SIZE / float2(columns, rows);
    float2 jitter = float2(unit_random(key), unit_random(key + 1u)) - 0.5;
    Creature creature = (Creature)0;
    creature.guid = int64_t(index) + 1;
    creature.position = (float2(local_index % columns, local_index / columns) + 0.5 + jitter * 0.6) * cell;
    creature.alive = true;
    creature.desired_target = -1;
    creature.radii = float2(0.5, 0.315);
    creature.max_speed = 12.0;
    creature.heading = unit_random(key + 2u) * 6.283185307179586;
    creature.color = cohort == 0u ? float3(0.3, 0.8, 0.7) : float3(0.9, 0.5, 0.3);
    creature.energy = INITIAL_ENERGY;
    creature.split_energy = 2u * INITIAL_ENERGY;
    creature.lineage = cohort + 1u;
    trial_creatures[index] = creature;
    for (uint parameter = 0; parameter < BRAIN_PARAMETERS; ++parameter)
        trial_weights[index * BRAIN_PARAMETERS + parameter] = cohort == 0u
            ? current_weights[source_index * BRAIN_PARAMETERS + parameter]
            : reference_weights[source_index * BRAIN_PARAMETERS + parameter];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void initialize_cohorts(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= 2u)
        return;
    uint current_count = 0u;
    uint reference_count = 0u;
    for (uint index = 0u; index < source_count; ++index)
    {
        current_count += current_creatures[index].alive ? 1u : 0u;
        reference_count += reference_creatures[index].alive ? 1u : 0u;
    }
    // Keep trial density and layout fixed even when the live population shrinks.
    // Reuse the shuffled living population only after visiting every member.
    uint cohort_size = current_count > 0u && reference_count > 0u ? sample_size : 0u;
    if (world_index == 0u)
    {
        counts[4] = current_count;
        counts[5] = reference_count;
    }
    World world = trial_worlds[world_index];
    world.num_creatures = int(2u * cohort_size);
    world.births = 0u;
    world.deaths = 0u;
    trial_worlds[world_index] = world;
    uint current_sample = 0u;
    uint reference_sample = 0u;
    for (uint order = 0u; current_sample < cohort_size || reference_sample < cohort_size; ++order)
    {
        uint source_index = sample_order[order % source_count];
        Creature current = current_creatures[source_index];
        Creature reference = reference_creatures[source_index];
        if (current.alive && current_sample < cohort_size)
            place_creature(source_index, current_sample++, 0u, cohort_size, world_index);
        if (reference.alive && reference_sample < cohort_size)
            place_creature(source_index, reference_sample++, 1u, cohort_size, world_index);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void copy_food_layout(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index < food_capacity)
        trial_food[food_capacity + index] = trial_food[index];
}

[shader("compute")]
[numthreads(1, 1, 1)]
void count_lineages(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= 2u)
        return;
    uint current = 0u;
    uint reference = 0u;
    for (uint local_index = 0u; local_index < trial_capacity; ++local_index)
    {
        Creature creature = trial_creatures[world_index * trial_capacity + local_index];
        if (!creature.alive)
            continue;
        current += creature.lineage == 1u ? 1u : 0u;
        reference += creature.lineage == 2u ? 1u : 0u;
    }
    counts[2u * world_index] = current;
    counts[2u * world_index + 1u] = reference;
}
