import world;
import brain_types;
import vision_types;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
RWStructuredBuffer<float> brain_weights;
RWStructuredBuffer<float> brain_activations;
RWStructuredBuffer<VisionSlot> vision_history;
RWStructuredBuffer<VisionFrame> vision_frames;
uint num_worlds;
uint creatures_per_world;
uint seed;
uint tick;
uint food_age_start;
uint food_age_fade;

// Rates are energy units per second at a one-metre body length.
public static const float IDLE_ENERGY_COST = 0.05;
public static const float SPEED_ENERGY_COST = 0.001;
public static const float WEIGHT_MUTATION_PROBABILITY = 0.01;
public static const float WEIGHT_MUTATION_STD = 0.02;

float mutate_weight(float value, uint key, uint parameter)
{
    uint random = hash_uint(key ^ hash_uint(parameter + 1u));
    float chance = float(random & 0x00ffffffu) / 16777216.0;
    if (chance >= WEIGHT_MUTATION_PROBABILITY)
        return value;

    // Box-Muller Gaussian noise. Keep u1 strictly positive for log().
    float u1 = (float(hash_uint(random ^ 0xa511e9b3u) & 0x00ffffffu) + 1.0) / 16777216.0;
    float u2 = float(hash_uint(random ^ 0x63d83595u) & 0x00ffffffu) / 16777216.0;
    float noise = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    return value + WEIGHT_MUTATION_STD * noise;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resolve_lifecycle(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds)
        return;

    uint first = world_index * creatures_per_world;
    World world = g_worlds[world_index];
    world.num_creatures = 0;

    // Collection has already credited energy. Settle costs and age limits before
    // deciding which dead slots are available for reproduction this tick.
    for (uint local_index = 0; local_index < creatures_per_world; ++local_index)
    {
        uint index = first + local_index;
        Creature creature = g_creatures[index];
        if (!creature.alive)
            continue;

        float size = 2.0 * creature.radii.x;
        float rate = size * (IDLE_ENERGY_COST + SPEED_ENERGY_COST * dot(creature.velocity, creature.velocity));
        float quanta = creature.energy_remainder + rate * TIME_STEP * float(ENERGY_PER_UNIT);
        uint64_t debit = uint64_t(floor(quanta));
        creature.energy_remainder = quanta - float(debit);
        uint age = tick - creature.age_start_tick;
        bool expired = age >= food_age_start && age - food_age_start >= food_age_fade;
        if (creature.energy <= debit || expired)
        {
            creature.energy = 0;
            creature.energy_remainder = 0.0;
            creature.alive = false;
            creature.velocity = 0.0;
            world.deaths += 1u;
        }
        else
        {
            creature.energy -= debit;
            world.num_creatures += 1;
        }
        g_creatures[index] = creature;
    }

    // Serial within a world, parallel across worlds: each slot can be claimed once.
    for (uint local_index = 0; local_index < creatures_per_world; ++local_index)
    {
        uint index = first + local_index;
        Creature parent = g_creatures[index];
        if (!parent.alive || parent.birth_tick >= tick || parent.energy < parent.split_energy)
            continue;

        uint child_index = first + creatures_per_world;
        for (uint candidate = first; candidate < first + creatures_per_world; ++candidate)
        {
            if (!g_creatures[candidate].alive)
            {
                child_index = candidate;
                break;
            }
        }
        if (child_index == first + creatures_per_world)
            break;

        Creature child = parent;
        child.guid = int64_t(num_worlds * creatures_per_world)
            + int64_t(world.births) * int64_t(num_worlds) + int64_t(world_index) + 1;
        child.parent_guid = parent.guid;
        child.generation = parent.generation + 1u;
        child.birth_tick = tick;
        child.age_start_tick = tick;
        parent.age_start_tick = tick;
        child.energy = parent.energy / 2u;
        child.energy_remainder = 0.0;
        child.desired_action = 0;
        child.desired_target = -1;
        // Split sideways with a quarter-body-length gap, facing away from each other.
        float heading = parent.heading;
        float2 side = float2(-sin(heading), cos(heading));
        float2 separation = side * (1.25 * parent.radii.x);
        child.position = parent.position + separation;
        parent.position -= separation;
        child.position -= WORLD_SIZE * floor(child.position / WORLD_SIZE);
        parent.position -= WORLD_SIZE * floor(parent.position / WORLD_SIZE);
        child.heading = heading + 1.570796326794897;
        parent.heading = heading - 1.570796326794897;
        child.heading -= 6.283185307179586 * floor((child.heading + 3.141592653589793) / 6.283185307179586);
        parent.heading -= 6.283185307179586 * floor((parent.heading + 3.141592653589793) / 6.283185307179586);
        parent.velocity = float2(parent.velocity.y, -parent.velocity.x);
        child.velocity = 0.0;
        parent.energy -= child.energy;
        g_creatures[index] = parent;
        g_creatures[child_index] = child;

        // Each birth gets its own stream, independent of reused slots and other worlds.
        uint mutation_key = hash_uint(seed ^ hash_uint(world_index + 1u) ^ 0xd1b54a35u);
        mutation_key = hash_uint(mutation_key ^ hash_uint(world.births + 1u));
        for (uint parameter = 0; parameter < BRAIN_PARAMETERS; ++parameter)
            brain_weights[child_index * BRAIN_PARAMETERS + parameter] =
                mutate_weight(brain_weights[index * BRAIN_PARAMETERS + parameter], mutation_key, parameter);
        for (uint neuron = 0; neuron < BRAIN_ACTIVATIONS; ++neuron)
            brain_activations[child_index * BRAIN_ACTIVATIONS + neuron] = 0.0;
        world.births += 1u;
        world.num_creatures += 1;
    }
    g_worlds[world_index] = world;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reset_newborn_history(uint3 thread_id : SV_DispatchThreadID)
{
    uint frame_index = thread_id.x;
    uint creature_count = num_worlds * creatures_per_world;
    if (frame_index >= creature_count * VISION_HISTORY)
        return;
    uint creature_index = frame_index % creature_count;
    Creature creature = g_creatures[creature_index];
    if (!creature.alive || creature.generation == 0u || creature.birth_tick != tick)
        return;

    VisionFrame frame = (VisionFrame)0;
    frame.tick = -1;
    vision_frames[frame_index] = frame;
    for (uint segment = 0; segment < VISION_SLOTS; ++segment)
        vision_history[frame_index * VISION_SLOTS + segment] = empty_vision_slot();
}
