import world;
import brain_types;
import vision_types;

RWStructuredBuffer<Creature> g_creatures;
RWStructuredBuffer<World> g_worlds;
RWStructuredBuffer<half> brain_weights;
RWStructuredBuffer<half> brain_activations;
RWStructuredBuffer<VisionSlot> vision_history;
RWStructuredBuffer<VisionFrame> vision_frames;

// Eligibility is a bitset so atomic scheduling cannot choose which parent gets
// a scarce slot. The serial structural pass visits set bits in ascending order.
public struct LifecycleCounts
{
    public uint split_count;
    public uint free_count;
    public uint birth_count;
}

public struct BirthAction
{
    // The original network remains here until both offspring are initialized.
    public uint parent_index;
    public uint child_index;
    public uint mutation_key;
}

RWStructuredBuffer<LifecycleCounts> lifecycle_counts;
RWStructuredBuffer<uint> lifecycle_free_slots;
RWStructuredBuffer<uint> lifecycle_split_slots;
RWStructuredBuffer<int64_t> lifecycle_split_guids;
RWStructuredBuffer<BirthAction> lifecycle_births;
uint num_worlds;
uint creatures_per_world;
uint seed;
uint tick;
uint food_age_start;
uint food_age_fade;
bool mutation_enabled;
float input_mutation_probability;
float policy_mutation_probability;

// Rates are energy units per second at a one-metre body length.
public static const float IDLE_ENERGY_COST = 0.05;
public static const float SPEED_ENERGY_COST = 0.001;
public static const float WEIGHT_MUTATION_STD = 0.02;

float mutate_weight(float value, uint key, uint parameter)
{
    uint random = hash_uint(key ^ hash_uint(parameter + 1u));
    float chance = float(random & 0x00ffffffu) / 16777216.0;
    // Separate rates let the large input layer and the biases/output layer
    // explore independently without changing the size of each mutation.
    float probability = parameter < 1024u ? input_mutation_probability : policy_mutation_probability;
    if (chance >= probability)
        return value;

    // Box-Muller Gaussian noise. Keep u1 strictly positive for log().
    float u1 = (float(hash_uint(random ^ 0xa511e9b3u) & 0x00ffffffu) + 1.0) / 16777216.0;
    float u2 = float(hash_uint(random ^ 0x63d83595u) & 0x00ffffffu) / 16777216.0;
    float noise = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    return value + WEIGHT_MUTATION_STD * noise;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_lifecycle(uint3 thread_id : SV_DispatchThreadID)
{
    uint words_per_world = max(1u, (creatures_per_world + 31u) / 32u);
    uint index = thread_id.x;
    if (index >= num_worlds * words_per_world)
        return;

    lifecycle_free_slots[index] = 0u;
    lifecycle_split_slots[index] = 0u;
    if (index % words_per_world == 0u)
    {
        uint world_index = index / words_per_world;
        lifecycle_counts[world_index] = (LifecycleCounts)0;
        g_worlds[world_index].num_creatures = 0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void settle_lifecycle(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;

    uint world_index = index / creatures_per_world;
    uint local_index = index % creatures_per_world;
    uint words_per_world = max(1u, (creatures_per_world + 31u) / 32u);
    uint word = world_index * words_per_world + local_index / 32u;
    uint bit = 1u << (local_index % 32u);
    Creature creature = g_creatures[index];

    // Food has already been credited. Every creature owns its cost arithmetic;
    // all deaths finish before any free slot is assigned to a child.
    if (creature.alive)
    {
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
            InterlockedAdd(g_worlds[world_index].deaths, 1u);
            // Brains were evaluated from the pre-resolution scene this tick.
            for (uint neuron = 0; neuron < BRAIN_ACTIVATIONS; ++neuron)
                brain_activations[index * BRAIN_ACTIVATIONS + neuron] = half(0);
        }
        else
        {
            creature.energy -= debit;
            InterlockedAdd(g_worlds[world_index].num_creatures, 1);
        }
        g_creatures[index] = creature;
    }

    if (!creature.alive)
    {
        InterlockedOr(lifecycle_free_slots[word], bit);
        InterlockedAdd(lifecycle_counts[world_index].free_count, 1u);
    }
    else if (creature.birth_tick < tick && creature.energy >= creature.split_energy)
    {
        lifecycle_split_guids[index] = creature.guid;
        InterlockedOr(lifecycle_split_slots[word], bit);
        InterlockedAdd(lifecycle_counts[world_index].split_count, 1u);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void resolve_lifecycle(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds)
        return;
    LifecycleCounts counts = lifecycle_counts[world_index];
    if (counts.split_count == 0u || counts.free_count == 0u)
        return;

    uint first = world_index * creatures_per_world;
    uint words_per_world = max(1u, (creatures_per_world + 31u) / 32u);
    uint first_word = world_index * words_per_world;
    uint free_word = 0u;
    uint free_bits = lifecycle_free_slots[first_word];
    uint birth_capacity = max(1u, creatures_per_world / 2u);
    World world = g_worlds[world_index];

    for (uint split_word = 0u; split_word < words_per_world; ++split_word)
    {
        uint split_bits = lifecycle_split_slots[first_word + split_word];
        while (split_bits != 0u)
        {
            uint local_index = 32u * split_word + uint(firstbitlow(split_bits));
            split_bits &= split_bits - 1u;
            uint index = first + local_index;
            Creature parent = g_creatures[index];
            // Earlier actions may invalidate a queued handle or its eligibility.
            if (!parent.alive || parent.guid != lifecycle_split_guids[index]
                || parent.birth_tick >= tick || parent.energy < parent.split_energy)
                continue;

            uint child_index = first + creatures_per_world;
            // This iterator only moves forward: free slots are traversed once,
            // rather than rescanning the entire capacity for every split.
            while (free_word < words_per_world)
            {
                if (free_bits == 0u)
                {
                    free_word += 1u;
                    if (free_word < words_per_world)
                        free_bits = lifecycle_free_slots[first_word + free_word];
                    continue;
                }
                uint candidate = first + 32u * free_word + uint(firstbitlow(free_bits));
                free_bits &= free_bits - 1u;
                if (!g_creatures[candidate].alive)
                {
                    child_index = candidate;
                    break;
                }
            }
            if (child_index == first + creatures_per_world)
            {
                g_worlds[world_index] = world;
                lifecycle_counts[world_index].birth_count = counts.birth_count;
                return;
            }

            Creature first_child = parent;
            first_child.guid = int64_t(num_worlds * creatures_per_world)
                + 2 * int64_t(world.births) * int64_t(num_worlds) + int64_t(world_index) + 1;
            first_child.parent_guid = parent.guid;
            first_child.generation = parent.generation + 1u;
            first_child.birth_tick = tick;
            first_child.age_start_tick = tick;
            first_child.energy_remainder = 0.0;
            first_child.desired_action = 0;
            first_child.desired_target = -1;
            first_child.velocity = 0.0;
            Creature child = first_child;
            child.guid += int64_t(num_worlds);
            child.energy = parent.energy / 2u;
            first_child.energy = parent.energy - child.energy;
            // Split sideways with a quarter-body-length gap, facing away.
            float heading = parent.heading;
            float2 side = float2(-sin(heading), cos(heading));
            float2 separation = side * (1.25 * parent.radii.x);
            child.position = parent.position + separation;
            first_child.position -= separation;
            child.position -= WORLD_SIZE * floor(child.position / WORLD_SIZE);
            first_child.position -= WORLD_SIZE * floor(first_child.position / WORLD_SIZE);
            child.heading = heading + 1.570796326794897;
            first_child.heading = heading - 1.570796326794897;
            child.heading -= 6.283185307179586 * floor((child.heading + 3.141592653589793) / 6.283185307179586);
            first_child.heading -= 6.283185307179586 * floor((first_child.heading + 3.141592653589793) / 6.283185307179586);
            g_creatures[index] = first_child;
            g_creatures[child_index] = child;

            // Each birth owns a stream independent of slot reuse and other worlds.
            uint mutation_key = hash_uint(seed ^ hash_uint(world_index + 1u) ^ 0xd1b54a35u);
            mutation_key = hash_uint(mutation_key ^ hash_uint(world.births + 1u));
            BirthAction action;
            action.parent_index = index;
            action.child_index = child_index;
            action.mutation_key = mutation_key;
            lifecycle_births[world_index * birth_capacity + counts.birth_count] = action;
            counts.birth_count += 1u;
            world.births += 1u;
            world.num_creatures += 1;
        }
    }
    g_worlds[world_index] = world;
    lifecycle_counts[world_index].birth_count = counts.birth_count;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_newborns(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint world_index = group_id.y;
    uint birth_index = group_id.x;
    if (world_index >= num_worlds || birth_index >= lifecycle_counts[world_index].birth_count)
        return;
    uint birth_capacity = max(1u, creatures_per_world / 2u);
    BirthAction action = lifecycle_births[world_index * birth_capacity + birth_index];
    uint first_key = hash_uint(action.mutation_key ^ 0xa511e9b3u);
    uint second_key = hash_uint(action.mutation_key ^ 0x63d83595u);

    for (uint parameter = lane; parameter < BRAIN_PARAMETERS; parameter += 64u)
    {
        // Each lane owns this parameter: read the original once before replacing
        // its slot, so neither offspring can inherit the other's mutations.
        float inherited = brain_weights[action.parent_index * BRAIN_PARAMETERS + parameter];
        brain_weights[action.parent_index * BRAIN_PARAMETERS + parameter] = half(mutation_enabled
            ? mutate_weight(inherited, first_key, parameter) : inherited);
        brain_weights[action.child_index * BRAIN_PARAMETERS + parameter] = half(mutation_enabled
            ? mutate_weight(inherited, second_key, parameter) : inherited);
    }
    // Neither offspring moves before its first observation and policy evaluation.
    for (uint neuron = lane; neuron < BRAIN_ACTIVATIONS; neuron += 64u)
    {
        brain_activations[action.parent_index * BRAIN_ACTIVATIONS + neuron] = half(0);
        brain_activations[action.child_index * BRAIN_ACTIVATIONS + neuron] = half(0);
    }

    uint creature_count = num_worlds * creatures_per_world;
    for (uint history_index = lane; history_index < VISION_HISTORY; history_index += 64u)
    {
        uint first_frame_index = history_index * creature_count + action.parent_index;
        uint frame_index = history_index * creature_count + action.child_index;
        VisionFrame frame = (VisionFrame)0;
        frame.tick = -1;
        vision_frames[first_frame_index] = frame;
        vision_frames[frame_index] = frame;
        for (uint segment = 0; segment < VISION_SLOTS; ++segment)
        {
            vision_history[first_frame_index * VISION_SLOTS + segment] = empty_vision_slot();
            vision_history[frame_index * VISION_SLOTS + segment] = empty_vision_slot();
        }
    }
}
