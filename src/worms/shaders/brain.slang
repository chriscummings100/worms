import world;
import vision_types;
import brain_types;

StructuredBuffer<Creature> g_creatures;
StructuredBuffer<VisionSlot> vision_history;
StructuredBuffer<VisionFrame> vision_frames;
RWStructuredBuffer<float> brain_weights;
RWStructuredBuffer<float> brain_activations;
uint num_worlds;
uint creatures_per_world;
uint seed;
uint tick;

float random_weight(uint key, uint parameter, float limit)
{
    float unit = float(hash_uint(key ^ hash_uint(parameter + 1u)) & 0x00ffffffu) / 16777216.0;
    return (2.0 * unit - 1.0) * limit;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void initialize_brains(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * creatures_per_world)
        return;
    uint world_index = index / creatures_per_world;
    uint local_index = index % creatures_per_world;
    // Adding worlds or changing population size leaves existing identities' brains unchanged.
    uint key = hash_uint(seed ^ hash_uint(world_index + 1u) ^ 0xb7e15162u);
    key = hash_uint(key ^ hash_uint(local_index + 123u));
    uint base = index * BRAIN_PARAMETERS;
    float hidden_limit = sqrt(6.0 / float(BRAIN_INPUTS + BRAIN_HIDDEN));
    float output_limit = sqrt(6.0 / float(BRAIN_HIDDEN + BRAIN_OUTPUTS));
    for (uint parameter = 0; parameter < BRAIN_PARAMETERS; ++parameter)
    {
        bool hidden_weight = parameter < 1024u;
        bool output_weight = parameter >= 1040u && parameter < 1072u;
        brain_weights[base + parameter] = hidden_weight || output_weight
            ? random_weight(key, parameter, hidden_weight ? hidden_limit : output_limit)
            : 0.0;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void evaluate_brains(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    uint creature_count = num_worlds * creatures_per_world;
    if (index >= creature_count)
        return;
    uint activation_base = index * BRAIN_ACTIVATIONS;
    uint frame_index = (tick % VISION_HISTORY) * creature_count + index;
    if (!g_creatures[index].alive || vision_frames[frame_index].tick != int(tick))
    {
        for (uint neuron = 0; neuron < BRAIN_ACTIVATIONS; ++neuron)
            brain_activations[activation_base + neuron] = 0.0;
        return;
    }

    float inputs[BRAIN_INPUTS] = {};
    float hidden[BRAIN_HIDDEN] = {};
    for (uint segment = 0; segment < VISION_SLOTS; ++segment)
    {
        VisionSlot slot = vision_history[frame_index * VISION_SLOTS + segment];
        bool food_present = slot.kind == VISION_FOOD;
        bool creature_present = slot.kind == VISION_CREATURE;
        bool present = food_present || creature_present;
        uint first = segment * 4u;
        inputs[first] = present ? saturate(1.0 - slot.distance / VISION_RANGE) : 0.0;
        inputs[first + 1u] = present ? slot.size / (slot.size + 1.0) : 0.0;
        inputs[first + 2u] = food_present ? 1.0 : 0.0;
        inputs[first + 3u] = creature_present ? 1.0 : 0.0;
    }
    uint weight_base = index * BRAIN_PARAMETERS;
    for (uint neuron = 0; neuron < BRAIN_HIDDEN; ++neuron)
    {
        float value = brain_weights[weight_base + 1024u + neuron];
        for (uint input = 0; input < BRAIN_INPUTS; ++input)
            value += inputs[input] * brain_weights[weight_base + neuron * BRAIN_INPUTS + input];
        hidden[neuron] = tanh(value);
    }
    for (uint input = 0; input < BRAIN_INPUTS; ++input)
        brain_activations[activation_base + input] = inputs[input];
    for (uint neuron = 0; neuron < BRAIN_HIDDEN; ++neuron)
        brain_activations[activation_base + BRAIN_INPUTS + neuron] = hidden[neuron];
    for (uint output = 0; output < BRAIN_OUTPUTS; ++output)
    {
        float value = brain_weights[weight_base + 1072u + output];
        for (uint neuron = 0; neuron < BRAIN_HIDDEN; ++neuron)
            value += hidden[neuron]
                * brain_weights[weight_base + 1040u + output * BRAIN_HIDDEN + neuron];
        brain_activations[activation_base + BRAIN_INPUTS + BRAIN_HIDDEN + output] =
            output == 0u ? tanh(value) : saturate(0.5 + value);
    }
}
