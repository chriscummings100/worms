import brain_types;
import world;

Texture2D<float4> labels;
float2 resolution;
float content_scale;
float2 content_offset;
StructuredBuffer<half> activations;
StructuredBuffer<Creature> creatures;
uint creature_index;
int64_t selected_guid;

static const float2 CORNERS[6] = {
    float2(-1, -1), float2(1, -1), float2(1, 1),
    float2(-1, -1), float2(1, 1), float2(-1, 1)
};

struct LabelVertex
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
LabelVertex label_vertex(uint vertex : SV_VertexID)
{
    float2 uv = CORNERS[vertex] * 0.5 + 0.5;
    LabelVertex output;
    output.position = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0, 1);
    output.uv = uv;
    return output;
}

[shader("fragment")]
float4 label_fragment(LabelVertex input) : SV_Target
{
    return labels.Load(int3(int2(input.uv * resolution), 0));
}

struct NeuronVertex
{
    float4 position : SV_Position;
    float2 local : TEXCOORD0;
    nointerpolation float value : TEXCOORD1;
    nointerpolation uint kind : TEXCOORD2;
};

[shader("vertex")]
NeuronVertex neuron_vertex(uint vertex : SV_VertexID, uint instance : SV_InstanceID)
{
    NeuronVertex output = (NeuronVertex)0;
    output.position = float4(2.0, 2.0, 0.0, 1.0);
    Creature tracked = creatures[creature_index];
    if (!tracked.alive || (selected_guid != 0 && tracked.guid != selected_guid))
        return output;
    float2 center, radius;
    uint activation = instance;
    uint kind = 0;
    if (instance < BRAIN_INPUTS)
    {
        center = float2(190 + (instance / 4) * 26, 70 + (instance % 4) * 26);
        radius = 7;
    }
    else if (instance < BRAIN_INPUTS + BRAIN_HIDDEN)
    {
        uint hidden = instance - BRAIN_INPUTS;
        center = float2(720 + (hidden % 8) * 25, 84 + (hidden / 8) * 46);
        radius = 9;
    }
    else if (instance < BRAIN_ACTIVATIONS)
    {
        center = float2(998, 84 + (instance - BRAIN_INPUTS - BRAIN_HIDDEN) * 46);
        radius = 11;
    }
    else
    {
        uint output_index = instance - BRAIN_ACTIVATIONS;
        activation = BRAIN_INPUTS + BRAIN_HIDDEN + output_index;
        kind = output_index + 1;
        center = float2(1136, 84 + output_index * 46);
        radius = float2(104, 5);
    }
    float2 local = CORNERS[vertex];
    float2 position = (content_offset + (center + local * radius) * content_scale) / resolution;
    output.position = float4(position.x * 2.0 - 1.0, 1.0 - position.y * 2.0, 0, 1);
    output.local = local;
    output.value = activations[creature_index * BRAIN_ACTIVATIONS + activation];
    output.kind = kind;
    return output;
}

[shader("fragment")]
float4 neuron_fragment(NeuronVertex input) : SV_Target
{
    float magnitude = saturate(abs(input.value));
    float3 positive = float3(1.0, 0.47, 0.12);
    float3 negative = float3(0.13, 0.53, 1.0);
    float3 color = input.value < 0 ? negative : positive;
    float3 dim = float3(0.035, 0.052, 0.066);
    if (input.kind != 0)
    {
        float x = input.local.x;
        bool filled = input.kind == 1
            ? x >= min(0.0, input.value) && x <= max(0.0, input.value)
            : x <= input.value * 2.0 - 1.0;
        float3 bar = filled ? color * 0.8 : dim;
        if (input.kind == 1 && abs(x) < 0.008)
            bar = float3(0.4, 0.5, 0.55);
        return float4(bar, 1);
    }
    float distance = length(input.local);
    float edge = max(fwidth(distance), 0.001);
    float coverage = 1.0 - smoothstep(1.0 - edge, 1.0, distance);
    float3 face = lerp(dim, color, pow(magnitude, 0.7) * 0.86);
    float3 rim = lerp(dim * 2.0, color, magnitude);
    color = lerp(face, rim, smoothstep(0.73, 0.9, distance));
    return float4(color * coverage, coverage);
}
