import world;

StructuredBuffer<Creature> creatures;
StructuredBuffer<World> worlds;
RWStructuredBuffer<int> result;
uint creatures_per_world;
uint world_index;
float2 pick_position;
float tolerance;
int selected;
int64_t selected_guid;
float2 follow_position;
bool follow;

// Picking is an occasional user action. Scan the small world in one thread,
// keeping tie-breaking deterministic and the CPU readback to one integer.
[shader("compute")]
[numthreads(1, 1, 1)]
void pick()
{
    float2 position = pick_position;
    if (follow)
    {
        float2 anchor = follow_position;
        if (selected >= 0 && uint(selected) < creatures_per_world)
        {
            Creature tracked = creatures[world_index * creatures_per_world + uint(selected)];
            if (tracked.alive && (selected_guid == 0 || tracked.guid == selected_guid))
                anchor = tracked.position;
        }
        position += anchor - WORLD_SIZE * 0.5;
    }
    result[0] = -1;
    float closest = 1e30;
    for (uint index = 0; index < creatures_per_world; ++index)
    {
        Creature creature = creatures[world_index * creatures_per_world + index];
        if (!creature.alive)
            continue;
        float2 delta = position - creature.position;
        delta -= WORLD_SIZE * floor(delta / WORLD_SIZE + 0.5);
        float2 direction = float2(cos(creature.heading), sin(creature.heading));
        float2 local = float2(dot(delta, direction), dot(delta, float2(-direction.y, direction.x)));
        float2 shape = local / (creature.radii + tolerance);
        float distance_squared = dot(delta, delta);
        if (dot(shape, shape) <= 1.0 && distance_squared < closest)
        {
            closest = distance_squared;
            result[0] = int(index);
        }
    }
}
