import world;
import vision_types;

StructuredBuffer<Creature> g_creatures;
StructuredBuffer<Food> g_food;
StructuredBuffer<int> grid_heads;
StructuredBuffer<int> grid_next;
RWStructuredBuffer<int> food_grid_heads;
RWStructuredBuffer<int> food_grid_next;
RWStructuredBuffer<VisionSlot> vision_history;
RWStructuredBuffer<VisionFrame> vision_frames;
uint num_worlds;
uint creatures_per_world;
uint food_capacity;
uint tick;

uint2 vision_cell_at(float2 position)
{
    position -= WORLD_SIZE * floor(position / WORLD_SIZE);
    return uint2(floor(position / GRID_CELL_SIZE));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_vision(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    uint frame_count = num_worlds * creatures_per_world * VISION_HISTORY;
    if (index >= frame_count)
        return;
    // Clear a complete frame per thread to keep dispatch dimensions modest.
    for (uint segment = 0; segment < VISION_SLOTS; ++segment)
        vision_history[index * VISION_SLOTS + segment] = empty_vision_slot();
    VisionFrame frame;
    frame.position = 0.0;
    frame.heading = 0.0;
    frame.tick = -1;
    vision_frames[index] = frame;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear_food_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index < num_worlds * GRID_CELL_COUNT)
        food_grid_heads[index] = -1;
}

// Each world has one owner, matching the creature grid's serial insertion.
[shader("compute")]
[numthreads(1, 1, 1)]
void build_food_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint world_index = thread_id.x;
    if (world_index >= num_worlds)
        return;
    uint food_base = world_index * food_capacity;
    uint grid_base = world_index * GRID_CELL_COUNT;
    for (uint local_index = 0; local_index < food_capacity; ++local_index)
    {
        uint index = food_base + local_index;
        food_grid_next[index] = -1;
        Food food = g_food[index];
        if (food.energy == 0u)
            continue;
        uint2 cell = vision_cell_at(food.position);
        uint head = grid_base + cell.y * GRID_SIZE + cell.x;
        food_grid_next[index] = food_grid_heads[head];
        food_grid_heads[head] = int(index);
    }
}

// Screen/world y increases downward, so positive bearings turn clockwise.
// The coincident-centre case has a defined bearing of zero.
int vision_segment(float2 position, float2 origin, float2 forward,
    out float distance, out float bearing)
{
    float2 delta = position - origin;
    delta -= WORLD_SIZE * floor(delta / WORLD_SIZE + 0.5);
    float distance_squared = dot(delta, delta);
    distance = sqrt(distance_squared);
    bearing = 0.0;
    if (distance_squared > VISION_RANGE * VISION_RANGE)
        return -1;
    if (distance_squared > 0.0)
        bearing = atan2(forward.x * delta.y - forward.y * delta.x, dot(forward, delta));
    if (abs(bearing) > VISION_FOV * 0.5)
        return -1;
    // Interior boundaries belong to the segment on their positive-bearing side.
    // The inclusive positive FOV endpoint belongs to the final segment.
    return min(int((bearing / VISION_FOV + 0.5) * float(VISION_SLOTS)), int(VISION_SLOTS) - 1);
}

bool nearer_object(VisionSlot previous, int previous_index, float distance, uint kind, int index)
{
    // Make exact ties independent of cell and linked-list traversal order.
    return previous.kind == VISION_NONE || distance < previous.distance
        || (distance == previous.distance && (kind < previous.kind
            || (kind == previous.kind && index < previous_index)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void capture_vision(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    uint creature_count = num_worlds * creatures_per_world;
    if (index >= creature_count)
        return;

    Creature observer = g_creatures[index];
    uint frame_index = (tick % VISION_HISTORY) * creature_count + index;
    VisionFrame frame;
    frame.position = observer.position;
    frame.heading = observer.heading;
    frame.tick = observer.alive ? int(tick) : -1;
    vision_frames[frame_index] = frame;

    VisionSlot slots[VISION_SLOTS] = {};
    int nearest_index[VISION_SLOTS] = {};
    for (uint segment = 0; segment < VISION_SLOTS; ++segment)
    {
        slots[segment] = empty_vision_slot();
        nearest_index[segment] = -1;
    }

    if (observer.alive)
    {
        uint world_index = index / creatures_per_world;
        uint grid_base = world_index * GRID_CELL_COUNT;
        int2 origin_cell = int2(vision_cell_at(observer.position));
        float2 forward = float2(cos(observer.heading), sin(observer.heading));
        // 24 m / 4 m = six cells each way. Every wrapped cell is visited once.
        int reach = int(ceil(VISION_RANGE / GRID_CELL_SIZE));
        for (int y = -reach; y <= reach; ++y)
        {
            for (int x = -reach; x <= reach; ++x)
            {
                int2 cell = (origin_cell + int2(x, y) + int(GRID_SIZE)) % int(GRID_SIZE);
                uint head = grid_base + uint(cell.y) * GRID_SIZE + uint(cell.x);
                int food_index = food_grid_heads[head];
                while (food_index >= 0)
                {
                    Food food = g_food[food_index];
                    float distance;
                    float bearing;
                    int segment = vision_segment(food.position, observer.position, forward, distance, bearing);
                    if (food.energy > 0u && segment >= 0)
                    {
                        VisionSlot slot = slots[segment];
                        if (nearer_object(slot, nearest_index[segment], distance, VISION_FOOD, food_index))
                        {
                            nearest_index[segment] = food_index;
                            slot.distance = distance;
                            slot.bearing = bearing;
                            slot.size = 2.0 * FOOD_RADIUS;
                            slot.food_energy = float(food.energy) / float(ENERGY_PER_UNIT);
                            slot.kind = VISION_FOOD;
                            slots[segment] = slot;
                        }
                    }
                    food_index = food_grid_next[food_index];
                }

                int creature_index = grid_heads[head];
                while (creature_index >= 0)
                {
                    Creature creature = g_creatures[creature_index];
                    if (creature.alive && uint(creature_index) != index)
                    {
                        float distance;
                        float bearing;
                        int segment = vision_segment(creature.position, observer.position, forward, distance, bearing);
                        if (segment >= 0)
                        {
                            VisionSlot slot = slots[segment];
                            if (nearer_object(slot, nearest_index[segment], distance, VISION_CREATURE, creature_index))
                            {
                                nearest_index[segment] = creature_index;
                                slot.distance = distance;
                                slot.bearing = bearing;
                                slot.size = 2.0 * creature.radii.x;
                                slot.food_energy = 0.0;
                                slot.kind = VISION_CREATURE;
                                slots[segment] = slot;
                            }
                        }
                    }
                    creature_index = grid_next[creature_index];
                }
            }
        }
    }

    for (uint segment = 0; segment < VISION_SLOTS; ++segment)
        vision_history[frame_index * VISION_SLOTS + segment] = slots[segment];
}
