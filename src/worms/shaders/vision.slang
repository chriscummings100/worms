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

// Food slots insert independently; capture starts after every link is written.
[shader("compute")]
[numthreads(64, 1, 1)]
void build_food_grid(uint3 thread_id : SV_DispatchThreadID)
{
    uint index = thread_id.x;
    if (index >= num_worlds * food_capacity)
        return;
    food_grid_next[index] = -1;
    Food food = g_food[index];
    if (food.energy == 0u)
        return;

    uint world_index = index / food_capacity;
    uint grid_base = world_index * GRID_CELL_COUNT;
    uint2 cell = vision_cell_at(food.position);
    uint head = grid_base + cell.y * GRID_SIZE + cell.x;
    int previous;
    InterlockedExchange(food_grid_heads[head], int(index), previous);
    food_grid_next[index] = previous;
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
    uint index = thread_id.x / VISION_SLOTS;
    uint segment = thread_id.x % VISION_SLOTS;
    uint creature_count = num_worlds * creatures_per_world;
    if (index >= creature_count)
        return;

    Creature observer = g_creatures[index];
    uint frame_index = (tick % VISION_HISTORY) * creature_count + index;
    if (segment == 0u)
    {
        VisionFrame frame;
        frame.position = observer.position;
        frame.heading = observer.heading;
        frame.tick = observer.alive ? int(tick) : -1;
        vision_frames[frame_index] = frame;
    }
    VisionSlot slot = empty_vision_slot();
    int nearest_index = -1;

    if (observer.alive)
    {
        uint world_index = index / creatures_per_world;
        uint grid_base = world_index * GRID_CELL_COUNT;
        int2 origin_cell = int2(vision_cell_at(observer.position));
        float2 forward = float2(cos(observer.heading), sin(observer.heading));
        float lower_angle = observer.heading - VISION_FOV * 0.5
            + float(segment) * (VISION_FOV / float(VISION_SLOTS));
        float upper_angle = lower_angle + VISION_FOV / float(VISION_SLOTS);
        float2 lower_edge = float2(cos(lower_angle), sin(lower_angle));
        float2 upper_edge = float2(cos(upper_angle), sin(upper_angle));
        float2 origin_in_cell = observer.position
            - GRID_CELL_SIZE * floor(observer.position / GRID_CELL_SIZE);
        // A circle enclosing the cell makes wedge/range rejection conservative,
        // including cells crossing angular boundaries and the world seams.
        // Padding covers float rounding; exact point tests below decide visibility.
        float cell_radius = GRID_CELL_SIZE * 0.7071067811865475 + 0.0001;
        float cell_range = VISION_RANGE + cell_radius;
        // 24 m / 4 m = six cells each way. Every wrapped cell is visited once.
        int reach = int(ceil(VISION_RANGE / GRID_CELL_SIZE));
        for (int y = -reach; y <= reach; ++y)
        {
            for (int x = -reach; x <= reach; ++x)
            {
                float2 cell_delta = (float2(x, y) + 0.5) * GRID_CELL_SIZE - origin_in_cell;
                float from_lower = lower_edge.x * cell_delta.y - lower_edge.y * cell_delta.x;
                float from_upper = upper_edge.x * cell_delta.y - upper_edge.y * cell_delta.x;
                if (dot(cell_delta, cell_delta) > cell_range * cell_range
                    || from_lower < -cell_radius || from_upper > cell_radius)
                    continue;
                int2 cell = (origin_cell + int2(x, y) + int(GRID_SIZE)) % int(GRID_SIZE);
                uint head = grid_base + uint(cell.y) * GRID_SIZE + uint(cell.x);
                int food_index = food_grid_heads[head];
                while (food_index >= 0)
                {
                    Food food = g_food[food_index];
                    float distance;
                    float bearing;
                    int found_segment = vision_segment(food.position, observer.position, forward, distance, bearing);
                    if (food.energy > 0u && found_segment == int(segment))
                    {
                        if (nearer_object(slot, nearest_index, distance, VISION_FOOD, food_index))
                        {
                            nearest_index = food_index;
                            slot.distance = distance;
                            slot.bearing = bearing;
                            slot.size = 2.0 * FOOD_RADIUS;
                            slot.food_energy = float(food.energy) / float(ENERGY_PER_UNIT);
                            slot.kind = VISION_FOOD;
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
                        int found_segment = vision_segment(creature.position, observer.position, forward, distance, bearing);
                        if (found_segment == int(segment))
                        {
                            if (nearer_object(slot, nearest_index, distance, VISION_CREATURE, creature_index))
                            {
                                nearest_index = creature_index;
                                slot.distance = distance;
                                slot.bearing = bearing;
                                slot.size = 2.0 * creature.radii.x;
                                slot.food_energy = 0.0;
                                slot.kind = VISION_CREATURE;
                            }
                        }
                    }
                    creature_index = grid_next[creature_index];
                }
            }
        }
    }

    vision_history[frame_index * VISION_SLOTS + segment] = slot;
}
