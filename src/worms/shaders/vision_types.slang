// Full observations are retained unchanged; brain input reduction is a later pass.
public static const uint VISION_SLOTS = 16u;
public static const uint VISION_HISTORY = 256u;
public static const float VISION_RANGE = 24.0;
public static const float VISION_FOV = 3.141592653589793;
public static const uint VISION_NONE = 0u;
public static const uint VISION_FOOD = 1u;
public static const uint VISION_CREATURE = 2u;

public struct VisionSlot
{
    public float distance;
    public float bearing;
    public float size;
    public float food_energy;
    public uint kind;
}

public struct VisionFrame
{
    public float2 position;
    public float heading;
    public int tick;
}

public VisionSlot empty_vision_slot()
{
    VisionSlot slot;
    slot.distance = -1.0;
    slot.bearing = 0.0;
    slot.size = 0.0;
    slot.food_energy = 0.0;
    slot.kind = VISION_NONE;
    return slot;
}
