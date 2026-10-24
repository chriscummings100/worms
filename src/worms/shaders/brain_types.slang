// Fixed network layout: current vision -> hidden layer -> turn/throttle.
public static const uint BRAIN_INPUTS = 64u;
public static const uint BRAIN_HIDDEN = 16u;
public static const uint BRAIN_OUTPUTS = 2u;
public static const uint BRAIN_PARAMETERS = 1074u;
public static const uint BRAIN_ACTIVATIONS = 82u;
public static const float MAX_TURN_RATE = 1.570796326794897;

// Parameters per creature: W1[16][64], b1[16], W2[2][16], b2[2].
// Activations per creature: inputs[64], hidden[16], turn, throttle.
