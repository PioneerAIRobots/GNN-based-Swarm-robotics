# Self-reported state of one swarm agent, broadcast over the comms graph.
std_msgs/Header header

string agent_id                 # "drone_1", "ugv_1", "dog_1"
uint8  platform                 # see constants below
uint8 PLATFORM_DRONE=0
uint8 PLATFORM_UGV=1
uint8 PLATFORM_QUADRUPED=2

geometry_msgs/Pose  pose        # in the shared `map` frame
geometry_msgs/Twist twist

float32 battery                 # [0..1] remaining energy
float32 max_speed               # m/s, platform envelope
float32 health                  # [0..1] mobility/sensor health (1 = nominal)

bool[]   capabilities           # capability mask, index == TaskSpec capability id
string   current_task_id        # "" when idle
uint8    agent_mode             # see constants below
uint8 MODE_IDLE=0
uint8 MODE_TRANSIT=1
uint8 MODE_SEARCH=2
uint8 MODE_CONFIRM=3
uint8 MODE_ACT=4
uint8 MODE_RELAY=5
uint8 MODE_RTB=6
uint8 MODE_LOST=7

float32 comms_quality           # [0..1] link margin to the nearest peer/relay
