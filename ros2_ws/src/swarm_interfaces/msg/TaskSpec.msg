# One atomic mission task that the GNN allocator can bid on.
string  task_id
uint8   task_type
uint8 TYPE_SEARCH_CELL=0        # sweep a coverage cell
uint8 TYPE_RF_TRIANGULATE=1     # take an RF bearing from a stand-off baseline
uint8 TYPE_VISUAL_CONFIRM=2     # overfly / image a candidate emitter
uint8 TYPE_CBRN_SAMPLE=3        # close-in chemical sample (dismount only)
uint8 TYPE_EMPLACE_JAMMER=4     # drive to emitter, drop jammer-marker
uint8 TYPE_COMMS_RELAY=5        # hold a station to bridge a broken link
uint8 TYPE_BDA=6                # battle damage assessment / silence check

geometry_msgs/Point position
float32 radius                  # acceptance radius (m)
float32 priority                # [0..1] higher is more urgent
float32 risk                    # [0..1] expected threat exposure at the task
float32 service_time            # seconds of on-station work once arrived
bool[]  required_capabilities   # capability mask a bidder must satisfy
float32 deadline                # seconds from mission start; <0 = none
bool    complete
