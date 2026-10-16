std_msgs/Header header
uint8 phase
uint8 PHASE_INIT=0
uint8 PHASE_INGRESS=1
uint8 PHASE_SEARCH=2
uint8 PHASE_LOCALIZE=3
uint8 PHASE_CONFIRM=4
uint8 PHASE_NEUTRALIZE=5
uint8 PHASE_BDA=6
uint8 PHASE_EGRESS=7
uint8 PHASE_COMPLETE=8

string  phase_name
float32 mission_time
float32 coverage_fraction        # [0..1] of the sector swept
int32   threats_detected
int32   threats_confirmed
int32   threats_neutralized
int32   open_tasks
string  narrative                # human-readable one-liner for the operator
