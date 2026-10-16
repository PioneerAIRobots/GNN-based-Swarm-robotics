# Fused belief about one emitter, produced by the Bayesian localizer.
std_msgs/Header header
string  threat_id
geometry_msgs/Point position
float32[4] covariance_xy        # row-major 2x2 position covariance
float32 confidence              # [0..1] posterior mass inside the acceptance radius
float32 frequency_mhz
uint8   emitter_class
uint8   state
uint8 STATE_CANDIDATE=0
uint8 STATE_LOCALIZED=1
uint8 STATE_CONFIRMED=2
uint8 STATE_NEUTRALIZED=3
uint8   confirm_count
