# A single sensor observation of a hostile RF emitter.
std_msgs/Header header
string  observer_id
geometry_msgs/Point observer_position
float32 rssi_dbm                # received power at the observer
float32 bearing_rad             # DF bearing in `map` frame; NaN if omni-only
float32 bearing_sigma_rad
float32 frequency_mhz
uint8   emitter_class
uint8 CLASS_UNKNOWN=0
uint8 CLASS_JAMMER=1
uint8 CLASS_IED_TRIGGER=2
uint8 CLASS_CIVILIAN=3
