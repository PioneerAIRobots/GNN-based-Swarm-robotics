# Flattened snapshot of the graph the GNN actually consumed, for visualisation.
std_msgs/Header header
string[]  agent_ids
string[]  task_ids
int32[]   edge_src              # node index, agents first then tasks
int32[]   edge_dst
float32[] edge_weight
uint8[]   edge_type             # 0 agent-agent, 1 agent-task, 2 task-task
float32[] attention             # learned attention per edge, same length as edge_src
int32     num_agent_nodes
