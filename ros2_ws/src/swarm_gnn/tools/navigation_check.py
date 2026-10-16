#!/usr/bin/env python3
"""Can each platform actually reach the places the mission sends it?

Runs the real planner and the real local controller against the real sector
geometry, with no ROS and no coordination layer, and reports whether each
journey completes. Run from the package root:

    python3 tools/navigation_check.py

This exists because navigation failures are invisible from the mission log:
the symptom is a robot holding a task with a distance that never falls, which
looks identical whether the cause is allocation, arrival testing, or the
controller. Driving each leg in isolation separates them.

It has caught two real failures:

  * Courtyard entry was impossible for both ground platforms. The repulsive
    influence radius (3.0 m for the quadruped) exceeded the half-width of a
    4 m alley, so the flanking buildings pushed the robot back out of the
    mouth. Both bounced at ~13 m from the goal indefinitely while open-route
    travel worked, which is why it was not obvious. Fixed by projecting the
    opposing component of repulsion out of the route direction.
  * Repulsion was summed over scan returns rather than averaged, making the
    force proportional to the scanner's angular resolution.
"""
import math
import sys

sys.path.insert(0, ".")
import numpy as np
from swarm_gnn.model import sector
from swarm_gnn.model.planner import GridPlanner, PathFollower
from swarm_gnn.model.navigation import config_for, potential_field_command, ScanObstacles, NavState
from swarm_gnn.model.spec import Platform

def synth_scan(pos, yaw, samples=120, rmax=25.0):
    boxes = np.array(sector.building_boxes(include_rubble=False))
    angles = -math.pi + 2*math.pi*np.arange(samples)/samples + yaw
    dx, dy = np.cos(angles), np.sin(angles)
    rng = np.full(samples, np.inf)
    for bx,by,hx,hy in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            tx1=(bx-hx-pos[0])/dx; tx2=(bx+hx-pos[0])/dx
            ty1=(by-hy-pos[1])/dy; ty2=(by+hy-pos[1])/dy
        tn=np.maximum(np.minimum(tx1,tx2),np.minimum(ty1,ty2))
        tf=np.minimum(np.maximum(tx1,tx2),np.maximum(ty1,ty2))
        hit=(tf>=np.maximum(tn,0.0))&np.isfinite(tn)&(tn>0.12)
        rng=np.where(hit&(tn<rng),tn,rng)
    v=np.isfinite(rng)&(rng<rmax)
    return ScanObstacles(angles[v]-yaw, rng[v])

def in_obs(x,y,radius):
    if abs(x)>sector.SECTOR_HALF or abs(y)>sector.SECTOR_HALF: return True
    return sector.blocked(x,y,radius,include_rubble=False)

def run(platform, start, goal, label, secs=150.0, dt=0.1, accept=2.0):
    cfg = config_for(platform, 1.0)
    planner=GridPlanner(); follower=PathFollower(); st=NavState()
    radius={Platform.UGV:0.75,Platform.QUADRUPED:0.4,Platform.DRONE:0.0}[platform]
    pos=[start[0],start[1],0.4]; yaw=0.0
    plan=planner.plan(platform,start,goal); follower.set_plan(plan.waypoints)
    best=math.dist(start,goal); last_replan=0.0; prog=float('inf'); prog_t=0.0
    for k in range(int(secs/dt)):
        t=k*dt; here=(pos[0],pos[1])
        tgt=follower.current(here)
        if tgt is None or (t-prog_t>6.0 and t-last_replan>1.5):
            plan=planner.plan(platform,here,goal); follower.set_plan(plan.waypoints)
            last_replan=t; prog=float('inf'); prog_t=t; tgt=follower.current(here) or goal
        rem=math.dist(here,tgt)+3.0*(len(follower.waypoints)-follower.index)
        if rem<prog-0.5: prog=rem; prog_t=t
        obs=synth_scan(pos,yaw)
        vx,vy,vz,wz,_=potential_field_command(tuple(pos),yaw,(tgt[0],tgt[1],0.0),cfg,obs,[],st,dt=dt)
        yaw+=wz*dt
        ndx=(vx*math.cos(yaw)-vy*math.sin(yaw))*dt; ndy=(vx*math.sin(yaw)+vy*math.cos(yaw))*dt
        if not in_obs(pos[0]+ndx,pos[1]+ndy,radius): pos[0]+=ndx; pos[1]+=ndy
        elif not in_obs(pos[0]+ndx,pos[1],radius): pos[0]+=ndx
        elif not in_obs(pos[0],pos[1]+ndy,radius): pos[1]+=ndy
        d=math.dist((pos[0],pos[1]),goal); best=min(best,d)
        if d<accept:
            print(f"  {label:36s} ARRIVED {k*dt:6.1f} s"); return True
    print(f"  {label:36s} STUCK final {d:5.1f} m best {best:5.1f} m (start {math.dist(start,goal):5.1f} m)")
    return False

print("real geometry + planner + field:")
ok=[]
ok.append(run(Platform.QUADRUPED,(-52.,-3.5),(-46.,20.),"dog -> civilian courtyard"))
ok.append(run(Platform.QUADRUPED,(-52.,-3.5),(-20.,-20.),"dog -> trigger SW courtyard"))
ok.append(run(Platform.QUADRUPED,(-52.,-3.5),(20.,46.),"dog -> trigger N courtyard"))
ok.append(run(Platform.UGV,(-52.,0.),(20.,0.),"ugv -> along the main route"))
ok.append(run(Platform.UGV,(-52.,0.),(-46.,20.),"ugv -> civilian courtyard"))
ok.append(run(Platform.UGV,(-52.,0.),(46.,20.),"ugv -> jammer courtyard"))
ok.append(run(Platform.UGV,(-52.,0.),(0.,52.),"ugv -> rally point"))
print(f"\n{sum(ok)}/{len(ok)} journeys completed")
