# UAV Light Show - Waypoint Navigation in Webots (Python, PID + Kalman)

## Overview
This project demonstrates a decentralized waypoint navigation system for a small UAV light show in simulation. Four drones fly simple formations (square and triangle) in Webots using Python controllers. Each drone runs its own script and follows its own waypoint list. Control is handled with two PID layers (position-to-velocity, and velocity/thrust/attitude) plus a simple Kalman filter for smoother state estimates.

Why this exists
- Built as a dissertation project to show an end-to-end control pipeline for formation flight.
- Focus on clear structure, easy reproducibility, and readable Python code.
- Designed for education and as a starting point for larger light show choreographies.

## Features
- Decentralized control: each drone has its own controller (no single master for the demo).
- Formation patterns: square and triangle, flown in a loop.
- Two-layer PID control:
  - Waypoint (position) PID converts position error into a desired velocity.
  - Velocity and attitude loop converts desired velocity into thrust and attitude/rate commands.
- Basic state estimation: a Kalman filter fuses noisy sensor signals (for example GPS and accelerometer) for smoother tracking.
- Practical touches: different cruise altitudes per drone to reduce collision risk; optional hover pauses for stability.

## System design (high level)
- Controllers (Drone1.py .. Drone4.py): each file sets up sensors, loads a waypoint route, applies the Kalman filter, and runs the PID loops each simulation step.
- PID modules (PID.py, cf_pid_control.py): reusable PID blocks and the higher-level control logic for waypoint and velocity control.
- Kalman filter (kalman_filter.py): light-weight filter to smooth and stabilize state estimates used by the controllers.

## Repository layout
.
- Drone1.py
- Drone2.py
- Drone3.py
- Drone4.py
- cf_pid_control.py
- PID.py
- kalman_filter.py
- reports/               
- media/
- README.md

## Requirements
- Webots (R2023 or newer recommended)
- Python 3.9 or newer
- numpy

If you run Python controllers from Webots, Webots already provides the controller API; you only need to make sure your Python environment has numpy available.

## Getting started
1) Install Webots and ensure Python controller support is enabled.
2) Place these files in your Webots project (for example under controllers/), or clone the repo directly inside your Webots project folder.
3) Open your world in Webots (for example, a world with four quadrotors or Crazyflie-like drones).
4) Assign each robot a controller script:
   - Drone 1 -> Drone1.py
   - Drone 2 -> Drone2.py
   - Drone 3 -> Drone3.py
   - Drone 4 -> Drone4.py
5) Ensure Webots can import the helper modules (cf_pid_control.py, PID.py, kalman_filter.py). The simplest way is to keep them in the same controller folder or add that folder to the controller PYTHONPATH.
6) Run the simulation. The drones take off, stabilise, and then start flying the waypoint pattern.

## How it works (brief)
- Waypoint tracking: for the current target point, compute position error and use a PID to produce a desired velocity vector.
- Velocity and attitude control: a second controller turns the desired velocity into thrust and attitude/rate commands for the quadrotor mixer.
- Kalman filter: fuses noisy sensor inputs to provide smoother estimates of position and velocity used by the PID loops.
- Formations: the waypoint lists describe corners of a square or triangle. Each drone repeats its path, with small altitude differences to reduce risk of conflict.

## Results (summary)
- Stable take-off and hover.
- Smooth transitions between waypoints and steady path tracking.
- Square and triangle formations closely match the planned shapes in simulation.
- Decentralized setup proved reliable for this small formation without a central supervisor.

## What to tweak
You can adjust these directly inside the DroneX.py files:
- Waypoint lists for square or triangle, or add your own shape.
- Hover time at each waypoint and cruise altitude.
- PID gains (position and velocity loops) to fine-tune responsiveness.
- Kalman filter noise settings to trade smoothness versus responsiveness.

## Business view
- Use cases: light shows, education, and prototyping multi-UAV choreography before hardware trials.
- Why decentralized: fewer single points of failure and simpler scaling at small sizes; easy to extend to more drones by adding more controllers and waypoints.
- Next steps to production: add a safety supervisor (collision checks, geofencing), robust radio or telemetry, and logging or analytics for show operations.

## Limitations and future work
- Simulation only; no wind or outdoor effects are modeled here.
- No formal multi-agent collision avoidance beyond altitude separation.
- A central supervisor with closed-loop corrections is a natural next step for larger fleets.
- Porting to real drones would require hardware-specific interfaces and safety features.

## Frequently asked questions
- Do I need anything beyond numpy?
  Webots provides the controller API. numpy is enough for the math used here.
- Can I run this without Webots?
  The control code is written for Webots controllers. For other sims or hardware, you would need adapters.
- Can I add more drones?
  Yes. Duplicate a drone controller script and assign new waypoints and altitude.

## License
MIT

## Acknowledgements
Built as part of a dissertation project. Thanks to the Webots team for a great simulator.
