VitalScan — RF Human Detection System
> **Portable, low-cost SDR-based human detection through walls for first responders.**  
> Built for ECE 510 (IoT/CPS Design) at Illinois Institute of Technology — May 2026.
---
The Problem
Firefighters and rescue teams regularly operate in zero-visibility environments where thermal cameras fail and manual search is slow and dangerous. Existing through-wall detection systems (e.g., Camero Xaver 400 at ~$47,500) are priced out of reach for most local and volunteer emergency services.
VitalScan delivers comparable detection capability for under $300 in hardware.
---
The Solution
VitalScan uses a ADALM-Pluto Software Defined Radio (SDR) operating at 5.8875 GHz as a bistatic CW Doppler radar (1 TX, 2 RX). By transmitting a continuous-wave RF signal and analyzing the Doppler frequency shift in reflections, the system detects human motion — crawling, walking, waving — even through walls and in smoke-filled environments.
Signal processing runs in Python on a connected laptop, with a planned Raspberry Pi 5 edge deployment for full tactical portability.
---
Hardware Architecture
Component	Role
ADALM-Pluto (AD9361, firmware-modified)	SDR core — 1 TX / 2 RX (U.Fl mod to unlock 2nd RX port)
12 dBi Planar Yagi Antennas (x3)	Directional TX and dual RX — improves wall penetration and rejects lateral clutter
EVAL-CN0523-EBZ (TX Amplifier)	+28 dB gain in 5.8 GHz ISM band
EVAL-CN0534-EBZ (LNA x2)	+23 dB gain, 2 dB NF — improves SNR on both RX chains
VBFZ-5500-S+ Band Pass Filters	4.9–6.2 GHz BPF, 30 dB rejection, 1.26 dB insertion loss
Raspberry Pi 5 (planned)	Edge compute host — self-contained Wi-Fi hotspot for field use
Anker 737 Power Bank (140W, 24K mAh)	Powers entire system (21–33W draw) via USB-C PD
Power budget: ~4.3–6.7A @ 5V. Pi 5 requires dedicated 5V/5A USB-C PD handshake to prevent performance throttling.
---
Software Stack
```
Language   : Python 3
SDR Library: pyadi-iio + libiio
DSP        : NumPy / SciPy
Display    : Matplotlib
Protocol   : USB 3.0 → IIO backend
Future     : Raspberry Pi 5 / Liquid-DSP edge deploy
```
---
Detection Algorithm — Two-Stage Classifier
Stage 1: Motion Classifier
Computes six features, z-scored against a CFAR (Constant False Alarm Rate) empty-room baseline:
Feature	Weight	Notes
`phase_rms` (1–60 Hz)	+6	Primary discriminant — dominant for all human motion
`temporal_cv`	+1	Temporal energy variability
`residual_mean`	+1	Mean clutter-subtracted Doppler energy
`p90p10_ratio`	0	Near-zero discrimination at current SNR
`frame_activity`	0	Redundant with phase_rms
`spec_entropy`	0	Near-zero discrimination at current SNR
Composite score → sigmoid → confidence value (0.0–1.0)
Stage 2: Clutter / Mechanical Rejection
Applies soft confidence penalties for mechanical sources (fans, HVAC):
High-Phase-Noise gate (>10 Hz)
Spectral-Width check (<30 Hz RMS)
Peak-Persistence (>70% bin stability)
Output per analysis chunk
```
motion_detected : bool
activity_level  : "LOW" | "MED" | "HIGH"
confidence      : float (0.0 – 1.0)
```
---
Usage
Installation
```bash
pip install pyadi-iio numpy scipy matplotlib
# Install libiio: https://github.com/analogdevicesinc/libiio/releases
# Connect ADALM-Pluto via USB (appears at ip:192.168.2.1 by default)
```
Running
```bash
# Step 1 — Calibrate with empty room (20 seconds)
python firefinder_engine.py --calibrate --calibrate_sec 20

# Step 2 — Live detection using saved calibration
python firefinder_engine.py

# Use pre-recorded empty files as clutter baseline
python firefinder_engine.py --clutter empty1.npz empty2.npz

# Offline demo/replay mode (no SDR hardware needed)
python firefinder_engine.py --demo file1.npz file2.npz --clutter empty.npz
```
CLI Flags
Flag	Default	Description
`--uri`	`ip:192.168.2.1`	Pluto URI
`--freq`	`5887500000`	Carrier frequency (Hz)
`--rx_gain`	`40`	RX gain (dB)
`--tx_gain`	`-40`	TX gain (dB)
`--sample_rate`	`2400000`	Hardware sample rate (Hz)
`--chunk_sec`	`2.0`	Seconds per analysis window
`--n_sigma`	`2.0`	CFAR multiplier
`--calibrate`	—	Run live auto-calibration
`--calibrate_sec`	`20`	Live calibration duration
`--clutter`	—	Empty room .npz file(s) for baseline
`--calib_file`	`firefinder_calibration.npz`	Save/load calibration path
`--demo`	—	Offline replay mode
`--adaptive`	—	Enable rolling clutter update
`--verbose`	—	Print feature breakdown each frame
---
RF System Design
Topology: Bistatic radar — 1 TX antenna, 2 RX antennas (linear array, 6-inch spacing)
Carrier: 5.8875 GHz (chosen to avoid ISM band interference)
Doppler range detected: 10–40 Hz shifts (crawling 0.5 m/s → walking 1.2 m/s)
Validated range: Up to 6 ft with LNA + TX PA active
Noise mitigation: Decoupling capacitors (100µF + 0.1µF) across TX/RX amplifier inputs to suppress switching regulator noise
Sample Doppler Calculations (Δf = 2fv/c)
Motion	Speed	Δf @ 5.8875 GHz
Crawling	0.5 m/s	~19.6 Hz
Walking	1.2 m/s	~47.1 Hz
Crawling (50% slowdown)	0.5 m/s	~39.3 Hz
Walking (80% slowdown)	1.2 m/s	~235.5 Hz
---
Key Challenges & Scope Decisions
Breathing detection abandoned: The ADALM-Pluto's internal noise floor is too high to resolve micro-Doppler signatures at breathing frequency (~0.3 Hz chest motion requires ~0.4 Hz Doppler resolution at 5.8875 GHz). System was scoped to robust active-motion detection instead.
Portability vs. isolation tradeoff: Antenna shroud cones (reflective foil + absorptive rim) were necessary to eliminate TX-to-RX leakage but doubled the expected chassis footprint.
---
Project Timeline
Phase	Status	Description
Phase 1: Foundation	✅ Complete	Procured hardware, validated CW Doppler baseline in open air
Phase 2: Hardware Mod	✅ Complete	Firmware-modified Pluto to enable 2nd RX port; scoped to motion detection
Phase 3: Barrier Testing	✅ Complete	Ported GNU Radio pipeline to Python (NumPy/Liquid-DSP); chassis design
Phase 4: Pi Integration	🔄 In Progress	Transitioning MTI pipeline to Raspberry Pi 5; stress testing <1s refresh
---
Future Work
Tactical portability: Redesign chassis with RF-absorbent lightweight materials to reduce horn footprint
Hardware accuracy: Transition from ADALM-Pluto to dedicated high-SNR FMCW transceiver for sub-Hz breathing/heartbeat resolution
Advanced UI: Tablet-native HUD with real-time 3D occupancy heatmaps for field responders
---
Market Comparison
Device	Technology	Target Market	Est. Cost
Camero Xaver 400	UWB Pulse Radar	Elite Military / SWAT	~$47,500
Seek FirePRO 300	Thermal Imaging	Fire Rescue	$1,200
NASA FINDER	Microwave Radar	Federal Disaster Response	Proprietary
VitalScan	SDR FMCW (ADALM-Pluto)	Local / Volunteer Responders	< $300
---
Team
Member	Role
Adrian Korwel	Team Captain & Primary Tester — DSP filter development, hardware/software integration
Elijah Johnson	Computing Lead — Raspberry Pi 5 optimization, edge processing strategy
Bryan Evans	Hardware & Design Lead — power distribution, component selection, CAD/3D printing
Alvaro Tenorio	Research Specialist — RF physics, signal propagation analysis
Course: ECE 510 — IoT/CPS Design Project  
Instructor: Dr. Jafar Saniie  
Institution: Illinois Institute of Technology  
Date: May 2026
---
Repository Structure
```
VitalScan/
├── firefinder_engine.py       # Main CW Doppler detection engine
├── docs/
│   ├── Final_Presentation.pdf # Final project presentation (ECE 510)
│   └── VitalScan_Report.pdf   # Full technical report
├── hardware/
│   └── schematic_v1.3.png     # 1TX/2RX bistatic power & signal schematic
└── README.md
```
---
References
ADALM-Pluto SDR
VBFZ-5500-S+ BPF Datasheet
CN0523 TX Amplifier
CN0534 LNA
NASA FINDER — Human Detection Through Concrete
Detecting Human Presence using FMCW Radar (IEEE)
