# Research System Specification (v2 — July 2026)

The proposed research system shall not operate as a simple pose estimation demonstrator but as a rigorous scientific benchmarking framework designed for publication-quality evaluation of monocular upper limb motion capture using a single RGB camera. The system shall process every input video frame simultaneously through three independent pose estimation frameworks — MediaPipe Pose, MoveNet, and PoseNet — and shall support bilateral tracking of both the left and right upper limbs including shoulder and elbow joint kinematics. Following preprocessing and landmark extraction, the framework shall provide the capability to visualize the exact same video sequence processed by all three pose estimation libraries side-by-side in synchronized playback windows, enabling direct qualitative comparison of landmark placement accuracy, tracking stability, occlusion handling, and temporal consistency under identical conditions. This visual comparison component is considered essential for scientific reproducibility and shall include frame synchronization, overlayed skeletal landmarks, confidence scores, frame timestamps, and error visualization relative to ground truth annotations where available.

The system shall be built upon publicly available benchmark datasets containing synchronized videos and validated ground truth joint positions or joint angles, ensuring that all reported results are scientifically defensible and comparable with prior literature. Candidate datasets include Human3.6M, MPI-INF-3DHP, TotalCapture, CMU Panoptic Studio, or rehabilitation-specific upper limb datasets depending on publication objectives. Dataset videos shall be directly used as inputs to the evaluation pipeline rather than relying on synthetically generated data or manually selected examples. The experimental methodology shall employ reproducible train-validation-test partitions or subject-independent cross-validation strategies such as Leave-One-Subject-Out Cross Validation to establish generalization performance across unseen participants. Training datasets shall be used only for parameter optimization, validation datasets for model and filter selection, and testing datasets shall remain completely isolated until final evaluation to eliminate data leakage and ensure statistically valid results.

For every frame, MediaPipe, MoveNet, and PoseNet shall independently estimate body landmarks and upper limb joint positions, after which biomechanical joint angles for both left and right shoulders and elbows shall be computed using vector geometry, coordinate transformations, and inverse kinematic formulations. These measurements shall then pass through an AngleFilterBank consisting of a two-state Kalman Filter, Savitzky-Golay Filter, and Moving Average Filter. The two-state Kalman Filter shall model both joint angle and angular velocity, allowing prediction of motion trajectories and significantly reducing latency during rapid movements while maintaining high smoothness. The Savitzky-Golay filter shall preserve sharp motion peaks and dynamic characteristics, while the Moving Average filter shall serve as a computationally inexpensive baseline method for comparison purposes. An ablation study shall quantify the effect of each filter on accuracy, jitter reduction, temporal stability, and latency.

The framework shall generate publication-quality quantitative outputs rather than arbitrary numerical values or visual estimates. All evaluation metrics must be derived directly from comparisons against dataset ground truth annotations and shall include Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Mean Per Joint Position Error (MPJPE), Percentage of Correct Keypoints (PCK), coefficient of determination (R²), temporal jitter metrics, latency measurements, confidence intervals, and inference throughput measured in frames per second. Statistical significance testing including paired t-tests, Wilcoxon signed-rank tests, and effect size calculations shall be performed to determine whether performance differences between models and filtering methods are meaningful rather than incidental. The framework shall automatically generate comparison plots, error distributions, time-series graphs, Bland-Altman plots, confusion analyses where applicable, and publication-ready visualizations suitable for direct inclusion in journal papers and conference proceedings.

In addition to graphical outputs, the system shall automatically export all experimental results into structured Excel spreadsheets, CSV files, and machine-readable reports containing frame-level predictions, ground truth values, error statistics, model confidence scores, inference times, and filter outputs for every evaluated sequence. Researchers shall be able to reproduce every result from these exported files without rerunning the experiments. The generated reports shall contain side-by-side model comparisons, ranking tables, summary statistics, and reproducibility metadata including dataset versions, model versions, parameter settings, and evaluation protocols. Accuracy shall always take precedence over producing visually appealing outputs, and no metric shall be generated from heuristics, arbitrary assumptions, or simulated values. Every reported number in the study must originate from validated dataset annotations, benchmark protocols, or experimentally measured quantities to ensure scientific integrity and publication readiness.

The final stage of the pipeline shall optionally transmit filtered bilateral joint angles through UDP communication to a Unity humanoid avatar for qualitative verification of motion reconstruction fidelity. However, the Unity visualization component shall remain secondary to the primary research contribution, which is the creation of a fully automated, reproducible, mathematically rigorous, and experimentally validated monocular upper limb motion capture benchmarking framework capable of producing publishable results and serving as a low-cost alternative to marker-based motion capture systems for rehabilitation robotics, digital twins, exoskeleton calibration, human-computer interaction, and biomechanical analysis.

--------------
# Original Project Proposal (v1)
Monocular Vision-Based Estimation of Human Arm Joint Angles for Real-Time Digital Avatar Control

Motivation
Restoring and assisting human arm movement is a major goal in rehabilitation robotics and wearable assistive technologies. Individuals recovering from stroke, spinal cord injury, or neuromuscular disorders often require repetitive therapy and motion assistance to regain functional use of their upper limbs. Soft wearable exoskeletons are emerging as safe and comfortable systems for providing such assistance. However, effective operation, evaluation, and personalization of these devices require reliable methods to estimate human joint motion in real time.
Traditional motion capture approaches rely on markers, multiple cameras, or wearable sensors, increasing system cost and complexity and limiting use outside laboratory environments. Recent advances in vision-based pose estimation now allow tracking of human body keypoints using a single RGB camera, enabling low-cost, non-contact motion tracking. Beyond visualization, such vision systems can serve as a calibration and validation reference for wearable robotic systems.
This project focuses on developing a monocular vision-based framework to estimate human arm joint angles in real time and use them to control a digital avatar. More importantly, the system is designed as a foundational platform for future integration with soft wearable arm exoskeletons. Vision-derived joint angles can serve as a reference for calibrating wearable sensors, validating exoskeleton motion, and supporting feedback strategies in assistive and rehabilitation contexts. The resulting technology has applications in rehabilitation training, assistive arm support, human-machine interaction, teleoperation, and digital human modeling.
Objective
The objective of this project is to design and develop a monocular camera-based system that estimates shoulder and elbow joint angles from 2D pose keypoints and uses them to drive a humanoid avatar in Unity in real time. The system will include temporal filtering to reduce jitter, a calibration procedure to map estimated human joint angles to avatar joint rotations, and a communication pipeline between the vision module and Unity. The architecture will be designed so that the vision-based joint angle estimates can serve as a future reference signal for calibrating and validating soft wearable arm exoskeleton systems.
System Description
The system consists of three primary subsystems. The first subsystem is a vision module that captures video from a monocular RGB camera and uses a 2D pose estimation framework (e.g., MediaPipe Pose or MoveNet) to extract upper-body keypoints. The second subsystem is a processing module that computes arm joint angles using geometric relationships and simplified kinematic constraints, followed by temporal filtering. The third subsystem is a Unity-based visualization module where a rigged humanoid avatar mirrors the user’s arm motion in real time. The system estimates arm joint rotations, visualizes them through the avatar, and logs joint angle data that can serve as a reference dataset for future wearable exoskeleton calibration.
Methodology and Timeline
Remote Tasks (2–4 Weeks)
Review background material on pose estimation, arm kinematics, and Unity humanoid rig control. Extend an existing Unity full body avatar animation so that joint rotations are driven by numeric data received from an external application rather than keyboard input.
Task 1 – Data Generator Application
Develop a small desktop application in Python that continuously transmits numeric angle values over a UDP or serial connection. These values will simulate shoulder and elbow joint angles.
Task 2 – Unity Communication Integration
Implement a communication module in Unity that receives incoming angle data, parses it, and assigns values to the avatar’s upper arm and lower arm bones.
Task 3 – Avatar Arm Animation
Configure a humanoid avatar in Unity. Joint rotations must be applied using Transform.localRotation and Quaternion. .Euler. Arm motion should be smooth and frame-rate independent.
Task 4 – Angle Filtering in Unity
Implement a simple smoothing filter within Unity to reduce jitter in incoming angle data and verify stable animation.
Communication Protocol (Arm Model)
Update rate should be at least 20–30 Hz. Each packet must end with a newline character. Each line represents one full arm pose.
General format
S,shoulder_pitch,shoulder_yaw,shoulder_roll,elbow_flex
All values are in degrees. The coordinate convention and rotation axes will be defined relative to the Unity avatar coordinate frame during integration.
Onsite Tasks
Month 1 – Pose Estimation Framework Evaluation and Vision Pipeline Setup
Set up camera capture and integrate an initial 2D pose estimation framework (e.g., MediaPipe Pose). Extract upper-body keypoints including shoulder, elbow, wrist, and hip, and display them overlaid on video. Integrate at least two additional frameworks (e.g., PoseNet and MoveNet) and run both systems under similar conditions. Compare them using practical engineering metrics such as frame rate, keypoint jitter, stability of computed elbow angles during static poses, robustness during arm motion, and computational load. Document results and select the most suitable framework for the remainder of the project. The outcome is a reliable real-time tracking pipeline and a justified model selection. The goal is comparative evaluation, not model modification or retraining.
Month 2 – Joint Angle Computation
Develop algorithms to compute elbow flexion and simplified shoulder angles using vector geometry. Treat the arm as a two-link kinematic chain. Shoulder motion will be represented using elevation (arm lift relative to torso) and horizontal rotation components derived from upper-arm vector orientation. Validate angle outputs using simple reference poses (arm straight, arm bent). The outcome is a working joint angle estimation module. 
Month 3 – Unity Integration
Replace simulated data with real-time computed angles. Establish stable communication between the vision application and Unity. The avatar’s arm should mirror the user’s arm motion in real time.
Month 4 – Temporal Filtering, Calibration and Mapping
Implement filtering techniques such as moving average, Savitzky–Golay smoothing, and Kalman filtering in the processing module. Compare filtered versus unfiltered angles. Static pose angle variance after filtering should remain within approximately ±3–5 degrees. Optimize update rate and reduce latency. The outcome is smooth and stable avatar motion and cleaner joint angle signals suitable for use as calibration references. Develop a calibration routine to map human joint angles to avatar joint limits. Users perform reference poses (arm down, arm forward, elbow flexed). Calibration parameters are stored and applied during operation. Document how the same joint angle outputs could be used as reference signals for calibrating wearable exoskeleton joint sensors in future systems. 
Month 5 – System Evaluation and Refinement
Evaluate performance in terms of latency, stability, and angle consistency. Improve robustness under moderate motion and partial occlusion. Implement data logging and visualization tools for joint angle time series. Prepare final demonstration and documentation highlighting the system’s role as a vision-based reference for wearable robotics calibration.
Technical Challenges
Challenges include noise and jitter in 2D keypoints, partial occlusion of joints, depth ambiguity in monocular vision, and mapping human motion to simplified kinematic models. Ensuring stable real-time animation requires robust filtering and constraint handling. Designing a calibration pipeline that produces consistent joint angles suitable for use as reference signals is also a key challenge. These engineering aspects form central learning objectives of the project.
Modeling Assumptions
For this project, the human arm is represented using a simplified kinematic model designed for vision-based motion estimation rather than anatomical precision. The arm is treated as two rigid segments: the upper arm (shoulder to elbow) and the forearm (elbow to wrist), whose lengths are assumed to remain constant for a given user, with small variations due to tracking noise ignored. The elbow joint is modeled as a single rotational joint, and its flexion angle is computed from the relative orientation of the upper arm and forearm vectors. Shoulder motion is represented using simplified rotational components derived from the orientation of the upper arm relative to the torso, without requiring detailed biomechanical modeling of the shoulder complex. The system focuses on estimating joint angles rather than precise 3D joint positions, and the objective is to achieve consistent and smooth angle estimates suitable for real-time avatar animation and use as a reference motion signal, rather than medical-grade measurement accuracy.
Performance Requirements
The system shall operate in real time with end-to-end latency below 100 milliseconds. The vision module should run at a minimum of 20 frames per second. Avatar motion should appear smooth without noticeable jitter after filtering. Joint angle outputs should remain stable during static poses. The system should run continuously for at least 10 minutes without crashes.
Hardware Scope and Constraints
A standard RGB webcam or laptop camera will be used. No depth sensors, motion capture markers, or wearable sensors are required for this project phase. Processing will run on a typical student laptop. The system is intended as a software-focused research prototype and a reference motion capture layer for future wearable exoskeleton integration.
Evaluation Criteria
Evaluation will consider real-time performance, smoothness of avatar motion, robustness of tracking under moderate motion, and effectiveness of filtering. The usefulness of the joint angle outputs as consistent and repeatable motion references will also be assessed. Code organization, modularity, and documentation quality form part of the evaluation. A final demonstration showing live arm motion mirrored by the Unity avatar is required.
Final Deliverables
A real-time vision-based arm tracking application
Joint angle estimation and filtering module
Unity application with avatar arm control
Calibration module
Joint angle data logging and visualization tools
Technical documentation and demonstration
Video recordings demonstrating system operation
Optional Extensions
Use of a pretrained 2D-to-3D lifting model for improved shoulder angle estimation
Gesture recognition using arm motion
Integration with VR or AR environments
Future integration of joint angle outputs with a wearable arm exoskeleton for calibration and validation
References
2D Pose Estimation
MediaPipe Pose: https://developers.google.com/mediapipe
MoveNet: https://www.tensorflow.org/hub/tutorials/movenet
PoseNet: https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html
Unity
Unity Game Engine: https://unity.com/
Unity Manual: Quaternion and Euler Rotations: https://docs.unity3d.com/Manual/QuaternionAndEulerRotationsInUnity.html
Example 3D Avatar models: https://sketchfab.com/3d-models/nepali-kurtha-surwal-499d1002614e4ad398e901c66befc889
https://sketchfab.com/3d-models/indian-human-avatar-7d7f61652bfc4ad69a88e7177c5b9ac7?
Human Pose and Kinematics Background
Koritnik, T., Bajd, T. & Munih, M. A Simple Kinematic Model of a Human Body for Virtual Environments. https://www.researchgate.net/publication/226438496
Biryukova, E.V., A. Roby-Brami, A.A. Frolov, and M. Mokhtari. “Kinematics of Human Arm Reconstructed from Spatial Tracking System Recordings.” Journal of Biomechanics 33, no. 8 (2000): 985–95. https://doi.org/10.1016/S0021-9290(00)00040-3.
Filtering and Signal Processing
Filtering techniques: https://dergipark.org.tr/en/download/article-file/4801294
Kalman filter: https://www.youtube.com/watch?v=HCd-leV8OkU, https://www.youtube.com/watch?v=qCZ2UTgLM_g, https://www.youtube.com/watch?v=DbE4PMgqp3s,
https://www.youtube.com/watch?v=F5m0riPln-o,
https://www.youtube.com/watch?v=VRKGRD3-_0U

Savitzky-Golay Filter: youtube.com/watch?time_continue=1&v=I_K7cVlg2Cc&embeds_ referring_euri=https%3A%2F%2Fchatgpt.com%2F&source_ve_path=Mjg2NjY
