# References

**DeepFusionPose — Monocular Vision-Based Arm Tracking**
*Organized by category with inline notes on how each source was applied in this project.*

---

## 2D Pose Estimation

**[R1]** Google. *MediaPipe Pose.*
https://developers.google.com/mediapipe
> Used as the primary real-time 2D keypoint extraction framework. Provides 33 body landmarks including shoulder, elbow, wrist, and hip. Selected as the production framework after Month 1 comparative evaluation due to lowest jitter and highest detection rate at 30+ fps. Implemented in `src/pose/mediapipe_runner.py`.

**[R2]** Google. *MoveNet: Ultra fast and accurate pose detection model.*
https://www.tensorflow.org/hub/tutorials/movenet
> Integrated as the second baseline framework for comparative evaluation. MoveNet's lightning variant was evaluated for frame rate and elbow angle stability under dynamic arm motion. Implemented in `src/pose/movenet_runner.py`.

**[R3]** Papandreou, G. et al. *PoseNet: Real-time Human Pose Estimation in the Browser.*
TensorFlow Blog, May 2018.
https://blog.tensorflow.org/2018/05/real-time-human-pose-estimation-in.html
> Integrated as the third baseline for comparative evaluation. PoseNet provides 17 COCO keypoints and was compared against MediaPipe and MoveNet using MAE, RMSE, and jitter metrics. Implemented in `src/pose/posenet_runner.py`.

---

## Unity & Avatar Animation

**[R4]** Unity Technologies. *Unity Game Engine (version 2022 LTS).*
https://unity.com/
> Unity was used to build the 4-avatar Digital Twin visualization. The humanoid rig system, Animator component, and Transform hierarchy were used for real-time arm animation driven by UDP angle data.

**[R5]** Unity Technologies. *Quaternion and Euler Rotations in Unity.*
Unity Manual.
https://docs.unity3d.com/Manual/QuaternionAndEulerRotationsInUnity.html
> The key reference for implementing `Transform.localRotation = Quaternion.Euler(...)` in `ArmAngleController.cs`. Euler angles from the vision pipeline are decomposed per-axis (pitch, roll, yaw) and applied to the upper arm and forearm bones following Unity's left-handed coordinate convention.

**[R6]** *Nepali Kurtha Surwal — 3D Avatar Model.*
Sketchfab.
https://sketchfab.com/3d-models/nepali-kurtha-surwal-499d1002614e4ad398e901c66befc889
> Reference avatar model for the Unity Digital Twin scene. Humanoid rig compatible with Unity's Mecanim system, enabling automatic bone mapping via `Animator.GetBoneTransform(HumanBodyBones.LeftUpperArm)`.

**[R7]** *Indian Human Avatar — 3D Model.*
Sketchfab.
https://sketchfab.com/3d-models/indian-human-avatar-7d7f61652bfc4ad69a88e7177c5b9ac7
> Secondary avatar model reference. Both models were evaluated for Humanoid rig compatibility with Unity's `AutoFind Humanoid Bones` feature implemented in `ArmRigSetup.cs`.

---

## Human Pose & Arm Kinematics

**[R8]** Koritnik, T., Bajd, T. & Munih, M. *A Simple Kinematic Model of a Human Body for Virtual Environments.*
ResearchGate.
https://www.researchgate.net/publication/226438496
> Provided the theoretical basis for representing the human arm as a two-link rigid kinematic chain (upper arm + forearm). The simplified model — ignoring wrist pronation/supination and shoulder complex anatomy — is directly adopted in `joint_angle_estimator.py`, where shoulder elevation and elbow flexion are computed from 3D vector geometry.

**[R9]** Biryukova, E.V., Roby-Brami, A., Frolov, A.A., & Mokhtari, M. *Kinematics of Human Arm Reconstructed from Spatial Tracking System Recordings.*
Journal of Biomechanics, 33(8), 985–995, 2000.
https://doi.org/10.1016/S0021-9290(00)00040-3
> Used to validate the decomposition of shoulder motion into elevation (pitch), abduction (roll), and internal/external rotation (yaw) components. The paper's 7-DOF shoulder model informed the coordinate convention adopted in this project's simplified 3-DOF shoulder representation.

---

## Filtering & Signal Processing

**[R10]** *Comparison of Filtering Techniques for Human Motion Analysis.*
DergiPark Academic Journal.
https://dergipark.org.tr/en/download/article-file/4801294
> General comparative reference for filtering methods applied to biomechanical angle signals. Informed the decision to implement all four filters (Moving Average, EMA, Kalman, Savitzky–Golay) in `angle_filter.py` and evaluate them against the ±3–5° static variance target from the project specification.

**[R11]** Michel van Biezen. *Kalman Filter — Introduction (Lecture Series).*
YouTube.
- https://www.youtube.com/watch?v=HCd-leV8OkU
- https://www.youtube.com/watch?v=qCZ2UTgLM_g
- https://www.youtube.com/watch?v=DbE4PMgqp3s
- https://www.youtube.com/watch?v=F5m0riPln-o
- https://www.youtube.com/watch?v=VRKGRD3-_0U
> Five-part lecture series used to implement the scalar 1D Kalman filter (`KalmanFilter1D` in `angle_filter.py`). The predict-update cycle (process noise Q, measurement noise R, Kalman gain K) follows the derivation from these lectures directly. Kalman was selected as the default filter due to optimal balance of noise reduction and response speed for real-time joint angle estimation.

**[R12]** *Savitzky–Golay Filter — Explanation & Implementation.*
YouTube.
https://www.youtube.com/watch?v=I_K7cVlg2Cc
> Used to understand the polynomial least-squares fitting approach underlying the Savitzky–Golay filter. The real-time causal variant implemented in `SavitzkyGolayFilter` buffers the last `window_length` samples and calls `scipy.signal.savgol_filter` each frame, returning the polynomial-smoothed value at the most recent point.

---

## Deep Learning & Model Architecture

**[R13]** Vaswani, A. et al. *Attention Is All You Need.*
NeurIPS 2017. https://arxiv.org/abs/1706.03762
> Foundational reference for the Transformer encoder used in `DeepFusionPoseModel`. The multi-head self-attention mechanism allows the model to attend to relevant frames in the 60-frame input sequence when predicting joint angles.

**[R14]** Ionescu, C., Papava, D., Olaru, V. & Sminchisescu, C. *Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural Environments.*
IEEE TPAMI, 36(7):1325–1339, 2014.
https://ieeexplore.ieee.org/document/6682899
> The Human3.6M dataset was used as 3D ground truth for training the DeepFusionPose model. 3D joint positions from motion capture were converted to anatomical angles (shoulder pitch/roll/yaw, elbow flexion) using vector geometry, providing the supervised learning signal in `build_h36m_dataset.py`.

**[R15]** Gal, Y. & Ghahramani, Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*
ICML 2016. https://arxiv.org/abs/1506.02142
> Theoretical basis for the Monte Carlo Dropout uncertainty estimation implemented in `mc_predict()` in `fusion_network.py`. Dropout remains active at inference time over N forward passes; the mean and standard deviation of outputs provide calibrated uncertainty estimates used in the exoskeleton confidence gating logic.

**[R16]** Goodfellow, I., Pouget-Abadie, J. et al. *Generative Adversarial Nets.*
NeurIPS 2014. https://arxiv.org/abs/1406.2661
> Foundation for the Temporal GAN implemented in `gan_refinery.py`. The generator (DeepFusionPoseModel fine-tuned adversarially) and discriminator (1D-Conv temporal classifier) follow the original minimax training formulation, extended with a temporal smoothness loss term to reduce angle jitter.

---

## Rehabilitation Robotics & Exoskeleton Integration

**[R17]** Maciejasz, P. et al. *A Survey on Robotic Devices for Upper Limb Rehabilitation.*
Journal of NeuroEngineering and Rehabilitation, 11(3), 2014.
https://doi.org/10.1186/1743-0003-11-3
> Provided motivation and context for the exoskeleton calibration module. The survey establishes that vision-based reference signals are a clinically relevant tool for validating wearable sensor readings in upper-limb rehabilitation devices, directly supporting the design of `exoskeleton_streamer.py`.

**[R18]** Polygerinos, P. et al. *Soft Robotics: Review of Fluid-Driven Intrinsically Soft Devices.*
Advanced Engineering Materials, 19(12), 2017.
https://doi.org/10.1002/adem.201700016
> Background reference on soft pneumatic exoskeleton actuator characteristics. Informed the APPLY/HOLD gating logic in `ExoskeletonStreamer` — when Bayesian confidence drops below 0.85, the streamer holds the last safe angle value rather than transmitting uncertain estimates to an actuator.

---

## Software Libraries

| Library | Version | Use |
|---------|---------|-----|
| MediaPipe | 0.10.x | Pose landmark extraction |
| OpenCV | 4.x | Camera capture, frame processing, overlay drawing |
| PyTorch | 2.x | DeepFusionPose, GAN, MC Dropout training & inference |
| NumPy | 1.x | Angle computation, vector geometry |
| SciPy | 1.x | Savitzky–Golay filter (`scipy.signal.savgol_filter`) |
| Matplotlib | 3.x | Angle time-series plot generation |
| TensorFlow Hub | 0.14.x | MoveNet, PoseNet model loading |
| Unity | 2022 LTS | Digital Twin visualization, avatar rig control |
