• 对应的是 npu_ptv3_cpu_profile_fresh.json：

  - 点数：64
  - grasp 数：1
  - diffusion eval steps：1
  - PTv3 encoder：CPU
  - diffusion head：NPU
  - 稳态总耗时：均值 8.6367 s（两次，8.4866–8.7868 s）
  - 其中 generator PTv3 约 4.06 s，discriminator PTv3 约 4.50 s

  输入维度如下（均为 float32）：

   输入                         benchmark 原始形状    collate/model 形状
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━
   点云 points                             [64, 3]            [1, 64, 3]
  ───────────────────────────  ────────────────────  ────────────────────
   inputs（XYZ + 全零 RGB）                [64, 6]            [1, 64, 6]
  ───────────────────────────  ────────────────────  ────────────────────
   sweep_volume_open_and_mid                  [12]               [1, 12]
  ───────────────────────────  ────────────────────  ────────────────────

   z_offset                                    [1]                [1, 1]
  ───────────────────────────  ────────────────────  ────────────────────
   初始噪声 initial_noise                   [1, 6]                [1, 6]

  这里是 r3_so3 表示，因此 grasp/sample 的维度是 6，而不是 9。单步 diffusion 的 timestep 在循环中是标量，进入 denoising head 后会扩展成 batch 维度 [1]。

  网络内部的关键维度是：

  PTv3 coord:              [64, 3]
  PTv3 feat:               [64, 3]
  PTv3 offset:             [1]
  object_embedding:        [1, 512]
  gripper_embedding:       [1, 512]
  observation_embedding:   [1, 1024]

  diffusion head:
    observation_embedding  [1, 1024]
    timestep               [1]（原始循环变量是标量）
    sample                 [1, 6]
    output                 [1, 6]

  generated grasp:         [1, 1, 4, 4]
  confidence:              [1, 1, 1]

  可直接复用当时保存的输入文件：npu_ptv3_cpu.npz。其中：

  points:        (64, 3), float32
  initial_noise: (1, 6), float32

  当时的初始噪声是：

  [
      [-0.03582049, 3.06401086, 0.78554410,
       1.39773226, 0.52216345, 0.95267230]
  ]

  点云的生成方式是：

  rng = np.random.default_rng(1234)
  points = rng.normal(0.0, 0.05, size=(64, 3)).astype(np.float32)
  points -= points.mean(axis=0, keepdims=True)

  初始噪声使用：

  noise = np.random.default_rng(5678).standard_normal(
      (1, 6), dtype=np.float32
  )

  复现时建议直接使用保存的 NPZ：

  GRASPGENX_PTV3_CPU_SERIALIZATION=1 \
  python3 tools/benchmark_device.py \
    --device npu \
    --checkpoint-root /home/zf/GraspGenX/ext/graspgenx_checkpoints/release \
    --assets-dir /home/zf/GraspGenX/assets \
    --gripper parallel_2f_v1_1014 \
    --warmup 1 \
    --duration 15 \
    --num-grasps 1 \
    --diffusion-steps 1 \
    --ptv3-cpu \
    --profile-stages \
    --input-npz /home/zf/GraspGenX/npu_ptv3_cpu.npz \
    --output /tmp/reproduce.json

  注意：这是 benchmark 直接构造的 64 点输入。正常 GraspGenXSampler/server 可能会按配置把点云重采样到 cfg.data.num_points（当前配置通常是 3500），那样就不会得到这次 64 点测试的同样维度。
