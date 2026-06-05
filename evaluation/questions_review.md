# QA Review Sheet

This file is for manual review only. The executable dataset remains
`evaluation/questions.jsonl`.

## Always Clear Depth

### 1. Main Contributions

**ID:** `always_clear_depth_contributions`

**Question:** Always Clear Depth 论文的主要贡献包括哪些？

**Gold Answer:** Always Clear Depth 的主要贡献包括：提出 multi-tuple degradation dataset generation scheme；提出 ACDepth 框架用于恶劣天气下的鲁棒单目深度估计；提出 multi-granularity knowledge distillation, MKD；引入 ordinal guidance distillation, OGD，使模型关注不确定区域并提升深度估计。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `multi_tuple_degradation_dataset_generation` | multi-tuple degradation dataset generation scheme | multi-tuple degradation dataset generation; multi-tuple degradation dataset generation scheme; degradation dataset generation scheme; elaborate data generation scheme |
| `acdepth_framework` | ACDepth framework | ACDepth; ACDepth framework; robust monocular depth estimation framework; ACDepth approach; novel approach named ACDepth |
| `multi_granularity_knowledge_distillation` | multi-granularity knowledge distillation | multi-granularity knowledge distillation; MKD; multi-granularity knowledge distillation strategy; effective multi-granularity knowledge distillation |
| `ordinal_guidance_distillation` | ordinal guidance distillation | ordinal guidance distillation; OGD |

**Candidate Items / Distractors:**

`multi_tuple_degradation_dataset_generation`, `acdepth_framework`, `multi_granularity_knowledge_distillation`, `ordinal_guidance_distillation`, `feature_consistency_constraint`, `lora_adapters`, `depth_anything_v2`, `md4all`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 2 | `multi_tuple_degradation_dataset_generation` | The Introduction contribution bullets state that the paper proposes a practical multi-tuple degradation dataset generation scheme. |
| 7 | `multi_tuple_degradation_dataset_generation` | The Conclusion states that ACDepth introduces an elaborate data generation scheme to produce the multi-tuple depth dataset under diverse degradation conditions. |
| 2 | `acdepth_framework` | The Introduction contribution bullets say the paper develops ACDepth for high-quality depth estimation under adverse weather. |
| 7 | `acdepth_framework` | The Conclusion states that the paper proposes a novel approach named ACDepth for robust monocular depth estimation under adverse weather conditions. |
| 2 | `multi_granularity_knowledge_distillation` | The Introduction contribution bullets identify MKD as a strategy for transferring and aligning teacher and student capabilities. |
| 7 | `multi_granularity_knowledge_distillation` | The Conclusion states that the paper constructs an effective multi-granularity knowledge distillation (MKD) strategy for robust model training and teacher-student capability alignment. |
| 2 | `ordinal_guidance_distillation` | The Introduction contribution bullets introduce OGD to encourage the network to focus on uncertain regions through differential ranking. |

### 2. Evaluation Datasets

**ID:** `always_clear_depth_eval_datasets`

**Question:** Always Clear Depth 在哪些数据集上进行了实验评估？

**Gold Answer:** Always Clear Depth 在 nuScenes 和 RobotCar 两个数据集上进行了实验评估。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `nuscenes` | nuScenes | nuScenes; nuScenes dataset |
| `robotcar` | RobotCar | RobotCar; Oxford RobotCar; RobotCar dataset |

**Candidate Items / Distractors:**

`nuscenes`, `robotcar`, `kitti`, `cityscapes`, `ddad`, `waymo`, `nighttime_synthetic_drive`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 5 | `nuscenes` | The Datasets section says nuScenes is used for training and comparison, with generated samples for training and testing. |
| 5 | `robotcar` | The Datasets section says RobotCar is used for training and comparison, with day and night samples for testing. |

### 3. Ablation Components

**ID:** `always_clear_depth_ablation_components`

**Question:** Always Clear Depth 的消融实验验证了哪些组件？

**Gold Answer:** Always Clear Depth 的主要设计组件消融实验验证了 DL、OGD 和 FCC 三个组件，分别对应 distillation learning、ordinal guidance distillation 和 feature consistency constraint。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `distillation_learning` | distillation learning | distillation learning; DL |
| `ordinal_guidance_distillation` | ordinal guidance distillation | ordinal guidance distillation; OGD |
| `feature_consistency_constraint` | feature consistency constraint | feature consistency constraint; FCC |

**Candidate Items / Distractors:**

`distillation_learning`, `ordinal_guidance_distillation`, `feature_consistency_constraint`, `multi_granularity_knowledge_distillation`, `lora_adapters`, `depth_anything_v2`, `resnet18`, `acdepth_framework`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 7 | `distillation_learning` | Table 3 is the ablation study of design components and defines DL as distillation learning. |
| 7 | `ordinal_guidance_distillation` | Table 3 defines OGD as ordinal guidance distillation and includes it in the design component ablation. |
| 7 | `feature_consistency_constraint` | Table 3 defines FCC as feature consistency constraint and includes it in the design component ablation. |

### 4. SoTA Comparison Methods

**ID:** `always_clear_depth_sota_comparison_methods`

**Question:** Always Clear Depth 在 Comparison with SoTAs 中与哪些方法进行了比较？

**Gold Answer:** Always Clear Depth 在 Comparison with SoTAs 中与 Monodepth2、PackNet-SfM、RNW、md4all-AD、md4all-DD、DMMDE、DeFeatNet、ADIDS 和 WSGD 等方法进行了比较。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `monodepth2` | Monodepth2 | Monodepth2; Godard et al., 2019 |
| `packnet_sfm` | PackNet-SfM | PackNet-SfM; Guizilini et al., 2020 |
| `rnw` | RNW | RNW; Regularizing Nighttime Weirdness; Wang et al., 2021 |
| `md4all_ad` | md4all-AD | md4all-AD |
| `md4all_dd` | md4all-DD | md4all-DD; md4all; Gasperini et al., 2023; Robust Monocular Depth Estimation under Challenging Conditions |
| `dmmde` | DMMDE | DMMDE; DMMDE v1; DMMDE v2; DMMDE v3; Tosi et al., 2025 |
| `defeatnet` | DeFeatNet | DeFeatNet; Spencer et al., 2020 |
| `adids` | ADIDS | ADIDS; Liu et al., 2021 |
| `wsgd` | WSGD | WSGD; Vankadari et al., 2023 |

**Candidate Items / Distractors:**

`monodepth2`, `packnet_sfm`, `rnw`, `md4all_ad`, `md4all_dd`, `dmmde`, `defeatnet`, `adids`, `wsgd`, `acdepth`, `depth_anything_v2`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 5 | `comparison_scope` | The Comparison with SoTAs section states that ACDepth is evaluated on nuScenes and RobotCar with qualitative and quantitative comparisons. |
| 5 | `monodepth2,packnet_sfm,rnw,md4all_ad,md4all_dd,dmmde` | The nuScenes comparison describes typical MDE methods Monodepth2 and PackNet-SfM, and robust depth estimation approaches RNW, md4all and DMMDE; Table 1 includes md4all-AD, md4all-DD and DMMDE variants. |
| 7 | `monodepth2,defeatnet,adids,rnw,wsgd,md4all_dd,dmmde` | Table 2 on RobotCar lists Monodepth2, DeFeatNet, ADIDS, RNW, WSGD, md4all-DD and DMMDE variants as comparison methods. |

## DepthDark

Draft for manual review. These QA pairs are not yet added to
`evaluation/questions.jsonl`.

### 1. Main Contributions

**ID:** `depthdark_contributions`

**Question:** DepthDark 论文的主要贡献包括哪些？

**Gold Answer:** DepthDark 的主要贡献包括：提出面向弱光单目深度估计的 DepthDark 方法；提出低光数据生成方案 LLDG，通过 flare-simulation 和 noise-simulation 合成弱光配对深度数据；提出低光参数高效微调策略 LLPEFT；在 LLPEFT 中结合 illumination guidance 和 multiscale feature fusion，以提升弱光环境下的鲁棒性。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `depthdark_model` | DepthDark | DepthDark; robust foundation model; robust foundation model for low-light monocular depth estimation |
| `low_light_data_generation` | low-light data generation | low-light data generation; LLDG; low-light synthesis approach; low-light data synthesis |
| `flare_simulation_module` | flare-simulation module | flare-simulation module; flare simulation; light source simulation |
| `noise_simulation_module` | noise-simulation module | noise-simulation module; noise simulation; physically decoupled noise simulation |
| `low_light_peft` | low-light PEFT | low-light PEFT; LLPEFT; parameter-efficient fine-tuning strategy; PEFT strategy |
| `illumination_guidance` | illumination guidance | illumination guidance; illumination perception |
| `multiscale_feature_fusion` | multiscale feature fusion | multiscale feature fusion; multi-scale feature fusion |

**Candidate Items / Distractors:**

`depthdark_model`, `low_light_data_generation`, `flare_simulation_module`, `noise_simulation_module`, `low_light_peft`, `illumination_guidance`, `multiscale_feature_fusion`, `depth_anything_v2`, `tddc`, `adds`, `hypersim`, `virtual_kitti`

**Gold Evidence:**

<table>
  <colgroup>
    <col style="width: 8%;">
    <col style="width: 26%;">
    <col style="width: 66%;">
  </colgroup>
  <thead>
    <tr>
      <th>Page</th>
      <th>Evidence For</th>
      <th>Evidence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td><code>depthdark_model</code>, <code>flare_simulation_module</code>, <code>noise_simulation_module</code>, <code>low_light_peft</code>, <code>illumination_guidance</code>, <code>multiscale_feature_fusion</code></td>
      <td>The Abstract introduces DepthDark for low-light monocular depth estimation and describes the flare-simulation module, noise-simulation module, low-light PEFT strategy, illumination guidance, and multiscale feature fusion.</td>
    </tr>
    <tr>
      <td>2</td>
      <td><code>low_light_data_generation</code>, <code>low_light_peft</code></td>
      <td>The contribution bullets state that the paper proposes a low-light image synthesis approach and designs an efficient parameter-efficient fine-tuning strategy.</td>
    </tr>
    <tr>
      <td>3</td>
      <td><code>low_light_data_generation</code>, <code>flare_simulation_module</code>, <code>noise_simulation_module</code></td>
      <td>The Method section states that LLDG is introduced for the first time and integrates flare-simulation and noise-simulation modules to simulate low-light imaging and produce high-quality paired depth data.</td>
    </tr>
    <tr>
      <td>3</td>
      <td><code>low_light_peft</code></td>
      <td>The Method section states that the paper proposes an efficient Low-Light Parameter-Efficient Fine-Tuning (LLPEFT) strategy tailored for low-light scenarios.</td>
    </tr>
    <tr>
      <td>5</td>
      <td><code>low_light_peft</code>, <code>illumination_guidance</code>, <code>multiscale_feature_fusion</code></td>
      <td>The LLPEFT method section states that the LLPEFT strategy integrates illumination guidance and multiscale feature fusion to guide and enhance model performance under challenging low-light conditions.</td>
    </tr>
    <tr>
      <td>8</td>
      <td><code>flare_simulation_module</code>, <code>noise_simulation_module</code>, <code>low_light_peft</code>, <code>illumination_guidance</code>, <code>multiscale_feature_fusion</code></td>
      <td>The Conclusion summarizes the flare/noise simulation techniques and the PEFT strategy with illumination guidance and multiscale feature fusion.</td>
    </tr>
  </tbody>
</table>

### 2. Evaluation Datasets

**ID:** `depthdark_eval_datasets`

**Question:** DepthDark 在哪些数据集上进行了实验评估？

**Gold Answer:** DepthDark 主要在两个弱光单目深度估计基准数据集上进行实验评估：nuScenes-Night 和 RobotCar-Night。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `nuscenes_night` | nuScenes-Night | nuScenes-Night; nuScenes Night; N-N |
| `robotcar_night` | RobotCar-Night | RobotCar-Night; RobotCar Night; R-N |

**Candidate Items / Distractors:**

`nuscenes_night`, `robotcar_night`, `hypersim`, `virtual_kitti`, `kitti`, `nuscenes`, `robotcar`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 1 | `nuscenes_night,robotcar_night` | The Abstract reports that DepthDark achieves state-of-the-art performance on the challenging nuScenes-Night and RobotCar-Night datasets. |
| 2 | `nuscenes_night,robotcar_night` | The Introduction contribution bullets state that DepthDark achieves state-of-the-art performance on nuScenes-Night and RobotCar-Night. |
| 6 | `nuscenes_night,robotcar_night` | The Evaluation Protocol states that the method is evaluated on nuScenes-Night and RobotCar-Night. |
| 6 | `nuscenes_night,robotcar_night` | Table 4.1 reports separate results for Test on nuScenes-Night and Test on RobotCar-Night. |
| 8 | `nuscenes_night,robotcar_night` | The ablation tables report results on both nuScenes-Night and RobotCar-Night. |

### 3. Training Datasets

**ID:** `depthdark_training_datasets`

**Question:** DepthDark 在哪些数据集上进行了训练？

**Gold Answer:** DepthDark 使用 Hypersim 和 Virtual KITTI 作为训练数据。论文说明这两个数据集分别提供室内和合成室外场景，并且由于它们只包含 daytime images，作者使用 Section 3.1 的 LLDG 方法将其合成为高质量、低光、深度对齐的训练数据。论文也提到 KITTI，但认为 KITTI 训练集在部分场景中 ground truth 不可靠、数据量有限且场景泛化能力较弱，因此没有将其作为 DepthDark 的主要训练数据。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `hypersim` | Hypersim | Hypersim; Hypersim dataset; H |
| `virtual_kitti` | Virtual KITTI | Virtual KITTI; Virtual KITTI dataset; VK |

**Candidate Items / Distractors:**

`hypersim`, `virtual_kitti`, `kitti`, `nuscenes_night`, `robotcar_night`, `flare7k`, `depth_anything_unlabeled_data`, `depth_anything_v2_real_data`, `depth_anything_v2_synthetic_data`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 6 | `hypersim,virtual_kitti` | Section 4.2.1 Training Datasets states that the paper selects the Hypersim indoor dataset and the Virtual KITTI synthetic outdoor dataset as training data. |
| 6 | `hypersim,virtual_kitti` | Section 4.2.1 explains that these datasets contain only daytime images, so Section 3.1 is used to synthesize a large-scale low-light depth-aligned dataset for training DepthDark. |
| 6 | `hypersim,virtual_kitti` | Table 4.1 marks Ours DepthDark as trained on H and VK, which the table caption defines as Hypersim and Virtual KITTI. |
| 6 | `kitti` | Section 4.2.1 mentions KITTI but says its training ground truth is unreliable in some scenarios and has limited data volume and weak scene generalization, so it is not ideal for training foundation models. |

### 4. Ablation Components

**ID:** `depthdark_ablation_components`

**Question:** DepthDark 的消融实验验证了哪些组件？

**Gold Answer:** DepthDark 的主要组件消融实验验证了 LLDG 和 LLPEFT 两个模块。LLDG 负责生成低光训练数据，LLPEFT 负责低光场景下的参数高效微调，并结合 illumination guidance 和 multiscale feature fusion。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `lldg` | LLDG | LLDG; low-light data generation; low-light data generation module |
| `llpeft` | LLPEFT | LLPEFT; low-light PEFT; low-light parameter-efficient fine-tuning |

**Candidate Items / Distractors:**

`lldg`, `llpeft`, `flare_simulation_module`, `noise_simulation_module`, `illumination_guidance`, `multiscale_feature_fusion`, `amfg`, `lora`, `depth_anything_v2`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 8 | `lldg,llpeft` | Table 4.2 compares Depth Anything V2, Only LLDG, Only LLPEFT, and DepthDark. |
| 8 | `lldg,llpeft` | Section 5.1 says the ablation experiments evaluate the gains introduced by the LLDG and LLPEFT modules. |
| 8 | `llpeft` | Section 5.2 compares LLPEFT with representative PEFT methods such as AMFG and LoRA. |

### 5. SoTA Comparison Methods

**ID:** `depthdark_sota_comparison_methods`

**Question:** DepthDark 与哪些方法进行了比较？

**Gold Answer:** DepthDark 与 MonoViT、WSGD、ITDFA、RNW、ADDS、MonoFormer、Depth Anything 和 Depth Anything V2 进行了比较。TDDC 也出现在定量结果表和比较文字中，但由于官方代码不可用，论文使用 TDDC 原文报告的实验结果，而没有在定性可视化结果中直接与 TDDC 进行比较；定性结果中重点展示了 DepthDark 与 ADDS、Depth Anything 和 Depth Anything V2 的对比。

**Gold Items:**

| ID | Name | Aliases |
| --- | --- | --- |
| `monovit` | MonoViT | MonoViT; MonoVit |
| `wsgd` | WSGD | WSGD |
| `itdfa` | ITDFA | ITDFA |
| `rnw` | RNW | RNW; Regularizing Nighttime Weirdness |
| `adds` | ADDS | ADDS |
| `monoformer` | MonoFormer | MonoFormer |
| `tddc` | TDDC | TDDC |
| `depth_anything` | Depth Anything | Depth Anything |
| `depth_anything_v2` | Depth Anything V2 | Depth Anything V2; Depth AnythingV2 |

**Candidate Items / Distractors:**

`monovit`, `wsgd`, `itdfa`, `rnw`, `adds`, `monoformer`, `tddc`, `depth_anything`, `depth_anything_v2`, `marigold`, `depthdark`

**Gold Evidence:**

| Page | Evidence For | Evidence |
| --- | --- | --- |
| 6 | `monovit,wsgd,itdfa,rnw,adds,monoformer,tddc,depth_anything,depth_anything_v2` | Table 4.1 lists these methods in the quantitative comparison on nuScenes-Night and RobotCar-Night; TDDC is included using the experimental results reported in the TDDC paper because its official code is unavailable. |
| 6 | `depth_anything,depth_anything_v2,tddc` | The comparison text says the paper further compares with state-of-the-art methods including Depth Anything, Depth Anything V2, and TDDC, while clarifying that TDDC's reported results are used. |
| 7 | `adds,depth_anything,depth_anything_v2` | The qualitative figures compare DepthDark with ADDS, Depth Anything, and Depth Anything V2. |
| 8 | `adds,depth_anything,depth_anything_v2` | The qualitative results section says ADDS is selected for qualitative comparison because TDDC's official code is unavailable. |
