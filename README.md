# RadioSight

<div align="center">
  <h2><strong>Predictive mmWave XR Network Optimization from Dynamic Neural Radio Fields</strong></h2>
  <p>
    <img alt="MobiCom 2026" src="https://img.shields.io/badge/ACM-MobiCom%202026-0A66C2?style=flat-square" />
    <img alt="mmWave" src="https://img.shields.io/badge/Domain-mmWave%20XR-8A2BE2?style=flat-square" />
    <img alt="Neural Radio Fields" src="https://img.shields.io/badge/Method-Neural%20Radio%20Fields-16a34a?style=flat-square" />
  </p>
</div>

## Abstract

RadioSight is a real-time, multimodal neural radio field system for predictive mmWave XR network optimization. It continuously models the dynamic wireless environment using vision, motion, and RF observations, and proactively anticipates future blockage and beam changes before link failures occur. By combining scene-aware geometry with lightweight radio-field synchronization, RadioSight identifies alternative non-line-of-sight reflection paths, predicts beam transitions, and schedules proactive beam updates for resilient mmWave connectivity in mobile XR scenarios.

## Project Description

RadioSight studies a practical problem in mobile XR networking: mmWave links are highly directional and can fail quickly when the user moves, rotates, or is temporarily blocked by objects in the environment. Instead of reacting only after the connection breaks, RadioSight builds a dynamic radio model of the scene and predicts how the wireless channel will change over time.

The key idea is to combine environmental geometry, motion information, and radio measurements into a single live representation of the wireless environment. From this representation, the system can estimate future channel conditions, identify alternative propagation paths, and select the appropriate beam before the link degrades. This enables proactive beam switching and more stable mmWave connectivity for immersive applications.

The project is motivated by real XR deployments, where reliable low-latency communication matters for high-throughput experiences such as AR/VR, cloud-rendering, and collaborative immersive systems. RadioSight is designed as a practical system for real-world 28 GHz deployments rather than a purely offline modeling study.

## Authors

Lihao Zhang, Paul Kudyba, Zhenlin An, and Haijian Sun

University of Georgia

## Citation

```bibtex
@inproceedings{zhang2026radiosight,
  title={RadioSight: Predictive mmWave XR Network Optimization from Dynamic Neural Radio Fields},
  author={Lihao Zhang and Paul Kudyba and Zhenlin An and Haijian Sun},
  booktitle={Proceedings of the 30th Annual International Conference on Mobile Computing and Networking},
  series={MobiCom '26},
  year={2026},
  publisher={ACM}
}
```