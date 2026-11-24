# External software

| Software | Version | Used in |
|---|---|---|
| MATLAB | R2026a (current server install) [confirm version used for fitting] | rDCM fitting, recovery simulations, voxel matching |
| SPM | SPM25 (current server install) [confirm version used for fitting] | DCM structures used by rDCM (`SPM_PATH`) |
| TAPAS rDCM | [version; see TAPAS README] | `tapas_rdcm_*` (`TAPAS_PATH`) |
| FreeSurfer | 7.4.1 | cortical surfaces |
| AFNI/SUMA | 25.3.03 (current server install) | `3dcalc`, `3dSurf2Vol`, `MapIcosahedron`, `3dinfo`, `3dcopy`, `3dBrickStat` (`01_anatomy_layers`) |
| LayNii | 2.9.0 | `LN2_LAYERS`, equivolumetric depths |
| ANTs | 2.6.4 (current server install) | `antsRegistration`, `antsApplyTransforms` (`02_registration_rois`) |
| qMRLab | [version] | qT1 maps from MP2RAGE |
| micapipe | [v0.2.x] | intracortical profiles and MPC (14 surfaces) |

Fill in the bracketed versions before release.
