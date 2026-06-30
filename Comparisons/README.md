# IMUHOI Comparison Baselines

This folder contains paper-method comparison implementations for:

- `dip18`: Deep Inertial Poser style BiRNN reconstruction with Gaussian pose output.
- `tip`: Transformer Inertial Poser style causal Transformer plus RNN state predictor.
- `transpose`: TransPose pose S1/S2/S3 plus translation B1/B2 branches.
- `globalpose`: GlobalPose PL/IK/VR network-supervised path, exposing the outputs needed by the paper's physics optimizer.

The shared adapter in `Comparisons/common/adapters.py` converts the current `IMUDataset` batch into each method's expected protocol. Object IMU is appended as an explicit HOI extension and object position is predicted with an added object-position loss.

Important boundary: once object IMU, object position output, and object loss are added, these are no longer strict original-paper reproductions. They are comparison baselines that preserve each method's human-motion protocol as closely as practical while adding the same object tracking task.

TIP note: the current IMUHOI dataset does not contain TIP's SBP terrain labels. The model keeps the SBP output dimensions by default, but the adapter fills absent SBP targets with `NaN` and the loss masks the SBP constraint term. If SBP labels are later added under `tip_sbp`, the constraint loss is used automatically.

GlobalPose note: the paper's physics optimizer is a per-frame non-neural inference procedure with external dynamics dependencies. The training baseline here preserves the PL/IK/VR prediction chain and object extension. Online physics refinement should wrap the `vr` outputs rather than be trained as a supervised network layer.

Run a smoke training command from the project root:

```bash
conda activate SAGE
python -m Comparisons.train_comparison --method transpose --debug --epochs 1
```

