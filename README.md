# GALFusion

This is the official code for the paper "GALFusion: Multi-Exposure Image Fusion via a Global–Local Aggregation Learning Network".

## Environment Preparing

```
python 3.8
pytorch 1.7.0
torchvision 0.8.1
einops 0.6.1
```

### Testing

We provide some example images for testing in `./dataset/test_data/`

```
python test_gray.py
```

### Reference

If you find our work useful in your research please consider citing our paper:

```
@article{lei2023galfusion,
  title={GALFusion: Multi-exposure Image Fusion via a Global-local Aggregation Learning Network},
  author={Lei, Jia and Li, Jiawei and Liu, Jinyuan and Zhou, Shihua and Zhang, Qiang and Kasabov, Nikola K},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
```
