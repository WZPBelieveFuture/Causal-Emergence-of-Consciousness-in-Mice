# Causal-Emergence-of-Consciousness-in-Mice

## About

A machine learning framework (NIS+) infers multiscale causal variables, dymamics and information integration mode from cellular-resolution imaging in mouse cortex  across awake, anesthetized, and recovery stages, revealing a top-level "conscious variable" with high emergent complexity that links neural activity to conscious states across scales.

For more details, please refer the following paper:
> Zhipeng Wang, Yingqi Rong, Kaiwei Liu, Mingzhe Yang, Jiang Zhang, Jing He: Causal Emergence of Consciousness through Learned Multiscale Neural Dynamics in Mice. https://arxiv.org/abs/2509.10891

## Schematic diagram

Schematic overview of the study. (a) Experimental setup with a mouse inhaling anesthetic gas and the corresponding brain region/neuron distribution. (b) Timeline of the three anesthesia stages. (c) Neuronal activity (calcium signals). (d) The NIS+ model learns to predict neural dynamics and maximize effective information. (e) Key outputs include causal effects, information integration strategy and learned dynamics.

![model](images/schematic_diagram.png)


## Results



## Usages

To enhance computational efficiency, we adopt a multistory framework : (i) parallel training of all forward dynamics across scales during stage one, using the average microscopic loss of all scales for gradient updates; and (ii) maximizing ($\mathcal{J}_d$) independently for each scale

**Stage One**
```
bash run_mice_micro_stage1.sh 
```

**Stage Two**
```
bash run_mice_micro_stage2.sh 
```


## Contact
If you have any question about the paper or the code, 
please contact us.
**Zhipeng Wang**, **19906810976@163.com**

Please cite that paper if you use this code. Thanks!