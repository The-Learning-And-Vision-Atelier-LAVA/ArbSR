# Scale-Arbitrary-SR

## Motivation
Although recent CNN-based single image SR networks (e.g., EDSR, RDN and RCAN) have achieved promising performance, they are developed for image SR with a single specific integer scale (e.g., x2, x3, x4). In real-world applications, non-integer SR (e.g., from 100x100 to 220x220) and asymmetric SR (e.g., from 100x100 to 220x420) are also necessary such that customers can zoom in an image arbitrarily for better view of details.

## Contributions
- We propose a plug-in module for existing single image SR networks to perform scale-arbitrary SR. Our module can be easily adapted to scale-specific networks with small additional computational and memory costs.
- We introduce a scale-aware knowledge transfer paradigm to transfer knowledge from scale-specific networks to a scale-arbitrary network.
- Experimental results show that baseline networks equipped with our module produce promising results for scale-arbitrary SR while maintaining the state-of-the-art performance for SR with integer scale factors.

## Visual Results

![non-integer](./Figs/non-integer.png)

![asymmetric](./Figs/asymmetric.png)

## Demo
