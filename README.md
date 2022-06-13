# Side-Channel AutoEncoder
Implementations of Side-Channel Autoencoders

Refer:
~~Improving Non-Profiled Side-Channel Attacks using Autoencoder-based Preprocessing [[pdf](https://eprint.iacr.org/2020/396.pdf)]~~

Non-Profiled Deep Learning-based Side-Channel Preprocessing with Autoencoders [[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9400816)]

Contact: <donggeun.kwon@gmail.com>


### Intro
This paper introduces a novel approach with deep learning for improving side-channel attacks, especially in a non-profiling scenario. It propose a new principle of training that trains autoencoders using noise-reduced labels. It notably diminishes the noise in measurements by modifying the autoencoder framework to the signal preprocessing.

### Source Code
The sample source code for the proposed methods in this paper is in the SCAE folder. Also, the code for side channel attacks using side-channel autoencoders is in the Appendix folder.
The structure of the source code is as follows:

* SCAE
    + SCAE_Denoise.py _(in Section 3-2)_
    + SCAE_Align.py _(in Section 3-3)_
    + SCAE_DK.py _(in Section 3-4)_
    + hyperparameters.py

* Appendix
    + main.py
    + DDLA.py _(in Appendix A)_ [[refer](https://doi.org/10.13154/tches.v2019.i2.107-131)]
    + SCAE_CPA.py _(in Appendix A)_
    + _SCAE_DDLA.py (~~Test~~)
    + loadh5.py
    + hyperparameters.py


<center><a href="http://crypto.korea.ac.kr" target="_blank"><img src="http://crypto.korea.ac.kr/wp-content/uploads/2019/01/Algorithm_trans.png" width="30%" height="30%" /></a></center>
