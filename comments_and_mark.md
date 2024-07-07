In this project “Generating synthetic data using generative adversarial networks” the author studied generation of synthetic data by using generative adversarial networks (GANs). Several variants of GANs are studied, including standard GAN, as well as GAN models with Wasserstein loss and gradient penalty. The performance is evaluated by using a dataset of cancer patient records, which contains information about patient features and cancer types. The performance is evaluated with respect to the probability distribution of patient features, and conditional probability distribution of patient features, conditional on cancer type. 

The introduction clearly defines the problem and summarises the main results. The research questions and methods are based on well-chosen set of references, covering the state-of-the-art references on the underlying methodology (GANs), and including a reference on using GANs to patient record data, and one reference for the dataset used in the study. Questions investigated were given after the description of the metrics used – this is a bit disorganised. Motivation is discussed in the next section.  

The solution concept is sound. GANs are in common use for synthetic data generation. The key concepts are defined in Section 3 Methods. The key concepts are overall well defined. However, there is some room for improvement in clarity and having a more self-contained presentation. In Section 3.2, KL divergence function is not defined and neither is function L:(). In Equation (4) - P_\vartheta, shouldn’t be P_g ? In Equation (5) – p(z) ? The key idea of using a gradient penalty could have been explained better, by providing some more discussion for the rationale of why choosing the specific form of the penalty term. 

The implementation is in TensorFlow using Keras. The implementation is structured around two main classes: generator and discriminator. In addition, the code contains various functions for evaluation of results, as well as Jupyter notebooks for training, evaluation, and plotting. The code is neat and well structured. 

The dataset consists of patent records with 34 categorical patient features and 116 cancer types. The total dataset size used is 280,000 patient records (20% of the Simulacrum database). The dataset is well described in Section 4 of the report. 

The numerical evaluation compares four GAN models, including standard GAN, Wasserstein GAN with gradient penalty, and two variants of conditional GANs. Numerical results are well summarised using visualisation means and tables with numerical results. 

The conclusion section provides a good summary of the results presented in the report. 

The presentation is clear and well structured, overall up to a high standard. 

**80**
