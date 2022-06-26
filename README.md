# Transfer Learning
### Why use Transfer Learning for Artificial Neural Networks?
Deep learning techniques extract and disentangle higher level factors base on their variations with respect to the input space. Training such a model is computationally expensive and might be unfeasible for a wide array of reasons(availability, addition of output labels). Transfer learning attempts to train a model using a prior pre-trained model, which can be fine-tuned to fit specific novel constraints.
<br><br>
<p align="center">
<img src="https://www.oreilly.com/library/view/intelligent-projects-using/9781788996921/assets/07387bba-04ab-4758-9ac4-8740ea2f1bea.png" width="500"></p>
  
### Techniques for Transfer Learning?
- Weight Summation - Having M already pretrained networks, the weights of the fused neural network W<sub>i</sub><sup>f</sup> can be constructed through a linear combination of the corresponding weights W<sub>i</sub><sup>m</sup>.<br>
![equation](http://www.sciweavers.org/tex2img.php?eq=W%5E%7B1%7D_%7Bf%7D%20%3D%20%20%5Csum_%7Bm%3D1%7D%5EM%20W%5E%7Bm%7D_%7B1%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) <br>
Assuming statistically independent weights W<sub>i</sub><sup>m</sup> of M neural networks and an equal probability of excitatory and inhibitory weights,
combination of the weights does not change the strength of the presynaptic signal on average as shown by the equation below.<br>

![equation](http://www.sciweavers.org/tex2img.php?eq=sgn%28w%5E%7Bf%7Df%20%3D%20w%5E%7B1%7Df%20%2B%20w%5E%7B2%7Df%20%2B%20....%29%20%3D%20sgn%28w%5E%7B1%7Df%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
  
- Elastic Weight Consolidation - This method was proposed to prevent catastrophic forgetting based on an approximation of the error surface of a trained neural network by a paraboloid in multidimensional space of weights. Fischer Information is used to approximate the loss function which is simplified to <br>
![equation](https://bit.ly/3HUzEKf)

<br>Further details can be found on [Arxiv](https://arxiv.org/pdf/1612.00796.pdf)

### Architecture / Implementation Details
<p align="center">
<img src="https://i.imgur.com/p1C3LFc.png" width="200">
</p>
<br>
<ul>
<li>Experiments are run on MNIST and CIFAR 10 datasets.
<li>Multiple VGGs are used for extraction of features. We build a base model using the architecture shown in Fig(1).
<li>We freeze all the trainable parameters in all layers which perform feature extraction(VGG). The final layer weights are unfrozen.
<li>2 different models are trained on tasks T1 and T2, each of which have 5 classes. We perform training for 50 epochs on the model for both WS and EWC.
<li>The fusion methods are only applied to the top fully connected layers. Convolutional layers can also be combined using incremental iterative combination. I have not incorporated this part, since it is a fine-tuning on feature set.
  
### Results

<p align="center">
<img src="https://i.imgur.com/TL3MGGe.png" width="500">
<img src="https://i.imgur.com/8Fi9FlW.png" width="500">
</p>
