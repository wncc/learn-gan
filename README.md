# Why the Hype around GANs? 

In summer of 2020, a reading project on Generative Adversarial Networks was undertaken as a part of WnCC's [Seasons of Code](https://www.wncc-iitb.org/soc/) initiative. 
Since the mentors of this project are WnCC members themselves, they decided to open source the project timeline for the benefit of anyone out there who's just starting out with Deep Learning.

The *only* prerequisite for this course is **basic Python proficiency**. With this being said, let's begin!

### Week 1 | Getting Started

- To get a basic understanding what a Neural Network is, watch this excellent playlist by [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) - [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). 

- Now, to build your own Neural Network, try completing this short course by Andrew NG - [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning). You can opt for Financial aid, if you need to.

- It is sometimes overwhelming to visualise how a neural network improves its performance over time. This website will allow you to do just the same - [Neural Network Playground](https://playground.tensorflow.org/).   
P.S. - You might come across new terms here. Instead of just overlooking them, try finding out what they mean. You could google them or just visit our [Wiki](https://www.wncc-iitb.org/wiki/) page on [Deep Learning](https://www.wncc-iitb.org/wiki/index.php/Deep_Learning).

- Exhausted by all the math? Here's an article to get you motivativated - [Applications of GANs](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900).


### Week 2-3 | Learning Pytorch

- Libraries like PyTorch and Tensorflow make implementing neural nets a bliss. PyTorch's [60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) will help you get started. It's recommended that you type your own code as well.

- Hopefully you would have got a clear understanding of what a neural network is. It is now time to tinker around with them to decrease training time, and improve accuracy. Do this course on [Hyperparameter Tuning](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning) to know more. You can skip the TensorFlow part if you wish to, since you already got an idea of PyTorch.

- You can now do further PyTorch [tutorials](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html). The above course would help you understand these examples better. Make your own [Google Colab](https://colab.research.google.com/) notebooks and tinker around. It's important to try out various values of hyperparameters for better practical learning.


### Week 4 | Attempting a Kaggle Problem

- [MNIST](http://yann.lecun.com/exdb/mnist/) dataset is a large database of handwritten digits. Pytorch has a [tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html) to train your NN on the MNIST dataset. You can leave the [CNN](https://pytorch.org/tutorials/beginner/nn_tutorial.html#switch-to-cnn) part for now.

- Kaggle is a community of data scientists where you can find a vast variety of datasets and competitions to hone your skills. Try attempting this Kaggle Challenge to get started - [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer).

### Week 5 | CNNs

- Convolutional Neural Networks have been considered revolutionary in processing images. Read either of these articles to get an understanding of how they work - 
	+ [CNN in PyTorch](https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/)
	+ [CNN in PyTorch (2)](https://algorithmia.com/blog/convolutional-neural-nets-in-pytorch)

- [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) is an established computer-vision dataset. Attempt a related challenge on Kaggle - [Object Recognition](https://www.kaggle.com/c/cifar-10).

- Try implementing CNN models for classification problems on your own. This article will guide you as to how you can [Create your own dataset](https://towardsdatascience.com/how-to-create-your-own-image-dataset-for-deep-learning-b53f1c22c443).


### Week 6 | GANs

- At last, we will now start with GANs. In case you have never read a research paper before, here is a guide to get you started - [How to Read a Research Paper](https://www.youtube.com/watch?v=SHTOI0KtZnU).

- It might be overwhelming to read this paper but it is strongly recommended that you do - [GANs](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf). 

- It is okay even if you do not understand all of it. These articles might come handy -
	+ [What are GANs](https://medium.com/@jonathan_hui/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)
	+ [Understanding GANs](https://towardsdatascience.com/understanding-generative-adversarial-networks-gans-cd6e4651a29)

### Week 7 | Implementing GANs

- Now that you have understood how a GAN works, you can try implementing GANs for simple datasets. You can refer to the code given here - [PyTorch GANs](https://github.com/tezansahu/PyTorch-GANs). You can leave DCGANs for now.

- Also read this article for for some [Tips to make GANs work](https://github.com/soumith/ganhacks).

### Week 8 | Tinkering Around

- Researchers have developed various types of GANs for specialised applications. We have listed some of the most popular ones - 
	+ [DCGAN](https://arxiv.org/pdf/1511.06434.pdf)
	+ [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
	+ [StackGAN](https://arxiv.org/abs/1612.03242)
	+ [InfoGAN](https://arxiv.org/abs/1606.03657)
	+ [WassersteinGAN](https://arxiv.org/abs/1701.07875)

- You can try implementing these from their research papers to get a better understanding. If you find it difficult to get started, PyTorch provides an [Implementation of DCGANs](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) as well.

- Finally, try attempting this [Assignment](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf). Here are the [Solutions](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip).

### Conclusion

We hope this plan helps you in getting a better understanding of **"the most interesting idea in the last 10 years in Machine Learning"** as described by Yann LeCun. If on your learning path you discover some more efficient resources, we would be more than happy to incorporate them here. Just [create a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) on this repository.

***

<p align="center">Created with :heart: by <a href="https://www.wncc-iitb.org/">WnCC</a></p>
