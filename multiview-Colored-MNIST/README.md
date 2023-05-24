# multiviewColoredMnist
In this repository, we performed experiments for learning disentangled representations by using DiME (Difference of matrix based entropies)
on the colored MNIST dataset. 
Given a pair of images belonging to the same class but from different views (colored background and colored foreground digits, 
see figure 1.a), we learned a set of shared features (figure 1.b) that represent the commonalities between 
the images and disentangle the exclusive features of each view (figure 1.d). Specifically, We maximize the mutual information (MI) via DiME between the shared representations from both views,
and minimize the MI of the shared and exclusive features captured by a separate encoder to encourage the disentanglement of the two components.
<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/84861891/197013668-791a1c13-71c5-4663-b17b-96ef552d4d66.png" width="900">
</p>

The experiments are performed in 'main.ipynb'. There, the shared representations are learned initially, then the exclusive features. With these representations, there are several examples of image retrieval by finding the nearest neighbors on either the shared (figure 1.c) or exclusive spaces (figure 1.e). We also show how the generative factors of the background and foreground colors are well disentangled from the class shape (figures 2.a and 2.b) and performed transfer style (2.c). Finally, we performed classification experiments in order to compere with state of the art methods.

<p align="center">
  <img align="center" src="https://user-images.githubusercontent.com/84861891/197015277-699af4c3-4745-4165-be07-74ef287f1d63.png" width="600">
</p>

Folder structure: 

```
data/*: data
figures/*: figures
old_scripts/*: first experiments performed
saved_models/*: Models already trained.
utils/*: Miscellaneous scripts to load the dataset, find pairs of different views and visual utils.
losses.py: File where all the loss functions are stored (including DiME)
main.ipynb: Main notebook wit all the experiments
models.py: Includes the models used in all the experiments
trainer.py: Includes the function to train the different models
```

Installation:

Can use the same conda environment as detailed in the multiview-MNIST folder.


