# Transfer Learning ExampleIn this sample, several teachers give training services to a "student" - a learning service. 

In this sample, a "Consumer" needs to classify set of images with the characters 7,8,9,0 while it has very few images to be used for training (42). The consumer uses a Dopamine service provided by the "Provider", that was trained on classifying totally different set of images (1-6), to create it's own classifier:

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/sample_04_2.jpg'>

We show that such a consumer that uses such an external service is expected to have higher performance compared to case where it did not use such an external service :

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/transfer_learning_accuracy.png'>


## Code Samples
- [Provider side code](https://github.com/DopamineAI/bootcamp/blob/19de0dcc74fb1213b7ab2336001eee149a4c23ea/04.%20Decentralized%20AutoML%20Service/provider.ipynb)
- [Consumer side code](https://github.com/DopamineAI/bootcamp/blob/19de0dcc74fb1213b7ab2336001eee149a4c23ea/04.%20Decentralized%20AutoML%20Service/consumer.ipynb)
