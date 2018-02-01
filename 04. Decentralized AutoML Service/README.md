# Transfer Learning Example

In this sample, a "Consumer" needs to classify set of images with the characters 7,8,9,0 and has very few images it can use to train itself (42). The Consumer uses a Dopamine service provided by the "Provider," which was trained to classify a totally different set of images (1-6), to create it's own classifier:

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/sample_04_2.jpg'>

Below is an example of how a Consumer that uses this type of external service can anticipate higher performance compared to a case where the Consumer did not use this kind of external service:

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/transfer_learning_accuracy.png'>

## Code Samples
- [Provider side code](https://github.com/DopamineAI/bootcamp/blob/19de0dcc74fb1213b7ab2336001eee149a4c23ea/04.%20Decentralized%20AutoML%20Service/provider.ipynb)
- [Consumer side code](https://github.com/DopamineAI/bootcamp/blob/19de0dcc74fb1213b7ab2336001eee149a4c23ea/04.%20Decentralized%20AutoML%20Service/consumer.ipynb)
