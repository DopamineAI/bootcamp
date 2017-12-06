# Simple AI Inference Service
In this example we show a scenario where a buyer (B below) purchases an AI service from a seller (A below) through the Dopamine network

<img src="https://github.com/DopamineAI/bootcamp/raw/3476cdaf0dd65ca4631f07eba80c09713dd8abb6/img/simeple_ai_inference_service.png">

We have tried to make the code as simple as possible, so there is no usage of dopamine certificates, we also assume that both sides know each others address.
The \"Service\" is classifying a given image (in our case, identifying the lion picture below). The classification is done using a pretrained VGG model, downloaded via tensorflow library.

<img src="http://www.slate.com/content/dam/slate/articles/health_and_science/science/2015/07/150730_SCI_Cecil_lion.jpg.CROP.promo-xlarge2.jpg">

## Code samples
- [Buyer side code](https://10.156.0.172:8889/notebooks/02.%20Single%20Entity%20AI%20Inference%20Service/buyer.ipynb)
- [Seller side code](https://10.156.0.172:8889/notebooks/02.%20Single%20Entity%20AI%20Inference%20Service/seller.ipynb)