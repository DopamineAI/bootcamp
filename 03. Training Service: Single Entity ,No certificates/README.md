# Training Service: Single Entity, No Certificates
In this sample, several teachers give training services to a "student" - a learning service. 
After every training batch, the student examines how much it improved based on 'secret' holdout data, and the program rewards the teacher accordingly. 

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/tutor_and_student_simple.png'>

The student's task is recognizing handwritten digits from the MNIST database:

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/mnist-900x506.png'>

According to the student's reward policy, which is available in Dopamine.ai, the expected learning rate would look like the below, and teachers would be rewarded accordingly: 

<img src='https://github.com/DopamineAI/bootcamp/blob/master/img/LogarithmicReward.png'>

## Code Samples
In the results of the sample code, it's clear that the teachers are rewarded quite fairly:
- [Student side code](https://github.com/DopamineAI/bootcamp/blob/master/03.%20Training%20Service:%20Single%20Entity%20%2CNo%20certificates/student.ipynb)
- [Teachers side code](https://github.com/DopamineAI/bootcamp/blob/master/03.%20Training%20Service:%20Single%20Entity%20%2CNo%20certificates/teachers.ipynb)
