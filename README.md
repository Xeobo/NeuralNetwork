# NeuralNetwork

## Author & Contributor List

Vladimir Zbiljić

All other known bugs and fixes can be sent to vladimir.zbiljic@gmail.com

Reported bugs/fixes will be submitted to correction.


## About

Represents simple neural network library for Android applications. 

For implementation of this library is used OpenCV Matrix abstractions, to optimize matrix calculations.

Project is done as part of mine BSc work. This library was used to make TV program recomendataions based on user's habits.


## How to run

Application comes with basic test example, with dummy sinusoidal data set. 

Once you have code cloned to your computer you could install it to your device/VM to test application.

The MainActivity will start ExecutionService (IntentService) which will calculate optimal Neural Network weights and return cost function.

Labels are put randomly so, you should expect cost to be less than 0.7. Results will be written in logcat.

## How to use

Application shoud provide it's data as IDataSource object. It's recomended to use one of  AbstractCVMultipleDataSource implementations (Fifo or Random), to extract train, test and cross validation data sets. In data set initail value of weights should be random small numbers. It's recomended to use SinusoidInitMatrix class from util subpackage.

Sample code should look like:

```

AbstractCVMultipleDataSource dataSource = new FifoCVMultipleDataSource(new MyImplementationDataSource());
IMachineLearningAlgorithm algorithm = new NeuralNetwork( new LogisticRegression() );

//train Neural Network
List<Mat> weights = algorithm.gradientDescent(
									dataSource.getXTrain(),
									dataSource.getYTrain(),
									LAMBDA_VALUE,CONVERGE_RATIO, 
									MAX_EXECUTION_TIME, 
									dataSource.getThetas().toArray(new Mat[dataSource.getThetas().size()])
								);

//calculate prediction
Mat predictions = algorithm.hOfTheta(
								dataSource.getXTest(),
								weights.toArray(new Mat[weights.size()])
							);
...

```
