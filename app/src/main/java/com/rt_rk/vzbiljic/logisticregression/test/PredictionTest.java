package com.rt_rk.vzbiljic.logisticregression.test;

import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Created by vzbiljic on 29.7.16..
 */
public class PredictionTest implements ITest{
    private static final String TAG ="LambdaCalculationTest";

    private static final double[] LAMBDAS = { 0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24 };

    private static final double LAMBDA = 0.0;

    private static final double ALPHA = 0.5;

    private static final double THRESHOLD = 0.5;

    private AbstractCVMultipleDataSource mDataSource;

    public PredictionTest(AbstractCVMultipleDataSource mds){
        mDataSource = mds;
    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        NeuralNetwork lNeuralNetwork = (NeuralNetwork) algorithm;

        List<Mat> thetas = mDataSource.getThetas();
        Mat theta1 = thetas.get(0);

        Mat theta2 = thetas.get(1);

        Mat X = mDataSource.getXTrain();


        Mat y = mDataSource.getYTrain();


        List<Mat>[] thetaOfLambda = new List[LAMBDAS.length];



        thetaOfLambda[0] = lNeuralNetwork.gradientDescent(X, y, ALPHA, LAMBDA,CONVERGE_RATIO,MAX_EXECUTION_TIME, theta1.clone(), theta2.clone());

//        for(int i= 0; i< LAMBDAS.length;i++) {
//            //suppose that theta values will be similar, we could take as initial values theta we
//            //have previously calculated
//            //in hope we would converge much faster
//
//            thetaOfLambda[i] = lNeuralNetwork.gradientDescent(1000, X, y, ALPHA, LAMBDAS[i], theta1.clone(), theta2.clone());
//        }


        X = mDataSource.getXCV();


        y = mDataSource.getYCV();

        int minLambda = 0;//-1;
        double minCost = 0;
//
//        for(int i=0;i<thetaOfLambda.length;i++) {
//
//            theta1 = thetaOfLambda[i].get(0);
//
//            theta2 = thetaOfLambda[i].get(1);
//
//            double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), LAMBDAS[i], theta1, theta2).val[0];
//
//            if(minLambda == -1 || minCost > currentCost){
//                minLambda = i;
//                minCost = currentCost;
//            }
//
//            Log.i(TAG, "-------------------------------------------------------");
//            Log.i(TAG,"\tcross validation cost: " + currentCost + ", for lambda: " + LAMBDAS[i]);
//        }

        Log.i(TAG,"===========================================================");
        Log.i(TAG,"\tminimum validation cost: " + minCost + ", for lambda: " + LAMBDAS[minLambda]);
        Log.i(TAG,"===========================================================");






        theta1 = thetaOfLambda[minLambda].get(0);

        theta2 = thetaOfLambda[minLambda].get(1);

        X = mDataSource.getXTest();


        y = mDataSource.getYTest();

        double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), theta1, theta2).val[0];


//        Log.i(TAG,"\ttraining  cost: " + currentCost + ", for lambda: " + LAMBDAS[minLambda]);
        Log.i(TAG,"===========================================================");


        Mat prediction = lNeuralNetwork.hOfTheta(X,theta1,theta2);

        int missed = 0;
        int truePositives = 0;
        int actualPositives = 0;
        int predictedPositives = 0;
        double maxPrediction = -1;
        double minPrediction = 1;
        for(int i=0; i< prediction.rows();i++)
            for(int j=0; j< prediction.cols();j++){
                int value = prediction.get(i,j)[0]> THRESHOLD ?1:0;

                if(maxPrediction < prediction.get(i,j)[0]){
                    maxPrediction = prediction.get(i,j)[0];
                }
                if(minPrediction > prediction.get(i,j)[0]){
                    minPrediction = prediction.get(i,j)[0];
                }

                if(y.get(i,j)[0] == 1){
                    actualPositives++;
                    if(value == 1){
                        truePositives++;
                    }
                }

                if(value == 1){
                    predictedPositives++;
                }

                if( value != y.get(i,j)[0]){
                    missed++;

                }
            }
        double m = y.cols();

        double precision = ((double)truePositives)/predictedPositives;

        double recall = ((double)truePositives)/actualPositives;

        double FScore = 2*(precision*recall)/(precision + recall);

        Log.i(TAG, "Hit ratio: " +  (m-missed)/m*100 + " %");

        Log.i(TAG, "precision : " + precision*100 + " %");

        Log.i(TAG, "recall : " +  recall*100 + " %");

        Log.i(TAG, "Fscore : " +  FScore*100 + " %");

        Log.i(TAG, "actualPositives : " +  actualPositives + ", minPrediction: " + minPrediction + ", maxPrediction: " + maxPrediction );

    }
}
