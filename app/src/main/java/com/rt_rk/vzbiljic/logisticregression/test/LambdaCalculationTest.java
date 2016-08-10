package com.rt_rk.vzbiljic.logisticregression.test;

import android.nfc.Tag;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.MultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.List;

/**
 * Created by vzbiljic on 20.7.16..
 */
public class LambdaCalculationTest implements ITest {

    private static final double LAMBDA = 0.6;

    private static final String TAG ="LambdaCalculationTest";

    private static final double[] LAMBDAS = { 0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1.28,2.56,5.12,10.24 };



    private static final double ALPHA = 2;

    private AbstractCVMultipleDataSource mDataSource;

    public LambdaCalculationTest(AbstractCVMultipleDataSource mds){
        mDataSource = mds;
    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        NeuralNetwork lNeuralNetwork = (NeuralNetwork) algorithm;

        List<Mat> thetas = mDataSource.getThetas();
        Mat theta1 = thetas.get(0);

        Mat theta2 = thetas.get(1);

        Mat X = mDataSource.getXTest();


        Mat y = mDataSource.getYTest();


        List<Mat>[] thetaOfLambda = new List[LAMBDAS.length];

        for(int i=0;i<LAMBDAS.length;i++) {
            //suppose that theta values will be similar, we could take as initial values theta we
            //have previously calculated
            //in hope we would converge much faster

            thetaOfLambda[i] = lNeuralNetwork.gradientDescent(X, y, ALPHA, LAMBDAS[i],CONVERGE_RATIO,MAX_EXECUTION_TIME, theta1.clone(), theta2.clone());

        }



        X = mDataSource.getXCV();


        y = mDataSource.getYCV();

        int minLambda = -1;
        double minCost = 0;

        for(int i=0;i<thetaOfLambda.length;i++) {

            theta1 = thetaOfLambda[i].get(0);

            theta2 = thetaOfLambda[i].get(1);

            double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), theta1, theta2).val[0];

            if(minLambda == -1 || minCost > currentCost){
                minLambda = i;
                minCost = currentCost;
            }

            Log.i(TAG,"-------------------------------------------------------");
            Log.i(TAG,"\tcross validation cost: " + currentCost + ", for lambda: " + LAMBDAS[i]);
        }

        Log.i(TAG,"===========================================================");
        Log.i(TAG,"\tminimum validation cost: " + minCost + ", for lambda: " + LAMBDAS[minLambda]);
        Log.i(TAG,"===========================================================");




        X = mDataSource.getXTrain();
        y = mDataSource.getYTrain();

        theta1 = thetaOfLambda[minLambda].get(0);

        theta2 = thetaOfLambda[minLambda].get(1);

        double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), theta1, theta2).val[0];




        Log.i(TAG,"\ttraining  cost: " + currentCost + ", for lambda: " + LAMBDAS[minLambda]);
        Log.i(TAG,"===========================================================");

    }

}
