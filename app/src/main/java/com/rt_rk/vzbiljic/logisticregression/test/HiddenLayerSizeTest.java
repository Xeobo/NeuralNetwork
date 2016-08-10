package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.MultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.RandomCVMultipleDataSource;

import org.opencv.core.Mat;

import java.util.List;

import javax.sql.DataSource;

/**
 * Created by vzbiljic on 29.7.16..
 */
public class HiddenLayerSizeTest implements ITest {

    private static final String TAG ="LambdaCalculationTest";

    private static final double ALPHA = 2;

    private static final double LAMBDA = 0;

    private final static int[] HIDDEN_LAYER_SIZES = { 3,4,5,6,7,8,9,10,15,20 };

    private Context context;

    public HiddenLayerSizeTest(Context context){
        this.context = context;
    }

    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        NeuralNetwork lNeuralNetwork = (NeuralNetwork) algorithm;

        AbstractCVMultipleDataSource lDataSource = new RandomCVMultipleDataSource(new WatchedDataSource(context));

        List<Mat> thetas = lDataSource.getThetas();


        Mat X = lDataSource.getXTest();


        Mat y = lDataSource.getYTest();

        Mat XTrain = lDataSource.getXTrain();

        Mat yTrain = lDataSource.getYTrain();

        Mat XCV = lDataSource.getXCV();

        Mat yCV = lDataSource.getYCV();


        List<Mat>[] thetaOfLayerSize = new List[HIDDEN_LAYER_SIZES.length];

        for(int i=0;i<HIDDEN_LAYER_SIZES.length;i++) {
            lDataSource = new RandomCVMultipleDataSource(new WatchedDataSource(context,HIDDEN_LAYER_SIZES[i]));

            Mat theta1 = lDataSource.getThetas().get(0);

            Mat theta2 = lDataSource.getThetas().get(1);

            thetaOfLayerSize[i] = lNeuralNetwork.gradientDescent(X, y, ALPHA, LAMBDA,CONVERGE_RATIO,MAX_EXECUTION_TIME, theta1, theta2);

        }



        X = XCV;


        y = yCV;

        int minLambda = -1;
        double minCost = 0;

        for(int i=0;i<thetaOfLayerSize.length;i++) {

            Mat theta1 = thetaOfLayerSize[i].get(0);

            Mat theta2 = thetaOfLayerSize[i].get(1);

            double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), theta1, theta2).val[0];

            if(minLambda == -1 || minCost > currentCost){
                minLambda = i;
                minCost = currentCost;
            }

            Log.i(TAG, "-------------------------------------------------------");
            Log.i(TAG,"\tcross validation cost: " + currentCost + ", for hidden layer size: " + HIDDEN_LAYER_SIZES[i]);
        }

        Log.i(TAG,"===========================================================");
        Log.i(TAG,"\tminimum validation cost: " + minCost + ", for hidden layer size: " + HIDDEN_LAYER_SIZES[minLambda]);
        Log.i(TAG,"===========================================================");




        X = XTrain;
        y = yTrain;

        Mat theta1 = thetaOfLayerSize[minLambda].get(0);

        Mat theta2 = thetaOfLayerSize[minLambda].get(1);

        double currentCost = lNeuralNetwork.JofThetaNoRegularization(X, y, lNeuralNetwork.hOfTheta(X, theta1, theta2), theta1, theta2).val[0];




        Log.i(TAG,"\ttraining  cost: " + currentCost + ", for hidden layer size: " + HIDDEN_LAYER_SIZES[minLambda]);
        Log.i(TAG,"===========================================================");    
    }

}
