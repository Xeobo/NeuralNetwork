package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.SinusDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.List;
import java.util.concurrent.Semaphore;

/**
 * Created by vzbiljic on 18.7.16..
 */
public class GradientCheckTest implements ITest {


    private static final int LAMBDA = 1;

    private IDataSource mDataSource;

    public GradientCheckTest(Context context){
        mDataSource = new SinusDataSource();
    }


    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        NeuralNetwork lNeuralNetwork = (NeuralNetwork) algorithm;

        List<Mat> thetas = mDataSource.getThetas();

        Mat theta1 = thetas.get(0);

        Mat theta2 = thetas.get(1);

        Mat X = mDataSource.getX();

        Mat y = mDataSource.getY();


        lNeuralNetwork.checkGradient(X,y,theta1,theta2,LAMBDA);


    }




}
