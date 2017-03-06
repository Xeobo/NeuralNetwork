package com.rt_rk.vzbiljic.logisticregression.test;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Created by vzbiljic on 20.7.16..
 */
public class AlgorithmTest implements ITest {

    private static final double LAMBDA = 0.6;
    private static final double ALPHA = 1;

    private IDataSource mDataSource;

    public AlgorithmTest(IDataSource mds){
        mDataSource = mds;
    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        NeuralNetwork lNeuralNetwork = (NeuralNetwork) algorithm;

        List<Mat> thetas = mDataSource.getThetas();
        Mat theta1 = thetas.get(0);

        Mat theta2 = thetas.get(1);

        Mat X = mDataSource.getX();


        Mat y = mDataSource.getY();


        lNeuralNetwork.gradientDescent(X, y, ALPHA, LAMBDA,CONVERGE_RATIO,MAX_EXECUTION_TIME, theta1, theta2);
    }

}
