package com.rt_rk.vzbiljic.logisticregression.test;

import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.algorithm.SVMAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.dataSource.MultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;

import org.opencv.core.Mat;


/**
 * Created by vzbiljic on 8.8.16..
 */
public class SVMTest implements ITest{

    private final AbstractCVMultipleDataSource mDataSource;

    public SVMTest(AbstractCVMultipleDataSource mds){
        mDataSource = mds;
    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        SVMAlgorithm svmAlgorithm = (SVMAlgorithm) algorithm;

        svmAlgorithm.gradientDescent(mDataSource.getXTrain(),mDataSource.getYTrain(),0,0,0,0);

        Mat prediction = svmAlgorithm.hOfTheta(mDataSource.getXTest());

        int missed = 0;
        int truePositives = 0;
        int actualPositives = 0;
        int predictedPositives = 0;
        double maxPrediction = -1;
        double minPrediction = 1;
        for(int i=0; i< prediction.rows();i++)
            for(int j=0; j< prediction.cols();j++){
                int value = prediction.get(i,j)[0]> 0.5 ?1:0;

                if(maxPrediction < prediction.get(i,j)[0]){
                    maxPrediction = prediction.get(i,j)[0];
                }
                if(minPrediction > prediction.get(i,j)[0]){
                    minPrediction = prediction.get(i,j)[0];
                }

                if(mDataSource.getYTest().get(i, j)[0] == 1){
                    actualPositives++;
                    if(value == 1){
                        truePositives++;
                    }
                }

                if(value == 1){
                    predictedPositives++;
                }

                if( value != mDataSource.getYTest().get(i, j)[0]){
                    missed++;

                }
            }
        double m = mDataSource.getYTest().cols();

        double precision = ((double)truePositives)/predictedPositives;

        double recall = ((double)truePositives)/actualPositives;

        double FScore = 2*(precision*recall)/(precision + recall);

        Log.i("SVMTest", "Hit ratio: " + (m - missed) / m * 100 + " %");

        Log.i("SVMTest", "precision : " + precision*100 + " %");

        Log.i("SVMTest", "recall : " +  recall*100 + " %");

        Log.i("SVMTest", "Fscore : " +  FScore*100 + " %");

        Log.i("SVMTest", "actualPositives : " +  actualPositives + ", minPrediction: " + minPrediction + ", maxPrediction: " + maxPrediction );

    }
}
