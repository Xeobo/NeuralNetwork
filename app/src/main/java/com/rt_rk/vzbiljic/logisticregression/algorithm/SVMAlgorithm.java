package com.rt_rk.vzbiljic.logisticregression.algorithm;

import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.util.MatrixPrint;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.ml.SVM;

import java.util.List;

/**
 * Created by vzbiljic on 12.7.16..
 */
public class SVMAlgorithm implements IMachineLearningAlgorithm{


    private static final double C = 0.1;
    private SVM mSvm = SVM.create();

    @Override
    public Mat hOfTheta(Mat inputData, Mat... thetas) {
        Mat result = new Mat(1,inputData.rows(),inputData.type());
        for(int i=0; i< inputData.rows();i++) {
            result.put(0, i, mSvm.predict(inputData.row(i)));

        }
        return result;
    }

    @Override
    public Scalar JofTheta(Mat X, Mat Y, Mat hOfTheta, double lambdaVal, Mat... thetas) {
        return null;
    }

    @Override
    public Scalar JofThetaNoRegularization(Mat X, Mat Y, Mat hOfTheta, Mat... thetas) {
        return null;
    }

    @Override
    public List<Mat> grad(Mat X, Mat Y, Mat hOfTheta, double lambdaVal, Mat... thetas) {
        return null;
    }

    @Override
    public List<Mat> gradientDescent(Mat X, Mat y, double alpha, double lambdaVal, double convergeRatio, long maxExecutionTime,Mat... thetas) {
        mSvm.setKernel(SVM.LINEAR);
        mSvm.setC(C);

        if(mSvm.train(X,0,y.t())){
            Log.i("SVM", "trained!");
        }else{
            Log.i("SVM", "not trained!");
        }

        return null;
    }

}
