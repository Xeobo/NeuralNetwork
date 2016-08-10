package com.rt_rk.vzbiljic.logisticregression.dataSource.multiple;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 10.8.16..
 */
public class RandomCVMultipleDataSource extends AbstractCVMultipleDataSource {


    public RandomCVMultipleDataSource(IDataSource dataSource) {
        super(dataSource);
    }

    //test if it should data be assigned to TestSet
    private boolean isTraining(double rand, double countTest, Mat XCV, double countCV, Mat Xtest, double countTrain, Mat Xraining){

        if(countCV == XCV.rows() && countTrain == Xtest.rows()){
            return true;
        }
        if(countTest == Xraining.rows()){
            return false;
        }
        return rand < 0.6;
    }
    //test if it should data be assigned to CVSet
    private boolean isCV(double rand, double countTest, Mat XCV, double countCV, Mat Xtest, double countTrain){
        if(countTrain == Xtest.rows()){
            return true;
        }
        if(countCV == XCV.rows()){
            return false;
        }
        return rand > 0.6 && rand < 0.8;
    }

    @Override
    protected List<Mat> init(IDataSource dataSource) {
        Mat X = dataSource.getX();
        Mat y = dataSource.getY();

        int testCount = 0;
        int trainCount = 0;
        int CVCount = 0;

        Mat Xtrain = new Mat((int)(X.rows()*0.6),X.cols(),X.type());
        Mat Ytrain = new Mat(y.rows(),(int)(y.cols()*0.6),y.type());

        Mat Xtest = new Mat((int)(X.rows()*0.2),X.cols(),X.type());
        Mat Ytest = new Mat(y.rows(),(int)(y.cols()*0.2),y.type());

        Mat XCV = new Mat(X.rows()- Xtest.rows() - Xtrain.rows(),X.cols(),X.type());
        Mat YCV = new Mat(y.rows(),y.cols()- Ytest.cols() - Ytrain.cols(),y.type());

        for(int i= 0; i< X.rows(); i++){
            double rand = Math.random();

            if(isTraining(rand, testCount, XCV, CVCount, Xtest, trainCount, Xtrain)){//training set
                X.row(i).copyTo(Xtrain.row(testCount));
                Ytrain.col(testCount++).setTo(new Scalar(y.get(0,i)[0]));

            }else if(isCV(rand, testCount, XCV, CVCount, Xtest, trainCount)){//cross validation set
                X.row(i).copyTo(XCV.row(CVCount));
                YCV.col(CVCount++).setTo(new Scalar(y.get(0,i)[0]));

            }else {//test set
                X.row(i).copyTo(Xtest.row(trainCount));
                Ytest.col(trainCount++).setTo(new Scalar(y.get(0,i)[0]));
            }
        }
        List<Mat> mats = new ArrayList<>();

        mats.add(Xtrain);
        mats.add(Ytrain);

        mats.add(Xtest);
        mats.add(Ytest);

        mats.add(XCV);
        mats.add(YCV);

        return mats;
    }


}
