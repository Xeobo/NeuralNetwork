package com.rt_rk.vzbiljic.logisticregression.dataSource;

import android.nfc.Tag;
import android.util.Log;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.List;

/**
 * Created by vzbiljic on 27.7.16..
 */
public class MultipleDataSource {
    /*private Mat Xtest;
    private Mat Xtrain;
    private Mat XCV;

    private Mat Ytest;
    private Mat Ytrain;
    private Mat YCV;

    private IDataSource mDataSource;

    //test if it should data be assigned to TestSet
    private boolean setTest(double rand,double countTest,double countCV,double countTrain){

        if(countCV == XCV.rows() && countTrain == Xtrain.rows()){
            return true;
        }
        if(countTest == Xtest.rows()){
            return false;
        }
        return rand < 0.6;
    }
    //test if it should data be assigned to CVSet
    private boolean setCV(double rand,double countTest,double countCV,double countTrain){
        if(countTrain == Xtrain.rows()){
            return true;
        }
        if(countCV == XCV.rows()){
            return false;
        }
        return rand > 0.6 && rand < 0.8;
    }

    //init data
    //data should be split in to 3 pieces
    // X test = 60%
    // X cross validation = 20%
    // X train = 20%
    private void init() {
        Mat X = mDataSource.getX();
        Mat y = mDataSource.getY();

        int testCount = 0;
        int trainCount = 0;
        int CVCount = 0;

        Xtest = new Mat((int)(X.rows()*0.6),X.cols(),X.type());
        Ytest = new Mat(y.rows(),(int)(y.cols()*0.6),y.type());

        Xtrain = new Mat((int)(X.rows()*0.2),X.cols(),X.type());
        Ytrain = new Mat(y.rows(),(int)(y.cols()*0.2),y.type());

        XCV = new Mat(X.rows()- Xtrain.rows() - Xtest.rows(),X.cols(),X.type());
        YCV = new Mat(y.rows(),y.cols()- Ytrain.cols() - Ytest.cols(),y.type());

        for(int i= 0; i< X.rows(); i++){
            double rand = Math.random();

            if(setTest(rand,testCount,CVCount,trainCount)){//test set
                X.row(i).copyTo(Xtest.row(testCount));
                Ytest.col(testCount++).setTo(new Scalar(y.get(0,i)[0]));

            }else if(setCV(rand, testCount, CVCount, trainCount)){//cross validation set
                X.row(i).copyTo(XCV.row(CVCount));
                YCV.col(CVCount++).setTo(new Scalar(y.get(0,i)[0]));

            }else {//training set
                X.row(i).copyTo(Xtrain.row(trainCount));
                Ytrain.col(trainCount++).setTo(new Scalar(y.get(0,i)[0]));
            }
        }


    }

    public MultipleDataSource(IDataSource ds){

        mDataSource = ds;
    }

    public Mat getXTest(){
        //if not initialized, initialize it first
        if(null == Xtest){
            init();
        }
        return Xtest;

    }

    public Mat getYTest(){
        //if not initialized, initialize it first
        if(null == Ytest){
            init();
        }
        return Ytest;
    }
    public Mat getXTrain(){
        //if not initialized, initialize it first
        if(null == Xtrain){
            init();
        }
        return Xtrain;
    }

    public Mat getYTrain(){
        //if not initialized, initialize it first
        if(null == Ytrain){
            init();
        }
        return Ytrain;
    }

    public Mat getXCV(){
        //if not initialized, initialize it first
        if(null == XCV){
            init();
        }
        return XCV;
    }

    public Mat getYCV(){
        if(null == YCV){
            init();
        }
        return YCV;
    }

    public List<Mat> getThetas(){
        return mDataSource.getThetas();
    }*/

}
