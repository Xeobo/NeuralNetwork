package com.rt_rk.vzbiljic.logisticregression.dataSource.multiple;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 15.8.16..
 */
public class FifoCVMulitpleDataSource extends AbstractCVMultipleDataSource {

    public FifoCVMulitpleDataSource(IDataSource dataSource) {
        super(dataSource);
    }

    @Override
    protected List<Mat> init(IDataSource dataSource) {
        Mat X = dataSource.getX();
        Mat y = dataSource.getY();


        Mat Xtrain = new Mat((int)(X.rows()*0.6),X.cols(),X.type());
        Mat Ytrain = new Mat(y.rows(),(int)(y.cols()*0.6),y.type());

        Mat Xtest = new Mat((int)(X.rows()*0.2),X.cols(),X.type());
        Mat Ytest = new Mat(y.rows(),(int)(y.cols()*0.2),y.type());

        Mat XCV = new Mat(X.rows()- Xtest.rows() - Xtrain.rows(),X.cols(),X.type());
        Mat YCV = new Mat(y.rows(),y.cols()- Ytest.cols() - Ytrain.cols(),y.type());

        for (int i = 0; i < X.rows(); i++) {
            if(i < Xtrain.rows()){//training set

                X.row(i).copyTo(Xtrain.row(i));
                Ytrain.col(i).setTo(new Scalar(y.get(0, i)[0]));

            }else if(i < Xtrain.rows() + XCV.rows()){//cross validation test

                int iCV = i - Xtrain.rows();
                X.row(i).copyTo(XCV.row(iCV));
                YCV.col(iCV).setTo(new Scalar(y.get(0,i)[0]));

            }else{//test set
                int iTest = i - Xtrain.rows() - XCV.rows();
                X.row(i).copyTo(Xtest.row(iTest));
                Ytest.col(iTest).setTo(new Scalar(y.get(0,i)[0]));
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
