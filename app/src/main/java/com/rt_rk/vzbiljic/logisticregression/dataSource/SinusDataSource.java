package com.rt_rk.vzbiljic.logisticregression.dataSource;

import com.rt_rk.vzbiljic.logisticregression.util.SinusoidInitMatrix;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 20.7.16..
 */
public class SinusDataSource implements IDataSource {

    private static int INPUT_LAYER_SIZE = 3;
    private static int HIDDEN_LAYER_SIZE = 5;
    private static int NUM_LABELS = 1;
    private static int M = 50;




    @Override
    public Mat getX() {
        Mat xHelp = SinusoidInitMatrix.debugInitializeWeights(M, INPUT_LAYER_SIZE - 1);

        Mat X = new Mat(M,INPUT_LAYER_SIZE+1,xHelp.type());


        X.col(0).setTo(new Scalar(1));

        for(int i=0; i< xHelp.rows(); i++){
            X.put(i,1,xHelp.get(i,0)[0]);
            X.put(i,2,xHelp.get(i,1)[0]);
            X.put(i,3,xHelp.get(i,2)[0]);
        }

        return X;
    }

    @Override
    public Mat getY() {
        Mat y = Mat.zeros(NUM_LABELS,M,CvType.CV_64FC1);

        for(int i= 0; i< y.cols(); i++)
            if(NUM_LABELS == 1){
                if(Math.random() > 0.5){
                    y.put(0,i, 1);
                }
            }else {
                y.put((int)(Math.random()*y.cols()),i,1);
            }

        return y;
    }

    @Override
    public List<Mat> getThetas() {
        Mat theta1 = SinusoidInitMatrix.debugInitializeWeights(HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE);

        Mat theta2 = SinusoidInitMatrix.debugInitializeWeights(NUM_LABELS, HIDDEN_LAYER_SIZE);

        List<Mat> lists = new ArrayList<>();

        lists.add(theta1);
        lists.add(theta2);

        return lists;
    }
}
