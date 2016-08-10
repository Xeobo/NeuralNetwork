package com.rt_rk.vzbiljic.logisticregression.test;


import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 27.6.16..
 */
public class MatTest implements ITest{

    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        Mat mat = new Mat(2,2, CvType.CV_64FC1);

        mat.put(0,0,1);

        mat.put(0,1,2);

        mat.put(1,0,3);

        mat.put(1, 1, 4);

        Mat sec = new Mat(2,2, CvType.CV_64FC1);

        sec.put(0,0,5);

        sec.put(0,1,6);

        sec.put(1,0,7);

        sec.put(1,1,8);


        mat = mat.reshape(0,1);

        sec = sec.reshape(0,1);

        Core.transpose(mat, mat);

        Core.transpose(sec,sec);


        List<Mat> list = new ArrayList<>();

        list.add(mat);

        list.add(sec);

        Core.vconcat(list,mat);


        for(int i= 0; i<mat.rows();i++)
            for(int j=0;j<mat.cols();j++){
                Log.i("MatTest","result[" + i + "][" + j + "]=" + mat.get(i,j)[0] );
            }


    }
}
