package com.rt_rk.vzbiljic.logisticregression.util;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

/**
 * Created by vzbiljic on 21.7.16..
 */
public class SinusoidInitMatrix {

    /**
     *
     * @param length
     * @param width
     * @return matrix with size length x (width + 1)
     */
    public static Mat debugInitializeWeights(int length, int width){



        Mat w = Mat.zeros(length, width + 1, CvType.CV_64FC1);


        for(int i = 0; i < w.rows();i++)
            for(int j=0;j < w.cols(); j++){
                w.put(i,j,w.rows()*j+i+1);
            }

        for(int i = 0; i < w.rows();i++)
            for(int j=0;j < w.cols(); j++){

                w.put(i, j, Math.sin(w.get(i, j)[0])/10);
            }

        return w;
    }
}
