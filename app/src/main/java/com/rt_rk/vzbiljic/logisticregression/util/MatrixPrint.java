package com.rt_rk.vzbiljic.logisticregression.util;

import android.util.Log;

import org.opencv.core.Mat;

/**
 * Created by vzbiljic on 21.7.16..
 */
public class MatrixPrint {
    /**
     *
     * @param mat matrix to be printed
     * @param label label used when printing matrix's data
     */
    public static void print(Mat mat,String label){
        for(int i=0; i<mat.rows();i++)
            for(int j=0; j<mat.cols();j++){
                Log.i("MatrixPrint", label + "[" + i + "][" + j + "]=" + mat.get(i, j)[0]);
            }

        Log.i("MatrixPrint", "-------------------------------------");
        Log.i("MatrixPrint", " \tsize of " + label + ": " + mat.rows() + " x " + mat.cols());
        Log.i("MatrixPrint", "-------------------------------------");
    }
}
