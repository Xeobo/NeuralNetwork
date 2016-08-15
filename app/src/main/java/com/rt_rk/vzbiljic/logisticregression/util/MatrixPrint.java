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

        printSize(mat,label);
    }

    public static void peak(Mat mat,String label){
        for(int i=0; i<mat.rows();i+=100){
            String logMsg = "{";
            for(int j=0; j<mat.cols();j++){
                if(i != 0){
                    logMsg += ", ";
                }
                logMsg += mat.get(i, j)[0];
            }
            logMsg += "}";

            Log.i("MatrixPrint",label + "["+ i +"]= " + logMsg );
        }

        printSize(mat,label);
    }

    public static void printSize(Mat mat,String label) {
        Log.i("MatrixPrint", "-------------------------------------");
        Log.i("MatrixPrint", " \tsize of " + label + ": " + mat.rows() + " x " + mat.cols());
        Log.i("MatrixPrint", "-------------------------------------");
    }
}
