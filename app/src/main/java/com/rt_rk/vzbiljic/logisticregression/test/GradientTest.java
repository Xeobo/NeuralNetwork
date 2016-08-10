package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;
import com.rt_rk.vzbiljic.logisticregression.util.MatrixPrint;

import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 27.7.16..
 */
public class GradientTest implements ITest {


    private IDataSource ds ;

    public GradientTest(Context context){
        ds = new WatchedDataSource(context);
    }

    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        List<Mat> thetas = ds.getThetas();

        List<Mat> grad = algorithm.grad(ds.getX(), ds.getY(), algorithm.hOfTheta(ds.getX(), thetas.get(0), thetas.get(1)), 1, thetas.get(0), thetas.get(1));


        Mat delta2 = grad.get(0);

        Mat delta3 = grad.get(1);


        Mat delta2Vector = new Mat(delta2.cols()* delta2.rows(),1, delta2.type());

        for(int i=0;i<delta2.rows();i++)
            for(int j=0;j<delta2.cols();j++){
                delta2Vector.put(j * delta2.rows() + i,0, delta2.get(i, j)[0]);
            }

        Mat delta3Vector = new Mat(delta3.cols()* delta3.rows(),1, delta3.type());

        for(int i=0;i<delta3.rows();i++)
            for(int j=0;j<delta3.cols();j++){
                delta3Vector.put(j * delta3.rows() + i,0, delta3.get(i, j)[0]);

                MatrixPrint.print(delta3,"delta3");
            }




        List<Mat> list = new ArrayList<>();

        list.add(delta2Vector);
        list.add(delta3Vector);


        //[delta2(:);delta3(:)]
        Core.vconcat(list, delta2);


        for(int i=0;i<delta2.rows();i++)
            for(int j=0;j<delta2.cols();j++){
                Log.i("GradTest", "results [" + i + "][" + j + "]=" + delta2.get(i, j)[0]);
            }
    }
}
