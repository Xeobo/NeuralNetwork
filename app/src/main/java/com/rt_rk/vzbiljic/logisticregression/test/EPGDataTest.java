package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;
import com.rt_rk.vzbiljic.logisticregression.util.MatrixPrint;

import org.opencv.core.Mat;

import javax.sql.DataSource;

/**
 * Created by vzbiljic on 26.7.16..
 */
public class EPGDataTest implements ITest {

    private Context context;
    private IDataSource ds;

    public EPGDataTest(Context context){
        this.context=context;
        ds = new WatchedDataSource(context);

    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        Mat X = ds.getX();

    }
}
