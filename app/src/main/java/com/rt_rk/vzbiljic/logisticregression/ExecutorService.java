package com.rt_rk.vzbiljic.logisticregression;

import android.app.IntentService;
import android.content.Intent;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.LogisticRegression;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.algorithm.SVMAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.AbstractDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.MultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.ExpendPositivesDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.ShrinkDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.TypeDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.AbstractCVMultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.multiple.RandomCVMultipleDataSource;
import com.rt_rk.vzbiljic.logisticregression.test.ITest;
import com.rt_rk.vzbiljic.logisticregression.test.PredictionTest;
import com.rt_rk.vzbiljic.logisticregression.test.SVMTest;


/**
 * When called executes given ITest
 */
public class ExecutorService extends IntentService {


    private static final String TAG = "ExecutorService";

    public ExecutorService() {
        super("ExecutorService");
    }

    @Override
    protected void onHandleIntent(Intent intent) {

        IDataSource filterSource = new ShrinkDataSourceFilter(
                new ExpendPositivesDataSourceFilter(
                    new WatchedDataSource(this,10)));


        ITest test =  new PredictionTest(new RandomCVMultipleDataSource(filterSource));

        long startTime = System.currentTimeMillis();

        test.execute(new NeuralNetwork(new LogisticRegression()));

        Log.i(TAG, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        Log.i(TAG,"\tExecution time : " + (System.currentTimeMillis() - startTime)/1000. + " s");
        Log.i(TAG, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++");

    }


}
