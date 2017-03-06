package com.rt_rk.vzbiljic.logisticregression;

import android.app.IntentService;
import android.content.Intent;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.LogisticRegression;
import com.rt_rk.vzbiljic.logisticregression.algorithm.NeuralNetwork;
import com.rt_rk.vzbiljic.logisticregression.dataSource.SinusDataSource;
import com.rt_rk.vzbiljic.logisticregression.test.ITest;
import com.rt_rk.vzbiljic.logisticregression.test.AlgorithmTest;


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


        ITest test =  new AlgorithmTest(new SinusDataSource());

        long startTime = System.currentTimeMillis();

        test.execute(new NeuralNetwork(new LogisticRegression()));

        Log.i(TAG, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++");
        Log.i(TAG,"\tExecution time : " + (System.currentTimeMillis() - startTime)/1000. + " s");
        Log.i(TAG, "+++++++++++++++++++++++++++++++++++++++++++++++++++++++");

    }


}
