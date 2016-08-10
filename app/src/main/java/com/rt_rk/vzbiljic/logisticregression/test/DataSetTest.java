package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.ExpendPositivesDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.ShrinkDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.dataSource.WatchedDataSource;

/**
 * Created by vzbiljic on 3.8.16..
 */
public class DataSetTest implements ITest {


    private Context context;

    public DataSetTest(Context context){
        this.context = context;
    }

    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        IDataSource source = new ShrinkDataSourceFilter(new ExpendPositivesDataSourceFilter(
                new WatchedDataSource(context,10)));

        source.getX();

    }
}
