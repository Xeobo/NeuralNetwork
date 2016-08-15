package com.rt_rk.vzbiljic.logisticregression.dataSource.multiple;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Created by vzbiljic on 10.8.16..
 */
public abstract class AbstractCVMultipleDataSource  extends AbstractTestMultipleDataSource{

    private Mat Xcv;
    private Mat ycv;

    public AbstractCVMultipleDataSource(IDataSource dataSource) {
        super(dataSource);
    }

    @Override
    protected final List<Mat> initData(){
        List<Mat> mats = super.initData();
        Xcv = mats.get(4);
        ycv = mats.get(5);
        return mats;
    }

    /**
     *
     * @param dataSource data source
     * @return List containing training data set matrix, training data set labels matrix,
     *          test data set matrix, test data set matrix labels matrix
     *          cross validation data set matrix and cross validation data set labels matrix
     *           respectively restored from data source.
     *          split in 60-20-20 ratio.
     */
    protected abstract List<Mat> init(IDataSource dataSource);

    /**
     *
     * @return Cross validation data set
     */
    public Mat getXCV(){
        if(null == Xcv){
            initData();
        }
        return Xcv;
    }

    /**
     *
     * @return Cross validation data set labels
     */
    public Mat getYCV(){
        if(null == ycv){
            initData();
        }
        return ycv;
    }
}
