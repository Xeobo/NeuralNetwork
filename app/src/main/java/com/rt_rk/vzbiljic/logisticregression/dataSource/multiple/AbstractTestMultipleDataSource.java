package com.rt_rk.vzbiljic.logisticregression.dataSource.multiple;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.AbstractDataSourceFilter;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Created by vzbiljic on 10.8.16..
 */
public abstract class AbstractTestMultipleDataSource implements IDataSource{

    private Mat Xtrain;
    private Mat ytrain;

    private Mat Xtest;
    private Mat ytest;

    protected IDataSource dataSource;

    protected List<Mat> initData(){
        List<Mat> mats = init(dataSource);

        Xtrain = mats.get(0);
        ytrain = mats.get(1);

        Xtest = mats.get(2);
        ytest = mats.get(3);

        return mats;
    }


    /**
     *
     * @param dataSource data source
     * @return List containing training data set matrix, training data set labels matrix,
     *          test data set matrix and test data set labels matrix respectively restored from data;
     *          split in 70-30 ratio.
     */
    protected abstract List<Mat> init(IDataSource dataSource);

    public AbstractTestMultipleDataSource(IDataSource dataSource){
        this.dataSource = dataSource;
    }

    /**
     *
     * @return Whole data set
     */
    @Override
    public Mat getX() {
        return dataSource.getX();
    }

    /**
     *
     * @return whole data set labels;
     */
    @Override
    public Mat getY() {
        return dataSource.getX();
    }
    @Override
    public List<Mat> getThetas() {
        return dataSource.getThetas();
    }

    /**
     *
     * @return Training data set
     */
    public Mat getXTrain(){
        if(null == Xtrain){
            initData();
        }
        return Xtrain;
    }

    /**
     *
     * @return Training data set labels
     */
    public Mat getYTrain(){
        if(null == ytrain){
            initData();
        }
        return ytrain;
    }

    /**
     *
     * @return Test data set
     */
    public Mat getXTest(){
        if(null == Xtest){
            initData();
        }
        return Xtest;
    }

    /**
     *
     * @return Test data set labels
     */
    public Mat getYTest(){
        if(null == ytest){
            initData();
        }
        return ytest;
    }

}
