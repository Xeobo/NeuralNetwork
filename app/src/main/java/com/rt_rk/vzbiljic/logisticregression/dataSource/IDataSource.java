package com.rt_rk.vzbiljic.logisticregression.dataSource;

import org.opencv.core.Mat;

import java.util.List;

/**
 * Created by vzbiljic on 20.7.16..
 */
public interface IDataSource {

    Mat getX();

    Mat getY();

    List<Mat> getThetas();

}
