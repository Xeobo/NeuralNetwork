package com.rt_rk.vzbiljic.logisticregression.dataSource.filter;

import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.AbstractDataSourceFilter;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 2.8.16..
 */
public class ShrinkDataSourceFilter extends AbstractDataSourceFilter {

    //shrink ratio set how much data should be shrunk from original data set
    //0 - non, 1 - all
    private static final double SHRINKRATIO = 0.9;

    public ShrinkDataSourceFilter(IDataSource filteredDataSource) {
        super(filteredDataSource);
    }

    @Override
    protected List<Mat> filter(Mat Xstart, Mat ystart) {
        int negativeExamples = 0;

        //count positive examples
        for(int i= 0;i<ystart.cols();i++){
            if(ystart.get(0,i)[0] == 0){
                negativeExamples++;
            }
        }

        //add every positive example to score |MULTIPLY_RATIO| times
        Mat X = new Mat((int)(Xstart.rows()*(1-SHRINKRATIO)),Xstart.cols(),Xstart.type());

        Mat y = new Mat(ystart.rows(),(int)(ystart.cols()*(1-SHRINKRATIO)),ystart.type());

        int newDataCursor = 0;
        for(int i=0;i<Xstart.rows();i++){
            //insert only (1-SHRINKRATIO)*100 % of data to data set
            if((Math.random() > SHRINKRATIO &&  newDataCursor < X.rows()) || (Xstart.rows() - i) <= (X.rows() - newDataCursor )){

                putRow(X,newDataCursor,Xstart,i);

                putColumn(y, newDataCursor++, ystart, i);
            }

        }



        List<Mat> list = new ArrayList<>();

        Log.i("ShrinkDataSourceFilter", "preslikano " + newDataCursor + " podataka. A broj redova je " + X.rows() + ".");

        list.add(X);

        list.add(y);

        return list;
    }
}
