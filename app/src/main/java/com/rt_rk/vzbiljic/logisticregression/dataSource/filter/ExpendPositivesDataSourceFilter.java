package com.rt_rk.vzbiljic.logisticregression.dataSource.filter;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.AbstractDataSourceFilter;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 2.8.16..
 */
public class ExpendPositivesDataSourceFilter extends AbstractDataSourceFilter {


    public ExpendPositivesDataSourceFilter(IDataSource filteredDataSource) {
        super(filteredDataSource);
    }

    @Override
    protected List<Mat> filter(Mat Xstart,Mat ystart){
        int positiveExamples = 0;

        //count positive examples
        for(int i= 0;i<ystart.cols();i++){
            if(ystart.get(0,i)[0] == 1){
                positiveExamples++;
            }
        }

        final int MULTIPLY_RATIO = ystart.cols()/positiveExamples;

        //add every positive example to score |MULTIPLY_RATIO| times
        Mat X = new Mat(Xstart.rows() + MULTIPLY_RATIO*positiveExamples,Xstart.cols(),Xstart.type());

        Mat y = new Mat(ystart.rows(),ystart.cols()+ MULTIPLY_RATIO*positiveExamples,ystart.type());

        int newDataCursor = 0;

        for(int i=0; i< Xstart.rows();i++){
            putRow(X, newDataCursor, Xstart, i);

            putColumn(y,newDataCursor++,ystart,i);

            //replicate positive examples
            if(ystart.get(0,i)[0] == 1){
                for(int j=0; j< MULTIPLY_RATIO;j++){

                    putRow(X, newDataCursor, Xstart, i);

                    putColumn(y,newDataCursor++,ystart,i);
                }
            }

        }
        //resets filterChange if get operations are called again
        //to skip initializing



        List<Mat> list = new ArrayList<>();
        list.add(X);
        list.add(y);
        return list;
    }


}
