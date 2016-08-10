package com.rt_rk.vzbiljic.logisticregression.dataSource.filter;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;
import com.rt_rk.vzbiljic.logisticregression.dataSource.filter.AbstractDataSourceFilter;
import com.rt_rk.vzbiljic.logisticregression.util.MatrixPrint;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 8.8.16..
 */
public class TypeDataSourceFilter extends AbstractDataSourceFilter {


    public TypeDataSourceFilter(IDataSource filteredDataSource) {
        super(filteredDataSource);
    }

    @Override
    protected List<Mat> filter(Mat Xstart, Mat ystart) {
        List<Mat> mats = new ArrayList<>();

        Mat X = new Mat(Xstart.rows(),Xstart.cols(), CvType.CV_32F);

        Mat y = new Mat(ystart.rows(),ystart.cols(), CvType.CV_32S);

        for(int i=0; i< X.rows();i++){
            y.put(0, i, ystart.get(0, i)[0]);
            for (int j = 0; j < X.cols(); j++) {
                X.put(i, j, Xstart.get(i, j)[0]);

            }
        }
        MatrixPrint.print(y,"y");
        mats.add(X);
        mats.add(y);

        return mats;

    }
}
