package com.rt_rk.vzbiljic.logisticregression.dataSource.filter;

import com.rt_rk.vzbiljic.logisticregression.dataSource.IDataSource;

import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 2.8.16..
 */
public abstract class AbstractDataSourceFilter implements IDataSource {

     private IDataSource filteredDataSource;

     private Mat X = null;
     private Mat y = null;
     private boolean filterChanged = true;




     private void init(){
          List<Mat> mats = filter(filteredDataSource.getX(),filteredDataSource.getY());
          X = mats.get(0);
          y = mats.get(1);
          filterChanged = false;
     }

     //helper function that can be used in derived classes
     protected void putRow(Mat to, int toRow, Mat from, int fromRow){
          for(int i=0;i<from.cols();i++){
               to.put(toRow,i,from.get(fromRow, i)[0]);
          }
     }
     //helper function that can be used in derived classes
     protected void putColumn(Mat to, int toColumn, Mat from, int fromColumn){
          for(int i=0;i<from.rows();i++){
               to.put(i,toColumn,from.get(i, fromColumn)[0]);
          }
     }

     //function should do filtering and than return new filtered X and y matrices
     protected abstract List<Mat> filter(Mat Xstart,Mat ystart);


     public AbstractDataSourceFilter(IDataSource filteredDataSource){
          this.filteredDataSource = filteredDataSource;
     }

     @Override
     public Mat getX() {
          if(filterChanged){
               init();
          }
          return  X;
     }

     @Override
     public Mat getY() {
          if(filterChanged){
              init();
          }
          return  y;
     }

     @Override
     public List<Mat> getThetas() {
          return filteredDataSource.getThetas();
     }

     public final void setDataSource(IDataSource filteredDataSource){
          this.filteredDataSource = filteredDataSource;

          filterChanged = true;
     }
}
