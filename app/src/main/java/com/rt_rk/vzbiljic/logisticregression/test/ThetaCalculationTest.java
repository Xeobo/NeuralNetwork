package com.rt_rk.vzbiljic.logisticregression.test;

import android.content.Context;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;

/**
 * Created by vzbiljic on 24.6.16..
 */
public class ThetaCalculationTest implements ITest {

    private Context context;

    private static final String TAG = "ThetaCalculationTest";

    private static final double LAMBDA_VAL = 0;
    private static final double ALPHA_VAL = 16;

    public ThetaCalculationTest(Context context){
        this.context = context;
    }
    @Override
    public void execute(IMachineLearningAlgorithm algorithm) {
        long start = System.currentTimeMillis();

        File f = new File( context.getApplicationInfo().dataDir + "/ex2data1.txt");

        //x will be  vector
        Mat x = new Mat(100,3, CvType.CV_64FC1);


        int xIndex = 0;
        //y will be  vector
        Mat y = new Mat(100,1, CvType.CV_64FC1);

        int yIndex = 0;

        try(BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)))){
            String line;
            //init data from file
            while((line = br.readLine()) != null) {

                String[] pars = line.split(",");

                Double xCurrent1 = Double.parseDouble(pars[0]);

                Double xCurrent2 = Double.parseDouble(pars[1]);

                Double yCurrent = Double.parseDouble(pars[2]);

                x.put(xIndex, 0, 1 );
                x.put(xIndex, 1, (xCurrent1-50)/50);

                x.put(xIndex++, 2, (xCurrent2-50)/50);


                y.put(yIndex++, 0, yCurrent);




            }




            Mat theta = Mat.zeros(3, 1, x.type());

            theta.put(0,0,0);
            theta.put(1,0,0);
            theta.put(2,0,0);


            Log.i(TAG, "THETA: " + theta.size() + ", X: " + x.size());

            theta = Mat.zeros(3, 1, x.type());

            theta.put(0,0,0);
            theta.put(1, 0, 0);
            theta.put(2, 0, 0);

            List<Mat> thetas = algorithm.gradientDescent(x, y, ALPHA_VAL, LAMBDA_VAL,CONVERGE_RATIO,MAX_EXECUTION_TIME, theta);

            theta = thetas.get(0);

            //output
            Log.i(TAG, "===========================================");

            Log.i(TAG, "\ttheta");
            for(int i=0;i<theta.rows();i++) {
                for (int j = 0; j < theta.cols(); j++) {
                    Log.i(TAG,"theta[" + i + "]["+ j +"] = " + theta.get(i,j)[0]);
                }
            }
            Log.i(TAG, "===========================================");



        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }catch (NumberFormatException e){
            e.printStackTrace();
        }


        Log.i(TAG, "Execution time: " + (System.currentTimeMillis() - start) + " ms");
    }
}
