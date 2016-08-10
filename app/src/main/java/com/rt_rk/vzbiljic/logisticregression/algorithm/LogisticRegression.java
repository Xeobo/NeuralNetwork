package com.rt_rk.vzbiljic.logisticregression.algorithm;

import android.content.Context;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 7.6.16.
 *
 */
public class LogisticRegression implements IMachineLearningAlgorithm {

    private static String TAG = "LogisticRegression";


    /**
     *
     *
     * hOfTheta = 1./(1+ exp(-X*theta'));
     *
     *  size(hOfTheta)=X.rows x theta.cols
     *
     * */
    public Mat hOfTheta(Mat x, Mat... thetasTrans) {
        if(thetasTrans.length == 0)
            return null;
        Mat thetaTrans = thetasTrans[0];

        Mat result = new Mat();


        //-x
        Core.subtract(Mat.zeros(x.rows(), x.cols(), x.type()), x, result);

        //-X*theta'
        Core.gemm(result, thetaTrans, 1, new Mat(), 0, result, 0);

        //exp(-x*theta')
        Core.exp(result, result);

        //1 + exp(-x*theta')
        Core.add(Mat.ones(result.rows(), result.cols(), result.type()), result, result);


        //1./(1+ exp(-x*theta')
        Core.divide(1, result, result);

        return result;
    }
    /**
     *
     *  m= y.rows
     * J= (1/m)*{sum(-y.*log(H) - (1-y).*log(1-H)) + 1/2*sum((lambda.*theta).*theta)}
     *
     *
     *
     * */
    @Override
    public Scalar JofTheta(Mat X, Mat Y, Mat hOfTheta,double lambdaVal,Mat... thetas){
        if(thetas.length == 0)
            return  null;
        Mat theta = thetas[0];

        int m = Y.rows();

        Mat J = new Mat();


        Mat logH = new Mat();

        //log(H)
        Core.log(hOfTheta, logH);

        //-y
        Core.subtract(Mat.zeros(Y.rows(), Y.cols(), Y.type()), Y, J);

        //-y.*log(H)
        J = J.mul(logH);

        Mat oneMinusY = new Mat();
        //(1-y)
        Core.subtract(Mat.ones(Y.rows(), Y.cols(), Y.type()), Y, oneMinusY);


        //1-H
        Core.subtract(Mat.ones(hOfTheta.rows(), hOfTheta.cols(), hOfTheta.type()), hOfTheta, logH);

        //log(1-H);
        Core.log(logH, logH);

        //(1-y).*log(1-H)
        oneMinusY = oneMinusY.mul(logH);

        // -y.*log(H) - (1-y).*log(1-H)
        Core.subtract(J, oneMinusY, J);

        //sum(-y.*log(H) - (1-y).*log(1-H))
        Scalar scalar = Core.sumElems(J);

        //lambda = [ 0 lambdaVal ... lambdaVal];
        Mat lambda = Mat.ones(theta.rows(),theta.cols(),theta.type());
        lambda.put(0,0,0);
        Core.multiply(lambda,new Scalar(lambdaVal),lambda);

        Mat result = new Mat();

        //lambda.*theta
        result = lambda.mul(theta);

        //(lambda.*theta).*theta
        result = result.mul(theta);

        //sum((lambda.*theta).*theta)
        Scalar sum = Core.sumElems(result);

        //sum(-y.*log(H) - (1-y).*log(1-H)) + 1/2*sum((lambda.*theta).*theta)
        scalar.val[0] += sum.val[0]/2;

        //(1/m)*{sum(-y.*log(H) - (1-y).*log(1-H)) + 1/2*sum((lambda.*theta).*theta)}
        scalar.val[0] /= m;

        return scalar;

    }
    /**
     *
     *  m= y.rows
     *  J= (1/m)*{sum(-y.*log(H) - (1-y).*log(1-H))}
     *
     *
     *
     * */
    @Override
    public Scalar JofThetaNoRegularization(Mat X, Mat Y, Mat hOfTheta, Mat... thetas) {
        if(thetas.length == 0)
            return  null;
        Mat theta = thetas[0];

        int m = Y.rows();

        Mat J = new Mat();


        Mat logH = new Mat();

        //log(H)
        Core.log(hOfTheta, logH);

        //-y
        Core.subtract(Mat.zeros(Y.rows(), Y.cols(), Y.type()), Y, J);

        //-y.*log(H)
        J = J.mul(logH);

        Mat oneMinusY = new Mat();
        //(1-y)
        Core.subtract(Mat.ones(Y.rows(), Y.cols(), Y.type()), Y, oneMinusY);


        //1-H
        Core.subtract(Mat.ones(hOfTheta.rows(), hOfTheta.cols(), hOfTheta.type()), hOfTheta, logH);

        //log(1-H);
        Core.log(logH, logH);

        //(1-y).*log(1-H)
        oneMinusY = oneMinusY.mul(logH);

        // -y.*log(H) - (1-y).*log(1-H)
        Core.subtract(J, oneMinusY, J);

        //sum(-y.*log(H) - (1-y).*log(1-H))
        Scalar scalar = Core.sumElems(J);

        //(1/m)*{ sum(-y.*log(H) - (1-y).*log(1-H)) }
        scalar.val[0] /= m;

        return scalar;
    }


    /**
     *
     *
     *      (1/m)*(X'*(H-y) + lambda.*theta)
     *
     *      size = (n+1)x1
     *
     */
    public List<Mat> grad(Mat X,Mat Y,Mat hOfTheta,double lambdaVal,Mat... thetas){
        if(thetas.length == 0)
            return  null;

        Mat theta = thetas[0];

        Mat result = new Mat();

        //number of data in data set
        int m = Y.rows();
        //H-y
        Core.subtract(hOfTheta, Y, result);

        //X'
        Mat XTrans = new Mat(X.rows(),X.cols(),X.type());


        Core.transpose(X, XTrans);

        //X'*(H-y)
        Core.gemm(XTrans, result, 1, new Mat(), 0, result, 0);

        //lambda = [ 0 ;1; ... 1;];
        Mat lambda = Mat.ones(theta.rows(),theta.cols(),theta.type());
        lambda.put(0, 0, 0);

        //lambda = [ 0 ;lambdaVal; ... lambdaVal;];
        Core.multiply(lambda, new Scalar(lambdaVal),lambda);

        //lambda.*theta
        lambda = lambda.mul(theta);

        //X'*(H-y) + lambda.*theta
        Core.add(result, lambda, result);

        //1/m*result
        Core.divide(result, new Scalar(m),result);

        ArrayList<Mat> list = new ArrayList<>();

        list.add(result);

        return list;
    }


    /**
     *
     *  loop util converge{
     *      Theta = Theta - alpha*dJ/dTheta;
     *   }
     */
    @Override
    public List<Mat> gradientDescent(Mat X, Mat y, double alpha, double lambda, double convergeRatio, long maxExecutionTime,Mat... initial_theta){
        if(initial_theta.length == 0)
            return  null;

        Mat theta = initial_theta[0];

        //cost in current iteration
        Scalar cost = new Scalar(0);

        //cost in last iteration
        Scalar lastCost = new Scalar(-1);
        int i=0;

        long startExecutionTime = System.currentTimeMillis();
        while(true){
            Mat hOfTheta = hOfTheta(X, theta);

            //J = 1/m * sum(-y.*log(H) - (1-y).*log(1-H));
            cost = JofTheta(X, y, hOfTheta,lambda,theta);


            Mat result = new Mat();

            //dJ/dTheta = grad = 1/m*(X'*(H-y))
            List<Mat> gradList = grad(X, y, hOfTheta,lambda,theta);

            Mat grad = gradList.get(0);

            //alpha.*grad
            Core.multiply(grad, Scalar.all(alpha), result);
            if(i % 100 == 0)
                Log.i(TAG, "current cost: " + cost.val[0] + ",in iteration: " + i);

            Core.subtract(theta, result, theta);

            //in first iteration just continue
            if(lastCost.val[0] == -1){
                lastCost.val[0] = cost.val[0];
                continue;
            }

            //if less than CONVERGE_RATIO algorithm converged!
            if(lastCost.val[0] - cost.val[0] < convergeRatio){
                Log.i(TAG,"Gradient descent has converged!");
                break;
            }
            //execution time is limited with MAX_EXECUTION TIME!
            if(System.currentTimeMillis() - startExecutionTime > maxExecutionTime){
                Log.v(TAG,"Gradient descent hasn't converged!");
                break;
            }

            lastCost.val[0] = cost.val[0];
        }
        Log.i(TAG, "minimized cost function: " + cost.val[0]);

        Log.i(TAG, "===========================================");
        Log.i(TAG, "\ttheta");
        for(int j=0;j<theta.rows();j++) {
            for(int z=0;z<theta.cols();z++) {
                Log.i(TAG, "currenttheta[" + j + "][" + z + "] = " + theta.get(j, z)[0]);
            }
        }
        Log.i(TAG, "===========================================");

        List<Mat> lista = new ArrayList<>();

        lista.add(theta);
        return lista;
    }
}
