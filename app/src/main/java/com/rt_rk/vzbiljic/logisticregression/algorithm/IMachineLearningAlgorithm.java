package com.rt_rk.vzbiljic.logisticregression.algorithm;

import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.List;

/**
 * Created by vzbiljic on 23.6.16..
 */
public interface IMachineLearningAlgorithm {

    /**
     *  Calculates prediction value for input data set, and chosen theta parameters.
     *  If classification problem it returns double (from 0 to 1) values that represent probability that training
     *  example is positive in two class classification problem.
     * @param x data set where rows contain  one individual training example.
     * @param thetas function parameters
     * @return prediction
     */
    Mat hOfTheta(Mat x, Mat... thetas);

    /**
     *  Calculates cost function WITH calculated regularization for machine learning algorithm for given input parameters and chosen theta values
     *
     * @param X data set where ROWS contain  one individual training example.
     * @param Y (row vector) data set where COLUMNS contain  one individual label for ROW with same index in data set (X).
     * @param hOfTheta result of calling hOfTheta function for same input parameters X and thetas as in this function call
     * @param lambdaVal regularization parameter for scaling level of regularization involved on function
     * @param thetas function parameters
     * @return cost as Scalar value
     */
    Scalar JofTheta(Mat X, Mat Y, Mat hOfTheta,double lambdaVal,Mat... thetas);

    /**
     *  Calculates cost function WITHOUT calculated regularization for machine learning algorithm for given input parameters and chosen theta values.
     *
     * @param X data set where ROWS contain  one individual training example.
     * @param Y (row vector) data set where COLUMNS contain  one individual label for ROW with same index in data set (X).
     * @param hOfTheta result of calling hOfTheta function for same input parameters X and thetas as in this function call
     * @param thetas function parameters
     * @return cost as Scalar value
     */
    Scalar JofThetaNoRegularization(Mat X, Mat Y, Mat hOfTheta,Mat... thetas);


    /**
     * Calculates differential value of cost function WITH regularization.
     * It represents calculating gradient for one step of gradientDescent algorithm
     *
     * @param X data set where ROWS contain  one individual training example.
     * @param Y (row vector) data set where COLUMNS contain  one individual label for ROW with same index in data set (X).
     * @param hOfTheta result of calling hOfTheta function for same input parameters X and thetas as in this function call
     * @param lambdaVal regularization parameter for scaling level of regularization involved on function
     * @param thetas function parameters
     * @return gradients
     */
    List<Mat> grad(Mat X,Mat Y,Mat hOfTheta,double lambdaVal,Mat... thetas);

    /**
     *  Minimization function for calculating optimal values of theta for given training set
     *
     * @param X data set where ROWS contain  one individual training example.
     * @param y (row vector) data set where COLUMNS contain  one individual label for ROW with same index in data set (X).
     * @param alpha defines how big steps should minimization function should make. If too large algorithm may diverge, if too small algorithm may take much time to converge
     * @param lambdaVal regularization parameter for scaling level of regularization involved on function
     * @param convergeRatio how small difference in cost functions should between two steps, so minimization function would know that algorithm converged
     * @param MaxExecutionTime maximal execution time of algorithm
     * @param thetas initial chosen theta parameters.
     * @return optimal value of theta parameters
     */
    List<Mat> gradientDescent(Mat X, Mat y, double alpha, double lambdaVal, double convergeRatio, long MaxExecutionTime,Mat... thetas);

}
