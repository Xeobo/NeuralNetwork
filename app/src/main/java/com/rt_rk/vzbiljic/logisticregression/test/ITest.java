package com.rt_rk.vzbiljic.logisticregression.test;

import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;

/**
 * Created by vzbiljic on 24.6.16..
 */
public interface ITest {
    //maximum time for gradient descent algorithm to run in millis
    static final long MAX_EXECUTION_TIME = 20*60*1000;

    //if cost function has changed to less than CONVERGE_RATIO
    //it indicates that gradient descent has converged
    static final double CONVERGE_RATIO = 0.00001;

    //executes specific test on machine learning algorithm
    void execute(IMachineLearningAlgorithm algorithm);

}
