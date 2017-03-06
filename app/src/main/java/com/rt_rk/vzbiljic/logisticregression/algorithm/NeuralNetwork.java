package com.rt_rk.vzbiljic.logisticregression.algorithm;


import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by vzbiljic on 23.6.16..
 */
public class NeuralNetwork implements IMachineLearningAlgorithm {
    private static final double E = 0.0001;

    private static final String TAG = "NeuralNetwork";


    /**
     *  represents basic algorithm that will be used to compute function in
     *  one node of Neural Network
     */
    private IMachineLearningAlgorithm algorithm = null;



    private Mat gradientNum(Mat X, Mat Y, Mat theta1, Mat theta2,double lambdaVal){
        Mat numgrad = Mat.zeros(theta1.rows()*theta1.cols() + theta2.rows()*theta2.cols(),1,theta1.type());

        Mat perturb = Mat.zeros(theta1.rows(), theta1.cols(), theta1.type());


        for(int i=0; i<theta1.rows();i++){
            for(int j=0; j<theta1.cols();j++) {
                perturb.put(i, j, E);

                Mat lessTheta1 = new Mat(theta1.rows(),theta1.cols(),theta1.type());



                Core.subtract(theta1, perturb, lessTheta1);

                Mat bigTheta1 = new Mat(theta1.rows(),theta1.cols(),theta1.type());

                Core.add(theta1, perturb, bigTheta1);

                Scalar lossSmall = JofTheta(X, Y, hOfTheta(X, lessTheta1, theta2), lambdaVal,lessTheta1,theta2);

                Scalar lossBig = JofTheta(X, Y, hOfTheta(X, bigTheta1, theta2), lambdaVal,bigTheta1,theta2);

                numgrad.put(j*theta1.rows() + i,0,(lossBig.val[0]-
                        lossSmall.val[0])/(2*E));

                perturb.put(i, j, 0);

            }


        }

        int theta1size = theta1.rows()*theta1.cols();

        perturb = Mat.zeros(theta2.rows(),theta2.cols(),theta1.type());

        for(int i=0; i<theta2.rows();i++){
            for(int j=0; j<theta2.cols();j++) {
                perturb.put(i, j, E);

                Mat lessTheta2 = new Mat(theta2.rows(),theta2.cols(),theta2.type());

                Core.subtract(theta2, perturb, lessTheta2);

                Mat bigTheta2 = new Mat(theta2.rows(),theta2.cols(),theta2.type());

                Core.add(theta2, perturb, bigTheta2);

                Scalar lossSmall = JofTheta(X, Y, hOfTheta(X, theta1, lessTheta2), lambdaVal,theta1,lessTheta2);

                Scalar lossBig = JofTheta(X, Y, hOfTheta(X, theta1, bigTheta2), lambdaVal,theta1,bigTheta2);

                numgrad.put(theta1size + j*theta2.rows() + i,0,(lossBig.val[0]-lossSmall.val[0])/(2*E));

                perturb.put(i, j, 0);

            }


        }

        return numgrad;

    }
    public NeuralNetwork(IMachineLearningAlgorithm algorithm){
        this.algorithm = algorithm;
    }

    public void checkGradient(Mat X,Mat y,Mat theta1,Mat theta2,double lambdaVal){
        List<Mat> grad = grad(X, y, hOfTheta(X, theta1, theta2), lambdaVal, theta1, theta2);
        Mat num = gradientNum(X, y, theta1, theta2, lambdaVal);


        Mat delta2 = grad.get(0);

        Mat delta3 = grad.get(1);


        Mat delta2Vector = new Mat(delta2.cols()* delta2.rows(),1, delta2.type());

        for(int i=0;i<delta2.rows();i++)
            for(int j=0;j<delta2.cols();j++){
                delta2Vector.put(j * delta2.rows() + i,0, delta2.get(i, j)[0]);
            }

        Mat delta3Vector = new Mat(delta3.cols()* delta3.rows(),1, delta3.type());

        for(int i=0;i<delta3.rows();i++)
            for(int j=0;j<delta3.cols();j++){
                delta3Vector.put(j * delta3.rows() + i,0, delta3.get(i, j)[0]);
            }




        List<Mat> list = new ArrayList<>();

        list.add(delta2Vector);
        list.add(delta3Vector);


        //[delta2(:);delta3(:)]
        Core.vconcat(list, delta2);


        for(int i=0;i<delta2.rows();i++)
            for(int j=0;j<delta2.cols();j++){
                Log.i(TAG,"results [" + i + "][" + j + "]=" + delta2.get(i,j)[0] + ", " +  num.get(i,j)[0]);
            }


        Mat sub = new Mat();

        Core.subtract(num, delta2, sub);

        Mat add = new Mat();
        Core.add(num,delta2,add);

        double norm = Core.norm(sub)/Core.norm(add);

        Log.i(TAG,"normalized (should bee less than 1e-9) : " + norm);

    }


    /**
     *
     * @param x
     * @param thetas
     * @return
     *      checked!
     */
    @Override
    public Mat hOfTheta(Mat x, Mat... thetas) {
        if(thetas.length <2)
            return null;

        Mat aX = x;
        Mat result = new Mat();

        for(int index=0; index< thetas.length;index++){
            result = algorithm.hOfTheta(aX,thetas[index].t());

            aX = new Mat(result.rows(),result.cols() + 1,result.type());

            aX.col(0).setTo(new Scalar(1));

            for(int i=0; i< result.rows(); i++){
                for(int j=0; j< result.cols(); j++) {
                    aX.put(i, j+1, result.get(i, j)[0]);
                }
            }
        }


        return  result.t();
    }

    /**
     *
     *
     *
     *      lambda_matrix = lambda * ones(hidden_layer_size, input_layer_size );
     *      lambda_matrix = [zeros(hidden_layer_size,1) lambda_matrix];
     *      mat1 = lambda_matrix.*Theta1.*Theta1;
     *
     *
     *      lambda_matrix_2 = lambda * ones(num_labels, hidden_layer_size);
     *      lambda_matrix_2 = [zeros(num_labels,1) lambda_matrix_2];
     *
     *       mat2 = lambda_matrix_2.*Theta2.*Theta2;
     *
     *
     *      J = 1/m*sum(sum((-y.*log(H) - ((1-y).*log(1- H))))) + 1/(2*m)*(sum(sum(mat1)) + sum(sum(mat2)));
     */

    @Override
    public Scalar JofTheta(Mat X, Mat Y, Mat hOfTheta, double lambdaVal, Mat... thetas) {
        if(thetas.length <2)
            return null;


        double m = Y.cols();


        Mat[] lambdaMatrices = new Mat[thetas.length];


        double sum = 0;

        for(int i=0; i< thetas.length;i++){

            lambdaMatrices[i] = Mat.ones(thetas[i].rows(), thetas[i].cols(), thetas[i].type());

            Core.multiply(lambdaMatrices[i], new Scalar(lambdaVal),lambdaMatrices[i]);

            Mat.zeros(lambdaMatrices[i].rows(),1,lambdaMatrices[i].type()).copyTo(lambdaMatrices[i].col(0));

            //mul = lambda_matrix.*ThetaTrans.*ThetaTrans
            Mat mul = lambdaMatrices[i].mul(thetas[i]).mul(thetas[i]);

            //(sum(mat1) + ... + sum(matX))
            sum += Core.sumElems(mul).val[0];

        }

        //(sum(mat1) + ... + sum(matX))/2
        sum = sum / 2;


        Mat result =  new Mat();

        //log(H)
        Core.log(hOfTheta, result);

        Mat minusY = new Mat();

        Mat yTrans = Y.t();
        //-y
        Core.subtract(Mat.zeros(Y.rows(), Y.cols(), Y.type()), Y, minusY);


        //-y.*log(H)
        result = result.mul(minusY);

        //1-H
        Mat oneMinusH = new Mat(hOfTheta.rows(),hOfTheta.cols(),hOfTheta.type());
        Core.subtract(Mat.ones(hOfTheta.rows(), hOfTheta.cols(), hOfTheta.type()), hOfTheta, oneMinusH);

        //log(1-H)
        Core.log(oneMinusH, oneMinusH);

        //1-y
        Mat oneMinusY = new Mat();
        Core.subtract(Mat.ones(Y.rows(),Y.cols(),Y.type()),Y,oneMinusY);

        //(1-y).*log(1-H);
        oneMinusY = oneMinusY.mul(oneMinusH);


        //-y.*log(H) - (1-y).*log(1-H);
        Core.subtract(result,oneMinusY,result);

        //sum(-y.*log(H) - (1-y).*log(1-H))
        Scalar s = Core.sumElems(result);


        //(sum(-y.*log(H) - (1-y).*log(1-H)) + (sum(mat1)+ ... + sum(matX))/2)/m
        s.val[0] = (s.val[0] + sum)/m;




        return s;
    }

    /**
     *
     *      J = 1/m*sum(sum((-y.*log(H) - ((1-y).*log(1- H))))) + 1/(2*m)*(sum(sum(mat1)) + sum(sum(mat2)));
     */

    @Override
    public Scalar JofThetaNoRegularization(Mat X, Mat Y, Mat hOfTheta, Mat... thetas) {
        if(thetas.length <2)
            return null;


        double m = Y.cols();

        Mat result =  new Mat();

        //log(H)
        Core.log(hOfTheta, result);

        Mat minusY = new Mat();

        Mat yTrans = Y.t();
        //-y
        Core.subtract(Mat.zeros(Y.rows(), Y.cols(), Y.type()), Y, minusY);


        //-y.*log(H)
        result = result.mul(minusY);


        //1-H
        Mat oneMinusH = new Mat(hOfTheta.rows(),hOfTheta.cols(),hOfTheta.type());
        Core.subtract(Mat.ones(hOfTheta.rows(), hOfTheta.cols(), hOfTheta.type()), hOfTheta, oneMinusH);

        //log(1-H)
        Core.log(oneMinusH, oneMinusH);

        //1-y
        Mat oneMinusY = new Mat();
        Core.subtract(Mat.ones(Y.rows(),Y.cols(),Y.type()),Y,oneMinusY);

        //(1-y).*log(1-H);
        oneMinusY = oneMinusY.mul(oneMinusH);


        //-y.*log(H) - (1-y).*log(1-H);
        Core.subtract(result,oneMinusY,result);



        //sum(-y.*log(H) - (1-y).*log(1-H))
        Scalar sumElems = Core.sumElems(result);

        //(1/m)*{sum(-y.*log(H) - (1-y).*log(1-H))}
        sumElems.val[0] /= m;

        return sumElems;
    }

    /**
     *  delta3 = H - y;
     *
     *   %bios numbers should be erased from system
     *
     *   helper = (Theta2'*delta3)(2:end,:);
     *
     *
     *   delta2 = helper.*sigmoidGradient(X*Theta1')';
     *
     *
     *
     *   Theta2_grad = 1/m*(delta3*a1) + 1/m * lambda_matrix_2 .* Theta2;
     *
     *   Theta1_grad = 1/m*(delta2*X) + 1/m * lambda_matrix .* Theta1;
     *
     *
     */
    @Override
    public List<Mat> grad(Mat X, Mat Y, Mat hOfTheta, double lambdaVal, Mat... thetas) {
        if(thetas.length < 2)
            return null;

        double m = Y.cols();


        Mat[] Theta_grad = new Mat[thetas.length];

        Mat[] deltas = new Mat[thetas.length];

        Mat[] aX = new Mat[thetas.length + 1];

        aX[0] = X;



        //forward propagation
        for(int index=1; index< aX.length;index++ ){

            Mat hOfLogistic = algorithm.hOfTheta(aX[index-1],thetas[index-1].t());

            aX[index] = new Mat(hOfLogistic.rows(),hOfLogistic.cols() + 1,hOfLogistic.type());
            aX[index].col(0).setTo(new Scalar(1));
            for(int i=0; i< hOfLogistic.rows(); i++){
                for(int j=0; j< hOfLogistic.cols(); j++) {
                    aX[index].put(i, j + 1, hOfLogistic.get(i, j)[0]);
                }
            }
        }

        //back propagation
        for(int index = Theta_grad.length-1;index>=0;index--){
            Theta_grad[index] = new Mat();
            if(Theta_grad.length - 1 == index){
                //Theta_grad[Theta_grad.length - 1 ] = H -y
                Core.subtract(hOfTheta, Y, Theta_grad[index]);
                deltas[index] = Theta_grad[index].clone();

            }else{
                //delta2 = (theta2'*delta3)(2:end,:)

                Core.gemm(thetas[index+1].t(), deltas[index+1], 1, new Mat(), 0, Theta_grad[index]);
                deltas[index] = Theta_grad[index].clone();
                Theta_grad[index] = Theta_grad[index].rowRange(1, Theta_grad[index].rows());

                Mat sigmoidGradient = new Mat();

                Mat hOfLogistic = algorithm.hOfTheta(aX[index], thetas[index].t());
                //(1-hOfTheta)
                Core.subtract(Mat.ones(hOfLogistic.size(), hOfLogistic.type()), hOfLogistic, sigmoidGradient);

                //(1-hOfTheta).* hOfTheta
                sigmoidGradient = hOfLogistic.mul(sigmoidGradient);


                //sigmoidGradient'
                Core.transpose(sigmoidGradient, sigmoidGradient);

                //helper.*sigmoidGradient(X*Theta1')'

                Theta_grad[index] = Theta_grad[index].mul(sigmoidGradient);
            }

            //delta[i]*aX[i]
            Core.gemm(Theta_grad[index], aX[index], 1, new Mat(), 0, Theta_grad[index]);

            Mat lambdaMatrix = Mat.ones(thetas[index].rows(), thetas[index].cols(), thetas[index].type());

            //lambda_matrix .* Thetas[i]
            Core.multiply(lambdaMatrix, new Scalar(lambdaVal),lambdaMatrix);

            Mat.zeros(lambdaMatrix.rows(),1,lambdaMatrix.type()).copyTo(lambdaMatrix.col(0));

            Mat temp = lambdaMatrix.mul(thetas[index]);

            //Theta_grad[i]*aX[i] + lambda_matrix .* Thetas[i]
            Core.add(Theta_grad[index], temp, temp);

            //(Theta_grad[i]*aX[i] + lambda_matrix .* Thetas[i])/m
            Core.divide(temp, new Scalar(m), Theta_grad[index]);


        }


        List<Mat> lista = new ArrayList<>();

        for(int i=0; i< Theta_grad.length;i++) {

            lista.add(Theta_grad[i]);

        }


        return lista;
    }

    /**
     *
     *  loop util converge{
     *      Theta[i] = Theta[i] - alpha*dJ/dTheta[i];
     *   }
     */
    @Override
    public List<Mat> gradientDescent(Mat X, Mat y, double alpha, double lambda, double convergeRatio,long maxExecutionTime,Mat... initial_theta) {
        if(initial_theta.length <2)
            return null;

        Mat[] thetas = new Mat[initial_theta.length];

        for(int i=0; i< thetas.length;i++){
            thetas[i] = initial_theta[i];
        }


        //cost in current iteration
        Scalar cost = new Scalar(0);

        //cost in last iteration
        Scalar lastCost = new Scalar(-1);
        int i=0;

        long startExecutionTime = System.currentTimeMillis();
        while(true){
            Mat hOfTheta = hOfTheta(X, thetas);

            //J = 1/m * sum(-y.*log(H) - (1-y).*log(1-H));
            cost = JofTheta(X, y, hOfTheta,lambda,thetas);


            Mat result = new Mat();

            //dJ/dTheta = grad = 1/m*(X'*(H-y))
            List<Mat> grad = grad(X, y, hOfTheta,lambda,thetas);


            for(int j = 0; j < grad.size(); j++){

                Core.multiply(grad.get(j),Scalar.all(alpha),result);

                Core.subtract(thetas[j],result,thetas[j]);
            }

            Log.i(TAG, "current cost: " + cost.val[0] + ", in iteration: " + ++i);


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
        Log.i(TAG, "minimized cost function: " + cost.val[0] + " for lambda: " + lambda);


        List<Mat> theta = new ArrayList<>();


        for(i=0;i<thetas.length;i++) {

            theta.add(thetas[i]);
        }
        return theta;
    }
}
