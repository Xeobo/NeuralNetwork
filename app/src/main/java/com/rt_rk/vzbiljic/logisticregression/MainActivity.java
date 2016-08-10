package com.rt_rk.vzbiljic.logisticregression;

import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;

import com.rt_rk.vzbiljic.logisticregression.algorithm.LogisticRegression;
import com.rt_rk.vzbiljic.logisticregression.algorithm.IMachineLearningAlgorithm;
import com.rt_rk.vzbiljic.logisticregression.test.GradientCheckTest;
import com.rt_rk.vzbiljic.logisticregression.test.ITest;
import com.rt_rk.vzbiljic.logisticregression.test.MatTest;
import com.rt_rk.vzbiljic.logisticregression.test.ThetaCalculationTest;

@Deprecated
public class MainActivity extends AppCompatActivity {

    static{
        System.loadLibrary("opencv_java3");

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
            }
        });

        findViewById(R.id.front);
    }

    @Override
    public void onStart(){
        super.onStart();


    }

}
