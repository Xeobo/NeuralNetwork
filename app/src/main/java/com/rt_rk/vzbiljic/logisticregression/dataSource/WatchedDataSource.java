package com.rt_rk.vzbiljic.logisticregression.dataSource;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.util.Log;

import com.rt_rk.vzbiljic.logisticregression.test.GradientCheckTest;
import com.rt_rk.vzbiljic.logisticregression.util.MatrixPrint;
import com.rt_rk.vzbiljic.logisticregression.util.SinusoidInitMatrix;

import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.HashMap;
import java.util.List;

/**
 * Created by vzbiljic on 20.7.16..
 */
public class WatchedDataSource implements IDataSource{



    private Context context;

    private static final int DEFAULT_MAT_DATA_TYPE = CvType.CV_64F;


    private Mat X=null;
    private Mat y=null;
    private List<Mat> thetas=null;
    //hidden layer size without bios unit
    private final int HIDDEN_LAYER_SIZE;
    //input layer size without bios unit
    private final int INPUT_LAYER_SIZE = 5;

    private final int OUTPUT_LAYER_SIZE = 1;


    private static final int DEFAULT_HIDDEN_LAYER_SIZE = 10;


    private static final String TAG =
            "WatchedDataSource";

    private final static Uri WATCHED_URI = Uri.parse("content://com.iwedia.example.tvinput/watched");
    private final static Uri EPG_URI = Uri.parse("content://com.iwedia.example.tvinput/epg");

    //columns in X matrix
    private static final int BIOS = 0;
    private static final int CHANNEL_ID = 1;
    private static final int START_TIME = 2;
    private static final int END_TIME = 3;
    private static final int GENRE = 4;
    private static final int DAY_OF_THE_WEEK = 5;
    private static final int TEMP_START_TIME_MILLIS = 6;

    //the value in witch Epsilon surrounding should be initial theta values
    private static final double CONVERGE_VALUE = 4;

    public static final String COLUMN_WATCH_START_TIME_UTC_MILLIS =
            "watch_start_time_utc_millis";

    public static final String COLUMN_WATCH_END_TIME_UTC_MILLIS = "watch_end_time_utc_millis";

    public static final String COLUMN_CHANNEL_ID = "channel_id";

    public static final String COLUMN_TITLE = "title";

    public static final String COLUMN_START_TIME_UTC_MILLIS = "start_time_utc_millis";

    public static final String COLUMN_END_TIME_UTC_MILLIS = "end_time_utc_millis";

    public static final String COLUMN_DESCRIPTION = "description";

    public static final String COLUMN_INTERNAL_TUNE_PARAMS = "tune_params";

    public static final String COLUMN_INTERNAL_SESSION_TOKEN = "session_token";

    public static final String COLUMN_BROADCAST_GENRE = "broadcast_genre";

    private void init(){
        //get all EPG data
        String[] lProjection = {
                COLUMN_CHANNEL_ID,
                COLUMN_START_TIME_UTC_MILLIS,
                COLUMN_END_TIME_UTC_MILLIS,
                COLUMN_BROADCAST_GENRE
        };
        Cursor cursor = context.getContentResolver().query(
                EPG_URI,
                lProjection,
                null,
                null,
                null
        );

        if (cursor != null) {
            //data set has all columns queried from ContentProvider
            //plus one more column, day of the week
            X = new Mat(cursor.getCount(),TEMP_START_TIME_MILLIS+1, DEFAULT_MAT_DATA_TYPE);

            int index = 0;

            int channelCount = 0;
            int genreCount = 0;

            HashMap<String,Integer> genreMapping = new HashMap<>();

            double minChannel = Double.MAX_VALUE;
            double maxChannel = 0;
            //put all data in matrix while performing --feature scaling-- where possible
            //all features are scaled in range of -0.5 to 0.5
            while(cursor.moveToNext()){
                //bios column
                X.put(index,BIOS,1);

                //channel_id --not scaled--
                int channelId = cursor.getInt(cursor.getColumnIndex(COLUMN_CHANNEL_ID));
                X.put(index,CHANNEL_ID,channelId);

                if(channelId > maxChannel){
                    maxChannel = channelId;
                }

                if(channelId < minChannel){
                    minChannel = channelId;
                }

                //start Time --scaled--
                long startTime = cursor.getLong(cursor.getColumnIndex(COLUMN_START_TIME_UTC_MILLIS));
                GregorianCalendar gc = new GregorianCalendar();
                gc.setTimeInMillis(startTime);
                X.put(index,START_TIME, (60*gc.get(Calendar.HOUR) + gc.get(Calendar.MINUTE)-720)/1440.);

                //end Time --scaled--
                long endTime = cursor.getLong(cursor.getColumnIndex(COLUMN_START_TIME_UTC_MILLIS));
                gc.setTimeInMillis(endTime);
                X.put(index,END_TIME, (60*gc.get(Calendar.HOUR) + gc.get(Calendar.MINUTE)-720)/1440.);

                //genre --not scaled--
                Integer genre = genreMapping.get(cursor.getString(cursor.getColumnIndex(COLUMN_BROADCAST_GENRE)));
                //if genre doesn't exists put it in map
                if(null == genre){
                    genre = genreCount;

                    genreMapping.put(cursor.getString(cursor.getColumnIndex(COLUMN_BROADCAST_GENRE)),genreCount++);


                }
                X.put(index,GENRE,genre);

                //day of the week --scaled--
                //take it as day of the week of start Time
                gc.setTimeInMillis(startTime);
                X.put(index,DAY_OF_THE_WEEK,(gc.get(Calendar.DAY_OF_WEEK)-3.5)/7);


                //last column is startTimeInMillis and is used only to init Y matrix properly
                //after that last column should be deleted
                X.put(index,TEMP_START_TIME_MILLIS,startTime);

                index++;
            }

            cursor.close();

            //scale channel_id and genre
            for(int i=0;i<X.rows();i++){
                X.put(i,CHANNEL_ID,(X.get(i,CHANNEL_ID)[0] - (maxChannel + minChannel)/2)/maxChannel);
                X.put(i,GENRE,(X.get(i,3)[0] - genreCount/2)/genreCount);
            }


            //get watched data
            lProjection = new String[]{
                    COLUMN_CHANNEL_ID,
                    COLUMN_START_TIME_UTC_MILLIS
            };



            cursor = context.getContentResolver().query(
                    WATCHED_URI,
                    lProjection,
                    null,
                    null,
                    null
            );

            if(null!= cursor ) {
                //set y to be all zeros
                y = Mat.zeros( 1,X.rows(), X.type());


                while (cursor.moveToNext()){
                    long startTime = cursor.getLong(cursor.getColumnIndex(COLUMN_START_TIME_UTC_MILLIS));

                    boolean found = false;
                    //search positive examples in all examples matrix
                    for(int i=0;i<X.rows();i++){
                        if(X.get(i,TEMP_START_TIME_MILLIS)[0] == startTime){
                            found = true;
                            y.put(0,i,1);
                            break;
                        }
                    }
                    //unexpected. If happens report warning
                    if(!found){
                        Log.wtf(TAG,"Positive training example not found in all examples matrix, " +
                                "for input values, startTimeMillis: " + startTime + " and channelID: "
                                + cursor.getInt(cursor.getColumnIndex(COLUMN_CHANNEL_ID)) );
                    }


                }

                cursor.close();



                //after y matrix init erase last column of X matrix
                X = X.colRange(BIOS,TEMP_START_TIME_MILLIS);


            }else{
                Log.i(TAG,"null cursor!");
            }
        }else{
            Log.i(TAG,"null cursor!");
        }
    }

    public WatchedDataSource(Context context){
        this(context,DEFAULT_HIDDEN_LAYER_SIZE);
    }

    public WatchedDataSource(Context context,int hds){
        this.context = context;
        HIDDEN_LAYER_SIZE = hds;
    }

    @Override
    public Mat getX() {
        //if data is already loaded, just return it.
        if(null != X)
            return X;

        init();
        return X;
    }

    @Override
    public Mat getY() {
        //if data is already loaded, just return it.
        if(null != y)
            return y;

        init();
        return y;
    }

    @Override
    public List<Mat> getThetas() {
        //if data is already loaded, just return it.
        if(null == thetas) {
            thetas = new ArrayList<>();

            //init theta to small different values around 0
            Mat theta1 = SinusoidInitMatrix.debugInitializeWeights(HIDDEN_LAYER_SIZE,INPUT_LAYER_SIZE);
            Mat theta2 = SinusoidInitMatrix.debugInitializeWeights(OUTPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE);


            for(int i=0;i< theta1.rows();i++)
                for(int j=0;j< theta1.cols();j++)
                    theta1.put(i,j,theta1.get(i,j)[0] + CONVERGE_VALUE);

            for(int i=0;i< theta2.rows();i++)
                for(int j=0;j< theta2.cols();j++)
                    theta2.put(i,j,theta2.get(i,j)[0] + CONVERGE_VALUE);

            thetas.add(theta1);
            thetas.add(theta2);
        }
        return thetas;
    }
}
