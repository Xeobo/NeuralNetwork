package com.rt_rk.vzbiljic.logisticregression.util;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Properties;

/**
 * Created by vzbiljic on 10.8.16..
 */
public class PropertiesUtil {

    private static final String TAG = "PropertiesUtil";


    private static Properties properties = new Properties();

    private static final String PROPERTY_FILE_LOCATION = "/settings.properties";

    private static final String DEFAULT_DATE_FORMAT = "yyyyMMdd";

    private static final String DATE_FORMAT_KEY = "DATE_FORMAT";


    public  static final String DEBUG_BOOLEAN_KEY = "DEBUG";
    //format YYYYMMDD
    public static final String DEBUG_START_TIME_KEY = "START_DATE";
    public static final String DEBUG_END_TIME_KEY = "END_DATE";

    public static final String WINDOW_SIZE_KEY = "WINDOW_SIZE";

    public static void init(Context context){
        try {

            properties.load(new FileInputStream(context.getApplicationContext().getFilesDir() + PROPERTY_FILE_LOCATION));
        } catch (IOException e) {
            Log.w(TAG,"property file not found!");
        }
    }


    private  static void logNotFound(String key){
        Log.w(TAG,"Property: " + key + " not found. Default value set instead. ");
    }
    public static boolean getBooleanProperty(String key){
        try {
            return Boolean.parseBoolean((String) properties.get(key));
        }catch (NumberFormatException e){
            Log.w(TAG,"Bad boolean format of property: " + key + ", in file: " + PROPERTY_FILE_LOCATION);
            return false;
        }
    }

    public static int getIntProperty(String key){
        try {
            return Integer.parseInt( properties.getProperty(key));
        }catch (NumberFormatException e){
            Log.w(TAG,"Bad boolean format of property: " + key + ", in file: " + PROPERTY_FILE_LOCATION);
            return 0;
        }
    }



    public static long getDateToSQLDateProperty(String key) {
        String dateFormat = properties.getProperty(DATE_FORMAT_KEY);

        if(null == dateFormat){
            logNotFound(DATE_FORMAT_KEY);
            dateFormat = DEFAULT_DATE_FORMAT;
        }
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(dateFormat);

        String date = properties.getProperty(key);

        long value;
        if(null == date){
            logNotFound(key);

            value = System.currentTimeMillis();
        }else{
            try {

                value = simpleDateFormat.parse(date).getTime();

            } catch (ParseException e) {
                Log.v(TAG, "Wrong date format! Current date format is : " + dateFormat +" . " +
                        "Selected non debug time instead.");

                value =  System.currentTimeMillis();
            }
        }

        return value;
    }
}
