package com.rt_rk.vzbiljic.logisticregression.util;

import android.util.Log;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

/**
 * Created by vzbiljic on 10.8.16..
 */
public class PropertiesUtil {

    private static final String TAG = "PropertiesUtil";

    private static Properties properties = new Properties();

    private static final String PROPERTY_FILE_LOCATION = "settings.properties";


    public  static final String DEBUG_KEY = "DEBUG";

    static {
        try {
            properties.load(new FileInputStream(PROPERTY_FILE_LOCATION));
        } catch (IOException e) {
            Log.v(TAG,"property file not found!");
        }
    }

    public static boolean getBooleanProperty(String key){
        try {
            return Boolean.parseBoolean((String) properties.get(key));
        }catch (NumberFormatException e){
            Log.v(TAG,"Bad boolean format of property: " + key + ", in file: " + PROPERTY_FILE_LOCATION);
            return false;
        }
    }
}
