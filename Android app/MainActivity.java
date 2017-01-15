package com.example.raviteja.cameratut;


import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity {


    static {
        //System.loadLibrary("opencv_java");
        if(!OpenCVLoader.initDebug()){
            Log.i("opencv","opencv initialization failed");
        }else{
            Log.i("opencv","opencv initialization successful");
        }
    }


    private static final int ACTIVITY_START_CAMERA_APP = 0;
    private ImageView mPhotoCapturedImageView;
    private String mImageFileLocation = "";
    private Bitmap photoSmall;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camara_intent);

        mPhotoCapturedImageView = (ImageView) findViewById(R.id.capturePhotoImageView);
    }

    File createImageFile()throws IOException{
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "IMAGE_" + timeStamp + "_";
        File storageDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);

        File image = File.createTempFile(imageFileName,".jpg",storageDirectory);
        mImageFileLocation = image.getAbsolutePath();
        return image;
    }

    // ** camera functionality **
    public void takePhoto(View view){
        Intent callCameraApplicationIntent = new Intent();
        callCameraApplicationIntent.setAction(MediaStore.ACTION_IMAGE_CAPTURE);

        File photo = null;
        try{
            photo = createImageFile();
        }catch (IOException e){
            e.printStackTrace();
        }
        callCameraApplicationIntent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(photo));
        startActivityForResult(callCameraApplicationIntent,ACTIVITY_START_CAMERA_APP);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        if(requestCode == ACTIVITY_START_CAMERA_APP && resultCode == RESULT_OK){
            //Bundle extras = data.getExtras();
            //Bitmap photoCapturedBitmap = (Bitmap) extras.get("data");
            //mPhotoCapturedImageView.setImageBitmap(photoCapturedBitmap);
            //Bitmap photoCapturedBitmap = BitmapFactory.decodeFile(mImageFileLocation);
            //mPhotoCapturedImageView.setImageBitmap(photoCapturedBitmap);
            reduceImage();
        }
    }

    void reduceImage(){
        Log.i("opencv","reached reduce");
        int vWidth = mPhotoCapturedImageView.getWidth();
        int vHeight = mPhotoCapturedImageView.getHeight();

        BitmapFactory.Options bmOps = new BitmapFactory.Options();
        // ** dont delete this
        bmOps.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(mImageFileLocation, bmOps);

        int camWidth = bmOps.outWidth;
        int camHeight = bmOps.outHeight;

        int scale = Math.max(camWidth/vWidth,camHeight/vHeight);
        bmOps.inSampleSize = scale;
        bmOps.inJustDecodeBounds = false;
        photoSmall = BitmapFactory.decodeFile(mImageFileLocation,bmOps);
        showImage();
    }

    // **** Functions for various options ***

    public boolean showImage(){
        mPhotoCapturedImageView.setImageBitmap(photoSmall);
        return true;
    }
    public boolean showGrayImage(){
        Bitmap newBmp = photoSmall.copy(photoSmall.getConfig(), true);
        Mat imgMAT = new Mat (photoSmall.getHeight(), photoSmall.getWidth(), CvType.CV_8UC1);;
        Utils.bitmapToMat(photoSmall, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_RGB2GRAY);
        Utils.matToBitmap(imgMAT, newBmp);
        mPhotoCapturedImageView.setImageBitmap(newBmp);
        return true;
    }

    public boolean showFeatures(){
        FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);
        Bitmap newBmp = photoSmall.copy(photoSmall.getConfig(), true);
        Mat imgMAT = new Mat (photoSmall.getHeight(), photoSmall.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(photoSmall, imgMAT);
        Imgproc.cvtColor(imgMAT, imgMAT, Imgproc.COLOR_RGB2GRAY);
        MatOfKeyPoint keyPoints = new MatOfKeyPoint();
        detector.detect(imgMAT, keyPoints);
        Features2d.drawKeypoints(imgMAT, keyPoints, imgMAT);
        Utils.matToBitmap(imgMAT, newBmp);
        mPhotoCapturedImageView.setImageBitmap(newBmp);
        return true;
    }

    public void removeBackground(){
        Mat thresholdImg = new Mat (photoSmall.getHeight(), photoSmall.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(photoSmall, thresholdImg);
        Mat hsvImg = new Mat();
        List<Mat> hsvPlanes = new ArrayList<Mat>();
        Imgproc.cvtColor(thresholdImg, hsvImg, Imgproc.COLOR_RGB2HSV);
        Core.split(hsvImg,hsvPlanes);

        MatOfInt histSize = new MatOfInt(180);
        double average = 0.0;
        Mat hist_hue = new Mat();
        List<Mat> hue = new ArrayList<>();
        hue.add(hsvPlanes.get(0));
        Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));
        for (int h = 0; h < 180; h++)
            average += (hist_hue.get(h, 0)[0] * h);
        average = average / hsvImg.size().height / hsvImg.size().width;
        double threshValue = average;
        Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);
        Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));
        Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
        Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);
        Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);

        Bitmap bgBmp = BitmapFactory.decodeResource(getResources(),R.drawable.logo);
        Mat bgMat = new Mat (bgBmp.getHeight(), bgBmp.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(bgBmp, bgMat);

        Mat foreground = new Mat(thresholdImg.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
        Size sz = new Size(thresholdImg.width(),thresholdImg.height());
        Imgproc.resize( bgMat, foreground, sz );

        Mat frame = new Mat (photoSmall.getHeight(), photoSmall.getWidth(), CvType.CV_8UC1);
        Utils.bitmapToMat(photoSmall, frame);
        frame.copyTo(foreground, thresholdImg);

        Bitmap newBmp = photoSmall.copy(photoSmall.getConfig(), true);
        Utils.matToBitmap(foreground, newBmp);
        mPhotoCapturedImageView.setImageBitmap(newBmp);


    }


    // ****   OPTIONS  ****
    private MenuItem grayItem;
    private MenuItem featuresItem;
    private MenuItem camItem;
    private MenuItem removebgItem;

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        camItem = menu.add("Preview Original");
        grayItem = menu.add("Preview Gray");
        featuresItem = menu.add("Find features");
        removebgItem = menu.add("GT Badge");
        return true;
    }
    public boolean onOptionsItemSelected(MenuItem item) {

        if (item == camItem) {
            showImage();
        } else if(item == grayItem) {
            showGrayImage();
        } else if (item == featuresItem) {
            showFeatures();
        } else if (item == removebgItem) {
            removeBackground();
        }

        return true;
    }

}
