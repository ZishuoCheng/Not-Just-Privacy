package al.omid.tfdroid;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Button;
import android.view.View;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    private static final String MODEL_FILE = "file:///android_asset/LargeNetwork.pb";
    private static final String INPUT_NODE = "input";
    private static final String OUTPUT_NODE = "output";

    private static final int INPUT_SIZE = 32;

    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);


        final Button button = (Button) findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                final EditText editNum3 = (EditText) findViewById(R.id.editNum3);

                float num3 = Float.parseFloat(editNum3.getText().toString());

                float[] resu;
                int numClasses = (int) inferenceInterface.graph().operation(OUTPUT_NODE).output(0).shape().size(1);
                resu = new float[numClasses];

                Bitmap bitmap=BitmapFactory.decodeResource(getResources(), R.drawable.svhn);
                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                int[] intValues;
                intValues = new int[INPUT_SIZE * INPUT_SIZE];
                bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
                float[] floatValues;
                floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];

                for (int i = 0; i < intValues.length; ++i) {
                    final int val = intValues[i];
                    floatValues[i * 3 + 0] = (val >> 16) & 0xFF;
                    floatValues[i * 3 + 1] = (val >> 8) & 0xFF;
                    floatValues[i * 3 + 2] = val & 0xFF;
                }

                long begin = System.currentTimeMillis();

                for (int i = 0; i < num3; i++) {
                    inferenceInterface.fillNodeFloat(INPUT_NODE, new int[]{1, INPUT_SIZE, INPUT_SIZE, 3}, floatValues);

                    inferenceInterface.runInference(new String[] {OUTPUT_NODE});

                    inferenceInterface.readNodeFloat(OUTPUT_NODE, resu);
                }

                long duration = System.currentTimeMillis() - begin;

                final TextView textViewR = (TextView) findViewById(R.id.txtViewResult);

                textViewR.setText(Float.toString(resu[0]) + ", " + Long.toString(duration));
            }
        });

    }
}
