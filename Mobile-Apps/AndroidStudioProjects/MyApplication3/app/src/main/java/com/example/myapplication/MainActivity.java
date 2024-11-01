package com.example.myapplication;

import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;
import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }

        Button startButton = findViewById(R.id.startButton);
        downlaodButton.setOnClickListener(new View.OnClickListener(){
            @Override
            public void onClick(View v){
                setUrlToDownload();
            }
        });
    }
    private void setUrlToDownload() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(50, 40, 50, 40);

        // Create an EditText to place inside the AlertDialog
        final EditText inputUrl = new EditText(this);
        inputUrl.setHint("Url");
        layout.addView(inputUrl);

        final EditText inputType = new EditText(this);
        inputType.setHint("Desired type");
        layout.addView(inputType);

        // Build the AlertDialog
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("Downlaod by the URL");
        builder.setMessage("Type the URL below:");
        builder.setView(input);

        Python py = Python.getInstance();
        PyObject pyModule = py.getModule("DownloadMusic");

        // Set up the buttons
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                String videoUrl = inputUrl.getText().toString();
                String type = inputType.getText().toString();
                pyModule.callAttr("download_music", videoUrl, type="music");
            }
        });

        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
            }
        });

        // Show the AlertDialog
        builder.show();
    }
}
